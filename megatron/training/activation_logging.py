# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Forward activation logging using forward hooks."""

from collections import defaultdict
import json
import logging
import os
import re
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.router import Router

from .checkpointing import save_grads
from .diagnostic_layer_selection import make_layer_name_filter
from .utils import unwrap_model


def _discover_te_types():
    """Discover available Transformer Engine layer types.

    Returns (all_types, grouped_types) where grouped_types is the subset of
    TEGroupedLinear variants used for tokens-per-expert capture.
    """
    all_types = []
    grouped_types = []

    try:
        from megatron.core.extensions.transformer_engine import (
            TELinear,
            TENorm,
            TEColumnParallelLinear,
            TERowParallelLinear,
            TELayerNormColumnParallelLinear,
        )

        all_types.extend(
            [
                TELinear,
                TENorm,
                TEColumnParallelLinear,
                TERowParallelLinear,
                TELayerNormColumnParallelLinear,
            ]
        )
    except ImportError:
        pass

    try:
        from megatron.core.extensions.transformer_engine import (
            TEGroupedLinear,
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )

        if TEGroupedLinear is not None:
            grouped = [
                TEGroupedLinear,
                TEColumnParallelGroupedLinear,
                TERowParallelGroupedLinear,
            ]
            all_types.extend(grouped)
            grouped_types.extend(grouped)
    except ImportError:
        pass

    return tuple(all_types), tuple(grouped_types)


_TE_TYPES, _GROUPED_LINEAR_TYPES = _discover_te_types()


def _discover_norm_and_layer_types():
    types = [nn.LayerNorm]
    rms_norm = getattr(nn, "RMSNorm", None)
    if rms_norm is not None:
        types.append(rms_norm)

    try:
        from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

        types.append(FusedLayerNorm)
    except ImportError:
        pass

    try:
        from megatron.core.extensions.transformer_engine import TENorm

        types.append(TENorm)
    except ImportError:
        pass

    try:
        import transformer_engine.pytorch as te

        for name in ("LayerNorm", "RMSNorm", "TEFusedResidualRMSNorm"):
            layer_type = getattr(te, name, None)
            if layer_type is not None:
                types.append(layer_type)
    except ImportError:
        pass

    try:
        from megatron.core.transformer.transformer_layer import TransformerLayer

        types.append(TransformerLayer)
    except ImportError:
        pass

    try:
        from megatron.core.ssm.mamba_layer import MambaLayer

        types.append(MambaLayer)
    except ImportError:
        pass

    return tuple(types)


LOGGABLE_TYPES = (
    nn.Linear,
    nn.Embedding,
    ColumnParallelLinear,
    RowParallelLinear,
    Router,
    *_TE_TYPES,
    *_discover_norm_and_layer_types(),
)
LINEAR_TYPES = LOGGABLE_TYPES


def _iter_tensors(value, prefix: str):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for idx, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}{idx}")


def _rms(finite: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(finite * finite))


def _abs_mean(finite: torch.Tensor) -> torch.Tensor:
    return finite.abs().mean()


def _std(finite: torch.Tensor) -> torch.Tensor:
    return finite.std(unbiased=False) if finite.numel() > 1 else torch.tensor(0.0)


def _abs_max(finite: torch.Tensor) -> torch.Tensor:
    return finite.abs().max()


def _abs_min(finite: torch.Tensor) -> torch.Tensor:
    return finite.abs().min()


_STAT_FNS = {
    "rms": _rms,
    "abs_mean": _abs_mean,
    "std": _std,
    "abs_max": _abs_max,
    "abs_min": _abs_min,
}


def _resolve_stat_fn():
    """Pick the per-tensor summary statistic.

    Honors ``MEGATRON_RESIDUAL_LOG_STAT`` (default ``rms``). Supported
    values: rms, abs_mean, std, abs_max, abs_min. The chosen statistic
    is computed once at hook time and stored as a 0-d CPU tensor, so
    the on-disk file format and any downstream postprocessor that calls
    a ``rms()``-like helper on the saved tensor stay backward
    compatible — ``rms()`` of a 0-d tensor returns ``|x|``, which is
    the statistic value we stored.
    """
    name = os.environ.get("MEGATRON_RESIDUAL_LOG_STAT", "rms").strip().lower()
    fn = _STAT_FNS.get(name)
    if fn is None:
        logger.warning(
            "MEGATRON_RESIDUAL_LOG_STAT=%r is not recognized; supported: %s. Falling back to rms.",
            name,
            ", ".join(sorted(_STAT_FNS)),
        )
        name = "rms"
        fn = _rms
    return name, fn


_STAT_NAME, _STAT_FN = _resolve_stat_fn()
_SUMMARY_CHUNK_NUMEL = 16 * 1024 * 1024


def _streaming_finite_summary(
    tensor: torch.Tensor,
    statistic: str,
    chunk_numel: int = _SUMMARY_CHUNK_NUMEL,
) -> torch.Tensor:
    """Compute a finite-only statistic without materializing a full-size copy.

    Activation tensors can contain billions of elements.  Casting the whole
    tensor to fp32 and boolean-indexing all finite values can require multiple
    additional GiB at a diagnostic boundary.  Reduce bounded fp32 views into
    scalar fp64 accumulators instead.
    """
    if chunk_numel < 1:
        raise ValueError("chunk_numel must be positive")

    flat = tensor.detach().reshape(-1)
    device = flat.device
    count = torch.zeros((), dtype=torch.int64, device=device)
    total = torch.zeros((), dtype=torch.float64, device=device)
    total_abs = torch.zeros((), dtype=torch.float64, device=device)
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    maximum = torch.full((), -torch.inf, dtype=torch.float32, device=device)
    minimum = torch.full((), torch.inf, dtype=torch.float32, device=device)

    for start in range(0, flat.numel(), chunk_numel):
        chunk = flat[start : start + chunk_numel].float()
        finite = torch.isfinite(chunk)
        count += finite.sum(dtype=torch.int64)
        safe = torch.where(finite, chunk, 0.0)

        if statistic in {"rms", "std"}:
            total_sq += (safe * safe).sum(dtype=torch.float64)
        if statistic == "std":
            total += safe.sum(dtype=torch.float64)
        elif statistic == "abs_mean":
            total_abs += safe.abs().sum(dtype=torch.float64)
        elif statistic == "abs_max":
            maximum = torch.maximum(
                maximum,
                torch.where(finite, chunk.abs(), -torch.inf).max(),
            )
        elif statistic == "abs_min":
            minimum = torch.minimum(
                minimum,
                torch.where(finite, chunk.abs(), torch.inf).min(),
            )

    count_value = int(count.item())
    if count_value == 0:
        return torch.tensor(float("nan"))

    denominator = count.to(dtype=torch.float64)
    if statistic == "rms":
        result = torch.sqrt(total_sq / denominator)
    elif statistic == "abs_mean":
        result = total_abs / denominator
    elif statistic == "std":
        if count_value == 1:
            result = torch.zeros((), dtype=torch.float64, device=device)
        else:
            variance = total_sq / denominator - (total / denominator).square()
            result = torch.sqrt(variance.clamp_min(0.0))
    elif statistic == "abs_max":
        result = maximum
    elif statistic == "abs_min":
        result = minimum
    else:
        raise ValueError(f"unsupported activation statistic: {statistic}")
    return result.float().cpu()


def _rms_summary(tensor: torch.Tensor) -> torch.Tensor:
    """Scalar summary statistic of *tensor* on its current device,
    returned as a 0-d CPU tensor.

    Replaces ``tensor.detach().cpu()`` in the activation hook to keep
    saved-state files small. Saving full tensors produces multi-GB ``.pth``
    files at every save interval (issue #10: 326 GB observed for a single
    8-layer MoE rung at 30 iters / log-every-3); a 0-d scalar is
    several orders of magnitude smaller and is the only statistic any of
    our downstream analyses consume. The plotter
    ``analysis/postprocess/visualize_norms.py`` is unchanged because its
    ``rms()`` of a 0-d tensor returns ``|x|`` == the value stored here.

    Name kept for backward compatibility with call sites; the actual
    statistic is governed by ``MEGATRON_RESIDUAL_LOG_STAT`` (default
    ``rms``).
    """
    return _streaming_finite_summary(tensor, _STAT_NAME)


def _update_streaming_scalar(
    state: dict, counts: dict, chunk_name: str, key: str, value
) -> None:
    """Update a low-memory scalar mean for one diagnostic key.

    Hooks fire once per microbatch. Keep only a running scalar average on CPU so
    activation/dgrad logs represent all observed microbatches without retaining
    tensors or per-microbatch records.
    """
    try:
        value = value.detach().float().cpu()
    except AttributeError:
        value = torch.tensor(float(value))
    previous = state[chunk_name].get(key)
    count = counts[chunk_name].get(key, 0)
    if previous is None:
        state[chunk_name][key] = value
    else:
        state[chunk_name][key] = previous + (value - previous) / (count + 1)
    counts[chunk_name][key] = count + 1


def _parse_tpe_module_name(module_name: str) -> Tuple[str, int | None, int] | None:
    """Parse a TPE-eligible module name into ``(block, mtp_idx, layer)``.

    Returns ``None`` if *module_name* matches neither the decoder nor the MTP pattern.

    Examples::

        decoder.layers.3.mlp.experts.linear_fc1                       -> ("decoder", None, 3)
        mtp.layers.0.mtp_model_layer.layers.1.mlp.experts.linear_fc1  -> ("mtp", 0, 1)
    """
    if m := re.fullmatch(
        r"decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc1", module_name
    ):
        return "decoder", None, int(m.group(1))
    if m := re.fullmatch(
        r"mtp\.layers\.(\d+)\.mtp_model_layer\.layers\.(\d+)\.mlp\.experts\.linear_fc1",
        module_name,
    ):
        return "mtp", int(m.group(1)), int(m.group(2))
    return None


def _register_hooks(model, module_types, hook_factory, *, name_filter=None):
    """Walk *model* and register a forward hook on every module matching *module_types*.

    Args:
        model: Iterable of model chunks (possibly wrapped).
        module_types: Tuple of types to match via ``isinstance``.
        hook_factory: ``(model_chunk_name, module_name) -> hook_fn``.
        name_filter: Optional ``str -> bool`` predicate on the module name.

    Returns:
        List of hook handles.
    """
    handles = []
    for model_chunk_id, model_chunk in enumerate(model):
        model_chunk_name = f"model_chunk{model_chunk_id}"
        unwrapped = unwrap_model(model_chunk)
        for module_name, module in unwrapped.named_modules():
            if isinstance(module, module_types) and (
                name_filter is None or name_filter(module_name)
            ):
                hook_fn = hook_factory(model_chunk_name, module_name)
                if hook_fn is None:
                    continue
                handle = module.register_forward_hook(hook_fn, with_kwargs=True)
                handles.append(handle)
    return handles


class ActivationLogger:
    """Captures and saves forward activations using forward hooks.

    Manages two independent hook sets:

    - **Full activation hooks** capture all inputs / outputs / kwargs for every
      ``LINEAR_TYPES`` module.
    - **Tokens-per-expert (TPE) hooks** are lightweight hooks that only capture
      the tokens-per-expert routing metadata from MoE.
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir

        # Full activation state.
        self._activations_state_dict: defaultdict = defaultdict(dict)
        self._activation_counts: defaultdict = defaultdict(dict)
        self._activation_hooks: List[torch.utils.hooks.RemovableHook] = []

        # Tokens-per-expert state: per-microbatch token counts.  Decoder entries
        # are keyed by ``layer``; MTP entries by ``(mtp_idx, inner_layer)``.
        self._decoder_tpe_records: dict[int, list[list[int]]] = defaultdict(list)
        self._mtp_tpe_records: dict[Tuple[int, int], list[list[int]]] = defaultdict(
            list
        )
        self._tpe_hooks: List[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Full activation hooks
    # ------------------------------------------------------------------

    def _make_activation_hook(
        self, model_chunk_name: str, module_name: str
    ) -> Callable:
        """Forward hook that captures all inputs, outputs and kwargs."""
        sd = self._activations_state_dict
        counts = self._activation_counts

        def hook(_, args, kwargs, output):
            input_tuple = args if isinstance(args, tuple) else (args,)
            for idx, inp in enumerate(input_tuple):
                if not isinstance(inp, torch.Tensor):
                    continue
                key = f"{module_name}/input{idx}"
                _update_streaming_scalar(
                    sd, counts, model_chunk_name, key, _rms_summary(inp)
                )
            output_tuple = output if isinstance(output, tuple) else (output,)
            for idx, output_value in enumerate(output_tuple):
                for suffix, out in _iter_tensors(output_value, f"output{idx}"):
                    key = f"{module_name}/{suffix}"
                    _update_streaming_scalar(
                        sd, counts, model_chunk_name, key, _rms_summary(out)
                    )
            for kwarg_key, kwarg_value in kwargs.items():
                for suffix, tensor in _iter_tensors(kwarg_value, kwarg_key):
                    key = f"{module_name}/{suffix}"
                    _update_streaming_scalar(
                        sd, counts, model_chunk_name, key, _rms_summary(tensor)
                    )

        return hook

    def register_activation_hooks(self, model, args=None):
        assert not self._activation_hooks
        name_filter = make_layer_name_filter(model, args, stream="diagnostic")
        self._activation_hooks = _register_hooks(
            model,
            LINEAR_TYPES,
            self._make_activation_hook,
            name_filter=name_filter,
        )

    def remove_activation_hooks(self):
        for hook in self._activation_hooks:
            hook.remove()
        self._activation_hooks.clear()

    def save_activations(self, iteration: int):
        if not self._activations_state_dict:
            return
        save_grads(
            self._save_dir, self._activations_state_dict, iteration, "activations"
        )
        self._maybe_log_wandb(
            self._activations_state_dict, iteration, "act"
        )  # CHAWKINS-WANDB-PER-TENSOR
        self._activations_state_dict.clear()
        self._activation_counts.clear()

    # ------------------------------------------------------------------
    # Live W&B push of per-tensor RMS scalars. # CHAWKINS-WANDB-PER-TENSOR
    # Mirrors dead_neuron_logging.DeadNeuronLogger._maybe_log_wandb:
    # shares Megatron's active ``wandb.run`` (initialised by Megatron
    # when ``wandb_project`` is set), namespaces keys under
    # ``<prefix>/<chunk>/<module>/<io>``, and swallows any logging-side
    # failure so a wandb hiccup never kills training. State dict values
    # are already 0-d RMS scalars on CPU (issue10 commit 24b50f1).
    # ------------------------------------------------------------------

    def _maybe_log_wandb(self, state, iteration: int, prefix: str) -> None:
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if getattr(wandb, "run", None) is None:
            return
        scalars: dict[str, float] = {}
        for chunk_name, mods in state.items():
            for key, tensor in mods.items():
                try:
                    val = float(tensor)
                except Exception:
                    continue
                if val != val:  # NaN filter
                    continue
                scalars[f"{prefix}/{chunk_name}/{key}"] = val
        if scalars:
            try:
                wandb.log(scalars, step=iteration)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Tokens-per-expert hooks
    # ------------------------------------------------------------------

    def _make_tpe_hook(
        self, _model_chunk_name: str, module_name: str
    ) -> Callable | None:
        """Forward hook that captures only the non-Tensor ``input1`` (tokens_per_expert).

        Attaches to main decoder MoE layers
        (``decoder.layers.<N>.mlp.experts.linear_fc1``) and MTP MoE layers
        (``mtp.layers.<N>.mtp_model_layer.layers.<M>.mlp.experts.linear_fc1``).
        Returns ``None`` (and logs a warning) for any other module name.
        """
        parsed = _parse_tpe_module_name(module_name)
        if parsed is None:
            logger.warning(
                "Cannot extract layer number from module name: %r — "
                "skipping tokens-per-expert hook for this module",
                module_name,
            )
            return None
        block, mtp_idx, layer = parsed
        if block == "decoder":
            records, key = self._decoder_tpe_records, layer
        else:
            records, key = self._mtp_tpe_records, (mtp_idx, layer)

        def hook(_, args, kwargs, output):
            input_tuple = args if isinstance(args, tuple) else (args,)
            if len(input_tuple) > 1 and input_tuple[1] is not None:
                inp = input_tuple[1]
                if not isinstance(inp, torch.Tensor):
                    records[key].append(list(inp))

        return hook

    def register_tpe_hooks(self, model):
        assert not self._tpe_hooks
        self._tpe_hooks = _register_hooks(
            model,
            _GROUPED_LINEAR_TYPES,
            self._make_tpe_hook,
            name_filter=lambda name: name.endswith("linear_fc1"),
        )

    def remove_tpe_hooks(self):
        for hook in self._tpe_hooks:
            hook.remove()
        self._tpe_hooks.clear()

    def save_tpe(self, iteration: int):
        """Append captured tokens-per-expert records as JSON Lines.

        Each rank writes to its own file under ``{save_dir}/tokens_per_expert/``,
        e.g. ``rank0.jsonl``, ``rank1.jsonl``.  Each line is a JSON object; the
        ``mtp_idx`` field is present only for MTP entries::

            {"iter": 100, "block": "decoder", "layer": 3, "tpe": [[128, 64], [96, 80]]}
            {"iter": 100, "block": "mtp", "mtp_idx": 0, "layer": 1, "tpe": [[50, 50]]}
        """
        if not self._decoder_tpe_records and not self._mtp_tpe_records:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        tpe_dir = os.path.join(self._save_dir, "tokens_per_expert")
        os.makedirs(tpe_dir, exist_ok=True)
        filepath = os.path.join(tpe_dir, f"rank{rank}.jsonl")

        lines = []
        for layer, microbatches in sorted(self._decoder_tpe_records.items()):
            lines.append(
                json.dumps(
                    {
                        "iter": iteration,
                        "block": "decoder",
                        "layer": layer,
                        "tpe": microbatches,
                    }
                )
                + "\n"
            )
        for (mtp_idx, layer), microbatches in sorted(self._mtp_tpe_records.items()):
            lines.append(
                json.dumps(
                    {
                        "iter": iteration,
                        "block": "mtp",
                        "mtp_idx": mtp_idx,
                        "layer": layer,
                        "tpe": microbatches,
                    }
                )
                + "\n"
            )

        with open(filepath, "a") as f:
            f.writelines(lines)
        self._decoder_tpe_records.clear()
        self._mtp_tpe_records.clear()


_LOGGER: ActivationLogger | None = None


def _get_logger(save_dir: str) -> ActivationLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = ActivationLogger(save_dir)
    return _LOGGER


def _require_logger() -> ActivationLogger:
    assert _LOGGER is not None, "No ActivationLogger has been initialised"
    return _LOGGER


# -- Full activation logging -------------------------------------------


def enable_activation_logging(model: torch.nn.Module, save_dir: str, args=None):
    _get_logger(save_dir).register_activation_hooks(model, args)


def disable_activation_logging():
    _require_logger().remove_activation_hooks()


def save_activations(iteration: int):
    _require_logger().save_activations(iteration)


# -- Tokens-per-expert logging ----------------------------------------


def enable_tokens_per_expert_logging(model: torch.nn.Module, save_dir: str):
    _get_logger(save_dir).register_tpe_hooks(model)


def disable_tokens_per_expert_logging():
    _require_logger().remove_tpe_hooks()


def save_tokens_per_expert(iteration: int):
    _require_logger().save_tpe(iteration)
