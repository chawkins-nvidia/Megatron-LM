# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

"""Dead-neuron logging — per-feature activation + dgrad accumulators.

Two complementary metrics, both keyed on the FFN hidden ("neuron") axis:

1. **Activation variance over a full global batch.** Hook the
   ``mlp.linear_fc2`` module and accumulate per-feature ``count, sum,
   sum_sq`` over its ``input0`` tensor across every microbatch. The
   post-activation tensor's per-neuron variance is ~0 for a saturated /
   stuck neuron (e.g. ReLU/squared-relu always-negative pre-activation).

2. **Pre-activation gradient RMS over a full global batch.** Hook the
   ``mlp.linear_fc1`` module and capture the gradient flowing into its
   ``output`` tensor via ``output.register_hook``. This grad is what
   remains *after* the activation function's backward; for a dead unit
   under a ReLU-family activation it is identically zero.

Together they let post-processing flag neurons as dead by either:
   ``act_var < var_thresh``  OR
   ``grad_rms < grad_thresh``

Storage is small: one 1-D tensor of FFN hidden dim per (layer, stat) per
save. For r=4 dense (8 layers, ffn_hidden ≈ 2.5k), a save is ~80 KB.

The hook semantics match ``activation_logging`` and ``dgrad_logging`` —
accumulators carry state across microbatches between
``save_dead_neuron_stats`` calls, and ``save_grads`` writes the reduced
state to ``{save_dir}/dead_neurons/iter_*/``.
"""

from collections import defaultdict
import logging
import os
from typing import Callable, Iterable, Optional

import torch

from megatron.training.activation_logging import LOGGABLE_TYPES  # noqa: F401

from .checkpointing import save_grads
from .utils import unwrap_model

logger = logging.getLogger(__name__)


def _flatten_features(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape an n-D tensor to ``[N, features]`` where features = last dim."""
    if tensor.dim() == 0 or tensor.numel() == 0:
        return tensor.new_zeros((0, 0))
    return tensor.reshape(-1, tensor.shape[-1])


def _default_act_filter(name: str, _module: torch.nn.Module) -> bool:
    """Hook the ``mlp.linear_fc2`` module per transformer block. Its
    ``input0`` is the post-activation tensor whose feature axis is FFN
    hidden dim — the natural "neuron" axis for variance-based dead-unit
    detection. Also covers the MoE expert path.
    """
    return name.endswith(".mlp.linear_fc2") or (
        ".mlp.experts." in name and name.endswith(".linear_fc2")
    )


def _default_grad_filter(name: str, _module: torch.nn.Module) -> bool:
    """Hook ``mlp.linear_fc1`` so the gradient flowing into its
    ``output`` (the pre-activation) can be captured *after* the
    activation backward has applied. The activation backward kills the
    gradient at saturated units, so per-feature RMS of this grad is the
    dead-neuron signal complementary to activation variance.
    """
    return name.endswith(".mlp.linear_fc1") or (
        ".mlp.experts." in name and name.endswith(".linear_fc1")
    )


class DeadNeuronLogger:
    """Per-feature accumulator for activation variance and pre-act grad RMS."""

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._act_state: defaultdict = defaultdict(dict)
        self._grad_state: defaultdict = defaultdict(dict)
        self._fwd_hooks: list[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_act_hook(self, chunk_name: str, module_name: str) -> Callable:
        """Forward hook: accumulate per-feature count/sum/sum_sq of input0."""
        def hook(_, args, kwargs, _output):
            input_tuple = args if isinstance(args, tuple) else (args,)
            if not input_tuple:
                return
            inp = input_tuple[0]
            if not isinstance(inp, torch.Tensor):
                return
            x2d = _flatten_features(inp.detach().float())
            if x2d.numel() == 0:
                return
            count = x2d.shape[0]
            s = x2d.sum(dim=0).cpu()
            sq = (x2d * x2d).sum(dim=0).cpu()
            entry = self._act_state[chunk_name].get(module_name)
            if entry is None:
                entry = {
                    "count": 0,
                    "sum": torch.zeros_like(s),
                    "sum_sq": torch.zeros_like(sq),
                }
                self._act_state[chunk_name][module_name] = entry
            entry["count"] += count
            entry["sum"] += s
            entry["sum_sq"] += sq

        return hook

    def _make_grad_hook(self, chunk_name: str, module_name: str) -> Callable:
        """Forward hook: install a backward hook on the module's *output* so
        the captured gradient is the one applied *after* the downstream
        activation backward (where dead-unit gradients are zeroed).
        """
        def fwd_hook(_, args, kwargs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
                return

            def grad_hook(grad: torch.Tensor):
                if grad is None:
                    return
                g2d = _flatten_features(grad.detach().float())
                if g2d.numel() == 0:
                    return
                count = g2d.shape[0]
                sq = (g2d * g2d).sum(dim=0).cpu()
                entry = self._grad_state[chunk_name].get(module_name)
                if entry is None:
                    entry = {"count": 0, "sum_sq": torch.zeros_like(sq)}
                    self._grad_state[chunk_name][module_name] = entry
                entry["count"] += count
                entry["sum_sq"] += sq

            tensor.register_hook(grad_hook)

        return fwd_hook

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_hooks(
        self,
        model: Iterable[torch.nn.Module],
        act_filter: Optional[Callable[[str, torch.nn.Module], bool]] = None,
        grad_filter: Optional[Callable[[str, torch.nn.Module], bool]] = None,
    ) -> None:
        """Walk *model* and install one act-hook per match of ``act_filter``
        and one grad-hook per match of ``grad_filter``. A module may
        appear in either, both, or neither.
        """
        assert not self._fwd_hooks, "register_hooks called twice without remove_hooks"
        act_filter = act_filter or _default_act_filter
        grad_filter = grad_filter or _default_grad_filter
        act_matched = grad_matched = 0
        for chunk_id, chunk in enumerate(model):
            chunk_name = f"model_chunk{chunk_id}"
            unwrapped = unwrap_model(chunk)
            for name, mod in unwrapped.named_modules():
                if act_filter(name, mod):
                    act_matched += 1
                    self._fwd_hooks.append(
                        mod.register_forward_hook(
                            self._make_act_hook(chunk_name, name),
                            with_kwargs=True,
                        )
                    )
                if grad_filter(name, mod):
                    grad_matched += 1
                    self._fwd_hooks.append(
                        mod.register_forward_hook(
                            self._make_grad_hook(chunk_name, name),
                            with_kwargs=True,
                        )
                    )
        if act_matched == 0 and grad_matched == 0:
            logger.warning(
                "DeadNeuronLogger.register_hooks matched 0 modules — "
                "check filters; dead-neuron stats will be empty."
            )

    def remove_hooks(self) -> None:
        for handle in self._fwd_hooks:
            handle.remove()
        self._fwd_hooks.clear()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, iteration: int) -> None:
        if not self._act_state and not self._grad_state:
            return
        out: dict[str, dict[str, torch.Tensor]] = {}

        for chunk_name, mods in self._act_state.items():
            chunk_out = out.setdefault(chunk_name, {})
            for module_name, st in mods.items():
                n = max(int(st["count"]), 1)
                mean = st["sum"] / n
                ex2 = st["sum_sq"] / n
                var = (ex2 - mean * mean).clamp(min=0.0)
                chunk_out[f"{module_name}/act_mean"] = mean
                chunk_out[f"{module_name}/act_var"] = var
                chunk_out[f"{module_name}/act_count"] = torch.tensor(n, dtype=torch.int64)

        for chunk_name, mods in self._grad_state.items():
            chunk_out = out.setdefault(chunk_name, {})
            for module_name, st in mods.items():
                n = max(int(st["count"]), 1)
                grad_rms = (st["sum_sq"] / n).sqrt()
                chunk_out[f"{module_name}/grad_rms"] = grad_rms
                chunk_out[f"{module_name}/grad_count"] = torch.tensor(n, dtype=torch.int64)

        save_grads(self._save_dir, out, iteration, "dead_neurons")
        self._maybe_log_wandb(out, iteration)
        self._act_state.clear()
        self._grad_state.clear()

    # ------------------------------------------------------------------
    # Live W&B push (shares Megatron's wandb.run when wandb_project is set)
    # ------------------------------------------------------------------

    def _maybe_log_wandb(self, out, iteration: int) -> None:
        """Push per-layer dead-neuron median scalars to W&B if a run is active.

        Megatron's built-in writer initializes ``wandb.run`` when
        ``wandb_project`` is set; we share that run by calling
        ``wandb.log`` on the same process. Metric keys are namespaced
        under ``dead_neurons/`` so they don't collide with Megatron's
        own keys. Silently a no-op when wandb isn't installed or no run
        is active (e.g. non-rank-0 processes).
        """
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if getattr(wandb, "run", None) is None:
            return
        import re

        layer_re = re.compile(r"layers\.(\d+)")
        scalars: dict[str, float] = {}
        for chunk_name, mods in out.items():
            for module_key, tensor in mods.items():
                # Skip count tensors and any 0-D scalar fields.
                if module_key.endswith("/act_count") or module_key.endswith("/grad_count"):
                    continue
                m = layer_re.search(module_key)
                if m is None:
                    continue
                layer = int(m.group(1))
                stat = module_key.rsplit("/", 1)[-1]
                if not hasattr(tensor, "dim") or tensor.dim() == 0:
                    continue
                try:
                    val = float(tensor.median().item())
                except Exception:
                    continue
                scalars[f"dead_neurons/{chunk_name}/L{layer}/{stat}_median"] = val
        if scalars:
            try:
                wandb.log(scalars, step=iteration)
            except Exception:
                # A logging error on a research metric must never bring
                # the trainer down. Swallow and continue.
                pass


# ----------------------------------------------------------------------
# Module-level singleton (parallels activation_logging/dgrad_logging).
# ----------------------------------------------------------------------

_LOGGER: Optional[DeadNeuronLogger] = None


def _get_logger(save_dir: str) -> DeadNeuronLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = DeadNeuronLogger(save_dir)
    return _LOGGER


def enable_dead_neuron_logging(
    model: torch.nn.Module,
    save_dir: str,
    act_filter: Optional[Callable[[str, torch.nn.Module], bool]] = None,
    grad_filter: Optional[Callable[[str, torch.nn.Module], bool]] = None,
) -> None:
    """Install per-neuron activation + pre-act grad hooks.

    Reads optional overrides:
      - ``MEGATRON_DEAD_NEURON_ACT_SUFFIX``  (default ``.mlp.linear_fc2``)
      - ``MEGATRON_DEAD_NEURON_GRAD_SUFFIX`` (default ``.mlp.linear_fc1``)

    Explicit callable args take precedence over the env-var overrides.
    """
    if act_filter is None:
        suffix = os.environ.get("MEGATRON_DEAD_NEURON_ACT_SUFFIX", "").strip()
        if suffix:
            act_filter = lambda name, _m, s=suffix: name.endswith(s)  # noqa: E731
    if grad_filter is None:
        suffix = os.environ.get("MEGATRON_DEAD_NEURON_GRAD_SUFFIX", "").strip()
        if suffix:
            grad_filter = lambda name, _m, s=suffix: name.endswith(s)  # noqa: E731
    _get_logger(save_dir).register_hooks(model, act_filter, grad_filter)


def disable_dead_neuron_logging() -> None:
    if _LOGGER is None:
        return
    _LOGGER.remove_hooks()


def save_dead_neuron_stats(iteration: int) -> None:
    if _LOGGER is None:
        return
    _LOGGER.save(iteration)
