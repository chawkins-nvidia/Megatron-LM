# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Dgrad logging using forward-time tensor hooks."""

from collections import defaultdict
import torch
import torch.nn as nn

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.router import Router
from megatron.training.activation_logging import LOGGABLE_TYPES

from .checkpointing import save_grads
from .utils import unwrap_model


def _get_linear_types():
    """Build tuple of linear layer types to capture gradients from."""
    types = [nn.Linear, nn.Embedding, ColumnParallelLinear, RowParallelLinear, Router]

    # Add Transformer Engine layers if available.
    try:
        from megatron.core.extensions.transformer_engine import (
            TELinear,
            TENorm,
            TEColumnParallelLinear,
            TERowParallelLinear,
            TELayerNormColumnParallelLinear,
        )
        types.extend([TELinear, TENorm, TEColumnParallelLinear, TERowParallelLinear,
                      TELayerNormColumnParallelLinear])
    except ImportError:
        pass

    try:
        from megatron.core.extensions.transformer_engine import (
            TEGroupedLinear,
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )
        if TEGroupedLinear is not None:
            types.extend([TEGroupedLinear, TEColumnParallelGroupedLinear,
                          TERowParallelGroupedLinear])
    except ImportError:
        pass

    return tuple(types)


LINEAR_TYPES = LOGGABLE_TYPES


def _iter_tensors(value, prefix: str):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for idx, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}{idx}")


# Import the activation logger's summary callable to keep the two
# hooks in sync. Configurable via MEGATRON_RESIDUAL_LOG_STAT
# (see activation_logging._resolve_stat_fn).
from megatron.training.activation_logging import _rms_summary  # noqa: E402, F401


class DataGradLogger:
    """Captures and saves gradients from loggable module tensors.
    
    NOTE: Right now, we only save the dgrads for the last microbatch in a batch on DP replica 0.
    The code below would need to be extended to save dgrads for all microbatches in a batch."""

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._dgrads_state_dict = defaultdict(dict)
        self._hooks = []

    def _save_hook(self, model_chunk_name: str, key: str):
        def hook(grad):
            if grad is not None:
                self._dgrads_state_dict[model_chunk_name][key] = _rms_summary(grad)
        return hook

    def _make_hook(self, model_chunk_name: str, module_name: str):
        """Create a forward hook that installs tensor gradient hooks."""
        def hook(_, args, kwargs, output):
            input_tuple = args if isinstance(args, tuple) else (args,)
            for idx, inp in enumerate(input_tuple):
                for suffix, tensor in _iter_tensors(inp, f"input{idx}"):
                    if tensor.requires_grad:
                        key = f"{module_name}/{suffix}"
                        tensor.register_hook(self._save_hook(model_chunk_name, key))
            for kwarg_key, kwarg_value in kwargs.items():
                for suffix, tensor in _iter_tensors(kwarg_value, kwarg_key):
                    if tensor.requires_grad:
                        key = f"{module_name}/{suffix}"
                        tensor.register_hook(self._save_hook(model_chunk_name, key))
            for suffix, tensor in _iter_tensors(output, "output"):
                if tensor.requires_grad:
                    key = f"{module_name}/{suffix}"
                    if suffix == "output":
                        key = f"{module_name}/output"
                    tensor.register_hook(self._save_hook(model_chunk_name, key))
        return hook

    def save(self, iteration: int):
        """Save captured gradients to disk and clear the buffer."""
        if not self._dgrads_state_dict:
            return
        save_grads(self._save_dir, self._dgrads_state_dict, iteration, "dgrads")
        self._maybe_log_wandb(self._dgrads_state_dict, iteration, "dgrad")  # CHAWKINS-WANDB-PER-TENSOR
        self._dgrads_state_dict.clear()

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

    def register_hooks(self, model: torch.nn.Module):
        """Find and register hooks on all linear layers."""
        assert len(self._hooks) == 0
        for model_chunk_id, model_chunk in enumerate(model):
            unwrapped_model_chunk = unwrap_model(model_chunk)
            for module_name, module in unwrapped_model_chunk.named_modules():
                if isinstance(module, LINEAR_TYPES):
                    model_chunk_name = f"model_chunk{model_chunk_id}"
                    handle = module.register_forward_hook(
                        self._make_hook(model_chunk_name, module_name),
                        with_kwargs=True,
                    )
                    self._hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


_LOGGER = None


def enable_dgrad_logging(model: torch.nn.Module, save_dir: str):
    """Enable dgrad logging on a model."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = DataGradLogger(save_dir)
    _LOGGER.register_hooks(model)


def disable_dgrad_logging():
    """Disable dgrad logging on a model."""
    global _LOGGER
    assert _LOGGER is not None
    _LOGGER.remove_hooks()


def save_dgrads(iteration: int):
    """Save dgrads to disk."""
    global _LOGGER
    assert _LOGGER is not None
    _LOGGER.save(iteration)
