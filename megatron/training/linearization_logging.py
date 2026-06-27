# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Minimal finite-epsilon linearization diagnostics for #118 stability smokes.

This stream measures whether the residual-stream displacement from one source block's real
optimizer update is well approximated by a local finite-epsilon linear response at the current
iterate. It supports single-rank model layouts only (TP=PP=CP=1). Data parallelism is allowed:
each replica measures its local mini-batch and scalar metrics are averaged across the DP group
before W&B/checkpoint logging.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch

from .activation_logging import _rms
from .diagnostics_common import REDUCE_REPLICA, save_diag_state
from .probe_logging import run_probe_forward
from .utils import unwrap_model

logger = logging.getLogger(__name__)

_STREAM = "linearization"
_WANDB_PREFIX = "lin"
_EPS_FLOOR = 1.0e-12
GLOBAL_LAYER_ATTR = "param_global_layer_number"
SOURCE_MODULE_PATTERNS = (
    ("qkv", ("self_attention.linear_qkv", "self_attention.query_key_value")),
    ("attention_proj", ("self_attention.linear_proj", "self_attention.dense")),
    ("mlp_up_fc1", ("mlp.linear_fc1", "mlp.fc1")),
    ("mlp_down_fc2", ("mlp.linear_fc2", "mlp.fc2")),
)


def _rms_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return _rms(tensor.detach().float()).cpu()


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    if float(den) <= _EPS_FLOOR:
        return torch.tensor(0.0)
    return (num / den).detach().cpu()


def _flatten_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_f = _flatten_fp32(a)
    b_f = _flatten_fp32(b)
    denom = a_f.norm() * b_f.norm()
    if float(denom) <= _EPS_FLOOR:
        return torch.tensor(0.0)
    return torch.clamp(torch.dot(a_f, b_f) / denom, -1.0, 1.0).detach().cpu()


def _eps_tag(eps: float) -> str:
    if eps == 0:
        return "eps0"
    text = f"{eps:.0e}".replace("+0", "").replace("+", "")
    text = text.replace("-0", "-")
    return "eps" + text.replace("-", "m")


def parse_linearization_eps(raw) -> tuple[float, ...]:
    """Parse ``1e-3,1e-2`` style config into positive finite epsilons."""
    if raw is None:
        return (1.0e-3, 1.0e-2)
    if isinstance(raw, str):
        pieces = [p.strip() for p in raw.split(",")]
    elif isinstance(raw, Iterable):
        pieces = list(raw)
    else:
        pieces = [raw]
    eps = []
    for piece in pieces:
        if piece == "":
            continue
        val = float(piece)
        if val <= 0 or not torch.isfinite(torch.tensor(val)):
            raise ValueError(
                f"linearization_eps entries must be positive finite values, got {piece!r}"
            )
        eps.append(val)
    if not eps:
        raise ValueError("linearization_eps must contain at least one positive epsilon")
    return tuple(eps)


def _metrics_from_residuals(
    h_pre: torch.Tensor,
    h_true: torch.Tensor,
    h_eps_by_eps: dict[float, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return scalar linearization metrics from target residual tensors.

    ``h_true`` is the source-block-only residual after the full optimizer update. ``h_eps_by_eps``
    stores residuals after ``eps * delta_theta_source``. The finite difference
    ``(h_eps - h_pre) / eps`` is compared to ``h_true - h_pre``.
    """
    dh_true = h_true.detach().float() - h_pre.detach().float()
    dh_true_norm = _rms_fp32(dh_true)
    metrics: dict[str, torch.Tensor] = {"dh_true_norm": dh_true_norm}
    for eps, h_eps in h_eps_by_eps.items():
        dh_lin = (h_eps.detach().float() - h_pre.detach().float()) / eps
        dh_diff = dh_true - dh_lin
        tag = _eps_tag(eps)
        dh_lin_norm = _rms_fp32(dh_lin)
        dh_diff_norm = _rms_fp32(dh_diff)
        metrics[f"dh_lin_norm::{tag}"] = dh_lin_norm
        metrics[f"dh_diff_norm::{tag}"] = dh_diff_norm
        metrics[f"R_lazy::{tag}"] = _safe_ratio(dh_diff_norm, dh_true_norm)
        metrics[f"cos_align::{tag}"] = _cosine(dh_true, dh_lin)
    return metrics


def _as_model_list(model):
    return model if isinstance(model, (list, tuple)) else [model]


def _parallel_layout_supported() -> bool:
    try:
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            return True
        context_parallel_world_size = 1
        if hasattr(mpu, "get_context_parallel_world_size"):
            context_parallel_world_size = mpu.get_context_parallel_world_size()
        return (
            mpu.get_tensor_model_parallel_world_size() == 1
            and mpu.get_pipeline_model_parallel_world_size() == 1
            and context_parallel_world_size == 1
        )
    except Exception:
        return True


def _data_parallel_world_size() -> int:
    try:
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            return 1
        return mpu.get_data_parallel_world_size()
    except Exception:
        return 1


def _mean_scalar_across_data_parallel(value: torch.Tensor) -> torch.Tensor:
    """Average one scalar diagnostic across the DP group and return it on CPU."""
    dp_world_size = _data_parallel_world_size()
    if dp_world_size <= 1 or not torch.distributed.is_available():
        return value.detach().cpu()
    if not torch.distributed.is_initialized():
        return value.detach().cpu()
    try:
        from megatron.core import parallel_state as mpu

        group = mpu.get_data_parallel_group()
    except Exception:
        group = None
    work = value.detach().float()
    if work.numel() != 1:
        work = work.mean()
    if torch.cuda.is_available():
        work = work.to(device=torch.device("cuda", torch.cuda.current_device()))
    torch.distributed.all_reduce(work, op=torch.distributed.ReduceOp.SUM, group=group)
    work /= dp_world_size
    return work.cpu()


def _mean_metrics_across_data_parallel(
    metrics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if _data_parallel_world_size() <= 1:
        return metrics
    return {key: _mean_scalar_across_data_parallel(value) for key, value in metrics.items()}


def _layer_from_name(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if match is None:
        return None
    return int(match.group(1)) + 1


def _param_layer(name: str, param: torch.nn.Parameter) -> int | None:
    stamped = getattr(param, GLOBAL_LAYER_ATTR, None)
    if stamped is not None:
        return int(stamped)
    return _layer_from_name(name)


def _source_module(name: str) -> str | None:
    for module, patterns in SOURCE_MODULE_PATTERNS:
        if any(pattern in name for pattern in patterns):
            return module
    return None


def _iter_trainable_params(model):
    for chunk_id, chunk in enumerate(_as_model_list(model)):
        chunk_name = f"model_chunk{chunk_id}"
        for name, param in unwrap_model(chunk).named_parameters():
            if param.requires_grad:
                yield chunk_name, name, param


def _select_depth(num_layers: int, fraction: float) -> int:
    if num_layers <= 0:
        return 1
    return max(1, min(num_layers, round(num_layers * fraction)))


def _target_residual_hooks(model, target_layer: int, capture):
    """Register hooks that capture target layer's post-mixer residual."""
    handles = []
    target_idx = target_layer - 1
    layer_pattern = re.compile(r"(.*\.layers\.(\d+))\.pre_mlp_layernorm")
    block_pattern = re.compile(r"(.*\.layers\.(\d+))$")
    for chunk in _as_model_list(model):
        for module_name, module in unwrap_model(chunk).named_modules():
            match = layer_pattern.fullmatch(module_name)
            if match is not None and int(match.group(2)) == target_idx:
                handles.append(
                    module.register_forward_pre_hook(lambda _m, args: capture(args[0]))
                )
                continue
            match = block_pattern.fullmatch(module_name)
            if (
                match is not None
                and int(match.group(2)) == target_idx
                and hasattr(module, "mixer")
                and not hasattr(module, "pre_mlp_layernorm")
            ):
                handles.append(
                    module.register_forward_hook(
                        lambda _m, _args, output: capture(
                            output[0] if isinstance(output, tuple) else output
                        )
                    )
                )
    return handles


@dataclass
class _LinearizationStep:
    eps: tuple[float, ...]
    source_layer: int
    target_layer: int
    theta_pre: dict[str, torch.Tensor]
    source_keys: set[str]
    source_keys_by_module: dict[str, set[str]]
    h_pre: torch.Tensor


class LinearizationLogger:
    def __init__(self):
        self._step: _LinearizationStep | None = None

    def _capture_target_residual(self, model, target_layer: int) -> torch.Tensor | None:
        captured: list[torch.Tensor] = []

        def capture(tensor):
            if isinstance(tensor, torch.Tensor):
                captured.append(tensor.detach().clone())

        handles = _target_residual_hooks(model, target_layer, capture)
        if not handles:
            return None
        try:
            run_probe_forward(model)
        finally:
            for handle in handles:
                handle.remove()
        return captured[0] if captured else None

    def pre(self, model, args) -> None:
        self._step = None
        if not _parallel_layout_supported():
            logger.warning(
                "linearization diagnostics require TP=PP=CP=1; pure DP replicas are supported"
            )
            return
        eps = parse_linearization_eps(getattr(args, "linearization_eps", None))
        num_layers = int(getattr(args, "num_layers", 0) or 0)
        source_layer = _select_depth(
            num_layers, float(getattr(args, "linearization_source_fraction", 0.25))
        )
        target_layer = _select_depth(
            num_layers, float(getattr(args, "linearization_target_fraction", 0.75))
        )
        if target_layer <= source_layer and num_layers > source_layer:
            target_layer = source_layer + 1
        theta_pre: dict[str, torch.Tensor] = {}
        source_keys: set[str] = set()
        source_keys_by_module: dict[str, set[str]] = {}
        for chunk_name, name, param in _iter_trainable_params(model):
            key = f"{chunk_name}/{name}"
            theta_pre[key] = param.detach().float().clone()
            if _param_layer(name, param) == source_layer:
                source_keys.add(key)
                if (module := _source_module(name)) is not None:
                    source_keys_by_module.setdefault(module, set()).add(key)
        if not source_keys:
            logger.warning(
                "linearization diagnostics found no trainable params at source layer %s",
                source_layer,
            )
            return
        h_pre = self._capture_target_residual(model, target_layer)
        if h_pre is None:
            logger.warning(
                "linearization diagnostics found no target residual hook for layer %s",
                target_layer,
            )
            return
        self._step = _LinearizationStep(
            eps=eps,
            source_layer=source_layer,
            target_layer=target_layer,
            theta_pre=theta_pre,
            source_keys=source_keys,
            source_keys_by_module=source_keys_by_module,
            h_pre=h_pre,
        )

    def _restore(self, model, values: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for chunk_name, name, param in _iter_trainable_params(model):
                key = f"{chunk_name}/{name}"
                value = values.get(key)
                if value is not None:
                    param.data.copy_(value.to(device=param.device, dtype=param.dtype))

    def _apply_source_scale(
        self, model, scale: float, source_keys: set[str] | None = None
    ) -> None:
        assert self._step is not None
        active_source_keys = self._step.source_keys if source_keys is None else source_keys
        with torch.no_grad():
            for chunk_name, name, param in _iter_trainable_params(model):
                key = f"{chunk_name}/{name}"
                base = self._step.theta_pre[key].to(device=param.device)
                if key in active_source_keys:
                    delta = param.detach().float() - base
                    value = base + scale * delta
                else:
                    value = base
                param.data.copy_(value.to(dtype=param.dtype))

    def _linearization_metrics_for_source_keys(
        self, model, post_values: dict[str, torch.Tensor], source_keys: set[str]
    ) -> dict[str, torch.Tensor] | None:
        assert self._step is not None
        self._restore(model, post_values)
        self._apply_source_scale(model, 1.0, source_keys)
        h_true = self._capture_target_residual(model, self._step.target_layer)
        if h_true is None:
            return None
        h_eps_by_eps = {}
        for eps in self._step.eps:
            self._restore(model, post_values)
            self._apply_source_scale(model, eps, source_keys)
            h_eps = self._capture_target_residual(model, self._step.target_layer)
            if h_eps is not None:
                h_eps_by_eps[eps] = h_eps
        if not h_eps_by_eps:
            return None
        return _metrics_from_residuals(self._step.h_pre, h_true, h_eps_by_eps)

    def post(self, model, save_dir: str, iteration: int) -> None:
        if self._step is None:
            return
        post_values = {
            f"{chunk_name}/{name}": param.detach().clone()
            for chunk_name, name, param in _iter_trainable_params(model)
        }
        try:
            metrics = self._linearization_metrics_for_source_keys(
                model, post_values, self._step.source_keys
            )
            if metrics is None:
                return
            key_prefix = f"source_block{self._step.source_layer}_to_resid{self._step.target_layer}"
            state = defaultdict(dict)
            metrics = _mean_metrics_across_data_parallel(metrics)
            for key, value in metrics.items():
                state["model_chunk0"][f"{key_prefix}/{key}"] = value
            for module, source_keys in sorted(self._step.source_keys_by_module.items()):
                module_metrics = self._linearization_metrics_for_source_keys(
                    model, post_values, source_keys
                )
                if module_metrics is None:
                    continue
                module_key_prefix = (
                    f"source_block{self._step.source_layer}_{module}"
                    f"_to_resid{self._step.target_layer}"
                )
                module_metrics = _mean_metrics_across_data_parallel(module_metrics)
                for key, value in module_metrics.items():
                    state["model_chunk0"][f"{module_key_prefix}/{key}"] = value
            save_diag_state(
                save_dir,
                stream=_STREAM,
                iteration=iteration,
                state=state,
                reductions={"": REDUCE_REPLICA},
                wandb_prefix=_WANDB_PREFIX,
            )
        finally:
            self._restore(model, post_values)
            self._step = None

    def reset(self) -> None:
        self._step = None


_LINEARIZATION: LinearizationLogger | None = None


def _get_linearization() -> LinearizationLogger:
    global _LINEARIZATION
    if _LINEARIZATION is None:
        _LINEARIZATION = LinearizationLogger()
    return _LINEARIZATION


def linearization_pre(model, args) -> None:
    _get_linearization().pre(model, args)


def linearization_post(model, save_dir: str, iteration: int) -> None:
    _get_linearization().post(model, save_dir, iteration)


def reset_linearization() -> None:
    if _LINEARIZATION is not None:
        _LINEARIZATION.reset()
