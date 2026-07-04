# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import hashlib
import inspect
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank

from .optimizer_config import ParamKey, ParamPredicate

try:
    from emerging_optimizers import registry
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    # It is necessary to import optimizers for the registry to work.
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import
    from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import

    try:
        # Standalone Shampoo is newer than the minimum supported package version. Importing
        # the module is what registers it with the package registry.
        from emerging_optimizers.shampoo import Shampoo  # pylint: disable=unused-import
    except ImportError:
        Shampoo = None

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object
    Shampoo = None


logger = logging.getLogger(__name__)

_CHECKPOINT_ADAPTER_SCHEMA = 1
_CHECKPOINT_ADAPTER_ATTR = "_megatron_checkpoint_adapter"
_SUPPORTED_LOCAL_AUX_OPTIMIZERS = {"soap", "shampoo"}
_SOAP_AUX_STATE_KEYS = {"L", "R", "Q_L", "Q_R"}
_SCHEDULER_OWNED_PARAM_GROUP_FIELDS = {"lr", "weight_decay"}


def _normalize_checkpoint_value(value: Any, context: str) -> Any:
    """Convert optimizer configuration and layout values to canonical JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{context} must be finite, got {value}")
        return value
    if isinstance(value, (tuple, list)):
        return [
            _normalize_checkpoint_value(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise RuntimeError(f"{context} has a non-string key {key!r}")
            normalized[key] = _normalize_checkpoint_value(value[key], f"{context}.{key}")
        return normalized
    raise RuntimeError(
        f"{context} has unsupported checkpoint value {value!r} ({type(value).__name__})"
    )


def _validate_local_aux_checkpoint_topology(topology: Mapping[str, int]) -> None:
    """Reject topology dimensions not covered by the local auxiliary-state adapter."""
    unsupported = {
        name: topology[name] for name in ("tp", "pp", "cp", "ep", "etp") if topology[name] != 1
    }
    if unsupported:
        raise RuntimeError(
            "SOAP/Shampoo local auxiliary-state checkpoints currently support only "
            f"TP=PP=CP=EP=ETP=1; got {unsupported}. Use no_save_optim or implement "
            "logical global-block checkpoint factories before enabling this topology."
        )
    if topology["dp"] != topology["world_size"]:
        raise RuntimeError(
            "SOAP/Shampoo local auxiliary-state checkpoint topology is inconsistent: "
            f"DP={topology['dp']} but world_size={topology['world_size']} with all model "
            "parallel dimensions equal to one"
        )


def _local_aux_checkpoint_topology() -> dict[str, int]:
    """Return the exact replicated-dense topology covered by the minimum adapter."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        topology = {"world_size": 1, "dp": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "etp": 1}
    else:
        topology = {
            "world_size": torch.distributed.get_world_size(),
            "dp": parallel_state.get_data_parallel_world_size(with_context_parallel=True),
            "tp": parallel_state.get_tensor_model_parallel_world_size(),
            "pp": parallel_state.get_pipeline_model_parallel_world_size(),
            "cp": parallel_state.get_context_parallel_world_size(),
            "ep": parallel_state.get_expert_model_parallel_world_size(),
            "etp": parallel_state.get_expert_tensor_parallel_world_size(),
        }
    if any(not isinstance(value, int) or value < 1 for value in topology.values()):
        raise RuntimeError(f"invalid SOAP/Shampoo checkpoint topology: {topology}")
    _validate_local_aux_checkpoint_topology(topology)
    return topology


def _effective_constructor_config(optimizer_cls: type, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Capture every effective constructor field needed to recreate an optimizer."""
    effective = {}
    for name, parameter in inspect.signature(optimizer_cls.__init__).parameters.items():
        if name in {"self", "params"}:
            continue
        if name in kwargs:
            value = kwargs[name]
        elif parameter.default is not inspect.Parameter.empty:
            value = parameter.default
        else:
            raise RuntimeError(
                f"cannot fingerprint required {optimizer_cls.__name__} constructor field {name!r}"
            )
        effective[name] = _normalize_checkpoint_value(value, f"{optimizer_cls.__name__}.{name}")
    return effective


@dataclass(frozen=True)
class LocalAuxStateCheckpointAdapter:
    """Topology-preserving checkpoint adapter for SOAP/Shampoo auxiliary state.

    This deliberately supports only replicated dense optimization. Auxiliary tensors are
    checkpointed as independent full tensors and replicated across DP in the same way as the
    owning model parameter. Topology-changing restore requires logical global-block factories
    and is rejected by the persisted fingerprint.
    """

    optimizer_identity: str
    optimizer_class: str
    constructor_config: Mapping[str, Any]
    schema_version: int = _CHECKPOINT_ADAPTER_SCHEMA

    def _is_aux_tensor_key(self, state_key: str) -> bool:
        if self.optimizer_identity == "soap":
            return state_key in _SOAP_AUX_STATE_KEYS
        if self.optimizer_identity == "shampoo":
            return (
                state_key.startswith("left_factor_")
                or state_key.startswith("right_factor_")
                or state_key.startswith("left_inverse_root_")
                or state_key.startswith("right_inverse_root_")
            )
        return False

    def shard_state_value(
        self,
        param_id: int,
        state_key: str,
        value: Any,
        model_param: ShardedTensor | ShardedTensorFactory,
    ) -> ShardedTensor | LocalNonpersistentObject | None:
        """Convert one non-parameter-shaped state value for ``torch_dist``."""
        model_data = model_param.data
        if torch.is_tensor(value) and torch.is_tensor(model_data):
            if self._is_aux_tensor_key(state_key):
                _local_aux_checkpoint_topology()
                return ShardedTensor.from_rank_offsets(
                    f"optimizer.state.{self.optimizer_identity}.{state_key}.param_{param_id}."
                    f"{model_param.key}",
                    value,
                    replica_id=model_param.replica_id,
                )
            if tuple(value.shape) != tuple(model_data.shape):
                raise RuntimeError(
                    f"unsupported {self.optimizer_identity} state tensor {state_key!r} for "
                    f"param {param_id}: state shape {tuple(value.shape)} differs from model "
                    f"shape {tuple(model_data.shape)}"
                )
            return None
        if self.optimizer_identity == "shampoo" and state_key == "block_ranges":
            return LocalNonpersistentObject(value)
        raise RuntimeError(
            f"unsupported non-tensor {self.optimizer_identity} optimizer state "
            f"{state_key!r} ({type(value).__name__}) for param {param_id}"
        )

    @staticmethod
    def _state_layout(optimizer, optimizer_state_dict: Mapping[str, Any]) -> dict[str, Any]:
        groups = []
        saved_groups = optimizer_state_dict["param_groups"]
        if len(saved_groups) != len(optimizer.param_groups):
            raise RuntimeError("optimizer param-group count changed while fingerprinting")
        for group_index, (live_group, saved_group) in enumerate(
            zip(optimizer.param_groups, saved_groups)
        ):
            groups.append(
                {
                    "config": _normalize_checkpoint_value(
                        {
                            key: value
                            for key, value in saved_group.items()
                            if key != "params" and key not in _SCHEDULER_OWNED_PARAM_GROUP_FIELDS
                        },
                        f"param_groups[{group_index}]",
                    ),
                    "params": [
                        {"shape": list(param.shape), "dtype": str(param.dtype)}
                        for param in live_group["params"]
                    ],
                }
            )

        states = {}
        for param_id, param_state in sorted(optimizer_state_dict["state"].items()):
            state_layout = {}
            for state_key, value in sorted(param_state.items()):
                if torch.is_tensor(value):
                    state_layout[state_key] = {
                        "kind": "tensor",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                elif state_key == "block_ranges":
                    state_layout[state_key] = {
                        "kind": "deterministic",
                        "value": _normalize_checkpoint_value(
                            value, f"state[{param_id}].{state_key}"
                        ),
                    }
                else:
                    state_layout[state_key] = {"kind": "scalar", "type": type(value).__name__}
            states[str(param_id)] = state_layout
        return {"param_groups": groups, "state": states}

    def fingerprint(self, optimizer, optimizer_state_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Build a value-independent schema/config/topology/layout fingerprint."""
        payload = {
            "schema_version": self.schema_version,
            "optimizer_identity": self.optimizer_identity,
            "optimizer_class": self.optimizer_class,
            "constructor_config": self.constructor_config,
            "topology": _local_aux_checkpoint_topology(),
            "layout": self._state_layout(optimizer, optimizer_state_dict),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {"payload": payload, "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}

    def sharded_fingerprint(
        self, optimizer, optimizer_state_dict: Mapping[str, Any]
    ) -> ShardedObject:
        """Wrap the common DP-replicated fingerprint for ``torch_dist``."""
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        return ShardedObject(
            key=f"optimizer.checkpoint_adapter.{self.optimizer_identity}",
            data=self.fingerprint(optimizer, optimizer_state_dict),
            global_shape=(1,),
            global_offset=(0,),
            replica_id=rank,
        )

    def validate_fingerprint(
        self, loaded: Any, optimizer, optimizer_state_dict: Mapping[str, Any]
    ) -> None:
        """Fail before optimizer load when schema, config, topology, or layout changed."""
        expected = self.fingerprint(optimizer, optimizer_state_dict)
        if not isinstance(loaded, Mapping) or loaded != expected:
            loaded_digest = loaded.get("sha256") if isinstance(loaded, Mapping) else None
            raise RuntimeError(
                f"{self.optimizer_identity} optimizer checkpoint fingerprint mismatch: "
                f"loaded={loaded_digest!r}, expected={expected['sha256']!r}. Same-topology "
                "restore requires identical optimizer config and state layout."
            )


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, ParamGroupOverride]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class.
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
        checkpoint_adapter_factory: Optional factory for optimizer-specific ``torch_dist``
            state mappings.
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    default_param_overrides: Dict[ParamKey, ParamGroupOverride] = field(
        default_factory=_default_param_overrides_factory
    )
    checkpoint_adapter_factory: Callable | None = None


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if entry.config_to_kwargs is not None:
        eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
    else:
        eopt_kwargs = _default_adam_based_eopt_config_to_kwargs(
            eopt_name, config, model_chunks, pg_collection
        )
    optimizer = entry.optimizer_cls(param_groups, **eopt_kwargs)
    if entry.checkpoint_adapter_factory is not None:
        checkpoint_adapter = entry.checkpoint_adapter_factory(
            eopt_name,
            entry.optimizer_cls,
            _effective_constructor_config(entry.optimizer_cls, eopt_kwargs),
        )
        setattr(optimizer, _CHECKPOINT_ADAPTER_ATTR, checkpoint_adapter)
    return optimizer, entry.init_state_fn


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _get_qkv_split_shapes(model_cfg) -> List[int]:
    """Compute QKV split shapes from model config."""
    return [
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels,
        model_cfg.kv_channels,
        model_cfg.kv_channels,
    ]


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Muon
# ===========================================================================


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        # Use explicit class call instead of super() so that subclasses with
        # multiple inheritance (e.g. TensorParallelAdaptiveMuon) don't route
        # through an intermediate class that doesn't accept scaled_orthogonalize_fn.
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, ' f'split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class TensorParallelAdaptiveMuon(TensorParallelMuon, AdaptiveMuon):
    """Tensor Parallel Adaptive Muon optimizer.

    This class extends Muon by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. This idea was first explored in D.E. Carlson,
    E. Collins, Ya-Ping Hsieh, L. Carin, and V. Cevher. *Preconditioned spectral
    descent for deep learning.* In Advances in neural information processing systems 28 (2015).
    The step() method is overridden to include second moment normalization logic.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: The exponential decay rate for momentum.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: Weight decay coefficient.
        use_decoupled_weight_decay: Whether to use decoupled weight decay.
        split_qkv: Whether to split QKV weights for orthogonalization.
        is_qkv_fn: Function to determine if a tensor is a QKV weight.
        qkv_split_shapes: Shapes for splitting QKV weights.
        fp32_matmul_prec: Precision for FP32 matrix multiplication.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update.
        extra_scale_factor: The additional scale factor to use for the update.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", or "distributed").
        moment2_method: Method for second moment accumulation ("adamuon" or "normuon").
        beta2: The exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )
        self.moment2_method = moment2_method

        for group in self.param_groups:
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step function"""
        return AdaptiveMuon.step(self, closure)


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _adaptive_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelAdaptiveMuon constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs


def _default_adam_based_eopt_config_to_kwargs(
    eopt_name, config, model_chunks, pg_collection
) -> Dict[str, Any]:
    """Convert OptimizerConfig to default emerging optimizer constructor kwargs."""
    kwargs = _kwargs_from_config(registry.get_optimizer_cls(eopt_name), eopt_name, config)
    kwargs["betas"] = (config.adam_beta1, config.adam_beta2)
    return kwargs


def _shampoo_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert ``OptimizerConfig`` to standalone Shampoo constructor kwargs.

    The package registry is the source of truth for the concrete class and constructor
    signature. Standalone Shampoo shares Megatron's global learning rate, weight decay,
    and Adam betas while exposing its algorithm-specific preconditioner and graft
    controls under the ``shampoo_`` prefix.
    """
    shampoo_cls = registry.get_optimizer_cls("shampoo")
    kwargs = _kwargs_from_config(shampoo_cls, "shampoo", config)
    if "betas" in inspect.signature(shampoo_cls.__init__).parameters:
        kwargs["betas"] = (config.adam_beta1, config.adam_beta2)
    return kwargs


def _local_aux_checkpoint_adapter_factory(
    optimizer_identity: str, optimizer_cls: type, constructor_config: Mapping[str, Any]
) -> LocalAuxStateCheckpointAdapter:
    """Create the bounded same-topology adapter for SOAP or Shampoo."""
    if optimizer_identity not in _SUPPORTED_LOCAL_AUX_OPTIMIZERS:
        raise RuntimeError(
            f"local auxiliary-state checkpoints are not registered for {optimizer_identity!r}"
        )
    return LocalAuxStateCheckpointAdapter(
        optimizer_identity=optimizer_identity,
        optimizer_class=f"{optimizer_cls.__module__}.{optimizer_cls.__qualname__}",
        constructor_config=constructor_config,
    )


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAdaptiveMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_adaptive_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
    }
)

if HAVE_EMERGING_OPTIMIZERS and "shampoo" in registry.get_optimizer_name_list():
    _EMERGING_OPTIMIZERS["shampoo"] = EmergingOptimizerEntry(
        optimizer_cls=registry.get_optimizer_cls("shampoo"),
        config_to_kwargs=_shampoo_config_to_kwargs,
        checkpoint_adapter_factory=_local_aux_checkpoint_adapter_factory,
    )

# Register soap with default config
# TODO(skyw): register all emerging optimizers.
if HAVE_EMERGING_OPTIMIZERS:
    for eopt_name in registry.get_optimizer_name_list():
        if eopt_name in _EMERGING_OPTIMIZERS:
            # skip already registered local versions, e.g. TensorParallel versions.
            continue
        _EMERGING_OPTIMIZERS[eopt_name] = EmergingOptimizerEntry(
            optimizer_cls=registry.get_optimizer_cls(eopt_name),
            checkpoint_adapter_factory=(
                _local_aux_checkpoint_adapter_factory if eopt_name == "soap" else None
            ),
        )
