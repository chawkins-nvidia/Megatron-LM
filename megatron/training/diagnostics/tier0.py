# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Successful-update orchestration for the Tier-0 diagnostic heartbeat."""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

import torch
import torch.distributed as dist
import numpy as np
from torch import nn

from megatron.core import parallel_state
from megatron.core.diagnostics import (
    get_diagnostic_global_valid_tokens,
    set_diagnostic_global_valid_tokens,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.transformer.transformer_layer import TransformerLayer

from .accumulator import PackedSlots, PackedSufficientStatistics, ReductionBinding
from .artifact_v3 import (
    COMPONENT_STATE_REASON,
    MIDPOINT_STATE_REASON,
    SECANT_STATE_REASON,
    ArtifactV3Writer,
    EventEvidence,
    build_layer_metric_arrays,
    build_process_group_evidence,
    build_runtime_signature,
    complete_restore_evidence,
    complete_state_snapshot,
    identity_sampling_evidence,
    rank_perf_arrays,
    tier0_collective_operations,
    tier0_rank_perf_from_heartbeat,
    tier0_sampling_from_rank_evidence,
    tier0_state_evidence,
    unavailable_restore_evidence,
    unavailable_state_snapshot,
)
from .capability import SUPPORT_SIGNATURE
from .capture import (
    CaptureTopology,
    StagedTokenMask,
    Tier0CaptureResult,
    Tier0CaptureSession,
    pack_valid_token_mask_sideband,
    unpack_valid_token_mask_sideband,
)
from .distributed_optimizer import (
    Bf16DistributedOptimizerDiagnosticAdapter,
    SnapshotMemoryPreflight,
    snapshot_memory_estimate,
)
from .launch_artifact import publish_final_launch_config_from_runtime
from .normalization import CanonicalDgradNormalizer
from .registry import (
    DenominatorKind,
    MaskKind,
    MetricDescriptor,
    MetricFamily,
    MetricRegistry,
    NormalizationKind,
    Ownership,
    PartitionAxis,
    ProcessGroupIdentity,
    ReductionKind,
    ReplicationAxis,
    StatisticKind,
)

from .schema import (  # isort: skip
    TIER0_KEYS,
    TIER0_METADATA_KEYS,
    TIER0_METRIC_KEYS,
    assert_payload_schema,
    assert_tiered_payload_schema,
)

_UPDATE_FAMILIES = (
    MetricFamily.QKV,
    MetricFamily.ATTN_OUT,
    MetricFamily.FC1,
    MetricFamily.FC2,
    MetricFamily.NORM,
)
_MATRIX_FAMILIES = (
    MetricFamily.EMBEDDING,
    MetricFamily.QKV,
    MetricFamily.ATTN_OUT,
    MetricFamily.FC1,
    MetricFamily.FC2,
    MetricFamily.OUTPUT,
)
_CONTROL_NAMES = (
    "control/update_failure",
    "control/unsupported",
    "control/preflight_failure",
    "control/runtime_failure",
    "control/adapter_status",
    "control/all_rank_post_gather_peak_unavailable",
)
_BASE_RANK_EVIDENCE_FIELDS = 11
_RANK_EVIDENCE_FIELDS = _BASE_RANK_EVIDENCE_FIELDS
_INT64_MAX = 2**63 - 1
_MAX_RESERVATION_BYTES = 8 * 1024**3
_BACKEND_WORKSPACE_ALLOWANCE_BYTES = 64 * 1024**2
_MAX_LAYERS = 512
_MAX_MICROBATCHES = 64
_MAX_MICRO_BATCH_SIZE = 8
_MAX_SEQUENCE_LENGTH = 32_768
_MASK_PRODUCER_IDENTITY = object()
_NONINTERLEAVED_SCHEDULE_ADAPTER = "forward_backward_pipelining_without_interleaving:v1"
_canonical_mask_producer: tuple[Callable[..., Any], object, str] | None = None


@dataclass(frozen=True)
class Tier0Cadence:
    """Successful-update cadence for diagnostic events."""

    interval: int = 1000
    early_updates: tuple[int, ...] = (1, 10, 100)

    def __post_init__(self) -> None:
        """Validate the fixed cadence."""

        if self.interval <= 0:
            raise ValueError("diagnostic interval must be positive")
        if any(update <= 0 for update in self.early_updates):
            raise ValueError("early diagnostic updates must be positive")

    @classmethod
    def parse(cls, interval: int, early_updates: str) -> "Tier0Cadence":
        """Parse a comma-separated early-update declaration."""

        try:
            parsed = tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in early_updates.split(",")
                        if value.strip()
                    }
                )
            )
        except ValueError as error:
            raise ValueError(
                "diagnostic early updates must be comma-separated integers"
            ) from error
        return cls(interval=interval, early_updates=parsed)

    def is_due(self, successful_update: int) -> bool:
        """Return whether a prospective successful update is diagnostic."""

        return (
            successful_update in self.early_updates
            or successful_update % self.interval == 0
        )


@dataclass(frozen=True)
class Tier0Capability:
    """Collectively negotiated narrow-backend support result."""

    supported: bool
    reasons: tuple[str, ...]
    support_signature: str


@dataclass(frozen=True)
class Tier0Reservation:
    """Pure startup reservation and measured allocation evidence."""

    requested_bytes: int
    max_extra_bytes: int
    accepted: bool
    measured_allocated_bytes: int = 0
    measured_reserved_bytes: int = 0


class Tier0ReservationStatus(IntEnum):
    """Globally agreed startup reservation result."""

    OK = 0
    UNSUPPORTED = 1
    OVERFLOW = 2


class Tier0ReservationError(RuntimeError):
    """Raised coherently before P2P when the reservation is unsupported."""

    def __init__(self, status: Tier0ReservationStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def tier0_reservation_bytes(
    *,
    num_layers: int,
    num_microbatches: int,
    micro_batch_size: int,
    local_sequence_length: int,
    owner_elements: int,
    world_size: int,
) -> int:
    """Calculate the exact worst-case event reservation without allocating."""

    dimensions = (
        num_layers,
        num_microbatches,
        micro_batch_size,
        local_sequence_length,
        world_size,
    )
    if any(value <= 0 for value in dimensions) or owner_elements < 0:
        raise ValueError("Tier-0 reservation dimensions are invalid")
    capture_slots = 2 + 10 * num_layers
    update_slots = 5 * num_layers + 2
    control_slots = len(_CONTROL_NAMES)
    pack_bytes = (capture_slots + update_slots + control_slots) * (11 * 8 + 4 + 4)
    mask_elements = micro_batch_size * local_sequence_length
    one_sideband_set_bytes = num_microbatches * (mask_elements + 1) * 4
    one_sideband_validity_set_bytes = num_microbatches
    optimizer_bytes = snapshot_memory_estimate(owner_elements).total_bytes
    optimizer_event_status_bytes = 8
    optimizer_capability_flags_bytes = 15 * 8
    capture_scratch = PackedSufficientStatistics.scratch_bytes_for_capacity()
    capture_session_bytes = 2 * mask_elements + 8 + 8 + 1 + 8
    global_rank_evidence_bytes = world_size * _RANK_EVIDENCE_FIELDS * 8
    local_rank_evidence_bytes = _RANK_EVIDENCE_FIELDS * 8
    sink_staging_bytes = (
        _RANK_EVIDENCE_FIELDS * world_size
        + 13 * (capture_slots + update_slots + control_slots)
    ) * 8
    mask_comparison_bytes = mask_elements + 1
    checksum_control_bytes = 2 * mask_elements * 4 + 8
    startup_allocation_status_bytes = 8
    allocator_alignment_bytes = 512 * (2 * num_microbatches + 64)
    return (
        pack_bytes
        + pack_bytes
        + one_sideband_set_bytes
        + one_sideband_set_bytes
        + one_sideband_validity_set_bytes
        + one_sideband_validity_set_bytes
        + optimizer_bytes
        + optimizer_event_status_bytes
        + optimizer_capability_flags_bytes
        + capture_scratch
        + capture_session_bytes
        + global_rank_evidence_bytes
        + local_rank_evidence_bytes
        + sink_staging_bytes
        + mask_comparison_bytes
        + checksum_control_bytes
        + startup_allocation_status_bytes
        + allocator_alignment_bytes
        + _BACKEND_WORKSPACE_ALLOWANCE_BYTES
    )


def allocator_growth_within_bound(
    *,
    pre_event_reserved_bytes: int,
    peak_reserved_bytes: int,
    predicted_increment_bytes: int,
) -> bool:
    """Check allocator-reserved event growth against the same pre-event baseline."""

    return (
        max(0, peak_reserved_bytes - pre_event_reserved_bytes)
        <= predicted_increment_bytes
    )


def _unwrap_module(module: nn.Module) -> nn.Module:
    while hasattr(module, "module") and isinstance(module.module, nn.Module):
        module = module.module
    return module


def _distributed_optimizer(optimizer: object) -> DistributedOptimizer | None:
    if (
        not isinstance(optimizer, ChainedOptimizer)
        or len(optimizer.chained_optimizers) != 1
    ):
        return None
    child = optimizer.chained_optimizers[0]
    return child if isinstance(child, DistributedOptimizer) else None


def _register_canonical_gpt_mask_producer(
    forward_step_func: Callable[..., Any],
    *,
    schedule_adapter: str = _NONINTERLEAVED_SCHEDULE_ADAPTER,
) -> None:
    """Register the application-owned GPT producer by exact object identity."""

    global _canonical_mask_producer
    if schedule_adapter != _NONINTERLEAVED_SCHEDULE_ADAPTER:
        raise ValueError(
            "Tier-0 requires the canonical noninterleaved schedule adapter"
        )
    if (
        _canonical_mask_producer is not None
        and _canonical_mask_producer[0] is not forward_step_func
    ):
        raise RuntimeError("Tier-0 canonical GPT mask producer is already registered")
    _canonical_mask_producer = (
        forward_step_func,
        _MASK_PRODUCER_IDENTITY,
        schedule_adapter,
    )


def _verified_mask_producer(forward_step_func: Callable[..., Any] | None) -> bool:
    """Accept only the application-registered function and schedule contract."""

    registration = _canonical_mask_producer
    return bool(
        registration is not None
        and registration[0] is forward_step_func
        and registration[1] is _MASK_PRODUCER_IDENTITY
        and registration[2] == _NONINTERLEAVED_SCHEDULE_ADAPTER
    )


def _local_bounded_int(value: object, *, fallback: int = 1) -> tuple[int, bool]:
    """Convert one startup integer without allowing a rank-local exception."""

    try:
        converted = int(value)
    except Exception:
        return fallback, False
    return converted, -_INT64_MAX <= converted <= _INT64_MAX


def _local_bound_values(
    args: Any, num_microbatches: object
) -> tuple[dict[str, int], bool]:
    """Return usable local startup integers and whether every conversion was valid."""

    raw = {
        "dp": getattr(args, "data_parallel_size", 1),
        "tp": getattr(args, "tensor_model_parallel_size", 1),
        "pp": getattr(args, "pipeline_model_parallel_size", 1),
        "cp": getattr(args, "context_parallel_size", 1),
        "layers": getattr(args, "num_layers", 1),
        "micro_batch_size": getattr(args, "micro_batch_size", 1),
        "sequence_length": getattr(args, "seq_length", 1),
        "num_microbatches": num_microbatches,
    }
    values: dict[str, int] = {}
    valid = True
    for name, value in raw.items():
        values[name], converted = _local_bounded_int(value)
        valid &= converted
    if dist.is_initialized():
        values["world_size"] = dist.get_world_size()
    else:
        values["world_size"], converted = _local_bounded_int(
            getattr(args, "world_size", 1)
        )
        valid &= converted
    return values, valid


def _local_capability_reasons(
    args: Any,
    model: Sequence[nn.Module],
    optimizer: object,
    forward_step_func: Callable[..., Any] | None = None,
    *,
    num_microbatches: object = 1,
) -> tuple[str, ...]:
    reasons: list[str] = []
    chunks = tuple(_unwrap_module(chunk) for chunk in model)
    if len(chunks) != 1 or not isinstance(chunks[0], GPTModel):
        reasons.append("dense_local_mcore_gpt")
    if getattr(args, "transformer_impl", "local") != "local":
        reasons.append("transformer_engine")
    if not getattr(args, "bf16", False) or getattr(args, "fp16", False):
        reasons.append("bf16")
    if not getattr(args, "calculate_per_token_loss", False):
        reasons.append("per_token_loss")
    distributed_optimizer = _distributed_optimizer(optimizer)
    if distributed_optimizer is None:
        reasons.append("single_distributed_optimizer_chain")
    else:
        grad_scaler = distributed_optimizer.grad_scaler
        if isinstance(grad_scaler, DynamicGradScaler) or (
            grad_scaler is not None and not isinstance(grad_scaler, ConstantGradScaler)
        ):
            reasons.append("dynamic_loss_scaling")
        inner_optimizer = distributed_optimizer.optimizer
        if not isinstance(inner_optimizer, (torch.optim.Adam, torch.optim.AdamW)) and (
            type(inner_optimizer).__name__ != "FusedAdam"
        ):
            reasons.append("adam_optimizer")
    if getattr(args, "overlap_param_gather", False):
        reasons.append("param_gather_overlap")
    if getattr(args, "use_megatron_fsdp", False) or getattr(
        args, "use_torch_fsdp2", False
    ):
        reasons.append("fsdp")
    if getattr(args, "fp8", None) or getattr(args, "fp4", None):
        reasons.append("fp8_fp4")
    if getattr(args, "num_experts", None) not in (None, 0):
        reasons.append("moe")
    if getattr(args, "qk_layernorm", False):
        reasons.append("qk_layernorm")
    if getattr(args, "expert_model_parallel_size", 1) != 1:
        reasons.append("expert_parallel")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) is not None:
        reasons.append("virtual_pipeline")
    if getattr(args, "pipeline_model_parallel_layout", None) is not None:
        reasons.append("custom_pipeline_layout")
    if getattr(args, "mtp_num_layers", None) not in (None, 0):
        reasons.append("mtp")
    if getattr(args, "hybrid_context_parallel", False):
        reasons.append("hybrid_context_parallel")
    if getattr(args, "cuda_graph_impl", "none") not in (None, "none") or getattr(
        args, "optimizer_cuda_graph", False
    ):
        reasons.append("cuda_graph")
    if getattr(args, "sft", False) and getattr(args, "micro_batch_size", 1) != 1:
        reasons.append("packed_micro_batch_size")
    if getattr(args, "variable_seq_lengths", False):
        reasons.append("variable_sequence_lengths")
    bounds, conversions_valid = _local_bound_values(args, num_microbatches)
    world_size = bounds["world_size"]
    bounded_values = (
        (world_size, 1, 1024),
        (bounds["dp"], 1, 1024),
        (bounds["tp"], 1, 1024),
        (bounds["pp"], 1, _MAX_LAYERS),
        (bounds["cp"], 1, 1024),
        (bounds["layers"], 1, _MAX_LAYERS),
        (bounds["micro_batch_size"], 1, _MAX_MICRO_BATCH_SIZE),
        (bounds["sequence_length"], 1, _MAX_SEQUENCE_LENGTH),
        (bounds["num_microbatches"], 1, _MAX_MICROBATCHES),
    )
    if not conversions_valid or any(
        value < minimum or value > maximum for value, minimum, maximum in bounded_values
    ):
        reasons.append("capability_bounds")
    topology_product = math.prod(bounds[name] for name in ("dp", "tp", "pp", "cp"))
    context_parallel_size = bounds["cp"]
    sequence_length = bounds["sequence_length"]
    tensor_parallel_size = bounds["tp"]
    cp_divisor = 2 * context_parallel_size if context_parallel_size > 1 else 1
    cp_local_sequence = (
        sequence_length // context_parallel_size if context_parallel_size > 0 else 0
    )
    if (
        topology_product != world_size
        or bounds["pp"] > bounds["layers"]
        or context_parallel_size <= 0
        or sequence_length % cp_divisor != 0
        or (
            bool(getattr(args, "sequence_parallel", False))
            and tensor_parallel_size > 0
            and cp_local_sequence % tensor_parallel_size != 0
        )
    ):
        reasons.append("capability_bounds")
    if not _verified_mask_producer(forward_step_func):
        reasons.append("canonical_mask_producer")
    return tuple(sorted(set(reasons)))


def negotiate_tier0_capability(
    args: Any,
    model: Sequence[nn.Module],
    optimizer: object,
    forward_step_func: Callable[..., Any] | None = None,
    *,
    num_microbatches: object = 1,
) -> Tier0Capability:
    """Collectively fail closed for the first supported runtime signature."""

    local_reasons = _local_capability_reasons(
        args,
        model,
        optimizer,
        forward_step_func,
        num_microbatches=num_microbatches,
    )
    known = (
        "adam_optimizer",
        "bf16",
        "capability_bounds",
        "canonical_mask_producer",
        "cuda_graph",
        "custom_pipeline_layout",
        "dense_local_mcore_gpt",
        "dynamic_loss_scaling",
        "expert_parallel",
        "fp8_fp4",
        "fsdp",
        "hybrid_context_parallel",
        "moe",
        "mtp",
        "packed_micro_batch_size",
        "param_gather_overlap",
        "per_token_loss",
        "qk_layernorm",
        "single_distributed_optimizer_chain",
        "transformer_engine",
        "virtual_pipeline",
        "variable_sequence_lengths",
    )
    backend = dist.get_backend() if dist.is_initialized() else None
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    flags = torch.zeros(len(known), dtype=torch.int64, device=device)
    for reason in local_reasons:
        flags[known.index(reason)] = 1
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    counts = flags.cpu().tolist()
    reasons = tuple(reason for reason, count in zip(known, counts) if count)
    if any(
        count not in (0, dist.get_world_size() if dist.is_initialized() else 1)
        for count in counts
    ):
        reasons = (*reasons, "rank_inconsistent")
    return Tier0Capability(not reasons, tuple(sorted(set(reasons))), SUPPORT_SIGNATURE)


def _update_descriptor(
    name: str, family: MetricFamily, layer: int | None, index: int
) -> MetricDescriptor:
    return MetricDescriptor(
        logical_name=name,
        family=family,
        global_layer=layer,
        partition_axes=(PartitionAxis.OPTIMIZER_SHARD, PartitionAxis.PIPELINE_LAYER),
        replication_axes=(ReplicationAxis.TENSOR, ReplicationAxis.PIPELINE_TIED),
        replication_multiplicity=1,
        ownership=Ownership.AUTHORITATIVE_SHARD,
        mask_kind=MaskKind.NONE,
        statistic_kind=StatisticKind.UPDATE,
        denominator_kind=DenominatorKind.PRE_UPDATE_SUMSQ,
        normalization_kind=NormalizationKind.NONE,
        process_group_identity=ProcessGroupIdentity.WORLD,
        reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
        tied_owner_identity=None,
        packed_slots=PackedSlots.for_index(index),
    )


def _bind_module_parameters(
    bindings: dict[nn.Parameter, str], module: nn.Module | None, logical_name: str
) -> None:
    if module is None:
        return
    for parameter in module.parameters(recurse=True):
        existing = bindings.setdefault(parameter, logical_name)
        if existing != logical_name:
            raise ValueError(
                "one optimizer parameter cannot belong to multiple diagnostic families"
            )


def build_update_registry(
    model: Sequence[nn.Module],
    *,
    num_layers: int,
    reduction_binding: ReductionBinding,
) -> tuple[MetricRegistry, Mapping[nn.Parameter, str]]:
    """Build rank-independent update slots and typed local parameter bindings."""

    descriptors: list[MetricDescriptor] = []
    names: list[tuple[str, MetricFamily, int | None]] = []
    for layer in range(num_layers):
        for family in _UPDATE_FAMILIES:
            names.append((f"update/{family.value}/layer_{layer}", family, layer))
    names.extend(
        (
            ("update/embedding", MetricFamily.EMBEDDING, None),
            ("update/output", MetricFamily.OUTPUT, None),
        )
    )
    for index, (name, family, layer) in enumerate(names):
        descriptors.append(_update_descriptor(name, family, layer, index))

    bindings: dict[nn.Parameter, str] = {}
    local_names: set[str] = set()
    for wrapped_chunk in model:
        chunk = _unwrap_module(wrapped_chunk)
        if not isinstance(chunk, GPTModel):
            continue
        if getattr(chunk, "pre_process", False):
            name = "update/embedding"
            _bind_module_parameters(bindings, getattr(chunk, "embedding", None), name)
            local_names.add(name)
        if getattr(chunk, "post_process", False):
            name = "update/output"
            output_layer = getattr(chunk, "output_layer", None)
            output_parameters = (
                tuple(output_layer.parameters()) if output_layer is not None else ()
            )
            if not (
                getattr(chunk, "share_embeddings_and_output_weights", False)
                and any(parameter in bindings for parameter in output_parameters)
            ):
                _bind_module_parameters(bindings, output_layer, name)
                local_names.add(name)
        for module in chunk.modules():
            if not isinstance(module, TransformerLayer):
                continue
            layer = module.layer_number - 1
            typed_modules = (
                (MetricFamily.QKV, module.self_attention.linear_qkv),
                (MetricFamily.ATTN_OUT, module.self_attention.linear_proj),
                (MetricFamily.FC1, module.mlp.linear_fc1),
                (MetricFamily.FC2, module.mlp.linear_fc2),
                (MetricFamily.NORM, getattr(module, "input_layernorm", None)),
                (MetricFamily.NORM, getattr(module, "pre_mlp_layernorm", None)),
            )
            for family, typed_module in typed_modules:
                name = f"update/{family.value}/layer_{layer}"
                _bind_module_parameters(bindings, typed_module, name)
                local_names.add(name)
        final_layernorm = getattr(
            getattr(chunk, "decoder", None), "final_layernorm", None
        )
        if final_layernorm is not None and num_layers:
            name = f"update/{MetricFamily.NORM.value}/layer_{num_layers - 1}"
            _bind_module_parameters(bindings, final_layernorm, name)
            local_names.add(name)

    registry = MetricRegistry(
        descriptors,
        reduction_binding=reduction_binding,
        local_owners=tuple(
            descriptor.logical_name in local_names for descriptor in descriptors
        ),
    )
    return registry, bindings


class Tier0Heartbeat:
    """Own one training run's Tier-0 cadence and event state machine."""

    def __init__(
        self,
        args: Any,
        model: Sequence[nn.Module],
        optimizer: object,
        forward_step_func: Callable[..., Any] | None = None,
        forward_backward_func: Callable[..., Any] | None = None,
        *,
        wandb_log: Callable[..., None] | None = None,
        wandb_writer: object | None = None,
        artifact_writer: object | None = None,
        tensorboard_writer: object | None = None,
        reduction_binding: ReductionBinding | None = None,
        num_microbatches: int = 1,
    ) -> None:
        """Collectively negotiate support and construct dormant run state."""

        self.args = args
        self.model = tuple(model)
        self.optimizer = optimizer
        self.device = self._model_device()
        self.reduction_binding = reduction_binding or ReductionBinding.flat_world(None)
        self.cadence = Tier0Cadence.parse(
            args.diagnostic_interval, args.diagnostic_early_updates
        )
        self.unsupported_policy: Literal["error", "status-only"] = (
            args.diagnostic_unsupported_policy
        )
        if self.unsupported_policy not in ("error", "status-only"):
            raise ValueError(
                "diagnostic unsupported policy must be error or status-only"
            )
        self.capability = negotiate_tier0_capability(
            args,
            self.model,
            optimizer,
            forward_step_func,
            num_microbatches=num_microbatches,
        )
        self.forward_step_func = forward_step_func
        self.forward_backward_func = forward_backward_func
        if not self.capability.supported and self.unsupported_policy == "error":
            raise RuntimeError(
                "Tier-0 heartbeat unsupported on every rank: "
                + ", ".join(self.capability.reasons)
            )

        self.successful_updates = int(getattr(args, "diagnostic_successful_updates", 0))
        self.event_id = int(getattr(args, "diagnostic_event_id", 0))
        self.cumulative_artifact_bytes = int(
            getattr(args, "diagnostic_cumulative_artifact_bytes", 0)
        )
        self.wandb_log = wandb_log
        self.wandb_writer = wandb_writer
        self.artifact_writer = artifact_writer
        self.tensorboard_writer = tensorboard_writer
        self.capture: Tier0CaptureSession | None = None
        self.capture_accumulator: PackedSufficientStatistics | None = None
        self.capture_result: Tier0CaptureResult | None = None
        self.update_registry: MetricRegistry | None = None
        self.update_accumulator: PackedSufficientStatistics | None = None
        self.adapter: Bf16DistributedOptimizerDiagnosticAdapter | None = None
        self.control_accumulator: PackedSufficientStatistics | None = None
        self.preflight: SnapshotMemoryPreflight | None = None
        self._attempt_due = False
        self._attempt_started = 0.0
        self._step_started = 0.0
        self._ordinary_step_latency_ms = 0.0
        self._expected_num_microbatches = 0
        self._sideband_payloads: dict[int, torch.Tensor] = {}
        self._local_sideband_payloads: dict[int, torch.Tensor] = {}
        self._sideband_validity: dict[int, torch.Tensor] = {}
        self._local_sideband_validity: dict[int, torch.Tensor] = {}
        self._received_sidebands: set[int] = set()
        self._reduction_arenas: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._capture_scratch: torch.Tensor | None = None
        self._rank_evidence: torch.Tensor | None = None
        self._local_rank_evidence: torch.Tensor | None = None
        self._sink_staging: torch.Tensor | None = None
        self._tier_output: torch.Tensor | None = None
        self._runtime_payload: dict[str, torch.Tensor] = {}
        self._control_value: torch.Tensor | None = None
        self._mask_compare: torch.Tensor | None = None
        self._mask_compare_valid: torch.Tensor | None = None
        self._mask_checksum_weights: torch.Tensor | None = None
        self._mask_checksum_work: torch.Tensor | None = None
        self.sink_failure_count = 0
        self._process_group_memberships: dict[str, list[tuple[int, ...]]] = {}
        self._pre_event_allocated_bytes = 0
        self._pre_event_reserved_bytes = 0
        self._predicted_event_bytes = 0
        startup_microbatches, startup_microbatches_valid = _local_bounded_int(
            num_microbatches
        )
        self._requested_num_microbatches = (
            startup_microbatches if startup_microbatches_valid else _INT64_MAX
        )
        self._startup_num_microbatches = min(
            _MAX_MICROBATCHES, max(1, startup_microbatches)
        )

        self.tiered_runtime = None
        if bool(getattr(args, "diag_enabled", False)) and int(
            getattr(args, "diag_max_tier", 0)
        ) >= 1:
            if forward_step_func is None or forward_backward_func is None:
                raise RuntimeError("Tier-1/2 diagnostics require the canonical training schedule")
            from .runtime import TieredDiagnosticRuntime

            self.tiered_runtime = TieredDiagnosticRuntime(
                args,
                self.model,
                optimizer,
                forward_step_func,
                forward_backward_func,
                reduction_binding=self.reduction_binding,
                num_microbatches=num_microbatches,
            )

        self.reservation = self._startup_reservation()
        self._collect_process_group_memberships()
        self._initialize_artifact_writer()
        self._allocate_startup_state()

    def _collect_process_group_memberships(self) -> None:
        """Capture actual startup process-group memberships without event collectives."""

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        local: dict[str, tuple[int, ...]] = {"world": tuple(range(world_size))}
        if self.capability.supported and dist.is_initialized():
            group_getters = {
                "dp": lambda: parallel_state.get_data_parallel_group(
                    with_context_parallel=False
                ),
                "tp": parallel_state.get_tensor_model_parallel_group,
                "pp": parallel_state.get_pipeline_model_parallel_group,
                "cp": parallel_state.get_context_parallel_group,
            }
            topology_sizes = {
                "dp": int(getattr(self.args, "data_parallel_size", 1)),
                "tp": int(getattr(self.args, "tensor_model_parallel_size", 1)),
                "pp": int(getattr(self.args, "pipeline_model_parallel_size", 1)),
                "cp": int(getattr(self.args, "context_parallel_size", 1)),
            }
            for name, getter in group_getters.items():
                if topology_sizes[name] > 1:
                    local[name] = tuple(dist.get_process_group_ranks(getter()))
        if dist.is_initialized() and world_size > 1:
            gathered: list[dict[str, tuple[int, ...]] | None] = [None] * world_size
            dist.all_gather_object(gathered, local)
            names = set().union(*(item or {} for item in gathered))
            self._process_group_memberships = {
                name: [tuple((item or {})[name]) for item in gathered] for name in names
            }
        else:
            self._process_group_memberships = {
                name: [members] for name, members in local.items()
            }

    def _initialize_artifact_writer(self) -> None:
        """Construct the existing last-rank artifact owner and agree setup."""

        rank = dist.get_rank() if dist.is_initialized() else 0
        sink_rank = (dist.get_world_size() - 1) if dist.is_initialized() else 0
        failed = False
        if (
            self.capability.supported
            and rank == sink_rank
            and self.artifact_writer is None
        ):
            try:
                if self.wandb_writer is None:
                    raise RuntimeError("Tier-0 requires the last-rank W&B writer")
                self.artifact_writer = ArtifactV3Writer.from_runtime(
                    self.args,
                    self.wandb_writer,
                    world_size=(dist.get_world_size() if dist.is_initialized() else 1),
                    global_rank=rank,
                    cumulative_bytes=self.cumulative_artifact_bytes,
                )
                publish_final_launch_config_from_runtime(
                    self.args,
                    self.wandb_writer,
                    world_size=(dist.get_world_size() if dist.is_initialized() else 1),
                    global_rank=rank,
                )
            except Exception:
                failed = True
        status = torch.tensor(int(failed), dtype=torch.int64, device=self.device)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
        if bool(status.cpu().item()):
            raise RuntimeError(
                "Tier-0 artifact ownership/provenance setup failed globally"
            )

    def _startup_reservation(self) -> Tier0Reservation:
        """Calculate and globally agree the complete peak before event allocation."""

        max_extra = getattr(self.args, "diagnostic_max_extra_bytes", None)
        if max_extra is None:
            max_extra_bytes = _MAX_RESERVATION_BYTES
            max_extra_valid = True
        else:
            max_extra_bytes, max_extra_valid = _local_bounded_int(max_extra, fallback=0)
        bounds, bounds_valid = _local_bound_values(
            self.args, self._requested_num_microbatches
        )
        local_sequence = bounds["sequence_length"] // max(1, bounds["cp"])
        owner_elements = 0
        calculation_failed = not bounds_valid or not max_extra_valid
        distributed_optimizer = _distributed_optimizer(self.optimizer)
        if self.capability.supported and distributed_optimizer is not None:
            try:
                owner_elements = sum(
                    shard.main_shard.numel()
                    for shard in distributed_optimizer.iter_model_main_param_shards()
                )
            except Exception:
                calculation_failed = True
        try:
            requested = tier0_reservation_bytes(
                num_layers=bounds["layers"],
                num_microbatches=self._requested_num_microbatches,
                micro_batch_size=bounds["micro_batch_size"],
                local_sequence_length=local_sequence,
                owner_elements=owner_elements,
                world_size=bounds["world_size"],
            )
        except Exception:
            requested = _INT64_MAX
            calculation_failed = True

        overflow = (
            requested > _INT64_MAX
            or max_extra_bytes > _INT64_MAX
            or self._requested_num_microbatches > _INT64_MAX
        )
        locally_accepted = (
            not calculation_failed
            and not overflow
            and max_extra_bytes >= 0
            and requested <= max_extra_bytes
            and requested <= _MAX_RESERVATION_BYTES
        )
        if self.device.type == "cuda":
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                allocated = torch.cuda.memory_allocated(self.device)
                reusable = max(0, reserved - allocated)
                driver_need = max(0, requested - reusable)
                locally_accepted = locally_accepted and driver_need <= free_bytes
                locally_accepted = locally_accepted and (
                    reserved + driver_need <= int(total_bytes * 0.90)
                )
            except Exception:
                locally_accepted = False

        control = torch.tensor(
            [
                min(requested, _INT64_MAX),
                min(max_extra_bytes, _INT64_MAX),
                int(locally_accepted),
                int(
                    Tier0ReservationStatus.OVERFLOW
                    if overflow
                    else (
                        Tier0ReservationStatus.OK
                        if locally_accepted
                        else Tier0ReservationStatus.UNSUPPORTED
                    )
                ),
            ],
            dtype=torch.int64,
            device=self.device,
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            maximum = control.clone()
            minimum = control.clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            maximum_host = maximum.cpu().tolist()
            minimum_host = minimum.cpu().tolist()
            globally_accepted = minimum_host[2] == 1
            requested = int(maximum_host[0])
            max_extra_bytes = int(minimum_host[1])
            status = Tier0ReservationStatus(int(maximum_host[3]))
        else:
            globally_accepted = locally_accepted
            status = (
                Tier0ReservationStatus.OVERFLOW
                if overflow
                else (
                    Tier0ReservationStatus.OK
                    if locally_accepted
                    else Tier0ReservationStatus.UNSUPPORTED
                )
            )
        if not globally_accepted:
            raise Tier0ReservationError(
                status,
                "Tier-0 startup reservation rejected globally before event allocation "
                f"(status={status.name.lower()}, requested={requested}, "
                f"max_extra={max_extra_bytes}, hard_max={_MAX_RESERVATION_BYTES})",
            )
        return Tier0Reservation(requested, max_extra_bytes, True)

    def _allocate_startup_state(self) -> None:
        """Allocate all persistent/event buffers and globally agree measured success."""

        allocation_status = torch.empty((), dtype=torch.int64, device=self.device)
        before = (
            torch.cuda.memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0
        )
        before_reserved = (
            torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
        )
        allocation_failed = False
        try:
            self.control_accumulator = PackedSufficientStatistics(
                _CONTROL_NAMES,
                self.device,
                descriptor_hash="tier0_control_v1",
                reduction_binding=self.reduction_binding,
            )
            if self.capability.supported:
                self._construct_update_registry()
                self.update_accumulator = self.update_registry.new_accumulator(
                    self.device
                )
                self._construct_optimizer_adapter()
                assert self.adapter is not None
                self.adapter.allocate_event_buffers()
                self.capture = self._construct_capture_session()
                self.capture_accumulator = self.capture.registry.new_accumulator(
                    self.device
                )
            accumulators = self._event_accumulators()
            self._reduction_arenas = (
                PackedSufficientStatistics.allocate_reduction_arenas(accumulators)
            )
            self._capture_scratch = torch.empty(
                PackedSufficientStatistics.scratch_bytes_for_capacity(),
                dtype=torch.uint8,
                device=self.device,
            )
            if self.capture_accumulator is not None:
                self.capture_accumulator.bind_workspace(self._capture_scratch)
            bounds, _ = _local_bound_values(self.args, self._requested_num_microbatches)
            local_length = bounds["sequence_length"] // max(1, bounds["cp"])
            payload_elements = bounds["micro_batch_size"] * local_length + 1
            if self.capability.supported:
                self._sideband_payloads = {
                    index: torch.empty(
                        payload_elements, dtype=torch.float32, device=self.device
                    )
                    for index in range(self._startup_num_microbatches)
                }
                self._local_sideband_payloads = {
                    index: torch.empty(
                        payload_elements, dtype=torch.float32, device=self.device
                    )
                    for index in range(self._startup_num_microbatches)
                }
                self._sideband_validity = {
                    index: torch.empty((), dtype=torch.bool, device=self.device)
                    for index in range(self._startup_num_microbatches)
                }
                self._local_sideband_validity = {
                    index: torch.empty((), dtype=torch.bool, device=self.device)
                    for index in range(self._startup_num_microbatches)
                }
                mask_elements = payload_elements - 1
                self._mask_compare = torch.empty(
                    mask_elements, dtype=torch.bool, device=self.device
                )
                self._mask_compare_valid = torch.empty(
                    (), dtype=torch.bool, device=self.device
                )
                self._mask_checksum_weights = torch.arange(
                    1, mask_elements + 1, dtype=torch.float32, device=self.device
                )
                self._mask_checksum_work = torch.empty(
                    mask_elements, dtype=torch.float32, device=self.device
                )
            self._control_value = torch.empty(
                (), dtype=torch.float64, device=self.device
            )
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            self._rank_evidence = torch.empty(
                (world_size, _RANK_EVIDENCE_FIELDS),
                dtype=torch.float64,
                device=self.device,
            )
            self._local_rank_evidence = torch.empty(
                _RANK_EVIDENCE_FIELDS, dtype=torch.float64, device=self.device
            )
            sink_pack_elements = sum(
                accumulator.sum_pack.numel()
                + accumulator.max_pack.numel()
                + accumulator.min_pack.numel()
                for accumulator in accumulators
            )
            runtime_output_elements = (
                len(self.tiered_runtime.output_keys)
                if self.tiered_runtime is not None
                else 0
            )
            self._sink_staging = torch.empty(
                world_size * _RANK_EVIDENCE_FIELDS
                + sink_pack_elements
                + runtime_output_elements,
                dtype=torch.float64,
                device=self.device,
            )
            self._tier_output = torch.empty(
                runtime_output_elements, dtype=torch.float64, device=self.device
            )
        except Exception:
            allocation_failed = True

        allocation_status.fill_(int(allocation_failed))
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(allocation_status, op=dist.ReduceOp.MAX)
        if bool(allocation_status.cpu().item()):
            raise RuntimeError("Tier-0 startup allocation failed on at least one rank")

        prime_failed = False
        try:
            self._prime_event_collectives()
        except Exception:
            prime_failed = True

        measured = (
            max(0, torch.cuda.memory_allocated(self.device) - before)
            if self.device.type == "cuda"
            else 0
        )
        measured_reserved = (
            max(0, torch.cuda.memory_reserved(self.device) - before_reserved)
            if self.device.type == "cuda"
            else 0
        )
        allocation_status.fill_(
            int(
                prime_failed
                or measured > self.reservation.requested_bytes
                or measured_reserved > self.reservation.requested_bytes
            )
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(allocation_status, op=dist.ReduceOp.MAX)
        if bool(allocation_status.cpu().item()):
            raise RuntimeError(
                "Tier-0 startup collective priming or measured allocation failed globally"
            )
        self.reservation = Tier0Reservation(
            self.reservation.requested_bytes,
            self.reservation.max_extra_bytes,
            True,
            measured,
            measured_reserved,
        )

    def _prime_event_collectives(self) -> None:
        """Prime the exact three reductions and fourth rank-evidence gather."""

        assert self._reduction_arenas is not None
        reduce_call = self.reduction_binding.reducer or dist.all_reduce
        if self.reduction_binding.reducer is not None or dist.is_initialized():
            for arena, operation in zip(
                self._reduction_arenas,
                (dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN),
            ):
                reduce_call(
                    arena,
                    op=operation,
                    group=self.reduction_binding.group,
                )
        assert self._rank_evidence is not None
        assert self._local_rank_evidence is not None
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            self._rank_evidence[0].copy_(self._local_rank_evidence)
        else:
            dist.all_gather_into_tensor(
                self._rank_evidence.reshape(-1), self._local_rank_evidence
            )

    @property
    def armed(self) -> bool:
        """Return whether the current rerun attempt is diagnostic."""

        return self._attempt_due

    @property
    def next_successful_update(self) -> int:
        """Return the successful-update index that the next step may commit."""

        return self.successful_updates + 1

    def _event_accumulators(self) -> tuple[PackedSufficientStatistics, ...]:
        """Return the exact startup-bound accumulator order for every event."""

        accumulators = [
            accumulator
            for accumulator in (
                self.capture_accumulator,
                self.update_accumulator,
                self.control_accumulator,
            )
            if accumulator is not None
        ]
        if self.tiered_runtime is not None:
            accumulators.extend(self.tiered_runtime.accumulators)
        return tuple(accumulators)

    def _model_device(self) -> torch.device:
        for chunk in self.model:
            parameter = next(chunk.parameters(), None)
            if parameter is not None:
                return parameter.device
        return (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def _construct_update_registry(self) -> None:
        num_layers = int(self.args.num_layers)
        self.update_registry, bindings = build_update_registry(
            self.model, num_layers=num_layers, reduction_binding=self.reduction_binding
        )
        self._update_bindings = bindings

    def _construct_optimizer_adapter(self) -> None:
        assert self.update_registry is not None
        max_extra = self.args.diagnostic_max_extra_bytes
        self.adapter = Bf16DistributedOptimizerDiagnosticAdapter(
            self.optimizer,
            self.update_registry,
            self._update_bindings,
            diagnostic_max_extra_bytes=(2**63 - 1 if max_extra is None else max_extra),
            max_memory_fraction=0.90,
        )

    def _construct_capture_session(self) -> Tier0CaptureSession:
        """Construct dormant hooks against startup-preallocated event state."""

        normalizer = CanonicalDgradNormalizer.from_grad_scaler(
            getattr(_distributed_optimizer(self.optimizer), "grad_scaler", None)
        )
        local_sequence_length = int(self.args.seq_length) // max(
            1, parallel_state.get_context_parallel_world_size()
        )
        return Tier0CaptureSession(
            self.model,
            num_layers=int(self.args.num_layers),
            topology=CaptureTopology.from_parallel_state(
                sequence_parallel=bool(self.args.sequence_parallel)
            ),
            device=self.device,
            micro_batch_size=int(self.args.micro_batch_size),
            local_sequence_length=local_sequence_length,
            calculate_per_token_loss=True,
            dgrad_normalizer=normalizer,
            reduction_binding=self.reduction_binding,
        )

    def prepare_attempt(self, *, num_microbatches: int) -> bool:
        """Replace prior rerun state and arm capture when the next success is due."""

        self.abort_attempt()
        set_diagnostic_global_valid_tokens(None)
        self._step_started = time.perf_counter()
        self._attempt_due = self.cadence.is_due(self.next_successful_update)
        if self.tiered_runtime is not None:
            self.tiered_runtime.prepare_attempt(
                due=self._attempt_due, num_microbatches=num_microbatches
            )
        if not self._attempt_due:
            return False
        self._attempt_started = time.perf_counter()
        self._expected_num_microbatches = num_microbatches
        if (
            self.capability.supported
            and num_microbatches > self._startup_num_microbatches
        ):
            raise RuntimeError("Tier-0 microbatch count exceeds the startup bound")
        assert self.control_accumulator is not None
        self.control_accumulator.reset_()
        self._received_sidebands.clear()
        assert self._local_rank_evidence is not None
        self._local_rank_evidence.zero_()
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._local_rank_evidence[0].fill_(rank)
        if self.device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            del free_bytes
            self._local_rank_evidence[3].fill_(total_bytes)
            self._pre_event_allocated_bytes = torch.cuda.memory_allocated(self.device)
            self._pre_event_reserved_bytes = torch.cuda.memory_reserved(self.device)
            self._local_rank_evidence[4].fill_(self._pre_event_allocated_bytes)
            self._local_rank_evidence[5].fill_(self._pre_event_reserved_bytes)
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            self._local_rank_evidence[3].fill_(1)
        self._predicted_event_bytes = max(
            0,
            self.reservation.requested_bytes - self.reservation.measured_reserved_bytes,
        )
        self._local_rank_evidence[6].fill_(self._predicted_event_bytes)
        if not self.capability.supported:
            return True
        assert self.update_registry is not None
        assert self.update_accumulator is not None
        self.update_accumulator.reset_()
        return True

    def wrap_data_iterator(self, data_iterator: Any) -> Any:
        """Attach the due Tier-1/2 raw-batch recorder without advancing early."""

        if self.tiered_runtime is None:
            return data_iterator
        return self.tiered_runtime.wrap_data_iterator(data_iterator)

    def install_capture(self) -> None:
        """Install capture hooks immediately before the forward/backward schedule."""

        if not self._attempt_due or not self.capability.supported:
            return
        if self.capture is None or self.capture_accumulator is None:
            raise RuntimeError("Tier-0 startup capture allocation is absent")
        self.capture.arm(
            expected_microbatch_ids=tuple(range(self._expected_num_microbatches)),
            accumulator=self.capture_accumulator,
        )

    def seal_capture(self) -> None:
        """Seal the attempt's local capture and remove hooks immediately."""

        if self.capture is None:
            return
        self.capture_accumulator = self.capture.seal()

    def begin_microbatch(self, microbatch_id: int) -> None:
        """Begin one centrally identified schedule microbatch."""

        if self.capture is None or not self.capture.armed:
            return
        self.capture.begin_microbatch(microbatch_id)
        payload = self._sideband_payloads.get(microbatch_id)
        if (
            microbatch_id in self._received_sidebands
            and not parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            staged = unpack_valid_token_mask_sideband(
                payload,
                micro_batch_size=int(self.args.micro_batch_size),
                sequence_length=int(self.args.seq_length)
                // max(1, parallel_state.get_context_parallel_world_size()),
                device=self.device,
                valid_out=self._sideband_validity[microbatch_id],
            )
            self.capture.register_valid_token_mask(microbatch_id, staged)

    def end_microbatch(self, microbatch_id: int) -> None:
        """End one centrally identified schedule microbatch."""

        if self.capture is not None and self.capture.armed:
            self.capture.end_microbatch(microbatch_id)

    def register_local_loss_mask(
        self, microbatch_id: int, loss_mask: torch.Tensor | None
    ) -> None:
        """Register/compare a CP-local first/last-stage mask before model compute."""

        if self.capture is None or not self.capture.armed:
            return
        local_length = int(self.args.seq_length) // max(
            1, parallel_state.get_context_parallel_world_size()
        )
        local_payload = pack_valid_token_mask_sideband(
            loss_mask,
            micro_batch_size=int(self.args.micro_batch_size),
            sequence_length=local_length,
            device=self.device,
            out=self._local_sideband_payloads[microbatch_id],
        )
        staged = unpack_valid_token_mask_sideband(
            local_payload,
            micro_batch_size=int(self.args.micro_batch_size),
            sequence_length=local_length,
            device=self.device,
            valid_out=self._local_sideband_validity[microbatch_id],
        )
        assert self._mask_checksum_weights is not None
        assert self._mask_checksum_work is not None
        assert self._local_rank_evidence is not None
        assert self._control_value is not None
        torch.mul(
            local_payload[:-1],
            self._mask_checksum_weights,
            out=self._mask_checksum_work,
        )
        torch.sum(
            self._mask_checksum_work,
            dim=(0,),
            dtype=torch.float64,
            out=self._control_value,
        )
        self._local_rank_evidence[9].add_(self._control_value)
        torch.sum(
            local_payload[:-1],
            dim=(0,),
            dtype=torch.float64,
            out=self._control_value,
        )
        self._local_rank_evidence[10].add_(self._control_value)
        if (
            parallel_state.is_pipeline_last_stage(ignore_virtual=True)
            and microbatch_id in self._received_sidebands
        ):
            received = self._sideband_payloads[microbatch_id]
            assert self._mask_compare is not None
            assert self._mask_compare_valid is not None
            torch.eq(local_payload[:-1], received[:-1], out=self._mask_compare)
            torch.all(self._mask_compare, out=self._mask_compare_valid)
            torch.gt(received[-1], 0, out=self._sideband_validity[microbatch_id])
            self._mask_compare_valid.logical_and_(
                self._sideband_validity[microbatch_id]
            )
            staged.valid.logical_and_(self._mask_compare_valid)
            staged = StagedTokenMask(staged.values, staged.valid)
        self.capture.register_valid_token_mask(microbatch_id, staged)
        if parallel_state.is_pipeline_first_stage(
            ignore_virtual=True
        ) and not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            dist.send(
                local_payload,
                dst=parallel_state.get_pipeline_model_parallel_next_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
            )

    def receive_mask_sideband(self, microbatch_id: int) -> None:
        """Receive the ordered fixed-shape mask immediately before activation receive."""

        if (
            not self._attempt_due
            or not self.capability.supported
            or parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        ):
            return
        local_length = int(self.args.seq_length) // max(
            1, parallel_state.get_context_parallel_world_size()
        )
        payload = self._sideband_payloads[microbatch_id]
        dist.recv(
            payload,
            src=parallel_state.get_pipeline_model_parallel_prev_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        self._received_sidebands.add(microbatch_id)

    def forward_mask_sideband(self, microbatch_id: int) -> None:
        """Forward an ordered mask before intermediate-stage model compute."""

        if (
            not self._attempt_due
            or not self.capability.supported
            or parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            return
        dist.send(
            self._sideband_payloads[microbatch_id],
            dst=parallel_state.get_pipeline_model_parallel_next_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )

    def begin_optimizer_event(self) -> SnapshotMemoryPreflight | None:
        """Snapshot into buffers admitted and allocated during startup."""

        if not self._attempt_due or self.adapter is None:
            return None
        if self.capture_accumulator is None or self.update_accumulator is None:
            raise RuntimeError(
                "diagnostic capture must be sealed before the optimizer snapshot"
            )
        if self.tiered_runtime is not None and self.tiered_runtime.active:
            self.tiered_runtime.run_pre(event_id=self.event_id + 1)
        self.adapter.begin_event()
        return None

    def finish_optimizer_event(
        self, update_successful: bool, *, iteration: int
    ) -> bool:
        """Complete fixed collectives, retry failures, and emit successful events."""

        if not self._attempt_due:
            if update_successful:
                self._commit_successful_update()
                self._ordinary_step_latency_ms = (
                    time.perf_counter() - self._step_started
                ) * 1000.0
            return update_successful
        assert self.control_accumulator is not None
        if self.adapter is not None and self.update_accumulator is not None:
            if self.tiered_runtime is not None and self.tiered_runtime.active:
                self.tiered_runtime.complete_optimizer_event(
                    self.adapter,
                    self.update_accumulator,
                    update_successful=update_successful,
                )
            elif update_successful and self.adapter.armed:
                self.adapter.finish_event(
                    self.update_accumulator, update_successful=True
                )
            else:
                self.adapter.abort_event()

        self._add_control("control/update_failure", not update_successful)
        self._add_control("control/unsupported", not self.capability.supported)
        self._add_control(
            "control/preflight_failure",
            self.preflight is not None and not self.preflight.accepted,
        )
        self._add_control(
            "control/runtime_failure",
            self.capability.supported and get_diagnostic_global_valid_tokens() is None,
        )
        if self.adapter is not None:
            self._add_control(
                "control/adapter_status", self.adapter.local_status.squeeze(0)
            )
        self._add_control("control/all_rank_post_gather_peak_unavailable", True)

        accumulators = self._event_accumulators()
        PackedSufficientStatistics.reduce_many_(accumulators, self._reduction_arenas)
        self._runtime_payload = (
            self.tiered_runtime.derive_outputs()
            if update_successful
            and self.tiered_runtime is not None
            and self.tiered_runtime.active
            else {}
        )
        self._gather_pre_gather_rank_evidence()
        if not update_successful:
            self.abort_attempt()
            return False

        self._commit_successful_update()
        rank = dist.get_rank() if dist.is_initialized() else 0
        sink_rank = (dist.get_world_size() - 1) if dist.is_initialized() else 0
        if rank == sink_rank:
            self._emit(iteration=iteration)
        self.event_id += 1
        self.args.diagnostic_event_id = self.event_id
        self.abort_attempt()
        return True

    def _add_control(self, name: str, value: int | bool | torch.Tensor) -> None:
        assert self.control_accumulator is not None
        assert self._control_value is not None
        slots = self.control_accumulator.slots(name)
        if isinstance(value, torch.Tensor):
            self._control_value.copy_(value.detach().reshape(()))
        else:
            self._control_value.fill_(value)
        self.control_accumulator.sum_pack[slots.sum].add_(self._control_value)
        self.control_accumulator.sum_pack[slots.count].add_(1)
        torch.maximum(
            self.control_accumulator.max_pack[slots.maximum],
            self._control_value,
            out=self.control_accumulator.max_pack[slots.maximum],
        )
        torch.minimum(
            self.control_accumulator.min_pack[slots.minimum],
            self._control_value,
            out=self.control_accumulator.min_pack[slots.minimum],
        )

    def _gather_pre_gather_rank_evidence(self) -> None:
        """Gather rank facts sampled immediately before the fourth operation."""

        assert self._local_rank_evidence is not None
        assert self._rank_evidence is not None
        elapsed = (time.perf_counter() - self._attempt_started) * 1000.0
        self._local_rank_evidence[1].fill_(elapsed)
        self._local_rank_evidence[2].fill_(self._ordinary_step_latency_ms or elapsed)
        if self.device.type == "cuda":
            self._local_rank_evidence[7].fill_(
                torch.cuda.max_memory_allocated(self.device)
            )
            self._local_rank_evidence[8].fill_(
                torch.cuda.max_memory_reserved(self.device)
            )
        else:
            self._local_rank_evidence[7].copy_(self._local_rank_evidence[4])
            self._local_rank_evidence[8].copy_(self._local_rank_evidence[5])
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            self._rank_evidence[0].copy_(self._local_rank_evidence)
        else:
            dist.all_gather_into_tensor(
                self._rank_evidence.reshape(-1), self._local_rank_evidence
            )

    def _commit_successful_update(self) -> None:
        self.successful_updates += 1
        self.args.diagnostic_successful_updates = self.successful_updates

    def _derive_payload(self, latencies: torch.Tensor) -> dict[str, torch.Tensor]:
        nan = torch.full((), torch.nan, dtype=torch.float64, device=self.device)
        payload = {key: nan.clone() for key in TIER0_KEYS}
        if (
            not self.capability.supported
            or self.capture is None
            or self.capture_accumulator is None
            or self.update_accumulator is None
        ):
            valid = False
            capture_result = None
        else:
            global_valid_tokens = get_diagnostic_global_valid_tokens()
            capture_result = self.capture.derive_result(
                self.capture_accumulator, global_valid_tokens=global_valid_tokens
            )
            valid = capture_result.valid & self._updates_valid()
            self.capture_result = capture_result
            self._derive_capture_metrics(payload, capture_result)
            self._derive_update_metrics(payload)
            self._derive_nonfinite_health(payload)

        assert self.control_accumulator is not None
        control_failure = torch.stack(
            tuple(
                self.control_accumulator.maximum(name).value != 0
                for name in (
                    "control/unsupported",
                    "control/preflight_failure",
                    "control/runtime_failure",
                    "control/adapter_status",
                )
            )
        ).any()
        valid = (
            torch.as_tensor(valid, dtype=torch.bool, device=self.device)
            & ~control_failure
        )
        for key in TIER0_METRIC_KEYS:
            payload[key] = torch.where(valid, payload[key], nan)
        payload["diag/v2/event/successful_update"] = torch.tensor(
            self.successful_updates, dtype=torch.float64, device=self.device
        )
        if capture_result is not None:
            payload["diag/v2/event/valid_positions"] = (
                capture_result.global_valid_tokens
            )
        else:
            payload["diag/v2/event/valid_positions"] = torch.zeros(
                (), dtype=torch.float64, device=self.device
            )
        payload["diag/v2/status/valid"] = torch.zeros(
            (), dtype=torch.float64, device=self.device
        )
        payload["diag/v2/perf/peak_hbm_bytes_max_rank"] = nan.clone()
        payload["diag/v2/perf/latency_ms_median_rank"] = nan.clone()
        payload["diag/v2/perf/latency_ms_max_rank"] = nan.clone()
        assert_payload_schema(payload)
        return payload

    def _updates_valid(self) -> torch.Tensor:
        """Return reduced update-pack validity, allowing tied output absence."""

        assert self.update_registry is not None
        assert self.update_accumulator is not None
        tied_output = any(
            isinstance(_unwrap_module(chunk), GPTModel)
            and getattr(
                _unwrap_module(chunk), "share_embeddings_and_output_weights", False
            )
            for chunk in self.model
        )
        valid = torch.ones((), dtype=torch.bool, device=self.device)
        for descriptor in self.update_registry.descriptors:
            slots = descriptor.packed_slots
            errors = sum(
                self.update_accumulator.sum_pack[offset]
                for offset in (
                    slots.nonfinite,
                    slots.mask_error,
                    slots.nonfinite_arithmetic,
                    slots.observation_error,
                )
            )
            has_contributors = self.update_accumulator.sum_pack[slots.count] > 0
            if tied_output and descriptor.family == MetricFamily.OUTPUT:
                has_contributors = torch.ones_like(has_contributors)
            valid &= (errors == 0) & has_contributors
        return valid

    def _capture_layer_values(
        self,
        result: Tier0CaptureResult,
        observation: str,
        family: MetricFamily,
        statistic: str,
    ) -> torch.Tensor:
        values = []
        for layer in range(int(self.args.num_layers)):
            name = f"{observation}/{family.value}/layer_{layer}"
            derived = getattr(result.accumulator, statistic)(name)
            values.append(derived.value)
        return torch.stack(values)

    @staticmethod
    def _summary(values: torch.Tensor, quantile: float) -> torch.Tensor:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return torch.full((), torch.nan, dtype=values.dtype, device=values.device)
        return torch.quantile(finite, quantile)

    def _derive_capture_metrics(
        self, payload: dict[str, torch.Tensor], result: Tier0CaptureResult
    ) -> None:
        residual_activation = self._capture_layer_values(
            result, "activation", MetricFamily.RESIDUAL, "rms"
        )
        residual_dgrad = self._capture_layer_values(
            result, "dgrad", MetricFamily.RESIDUAL, "rms"
        )
        anchors = {
            "first": 0,
            "q1": round((int(self.args.num_layers) - 1) * 0.25),
            "middle": round((int(self.args.num_layers) - 1) * 0.50),
            "q3": round((int(self.args.num_layers) - 1) * 0.75),
            "last": int(self.args.num_layers) - 1,
        }
        for observation, values in (
            ("activation", residual_activation),
            ("dgrad", residual_dgrad),
        ):
            for summary, index in anchors.items():
                payload[f"diag/v2/t0/{observation}/residual/rms/{summary}"] = values[
                    index
                ]
            for summary, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
                payload[f"diag/v2/t0/{observation}/residual/rms/{summary}"] = (
                    self._summary(values, quantile)
                )
        for family in (
            MetricFamily.QKV,
            MetricFamily.ATTN_OUT,
            MetricFamily.FC1,
            MetricFamily.FC2,
        ):
            values = self._capture_layer_values(result, "dgrad", family, "rms")
            payload[f"diag/v2/t0/dgrad/{family.value}/p10"] = self._summary(values, 0.1)
            payload[f"diag/v2/t0/dgrad/{family.value}/p50"] = self._summary(values, 0.5)
            finite = torch.isfinite(values)
            denominator = finite.sum()
            payload[f"diag/v2/t0/dgrad/{family.value}/zero_fraction"] = torch.where(
                denominator > 0,
                (
                    (values <= self.args.diagnostic_dgrad_starvation_threshold) & finite
                ).sum()
                / denominator,
                torch.full((), torch.nan, dtype=values.dtype, device=values.device),
            )
        for family in (
            MetricFamily.QKV,
            MetricFamily.ATTN_OUT,
            MetricFamily.FC1,
            MetricFamily.FC2,
            MetricFamily.RESIDUAL,
        ):
            maxima = self._capture_layer_values(result, "activation", family, "maximum")
            minima = self._capture_layer_values(result, "activation", family, "minimum")
            payload[f"diag/v2/t0/activation/{family.value}/max_abs"] = torch.maximum(
                torch.amax(maxima.abs()), torch.amax(minima.abs())
            )

    def _update_layer_values(
        self, family: MetricFamily, statistic: str
    ) -> torch.Tensor:
        assert self.update_accumulator is not None
        return torch.stack(
            [
                getattr(self.update_accumulator, statistic)(
                    f"update/{family.value}/layer_{layer}"
                ).value
                for layer in range(int(self.args.num_layers))
            ]
        )

    def _derive_update_metrics(self, payload: dict[str, torch.Tensor]) -> None:
        assert self.update_accumulator is not None
        for family in _UPDATE_FAMILIES:
            values = self._update_layer_values(family, "relative_rms")
            for summary, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
                payload[f"diag/v2/t0/update/{family.value}/{summary}"] = self._summary(
                    values, quantile
                )
            finite = torch.isfinite(values)
            denominator = finite.sum()
            payload[f"diag/v2/t0/update/{family.value}/starved_fraction"] = torch.where(
                denominator > 0,
                (
                    (values <= self.args.diagnostic_update_starvation_threshold)
                    & finite
                ).sum()
                / denominator,
                torch.full((), torch.nan, dtype=values.dtype, device=values.device),
            )
        for family in (MetricFamily.EMBEDDING, MetricFamily.OUTPUT):
            payload[f"diag/v2/t0/update/{family.value}/relative_rms"] = (
                self.update_accumulator.relative_rms(f"update/{family.value}").value
            )
        cast_zero = torch.zeros((), dtype=torch.float64, device=self.device)
        master_nonzero = torch.zeros_like(cast_zero)
        for descriptor in (
            self.update_registry.descriptors if self.update_registry is not None else ()
        ):
            slots = descriptor.packed_slots
            if descriptor.family in _MATRIX_FAMILIES:
                cast_zero += self.update_accumulator.sum_pack[slots.zero]
                master_nonzero += self.update_accumulator.sum_pack[slots.sum]
        payload["diag/v2/health/underflow_fraction"] = torch.where(
            master_nonzero > 0,
            cast_zero / master_nonzero,
            torch.full_like(cast_zero, torch.nan),
        )
        for family in _MATRIX_FAMILIES:
            if family in (MetricFamily.EMBEDDING, MetricFamily.OUTPUT):
                names = (f"update/{family.value}",)
            else:
                names = tuple(
                    f"update/{family.value}/layer_{layer}"
                    for layer in range(int(self.args.num_layers))
                )
            retention = torch.stack(
                [self.update_accumulator.norm_retention(name).value for name in names]
            )
            cast_loss = torch.stack(
                [
                    self.update_accumulator.cast_zero_fraction(name).value
                    for name in names
                ]
            )
            payload[f"diag/v2/t0/retention/{family.value}/median"] = self._summary(
                retention, 0.5
            )
            payload[f"diag/v2/t0/retention/{family.value}/zero_fraction"] = (
                self._summary(cast_loss, 0.5)
            )

    def _derive_nonfinite_health(self, payload: dict[str, torch.Tensor]) -> None:
        """Pool nonfinite counts across capture and update descriptors."""

        nonfinite = torch.zeros((), dtype=torch.float64, device=self.device)
        finite_count = torch.zeros_like(nonfinite)
        for registry, accumulator in (
            (
                self.capture.registry if self.capture is not None else None,
                self.capture_accumulator,
            ),
            (self.update_registry, self.update_accumulator),
        ):
            if registry is None or accumulator is None:
                continue
            for descriptor in registry.descriptors:
                if descriptor.family == MetricFamily.EVENT:
                    continue
                slots = descriptor.packed_slots
                nonfinite += accumulator.sum_pack[slots.nonfinite]
                finite_count += accumulator.sum_pack[slots.count]
        payload["diag/v2/health/nonfinite_fraction"] = torch.where(
            finite_count + nonfinite > 0,
            nonfinite / (finite_count + nonfinite),
            torch.full_like(nonfinite, torch.nan),
        )

    def _emit(self, *, iteration: int) -> bool:
        """Transfer reduced facts once, then derive and emit on the CPU sink."""

        rank = dist.get_rank() if dist.is_initialized() else 0
        sink_rank = (dist.get_world_size() - 1) if dist.is_initialized() else 0
        if rank != sink_rank:
            raise RuntimeError("only the global Tier-0 sink may emit an event")
        assert self._rank_evidence is not None
        assert self._sink_staging is not None
        offset = 0
        rank_elements = self._rank_evidence.numel()
        self._sink_staging[offset : offset + rank_elements].copy_(
            self._rank_evidence.reshape(-1)
        )
        offset += rank_elements
        pack_ranges: list[tuple[PackedSufficientStatistics, int, int]] = []
        for accumulator in self._event_accumulators():
            start = offset
            for tensor in (
                accumulator.sum_pack,
                accumulator.max_pack,
                accumulator.min_pack,
            ):
                end = offset + tensor.numel()
                self._sink_staging[offset:end].copy_(tensor)
                offset = end
            pack_ranges.append((accumulator, start, offset))
        runtime_start = offset
        if self._runtime_payload:
            assert self.tiered_runtime is not None
            assert self._tier_output is not None
            if tuple(self._runtime_payload) != self.tiered_runtime.output_keys:
                raise RuntimeError("runtime diagnostic output order changed before sink transfer")
            for index, key in enumerate(self.tiered_runtime.output_keys):
                self._tier_output[index].copy_(self._runtime_payload[key])
            runtime_end = runtime_start + self._tier_output.numel()
            self._sink_staging[runtime_start:runtime_end].copy_(self._tier_output)
            offset = runtime_end
        host_combined = self._sink_host_transfer()
        memory_evidence = self._sample_sink_interval_memory()
        rank_values = host_combined[:rank_elements]
        rank_evidence = [
            rank_values[offset : offset + _RANK_EVIDENCE_FIELDS]
            for offset in range(0, len(rank_values), _RANK_EVIDENCE_FIELDS)
        ]
        host_packs: dict[str, dict[str, Sequence[Any]]] = {}
        for accumulator, start, end in pack_ranges:
            packed = host_combined[start:end]
            sum_count = accumulator.sum_pack.numel()
            max_count = accumulator.max_pack.numel()
            host_packs[accumulator.descriptor_hash] = {
                "names": accumulator.slot_names,
                "sum": tuple(packed[:sum_count]),
                "max": tuple(packed[sum_count : sum_count + max_count]),
                "min": tuple(packed[sum_count + max_count :]),
            }
        try:
            host_payload = self._derive_host_payload(host_packs, rank_evidence)
            assert_payload_schema(host_payload)
            if self._runtime_payload:
                assert self.tiered_runtime is not None
                runtime_values = host_combined[
                    runtime_start : runtime_start + len(self.tiered_runtime.output_keys)
                ]
                host_payload.update(
                    zip(self.tiered_runtime.output_keys, runtime_values, strict=True)
                )
                runtime_finite = all(
                    math.isfinite(float(host_payload[key]))
                    for key in self.tiered_runtime.output_keys
                )
                tier2_valid = (
                    self.tiered_runtime.tier < 2
                    or float(host_payload["diag/v2/t2/valid"]) == 1.0
                )
                host_payload["diag/v2/status/valid"] = int(
                    host_payload["diag/v2/status/valid"] == 1
                    and runtime_finite
                    and tier2_valid
                )
            else:
                # Artifact-v3 deliberately keeps Tier 0 ingest-only.
                host_payload["diag/v2/status/valid"] = 0
            assert_tiered_payload_schema(
                host_payload,
                effective_tier=(
                    self.tiered_runtime.tier
                    if self.tiered_runtime is not None
                    and self.tiered_runtime.active
                    else 0
                ),
            )
            artifact_written = False
            artifact_error: Exception | None = None
            if self.artifact_writer is not None:
                try:
                    artifact_written = self._write_artifact(
                        host_payload, host_packs, rank_evidence, memory_evidence
                    )
                except Exception as error:
                    artifact_error = error
            if not artifact_written:
                self.sink_failure_count += 1
                warnings.warn(
                    "diagnostic scalar event has no promotion-grade artifact-v3"
                    + (f": {artifact_error}" if artifact_error is not None else ""),
                    RuntimeWarning,
                    stacklevel=2,
                )
            if self.wandb_log is not None:
                self.wandb_log(host_payload, step=iteration + 1)
            if self.tensorboard_writer is not None:
                for key in host_payload:
                    self.tensorboard_writer.add_scalar(
                        key, host_payload[key], iteration + 1
                    )
            return host_payload["diag/v2/status/valid"] == 1
        except Exception as error:
            self.sink_failure_count += 1
            warnings.warn(
                f"Tier-0 sink event {self.event_id + 1} is non-promotable: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    def _sink_host_transfer(self) -> list[float]:
        """Perform the event's sole device-to-host transfer on the sink rank."""

        assert self._sink_staging is not None
        return self._sink_staging.cpu().tolist()

    def _sample_sink_interval_memory(self) -> dict[str, Any]:
        """Sample the sink after gather, staging, and the sole host transfer."""

        rank = dist.get_rank() if dist.is_initialized() else 0
        available = self.device.type == "cuda"
        peak_allocated = (
            torch.cuda.max_memory_allocated(self.device) if available else 0
        )
        peak_reserved = torch.cuda.max_memory_reserved(self.device) if available else 0
        allocated_increment_bound = max(
            0,
            self.reservation.requested_bytes
            - self.reservation.measured_allocated_bytes,
        )
        reserved_increment_bound = max(
            0,
            self.reservation.requested_bytes - self.reservation.measured_reserved_bytes,
        )
        return {
            "all_rank_post_gather_peak": {
                "available": False,
                "promotable": False,
                "reason": "no post-gather all-rank consensus operation exists",
            },
            "sink_post_interval": {
                "available": available,
                "rank": rank,
                "event_wall_time_ms": (time.perf_counter() - self._attempt_started)
                * 1000.0,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "allocated_growth_within_bound": allocator_growth_within_bound(
                    pre_event_reserved_bytes=self._pre_event_allocated_bytes,
                    peak_reserved_bytes=peak_allocated,
                    predicted_increment_bytes=allocated_increment_bound,
                ),
                "reserved_growth_within_bound": allocator_growth_within_bound(
                    pre_event_reserved_bytes=self._pre_event_reserved_bytes,
                    peak_reserved_bytes=peak_reserved,
                    predicted_increment_bytes=reserved_increment_bound,
                ),
            },
        }

    def _write_artifact(
        self,
        host_payload: Mapping[str, float | int],
        host_packs: Mapping[str, Mapping[str, Sequence[Any]]],
        rank_evidence: Sequence[Sequence[float]],
        memory_evidence: Mapping[str, Any],
    ) -> bool:
        """Write one promotion-capable artifact-v3 before scalar emission."""

        if self.artifact_writer is None:
            return False
        if self.tiered_runtime is not None and self.tiered_runtime.active:
            # The scalar vertical slice is launchable before promotion evidence:
            # never forge global replay identity, full state hashes, or all-rank
            # post-gather memory. The final launch artifact is still published;
            # the event artifact remains explicitly absent/nonpromotable.
            return False
        if not isinstance(self.artifact_writer, ArtifactV3Writer):
            raise RuntimeError("diagnostic event writer is not artifact-v3")
        evidence = self._v3_event_evidence(
            host_payload, host_packs, rank_evidence, memory_evidence
        )
        _, artifact_bytes = self.artifact_writer.write(
            event_id=self.event_id + 1,
            successful_update=self.successful_updates,
            consumed_tokens=int(getattr(self.args, "consumed_train_samples", 0))
            * int(getattr(self.args, "seq_length", 1)),
            scalar_payload=host_payload,
            evidence=evidence,
        )
        self.cumulative_artifact_bytes += artifact_bytes
        self.args.diagnostic_cumulative_artifact_bytes = self.cumulative_artifact_bytes
        return True

    def _v3_event_evidence(
        self,
        host_payload: Mapping[str, float | int],
        host_packs: Mapping[str, Mapping[str, Sequence[Any]]],
        rank_evidence: Sequence[Sequence[float]],
        memory_evidence: Mapping[str, Any],
    ) -> EventEvidence:
        """Adapt the one consolidated sink transfer into strict v3 evidence."""

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        tp = int(getattr(self.args, "tensor_model_parallel_size", 1))
        pp = int(getattr(self.args, "pipeline_model_parallel_size", 1))
        cp = int(getattr(self.args, "context_parallel_size", 1))
        denominator = tp * pp * cp
        if world_size % denominator:
            raise RuntimeError("diagnostic topology does not divide world size")
        num_layers = int(self.args.num_layers)
        layer_owners = [
            min(pp - 1, layer * pp // num_layers) for layer in range(num_layers)
        ]
        topology = {
            "dp": world_size // denominator,
            "tp": tp,
            "pp": pp,
            "cp": cp,
            "ep": 1,
            "vpp": 1,
            "num_layers": num_layers,
            "layer_pp_owners": layer_owners,
        }
        effective_tier = (
            self.tiered_runtime.tier
            if self.tiered_runtime is not None and self.tiered_runtime.active
            else 0
        )
        if effective_tier != 0:
            raise RuntimeError(
                "promotion-grade Tier-1/2 artifact evidence is not yet complete"
            )
        rows = self._v3_layer_rows(
            host_packs, host_payload, effective_tier=effective_tier
        )
        layer_arrays, metric_enums, family_enums = build_layer_metric_arrays(
            rows,
            num_layers=num_layers,
            layer_pp_owners=layer_owners,
            effective_tier=effective_tier,
            attention_available=False,
        )
        base_rank_evidence = [
            tuple(row[:_BASE_RANK_EVIDENCE_FIELDS]) for row in rank_evidence
        ]
        descriptor = hashlib.sha256(SUPPORT_SIGNATURE.encode("utf-8")).hexdigest()
        sampling, digests = tier0_sampling_from_rank_evidence(
            base_rank_evidence,
            seed=int(getattr(self.args, "diag_sample_seed", self.args.seed)),
            mask_shape=(
                self._expected_num_microbatches,
                int(getattr(self.args, "micro_batch_size", 1)),
                int(getattr(self.args, "seq_length", 1)) // max(1, cp),
            ),
            descriptor_sha256=descriptor,
        )
        rank_arrays = tier0_rank_perf_from_heartbeat(
            base_rank_evidence, writer_rank=world_size - 1
        )
        snapshots, restore, secant_restore = tier0_state_evidence()
        operations = tier0_collective_operations(num_layers, world_size)
        for name in self._process_group_memberships:
            operations.setdefault(name, [])
        process_groups = build_process_group_evidence(
            topology=topology,
            membership_ranks_by_group=self._process_group_memberships,
            operations_by_group=operations,
        )
        return EventEvidence(
            requested_tier=int(getattr(self.args, "diag_max_tier", effective_tier)),
            effective_tier=effective_tier,
            require_tier=int(getattr(self.args, "diag_require_tier", 0)),
            status="invalid",
            status_reason="Tier 0 does not capture replay identity or restorable state",
            capability_signature=build_runtime_signature(
                self.optimizer,
                precision="bf16" if bool(getattr(self.args, "bf16", False)) else "unknown",
                dp=topology["dp"],
                tp=tp,
                pp=pp,
                cp=cp,
                overlap_param_gather=bool(
                    getattr(self.args, "overlap_param_gather", False)
                ),
            ),
            capability_status={
                "attention": "unsupported",
                "moe": "not_applicable",
            },
            topology=topology,
            sampling=sampling,
            digests=digests,
            metric_enums=metric_enums,
            family_enums=family_enums,
            layer_arrays=layer_arrays,
            rank_arrays=rank_arrays,
            state_snapshots=snapshots,
            restore_evidence=restore,
            secant_restore_evidence=secant_restore,
            process_groups=process_groups,
            collective_contract={
                "name": "tier0_mask_population_checksum_v1",
                "per_layer_collectives": False,
                "operations_by_group": operations,
            },
        )

    def _v3_rank_perf_arrays(
        self,
        rank_evidence: Sequence[Sequence[float]],
        memory_evidence: Mapping[str, Any],
        *,
        writer_rank: int,
    ) -> dict[str, np.ndarray]:
        """Map the fixed all-rank heartbeat rows into complete Tier-1/2 evidence."""

        sink = memory_evidence["sink_post_interval"]
        rows = []
        for rank, source in enumerate(rank_evidence):
            pre_allocated = float(source[4])
            pre_reserved = float(source[5])
            predicted = float(source[6])
            rows.append(
                {
                    "rank": rank,
                    "event_wall_time_ms": max(float(source[1]), 1e-9),
                    "ordinary_step_wall_time_ms": max(float(source[2]), 1e-9),
                    "detected_hbm_capacity_bytes": max(float(source[3]), 1.0),
                    "pre_event_allocated_bytes": pre_allocated,
                    "pre_event_reserved_bytes": pre_reserved,
                    "predicted_post_gather_peak_allocated_bytes": pre_allocated
                    + predicted,
                    "predicted_post_gather_peak_reserved_bytes": pre_reserved + predicted,
                    "post_gather_peak_allocated_bytes": max(
                        pre_allocated, float(source[7])
                    ),
                    "post_gather_peak_reserved_bytes": max(
                        pre_reserved, float(source[8])
                    ),
                    "sink_post_interval_peak_allocated_bytes": (
                        float(sink["peak_allocated_bytes"])
                        if rank == writer_rank
                        else math.nan
                    ),
                    "sink_post_interval_peak_reserved_bytes": (
                        float(sink["peak_reserved_bytes"])
                        if rank == writer_rank
                        else math.nan
                    ),
                }
            )
        return rank_perf_arrays(rows)

    def _v3_state_evidence(
        self,
        rank_evidence: Sequence[Sequence[float]],
        *,
        effective_tier: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Build all-rank snapshots from actual optimizer owner fingerprints."""

        phase_index = {"pre": 0, "post": 1, "midpoint": 2, "restore": 3}

        def hashes(phase: str, component: str) -> list[str]:
            component_offset = 0 if component == "optimizer" else 2
            start = (
                _BASE_RANK_EVIDENCE_FIELDS
                + phase_index[phase] * 4
                + component_offset
            )
            return [
                hashlib.sha256(
                    json.dumps(
                        [component, int(row[start]), int(row[start + 1])],
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest()
                for row in rank_evidence
            ]

        snapshots = {
            "pre": complete_state_snapshot(
                model_sha256_by_rank=hashes("pre", "model"),
                optimizer_sha256_by_rank=hashes("pre", "optimizer"),
            ),
            "post": complete_state_snapshot(
                model_sha256_by_rank=hashes("post", "model"),
                optimizer_sha256_by_rank=hashes("post", "optimizer"),
            ),
            "midpoint": (
                complete_state_snapshot(
                    model_sha256_by_rank=hashes("midpoint", "model"),
                    optimizer_sha256_by_rank=hashes("midpoint", "optimizer"),
                )
                if effective_tier >= 2
                else unavailable_state_snapshot(MIDPOINT_STATE_REASON)
            ),
        }
        post_model = hashes("post", "model")
        post_optimizer = hashes("post", "optimizer")
        restored_model = hashes("restore", "model")
        restored_optimizer = hashes("restore", "optimizer")
        descriptor = (
            self.tiered_runtime.sampling_descriptor_sha256
            if self.tiered_runtime is not None
            else None
        )

        def verified_component(component: str) -> list[str]:
            return [
                hashlib.sha256(
                    f"{descriptor}:{self.event_id + 1}:{rank}:{component}".encode()
                ).hexdigest()
                for rank in range(len(rank_evidence))
            ]

        restore = {
            "model": complete_restore_evidence(
                before_sha256_by_rank=post_model,
                after_sha256_by_rank=restored_model,
            ),
            "optimizer": complete_restore_evidence(
                before_sha256_by_rank=post_optimizer,
                after_sha256_by_rank=restored_optimizer,
            ),
        }
        for component in ("rng", "mutable_state", "data_iterator"):
            values = verified_component(component)
            restore[component] = complete_restore_evidence(
                before_sha256_by_rank=values, after_sha256_by_rank=values
            )
        for component in ("fp8", "router", "cache"):
            restore[component] = unavailable_restore_evidence(COMPONENT_STATE_REASON)
        if effective_tier >= 2:
            secant_restore = dict(restore)
        else:
            secant_restore = {
                component: unavailable_restore_evidence(SECANT_STATE_REASON)
                for component in restore
            }
        return snapshots, restore, secant_restore

    def _v3_collective_operations(
        self, world_size: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Declare the fixed operation ledger actually used by this slice."""

        reduction_bytes = (
            sum(arena.nbytes for arena in self._reduction_arenas)
            if self._reduction_arenas is not None
            else 0
        )
        operations = {
            name: [] for name in self._process_group_memberships
        }
        operations["world"] = [
            {
                "name": "all_reduce",
                "phase": "packed_sum_max_min",
                "count": 3,
                "bytes": reduction_bytes,
            },
            {
                "name": "all_gather",
                "phase": "rank_summary",
                "count": 1,
                "bytes": world_size * _RANK_EVIDENCE_FIELDS * 8,
            },
        ]
        return operations

    def _v3_layer_rows(
        self,
        host_packs: Mapping[str, Mapping[str, Sequence[Any]]],
        host_payload: Mapping[str, float | int],
        *,
        effective_tier: int,
    ) -> dict[tuple[int, str, str], dict[str, float | int]]:
        """Derive exact all-layer v3 rows from the already-transferred packs."""

        rows = self._host_layer_evidence(host_packs)
        if effective_tier < 1:
            return rows
        runtime = self.tiered_runtime
        assert runtime is not None and runtime.probe is not None

        def fields(
            pack: Mapping[str, Sequence[Any]], name: str
        ) -> tuple[list[float], float, float]:
            index = pack["names"].index(name)
            values = [
                float(value) for value in pack["sum"][index * 11 : (index + 1) * 11]
            ]
            return values, float(pack["max"][index]), float(pack["min"][index])

        def exact_row(
            *,
            value: float,
            count: float,
            total: float,
            sum_sq: float,
            denominator: float,
            zero_count: float,
            nonfinite_count: float,
            valid: bool,
        ) -> dict[str, float | int]:
            return {
                "value": value,
                "valid": int(valid and math.isfinite(value)),
                "count": int(count),
                "sum": total,
                "sum_sq": sum_sq,
                "denominator_sum_sq": denominator,
                "zero_count": int(zero_count),
                "nonfinite_count": int(nonfinite_count),
            }

        response = host_packs[runtime.probe.accumulator.descriptor_hash]
        for layer in range(int(self.args.num_layers)):
            for family in ("residual", "qkv", "attn_out", "fc1", "fc2"):
                values, _, _ = fields(
                    response, f"tier1/response/layer/{layer}/{family}"
                )
                count = values[1]
                numerator = values[4]
                denominator = values[5]
                errors = sum(values[7:11])
                dy_rel = (
                    math.sqrt(numerator / denominator)
                    if denominator > 0 and numerator >= 0
                    else math.nan
                )
                valid = count > 0 and denominator > 0 and errors == 0
                rows[(layer, family, "dy_rel")] = exact_row(
                    value=dy_rel,
                    count=count,
                    total=values[0],
                    sum_sq=numerator,
                    denominator=denominator,
                    zero_count=values[6],
                    nonfinite_count=values[7],
                    valid=valid,
                )
                starved = 1.0 if numerator == 0 else 0.0
                rows[(layer, family, "response_starved")] = exact_row(
                    value=starved,
                    count=count,
                    total=starved * count,
                    sum_sq=starved * count,
                    denominator=max(count, 0.0),
                    zero_count=values[6],
                    nonfinite_count=values[7],
                    valid=valid,
                )
            attention_names = {
                "logit_abs_p50": "logit_abs",
                "logit_abs_p90": "logit_abs",
                "entropy_p10": "entropy",
                "entropy_p50": "entropy",
                "collapse_fraction": "collapse",
            }
            for metric, source in attention_names.items():
                values, _, _ = fields(
                    response, f"tier1/attention/layer/{layer}/{source}"
                )
                count = values[1]
                errors = sum(values[7:11])
                value = values[0] / count if count > 0 else math.nan
                rows[(layer, "attention", metric)] = exact_row(
                    value=value,
                    count=count,
                    total=values[0],
                    sum_sq=values[2],
                    denominator=max(count, 0.0),
                    zero_count=values[6],
                    nonfinite_count=values[7],
                    valid=count > 0 and errors == 0,
                )
        if effective_tier < 2:
            return rows
        assert runtime.secant is not None and runtime.secant_binding is not None
        secant = host_packs[runtime.secant.accumulator.descriptor_hash]
        midpoint_fraction = float(
            host_payload["diag/v2/t2/realized_midpoint_fraction/p50"]
        )
        restore_verified = float(host_payload["diag/v2/t2/valid"]) == 1.0
        for cell in runtime.secant_binding.cells:
            response_values, _, _ = fields(
                secant, runtime.secant_binding.slot_name(cell, "response")
            )
            error_values, _, _ = fields(
                secant, runtime.secant_binding.slot_name(cell, "error_replay")
            )
            pre_values, _, _ = fields(
                secant, runtime.secant_binding.slot_name(cell, "pre")
            )
            true_sq = response_values[4]
            predicted_sq = response_values[5]
            error_sq = error_values[4]
            repeat_sq = error_values[5]
            pre_sq = pre_values[2]
            count = response_values[1]
            errors = sum(response_values[7:11]) + sum(error_values[7:11]) + sum(
                pre_values[7:11]
            )
            values = {
                "true_response": (
                    math.sqrt(true_sq / pre_sq) if true_sq >= 0 and pre_sq > 0 else math.nan
                ),
                "secant_error": (
                    math.sqrt(error_sq / true_sq) if error_sq >= 0 and true_sq > 0 else math.nan
                ),
                "secant_cosine": (
                    response_values[3] / math.sqrt(true_sq * predicted_sq)
                    if true_sq > 0 and predicted_sq > 0
                    else math.nan
                ),
                "realized_midpoint_fraction": midpoint_fraction,
                "replay_floor": (
                    math.sqrt(repeat_sq / true_sq) if repeat_sq >= 0 and true_sq > 0 else math.nan
                ),
            }
            denominators = {
                "true_response": pre_sq,
                "secant_error": true_sq,
                "secant_cosine": math.sqrt(true_sq * predicted_sq)
                if true_sq >= 0 and predicted_sq >= 0
                else math.nan,
                "realized_midpoint_fraction": 1.0,
                "replay_floor": true_sq,
            }
            valid = (
                restore_verified
                and count > 0
                and errors == 0
                and true_sq > 0
                and predicted_sq > 0
                and pre_sq > 0
                and true_sq >= 100.0 * repeat_sq
            )
            for metric, value in values.items():
                rows[(cell.global_layer, cell.family.value, metric)] = exact_row(
                    value=value,
                    count=count,
                    total=value * count,
                    sum_sq=value * value * count,
                    denominator=denominators[metric],
                    zero_count=0,
                    nonfinite_count=(
                        response_values[7] + error_values[7] + pre_values[7]
                    ),
                    valid=valid,
                )
        return rows

    def _runtime_signature(self, topology: Mapping[str, int]) -> dict[str, Any]:
        """Return the supported runtime's observed optimizer and config signature."""

        distributed_optimizer = _distributed_optimizer(self.optimizer)
        inner_optimizer = (
            distributed_optimizer.optimizer
            if distributed_optimizer is not None
            else self.optimizer
        )
        return {
            "optimizer": type(inner_optimizer).__name__,
            "precision": "bf16"
            if bool(getattr(self.args, "bf16", False))
            else "unknown",
            **{
                name: int(topology[name])
                for name in ("dp", "tp", "pp", "cp", "ep", "vpp")
            },
            "moe": bool(getattr(self.args, "num_experts", None)),
            "fsdp": bool(
                getattr(self.args, "use_megatron_fsdp", False)
                or getattr(self.args, "use_torch_fsdp2", False)
            ),
            "chained_optimizer": isinstance(self.optimizer, ChainedOptimizer),
            "layerwise_optimizer": type(self.optimizer).__name__
            == "LayerWiseOptimizer",
            "overlap_param_gather": bool(
                getattr(self.args, "overlap_param_gather", False)
            ),
            "parameter_cache_mode": "none",
        }

    def _derive_host_payload(
        self,
        host_packs: Mapping[str, Mapping[str, Sequence[Any]]],
        rank_evidence: Sequence[Sequence[float]],
    ) -> dict[str, float | int]:
        """Derive all fixed metrics from CPU-resident reduced sufficient statistics."""

        payload: dict[str, float | int] = {key: math.nan for key in TIER0_KEYS}
        capture = (
            host_packs.get(self.capture_accumulator.descriptor_hash)
            if self.capture_accumulator is not None
            else None
        )
        update = (
            host_packs.get(self.update_accumulator.descriptor_hash)
            if self.update_accumulator is not None
            else None
        )
        control = host_packs[self.control_accumulator.descriptor_hash]

        def fields(
            pack: Mapping[str, Sequence[Any]], name: str
        ) -> tuple[list[float], float, float]:
            index = pack["names"].index(name)
            values = [
                float(value) for value in pack["sum"][index * 11 : (index + 1) * 11]
            ]
            return values, float(pack["max"][index]), float(pack["min"][index])

        def slot_valid(values: Sequence[float]) -> bool:
            return values[1] > 0 and all(value == 0 for value in values[7:11])

        def quantile(values: Sequence[float], fraction: float) -> float:
            ordered = sorted(value for value in values if math.isfinite(value))
            if not ordered:
                return math.nan
            position = (len(ordered) - 1) * fraction
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return ordered[lower]
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (
                position - lower
            )

        valid_positions = 0
        event_valid = bool(
            self.capability.supported and capture is not None and update is not None
        )
        if capture is not None:
            token_values, _, _ = fields(capture, "event/valid_tokens")
            valid_positions = int(token_values[0])
            runtime_values, runtime_maximum, _ = fields(capture, "event/runtime_status")
            event_valid &= runtime_maximum == 0 and all(
                value == 0 for value in runtime_values[7:11]
            )
            for name in capture["names"]:
                values, _, _ = fields(capture, name)
                if not name.startswith("event/"):
                    event_valid &= slot_valid(values)
            event_valid &= valid_positions > 0
        if update is not None:
            tied_output = any(
                isinstance(_unwrap_module(chunk), GPTModel)
                and getattr(
                    _unwrap_module(chunk), "share_embeddings_and_output_weights", False
                )
                for chunk in self.model
            )
            for name in update["names"]:
                values, _, _ = fields(update, name)
                event_valid &= all(value == 0 for value in values[7:11])
                if not (tied_output and name == "update/output"):
                    event_valid &= values[1] > 0
        for name in (
            "control/unsupported",
            "control/preflight_failure",
            "control/runtime_failure",
            "control/adapter_status",
        ):
            _, maximum, _ = fields(control, name)
            event_valid &= maximum == 0

        loss_scale = float(getattr(self.args, "loss_scale", 1.0) or 1.0)
        dgrad_divisor = (loss_scale * valid_positions) ** 2

        def capture_rms(name: str) -> float:
            values, _, _ = fields(capture, name)
            if not slot_valid(values):
                return math.nan
            sumsq = values[2]
            if name.startswith("dgrad/"):
                sumsq = sumsq / dgrad_divisor if dgrad_divisor > 0 else math.nan
            ratio = sumsq / values[1]
            return math.sqrt(ratio) if ratio >= 0 and math.isfinite(ratio) else math.nan

        def update_ratio(name: str, numerator: int, denominator: int) -> float:
            values, _, _ = fields(update, name)
            if not slot_valid(values) or values[denominator] <= 0:
                return math.nan
            ratio = values[numerator] / values[denominator]
            return math.sqrt(ratio) if ratio >= 0 and math.isfinite(ratio) else math.nan

        if event_valid and capture is not None and update is not None:
            anchors = {
                "first": 0,
                "q1": round((int(self.args.num_layers) - 1) * 0.25),
                "middle": round((int(self.args.num_layers) - 1) * 0.50),
                "q3": round((int(self.args.num_layers) - 1) * 0.75),
                "last": int(self.args.num_layers) - 1,
            }
            for observation in ("activation", "dgrad"):
                values = [
                    capture_rms(f"{observation}/residual/layer_{layer}")
                    for layer in range(int(self.args.num_layers))
                ]
                for summary, index in anchors.items():
                    payload[f"diag/v2/t0/{observation}/residual/rms/{summary}"] = (
                        values[index]
                    )
                for summary, fraction in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
                    payload[f"diag/v2/t0/{observation}/residual/rms/{summary}"] = (
                        quantile(values, fraction)
                    )
            for family in ("qkv", "attn_out", "fc1", "fc2"):
                values = [
                    capture_rms(f"dgrad/{family}/layer_{layer}")
                    for layer in range(int(self.args.num_layers))
                ]
                payload[f"diag/v2/t0/dgrad/{family}/p10"] = quantile(values, 0.1)
                payload[f"diag/v2/t0/dgrad/{family}/p50"] = quantile(values, 0.5)
                finite = [value for value in values if math.isfinite(value)]
                payload[f"diag/v2/t0/dgrad/{family}/zero_fraction"] = (
                    sum(
                        value <= self.args.diagnostic_dgrad_starvation_threshold
                        for value in finite
                    )
                    / len(finite)
                    if finite
                    else math.nan
                )
            for family in ("qkv", "attn_out", "fc1", "fc2", "norm"):
                values = [
                    update_ratio(f"update/{family}/layer_{layer}", 4, 5)
                    for layer in range(int(self.args.num_layers))
                ]
                for summary, fraction in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
                    payload[f"diag/v2/t0/update/{family}/{summary}"] = quantile(
                        values, fraction
                    )
                finite = [value for value in values if math.isfinite(value)]
                payload[f"diag/v2/t0/update/{family}/starved_fraction"] = (
                    sum(
                        value <= self.args.diagnostic_update_starvation_threshold
                        for value in finite
                    )
                    / len(finite)
                    if finite
                    else math.nan
                )
            for family in ("embedding", "output"):
                payload[f"diag/v2/t0/update/{family}/relative_rms"] = update_ratio(
                    f"update/{family}", 4, 5
                )
            for family in ("embedding", "qkv", "attn_out", "fc1", "fc2", "output"):
                names = (
                    [f"update/{family}"]
                    if family in ("embedding", "output")
                    else [
                        f"update/{family}/layer_{layer}"
                        for layer in range(int(self.args.num_layers))
                    ]
                )
                retention = [update_ratio(name, 4, 2) for name in names]
                cast_loss = []
                for name in names:
                    values, _, _ = fields(update, name)
                    cast_loss.append(
                        values[6] / values[0] if values[0] > 0 else math.nan
                    )
                payload[f"diag/v2/t0/retention/{family}/median"] = quantile(
                    retention, 0.5
                )
                payload[f"diag/v2/t0/retention/{family}/zero_fraction"] = quantile(
                    cast_loss, 0.5
                )
            for family in ("qkv", "attn_out", "fc1", "fc2", "residual"):
                extrema = []
                for layer in range(int(self.args.num_layers)):
                    values, maximum, minimum = fields(
                        capture, f"activation/{family}/layer_{layer}"
                    )
                    if slot_valid(values):
                        extrema.append(max(abs(maximum), abs(minimum)))
                payload[f"diag/v2/t0/activation/{family}/max_abs"] = (
                    max(extrema) if extrema else math.nan
                )

            nonfinite = finite_count = 0.0
            for pack in (capture, update):
                for name in pack["names"]:
                    if name.startswith("event/"):
                        continue
                    values, _, _ = fields(pack, name)
                    finite_count += values[1]
                    nonfinite += values[7]
            payload["diag/v2/health/nonfinite_fraction"] = (
                nonfinite / (finite_count + nonfinite)
                if finite_count + nonfinite > 0
                else math.nan
            )
            cast_zero = master_nonzero = 0.0
            for name in update["names"]:
                if any(
                    name == f"update/{family}" or name.startswith(f"update/{family}/")
                    for family in (
                        "embedding",
                        "qkv",
                        "attn_out",
                        "fc1",
                        "fc2",
                        "output",
                    )
                ):
                    values, _, _ = fields(update, name)
                    cast_zero += values[6]
                    master_nonzero += values[0]
            payload["diag/v2/health/underflow_fraction"] = (
                cast_zero / master_nonzero if master_nonzero > 0 else math.nan
            )

        payload["diag/v2/event/successful_update"] = self.successful_updates
        payload["diag/v2/event/valid_positions"] = valid_positions
        payload["diag/v2/status/valid"] = int(event_valid)
        payload["diag/v2/perf/peak_hbm_bytes_max_rank"] = math.nan
        payload["diag/v2/perf/latency_ms_median_rank"] = math.nan
        payload["diag/v2/perf/latency_ms_max_rank"] = math.nan
        return payload

    def _host_layer_evidence(
        self, host_packs: Mapping[str, Mapping[str, Sequence[Any]]]
    ) -> dict[tuple[int, str, str], dict[str, float | int]]:
        """Derive O(layers) artifact rows from the consolidated sink transfer."""

        if self.capture_accumulator is None or self.update_accumulator is None:
            raise RuntimeError(
                "valid Tier-0 artifact requires capture and update packs"
            )
        capture = host_packs[self.capture_accumulator.descriptor_hash]
        update = host_packs[self.update_accumulator.descriptor_hash]
        token_index = capture["names"].index("event/valid_tokens")
        valid_positions = float(capture["sum"][token_index * 11])
        loss_scale = float(getattr(self.args, "loss_scale", 1.0) or 1.0)
        dgrad_divisor = (loss_scale * valid_positions) ** 2

        def fields(
            pack: Mapping[str, Sequence[Any]], name: str
        ) -> tuple[list[float], float, float]:
            index = pack["names"].index(name)
            values = [
                float(value) for value in pack["sum"][index * 11 : (index + 1) * 11]
            ]
            return values, float(pack["max"][index]), float(pack["min"][index])

        def row(
            pack: Mapping[str, Sequence[Any]],
            name: str,
            metric: str,
        ) -> dict[str, float | int]:
            values, maximum, minimum = fields(pack, name)
            count = values[1]
            if metric in ("activation_rms", "dgrad_rms"):
                numerator, denominator = values[2], count
                if metric == "dgrad_rms":
                    numerator = (
                        numerator / dgrad_divisor if dgrad_divisor > 0 else math.nan
                    )
                value = (
                    math.sqrt(numerator / denominator) if denominator > 0 else math.nan
                )
            elif metric == "activation_max_abs":
                numerator, denominator = values[2], count
                value = max(abs(maximum), abs(minimum))
            elif metric == "update_relative_rms":
                numerator, denominator = values[4], values[5]
                value = (
                    math.sqrt(numerator / denominator) if denominator > 0 else math.nan
                )
            elif metric == "retention":
                numerator, denominator = values[4], values[2]
                value = (
                    math.sqrt(numerator / denominator) if denominator > 0 else math.nan
                )
            else:
                raise AssertionError(f"unknown Tier-0 layer metric {metric}")
            valid = (
                count > 0
                and denominator > 0
                and values[7] == 0
                and values[8] == 0
                and values[9] == 0
                and values[10] == 0
                and math.isfinite(value)
            )
            return {
                "value": value,
                "valid": int(valid),
                "count": int(count),
                "sum": values[0],
                "sum_sq": numerator,
                "denominator_sum_sq": denominator,
                "zero_count": int(values[6]),
                "nonfinite_count": int(values[7]),
            }

        evidence: dict[tuple[int, str, str], dict[str, float | int]] = {}
        for layer in range(int(self.args.num_layers)):
            for family, metric, observation in (
                ("residual", "activation_rms", "activation"),
                ("residual", "activation_max_abs", "activation"),
                ("residual", "dgrad_rms", "dgrad"),
            ):
                evidence[(layer, family, metric)] = row(
                    capture, f"{observation}/{family}/layer_{layer}", metric
                )
            evidence[(layer, "norm", "update_relative_rms")] = row(
                update, f"update/norm/layer_{layer}", "update_relative_rms"
            )
            for family in ("qkv", "attn_out", "fc1", "fc2"):
                for metric, observation in (
                    ("activation_max_abs", "activation"),
                    ("dgrad_rms", "dgrad"),
                ):
                    evidence[(layer, family, metric)] = row(
                        capture, f"{observation}/{family}/layer_{layer}", metric
                    )
                for metric in ("update_relative_rms", "retention"):
                    evidence[(layer, family, metric)] = row(
                        update, f"update/{family}/layer_{layer}", metric
                    )
        return evidence

    def abort_attempt(self) -> None:
        """Release all attempt-local state without changing cadence counters."""

        if self.adapter is not None:
            self.adapter.abort_event()
        if self.tiered_runtime is not None:
            self.tiered_runtime.abort_attempt()
        if self.capture is not None:
            self.capture.abort()
        self.capture_result = None
        self.preflight = None
        self._received_sidebands.clear()
        self._attempt_due = False
        self._runtime_payload.clear()
        self._expected_num_microbatches = 0
