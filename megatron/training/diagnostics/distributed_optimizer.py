# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Authoritative BF16 distributed-optimizer update diagnostics."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import IntEnum

import torch
import torch.distributed as dist

from megatron.core.optimizer.distrib_optimizer import (
    DistributedOptimizer as MegatronDistributedOptimizer,
)
from megatron.core.optimizer.distrib_optimizer import ModelMainParamShard
from megatron.core.optimizer.optimizer import ChainedOptimizer

from .accumulator import PackedReducer, PackedSufficientStatistics
from .registry import MetricRegistry, StatisticKind


class DistributedOptimizerDiagnosticReason(IntEnum):
    """Stable fail-closed reasons for the first Tier-0 optimizer backend."""

    NONE = 0
    OPTIMIZER_TYPE = 1
    CHAIN_ARITY = 2
    STUB_OPTIMIZER = 3
    NOT_BF16 = 4
    NON_FP32_MAIN = 5
    FSDP = 6
    PRECISION_AWARE = 7
    FP8 = 8
    FP4 = 9
    MOE = 10
    LAYERWISE = 11
    CPU_OFFLOAD = 12
    OVERLAP_PARAM_GATHER = 13
    RANK_INCONSISTENT = 14


class DistributedOptimizerEventStatus(IntEnum):
    """Stable local event states packed by the later heartbeat consensus."""

    OK = 0
    CONSTRUCTOR_ITERATOR_FAILED = 1
    CONSTRUCTOR_BINDING_FAILED = 2
    CONSTRUCTOR_OWNERSHIP_FAILED = 3
    BEGIN_ALREADY_ARMED = 4
    BEGIN_ITERATOR_FAILED = 5
    BEGIN_IDENTITY_CHANGED = 6
    BEGIN_PREFLIGHT_REJECTED = 7
    BEGIN_NO_SHARDS = 8
    BEGIN_DEVICE_MISMATCH = 9
    BEGIN_FIRST_ALLOCATION_FAILED = 10
    BEGIN_SECOND_ALLOCATION_FAILED = 11
    BEGIN_COPY_FAILED = 12
    FINISH_NOT_ARMED = 13
    UPDATE_SKIPPED = 14
    FINISH_ITERATOR_FAILED = 15
    FINISH_SHARD_COUNT_CHANGED = 16
    FINISH_SHARD_IDENTITY_CHANGED = 17
    FINISH_SCRATCH_ALLOCATION_FAILED = 18
    MATERIALIZATION_MISMATCH = 19
    ACCUMULATION_FAILED = 20
    ALLOCATOR_QUERY_FAILED = 21
    BEGIN_UNEXPECTED_FAILED = 22
    FINISH_UNEXPECTED_FAILED = 23
    SECANT_DELTA_NOT_ARMED = 24
    SECANT_DELTA_IDENTITY_CHANGED = 25
    SECANT_DELTA_NONFINITE = 26
    SECANT_DELTA_FAILED = 27
    SECANT_MIDPOINT_UNAVAILABLE = 28
    SECANT_MIDPOINT_INSTALL_FAILED = 29
    SECANT_RESTORE_COPY_FAILED = 30
    SECANT_RESTORE_VERIFY_FAILED = 31
    SECANT_INVALID_PHASE = 32


class _SecantAdapterPhase(IntEnum):
    ARMED = 0
    DELTA_COMMITTING = 1
    DELTA_COMMITTED = 2
    MIDPOINT_INSTALLED = 3
    RESTORED = 4
    FAILED = 5


class SnapshotMemoryReason(IntEnum):
    """Stable reasons why complete event allocation is rejected."""

    NONE = 0
    MAX_EXTRA_BYTES = 1
    DEVICE_MEMORY_FRACTION = 2
    DEVICE_HEADROOM = 3
    ALLOCATION_FAILED = 4


@dataclass(frozen=True)
class DistributedOptimizerCapabilityReport:
    """Collectively negotiated support result.

    Capability negotiation is the one explicitly startup-only host synchronization
    in this adapter. Every rank reads one supported scalar; detailed flags are
    decoded only on the unsupported reporting path. Runtime event failures remain
    device resident for the heartbeat's fixed packed consensus.

    Attributes:
        supported: Whether every rank supports the first-backend contract.
        reasons: Sorted global fail-closed reasons.
        local_reasons: Sorted reasons detected locally before negotiation.
        world_size: Number of ranks participating in negotiation.
        device_flags: Packed global reason counts on the negotiation device.
        startup_host_sync: Whether startup support selection synchronized with the host.
    """

    supported: bool
    reasons: tuple[DistributedOptimizerDiagnosticReason, ...]
    local_reasons: tuple[DistributedOptimizerDiagnosticReason, ...]
    world_size: int
    device_flags: torch.Tensor
    startup_host_sync: bool


@dataclass(frozen=True)
class SnapshotMemoryEstimate:
    """Exact logical peak bytes requested by an armed optimizer event."""

    fp32_master_bytes: int
    bf16_applied_bytes: int
    post_fingerprint_bytes: int
    finish_elementwise_bytes: int
    finish_scalar_bytes: int
    finish_scratch_bytes: int
    snapshot_bytes: int
    total_bytes: int
    owner_elements: int
    finish_chunk_elements: int


def snapshot_memory_estimate(
    owner_elements: int, *, finish_chunk_elements: int = 65_536
) -> SnapshotMemoryEstimate:
    """Calculate the exact optimizer event reservation without allocating tensors."""

    if owner_elements < 0 or finish_chunk_elements <= 0:
        raise ValueError("snapshot reservation dimensions are invalid")
    chunk_elements = min(owner_elements, finish_chunk_elements)
    fp32_master_bytes = owner_elements * 4
    bf16_applied_bytes = owner_elements * 2
    post_fingerprint_bytes = 4 * 8
    layout = _scratch_layout(chunk_elements)
    snapshot_bytes = (
        fp32_master_bytes + bf16_applied_bytes + post_fingerprint_bytes
    )
    return SnapshotMemoryEstimate(
        fp32_master_bytes=fp32_master_bytes,
        bf16_applied_bytes=bf16_applied_bytes,
        post_fingerprint_bytes=post_fingerprint_bytes,
        finish_elementwise_bytes=layout.elementwise_bytes,
        finish_scalar_bytes=layout.scalar_bytes,
        finish_scratch_bytes=layout.total_bytes,
        snapshot_bytes=snapshot_bytes,
        total_bytes=snapshot_bytes + layout.total_bytes,
        owner_elements=owner_elements,
        finish_chunk_elements=chunk_elements,
    )


@dataclass(frozen=True)
class DeviceMemoryState:
    """Device allocator and driver memory state used by event preflight."""

    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class SnapshotMemoryPreflight:
    """Complete event estimate evaluated against configured HBM limits."""

    estimate: SnapshotMemoryEstimate
    memory: DeviceMemoryState | None
    max_extra_bytes: int
    max_memory_fraction: float
    available_bytes: int | None
    requested_bytes: int
    reusable_bytes: int | None
    driver_need_bytes: int | None
    projected_reserved_bytes: int | None
    accepted: bool
    reason: SnapshotMemoryReason


@dataclass(frozen=True)
class SnapshotMemoryMeasurement:
    """Measured retained payload and allocator peak through event completion."""

    estimated_bytes: int
    payload_bytes: int
    finish_scratch_bytes: int
    allocator_delta_bytes: int | None
    allocator_peak_delta_bytes: int | None


class DistributedOptimizerDiagnosticUnsupportedError(RuntimeError):
    """Raised collectively when the optimizer backend is unsupported."""


class SnapshotMemoryError(RuntimeError):
    """Legacy typed exception retained for API compatibility.

    Runtime begin/finish paths now expose :class:`DistributedOptimizerEventStatus`
    instead of raising this exception rank-locally.
    """

    def __init__(self, reason: SnapshotMemoryReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class _SnapshotState:
    shards: tuple[ModelMainParamShard, ...]
    offsets: tuple[tuple[int, int], ...]
    master_before: torch.Tensor
    applied_before: torch.Tensor
    measurement: SnapshotMemoryMeasurement
    allocator_before: int | None
    unique_capture_indices: tuple[int, ...]
    phase: _SecantAdapterPhase = _SecantAdapterPhase.ARMED
    post_fingerprints: torch.Tensor | None = None
    commit_measurement: SnapshotMemoryMeasurement | None = None

    @property
    def delta_ready(self) -> bool:
        return self.phase in (
            _SecantAdapterPhase.DELTA_COMMITTED,
            _SecantAdapterPhase.MIDPOINT_INSTALLED,
            _SecantAdapterPhase.RESTORED,
        )

    @property
    def midpoint_installed(self) -> bool:
        return self.phase == _SecantAdapterPhase.MIDPOINT_INSTALLED


@dataclass(frozen=True)
class _FinishScratchLayout:
    expected_bf16: tuple[int, int]
    applied_before_fp32: tuple[int, int]
    applied_after_fp32: tuple[int, int]
    master_delta_fp32: tuple[int, int]
    applied_delta_fp32: tuple[int, int]
    work_fp64: tuple[int, int]
    hash_position_int64: tuple[int, int]
    finite: tuple[int, int]
    auxiliary: tuple[int, int]
    moment_fp64: tuple[int, int]
    moment_fp32: tuple[int, int]
    materialization_valid: tuple[int, int]
    scalar_bool: tuple[int, int]
    status_int64: tuple[int, int]
    elementwise_bytes: int
    scalar_bytes: int
    total_bytes: int


@dataclass
class _FinishScratch:
    storage: torch.Tensor
    expected_bf16: torch.Tensor
    applied_before_fp32: torch.Tensor
    applied_after_fp32: torch.Tensor
    master_delta_fp32: torch.Tensor
    applied_delta_fp32: torch.Tensor
    work_fp64: torch.Tensor
    hash_position_int64: torch.Tensor
    finite: torch.Tensor
    auxiliary: torch.Tensor
    moment_fp64: torch.Tensor
    moment_fp32: torch.Tensor
    materialization_valid: torch.Tensor
    scalar_bool: torch.Tensor
    status_int64: torch.Tensor


MemoryStateProvider = Callable[[torch.device], DeviceMemoryState]


def _unwrap_distributed_optimizer(
    optimizer: object,
) -> tuple[MegatronDistributedOptimizer | None, tuple[DistributedOptimizerDiagnosticReason, ...]]:
    if isinstance(optimizer, MegatronDistributedOptimizer):
        return optimizer, ()
    if isinstance(optimizer, ChainedOptimizer):
        if len(optimizer.chained_optimizers) != 1:
            return None, (DistributedOptimizerDiagnosticReason.CHAIN_ARITY,)
        child = optimizer.chained_optimizers[0]
        if isinstance(child, MegatronDistributedOptimizer):
            return child, ()
    return None, (DistributedOptimizerDiagnosticReason.OPTIMIZER_TYPE,)


def _model_parameters(optimizer: MegatronDistributedOptimizer) -> tuple[torch.nn.Parameter, ...]:
    groups = getattr(optimizer, "model_float16_groups", ())
    return tuple(parameter for group in groups for parameter in group)


def _local_capability_reasons(
    optimizer: object,
) -> tuple[MegatronDistributedOptimizer | None, tuple[DistributedOptimizerDiagnosticReason, ...]]:
    distributed_optimizer, unwrap_reasons = _unwrap_distributed_optimizer(optimizer)
    if distributed_optimizer is None:
        return None, unwrap_reasons

    reasons: list[DistributedOptimizerDiagnosticReason] = []
    config = distributed_optimizer.config
    ddp_config = distributed_optimizer.ddp_config
    if distributed_optimizer.is_stub_optimizer:
        reasons.append(DistributedOptimizerDiagnosticReason.STUB_OPTIMIZER)
    if not config.bf16 or config.fp16:
        reasons.append(DistributedOptimizerDiagnosticReason.NOT_BF16)
    if ddp_config.use_megatron_fsdp:
        reasons.append(DistributedOptimizerDiagnosticReason.FSDP)
    if config.use_precision_aware_optimizer:
        reasons.append(DistributedOptimizerDiagnosticReason.PRECISION_AWARE)
    if ddp_config.fp8_param_gather or config.fp8_recipe is not None:
        reasons.append(DistributedOptimizerDiagnosticReason.FP8)
    if ddp_config.fp4_param_gather:
        reasons.append(DistributedOptimizerDiagnosticReason.FP4)
    if any(
        not getattr(parameter, "allreduce", True)
        for parameter in _model_parameters(distributed_optimizer)
    ):
        reasons.append(DistributedOptimizerDiagnosticReason.MOE)
    if config.use_layer_wise_distributed_optimizer:
        reasons.append(DistributedOptimizerDiagnosticReason.LAYERWISE)
    if config.optimizer_cpu_offload:
        reasons.append(DistributedOptimizerDiagnosticReason.CPU_OFFLOAD)
    if (
        ddp_config.overlap_param_gather
        or config.overlap_param_gather
        or config.overlap_param_gather_with_optimizer_step
    ):
        reasons.append(DistributedOptimizerDiagnosticReason.OVERLAP_PARAM_GATHER)

    main_groups = getattr(distributed_optimizer, "shard_fp32_from_float16_groups", ())
    if any(
        parameter is None or parameter.dtype != torch.float32
        for group in main_groups
        for parameter in group
    ):
        reasons.append(DistributedOptimizerDiagnosticReason.NON_FP32_MAIN)
    model_groups = getattr(distributed_optimizer, "model_float16_groups", ())
    if any(
        parameter.dtype != torch.bfloat16 for group in model_groups for parameter in group
    ) or any(getattr(distributed_optimizer, "model_fp32_groups", ())):
        reasons.append(DistributedOptimizerDiagnosticReason.NOT_BF16)
    return distributed_optimizer, tuple(sorted(set(reasons), key=int))


def _negotiation_device(
    optimizer: MegatronDistributedOptimizer | None, process_group: object | None
) -> torch.device:
    backend = dist.get_backend(process_group) if dist.is_initialized() else None
    if optimizer is not None:
        parameters = _model_parameters(optimizer)
        if parameters:
            device = parameters[0].device
            if backend != "nccl" or device.type == "cuda":
                return device
    if backend == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _default_memory_state(device: torch.device) -> DeviceMemoryState:
    if device.type != "cuda":
        raise RuntimeError("HBM memory state is available only for CUDA devices")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return DeviceMemoryState(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        free_bytes=free_bytes,
        total_bytes=total_bytes,
    )


def _align(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _scratch_layout(elements: int) -> _FinishScratchLayout:
    offset = 0

    def reserve(count: int, bytes_per_element: int, alignment: int) -> tuple[int, int]:
        nonlocal offset
        offset = _align(offset, alignment)
        start = offset
        offset += count * bytes_per_element
        return start, offset

    expected_bf16 = reserve(elements, 2, 2)
    applied_before_fp32 = reserve(elements, 4, 4)
    applied_after_fp32 = reserve(elements, 4, 4)
    master_delta_fp32 = reserve(elements, 4, 4)
    applied_delta_fp32 = reserve(elements, 4, 4)
    work_fp64 = reserve(elements, 8, 8)
    hash_position_int64 = reserve(elements, 8, 8)
    finite = reserve(elements, 1, 1)
    auxiliary = reserve(elements, 1, 1)
    elementwise_bytes = offset

    # Ten FP64 moment/error scalars, two FP32 extrema, and the boolean/int64
    # lifecycle scalars remain live through the accumulator update. Keeping all
    # of them in this storage makes the preflight bound complete and exact.
    moment_fp64 = reserve(10, 8, 8)
    moment_fp32 = reserve(2, 4, 4)
    materialization_valid = reserve(1, 1, 1)
    scalar_bool = reserve(1, 1, 1)
    status_int64 = reserve(1, 8, 8)
    return _FinishScratchLayout(
        expected_bf16=expected_bf16,
        applied_before_fp32=applied_before_fp32,
        applied_after_fp32=applied_after_fp32,
        master_delta_fp32=master_delta_fp32,
        applied_delta_fp32=applied_delta_fp32,
        work_fp64=work_fp64,
        hash_position_int64=hash_position_int64,
        finite=finite,
        auxiliary=auxiliary,
        moment_fp64=moment_fp64,
        moment_fp32=moment_fp32,
        materialization_valid=materialization_valid,
        scalar_bool=scalar_bool,
        status_int64=status_int64,
        elementwise_bytes=elementwise_bytes,
        scalar_bytes=offset - elementwise_bytes,
        total_bytes=offset,
    )


def _typed_view(
    storage: torch.Tensor, byte_range: tuple[int, int], dtype: torch.dtype
) -> torch.Tensor:
    start, end = byte_range
    return storage[start:end].view(dtype)


class Bf16DistributedOptimizerDiagnosticAdapter:
    """Capture verified BF16 updates from authoritative optimizer owner shards."""

    def __init__(
        self,
        optimizer: object,
        registry: MetricRegistry,
        metric_name_by_parameter: Mapping[torch.nn.Parameter, str],
        *,
        diagnostic_max_extra_bytes: int,
        max_memory_fraction: float = 0.9,
        finish_chunk_elements: int = 65_536,
        process_group: object | None = None,
        capability_reducer: PackedReducer | None = None,
        capability_world_size: int | None = None,
        memory_state_provider: MemoryStateProvider | None = None,
    ) -> None:
        """Negotiate support and bind typed owner shards without local raises.

        Args:
            optimizer: One real distributed optimizer or a trivial one-child chain.
            registry: Existing owner-aware metric registry.
            metric_name_by_parameter: Typed parameter-to-update-descriptor bindings.
            diagnostic_max_extra_bytes: Hard complete-event allocation limit.
            max_memory_fraction: Maximum post-allocation device-memory fraction.
            finish_chunk_elements: Maximum owner elements processed in finish scratch.
            process_group: Startup capability-negotiation group, defaulting to world.
            capability_reducer: Optional injected SUM reducer for tests.
            capability_world_size: Required injected world size with a reducer.
            memory_state_provider: Optional exact device-memory provider for tests.

        Raises:
            DistributedOptimizerDiagnosticUnsupportedError: If startup negotiation
                collectively rejects the backend.
            ValueError: If scalar adapter configuration is invalid.
        """

        if diagnostic_max_extra_bytes < 0:
            raise ValueError("diagnostic_max_extra_bytes must be nonnegative")
        if not 0 < max_memory_fraction <= 1:
            raise ValueError("max_memory_fraction must be in (0, 1]")
        if finish_chunk_elements <= 0:
            raise ValueError("finish_chunk_elements must be positive")
        report = self.negotiate_capabilities(
            optimizer,
            process_group=process_group,
            reducer=capability_reducer,
            world_size=capability_world_size,
        )
        if not report.supported:
            reason_names = ", ".join(reason.name for reason in report.reasons)
            raise DistributedOptimizerDiagnosticUnsupportedError(
                f"unsupported BF16 distributed-optimizer diagnostics: {reason_names}"
            )
        distributed_optimizer, _ = _unwrap_distributed_optimizer(optimizer)
        assert distributed_optimizer is not None

        self.optimizer = distributed_optimizer
        self.registry = registry
        supplied_metric_names = dict(metric_name_by_parameter)
        self.metric_name_by_parameter: dict[torch.nn.Parameter, str] = {}
        self.diagnostic_max_extra_bytes = diagnostic_max_extra_bytes
        self.max_memory_fraction = max_memory_fraction
        self.finish_chunk_elements = finish_chunk_elements
        self.capability_report = report
        self.memory_state_provider = memory_state_provider
        self._snapshot: _SnapshotState | None = None
        self._master_before_buffer: torch.Tensor | None = None
        self._applied_before_buffer: torch.Tensor | None = None
        self._finish_scratch: _FinishScratch | None = None
        self._last_memory_reason = SnapshotMemoryReason.NONE
        self._status = torch.zeros(
            1, dtype=torch.int64, device=_negotiation_device(distributed_optimizer, process_group)
        )
        self._construction_status = DistributedOptimizerEventStatus.OK
        self._bound_shards: tuple[ModelMainParamShard, ...] = ()
        self._bound_offsets: tuple[tuple[int, int], ...] = ()
        self._unique_capture_indices: tuple[int, ...] = ()

        try:
            shards = tuple(self.optimizer.iter_model_main_param_shards())
        except Exception:
            self._record_construction_failure(
                DistributedOptimizerEventStatus.CONSTRUCTOR_ITERATOR_FAILED
            )
            return
        try:
            descriptor_kinds = {
                descriptor.logical_name: descriptor.statistic_kind
                for descriptor in registry.descriptors
            }
            shard_parameters = {shard.model_param for shard in shards}
            if any(
                descriptor_kinds.get(logical_name) != StatisticKind.UPDATE
                for logical_name in supplied_metric_names.values()
            ):
                raise ValueError("invalid typed owner binding")
            local_metric_names = {
                parameter: logical_name
                for parameter, logical_name in supplied_metric_names.items()
                if parameter in shard_parameters
            }
            if any(
                shard.logical_owner and shard.model_param not in local_metric_names
                for shard in shards
            ):
                raise ValueError("missing typed owner binding")
        except Exception:
            self._record_construction_failure(
                DistributedOptimizerEventStatus.CONSTRUCTOR_BINDING_FAILED
            )
            return
        try:
            self._validate_unique_local_ownership(shards)
            offsets, unique_capture_indices = self._capture_layout(shards)
        except Exception:
            self._record_construction_failure(
                DistributedOptimizerEventStatus.CONSTRUCTOR_OWNERSHIP_FAILED
            )
            return
        self._bound_shards = shards
        self._bound_offsets = offsets
        self._unique_capture_indices = unique_capture_indices
        self.metric_name_by_parameter = local_metric_names

    @staticmethod
    def negotiate_capabilities(
        optimizer: object,
        *,
        process_group: object | None = None,
        reducer: PackedReducer | None = None,
        world_size: int | None = None,
    ) -> DistributedOptimizerCapabilityReport:
        """Collectively negotiate the narrow first-backend capability contract."""

        distributed_optimizer, local_reasons = _local_capability_reasons(optimizer)
        if reducer is not None:
            if world_size is None or world_size <= 0:
                raise ValueError("an injected capability reducer requires a positive world size")
            negotiated_world_size = world_size
        elif dist.is_available() and dist.is_initialized():
            negotiated_world_size = dist.get_world_size(group=process_group)
        else:
            negotiated_world_size = 1

        reason_count = max(reason.value for reason in DistributedOptimizerDiagnosticReason) + 1
        flags = torch.zeros(
            reason_count,
            dtype=torch.int64,
            device=_negotiation_device(distributed_optimizer, process_group),
        )
        for reason in local_reasons:
            flags[reason.value] = 1
        if reducer is not None:
            reducer(flags, op=dist.ReduceOp.SUM, group=process_group)
        elif negotiated_world_size > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.SUM, group=process_group)

        supported = not bool(torch.any(flags).item())
        reasons: tuple[DistributedOptimizerDiagnosticReason, ...] = ()
        if not supported:
            counts = flags.detach().cpu().tolist()
            decoded = [
                reason
                for reason in DistributedOptimizerDiagnosticReason
                if reason != DistributedOptimizerDiagnosticReason.NONE and counts[reason.value] > 0
            ]
            if any(0 < count < negotiated_world_size for count in counts):
                decoded.append(DistributedOptimizerDiagnosticReason.RANK_INCONSISTENT)
            reasons = tuple(sorted(set(decoded), key=int))
        return DistributedOptimizerCapabilityReport(
            supported=supported,
            reasons=reasons,
            local_reasons=local_reasons,
            world_size=negotiated_world_size,
            device_flags=flags,
            startup_host_sync=True,
        )

    @property
    def armed(self) -> bool:
        """Return whether pre-step owner snapshots are currently retained."""

        return self._snapshot is not None

    @property
    def local_status(self) -> torch.Tensor:
        """Return the device-resident local status packed by the heartbeat."""

        return self._status

    @property
    def last_memory_reason(self) -> SnapshotMemoryReason:
        """Return the most recent typed preflight or allocation reason."""

        return self._last_memory_reason

    def status_for_event_consensus(self) -> torch.Tensor:
        """Return one fixed-shape status tensor without launching a collective."""

        return self._status

    def iter_owner_shards(self) -> tuple[ModelMainParamShard, ...]:
        """Return the construction-time stable owner-shard sequence."""

        return self._bound_shards

    @property
    def secant_delta_ready(self) -> bool:
        """Return whether the shared FP32 pre buffer now stores authoritative deltas."""

        return self._snapshot is not None and self._snapshot.delta_ready

    @property
    def secant_delta_buffer(self) -> torch.Tensor | None:
        """Return the shared FP32 pre buffer after its in-place delta transformation."""

        if self._snapshot is None or not self._snapshot.delta_ready:
            return None
        return self._snapshot.master_before

    @property
    def secant_applied_pre_buffer(self) -> torch.Tensor | None:
        """Return the BF16 pre snapshot until commit reuses it for immutable post bytes."""

        if self._snapshot is None or self._snapshot.phase != _SecantAdapterPhase.ARMED:
            return None
        return self._snapshot.applied_before

    def estimate_snapshot_memory(self) -> SnapshotMemoryEstimate:
        """Return exact retained and bounded peak bytes through accumulation."""

        owner_elements = sum(
            self._bound_shards[index].main_shard.numel()
            for index in self._unique_capture_indices
        )
        return snapshot_memory_estimate(
            owner_elements, finish_chunk_elements=self.finish_chunk_elements
        )

    def allocate_event_buffers(self) -> None:
        """Allocate every optimizer event buffer once during startup."""

        if self._master_before_buffer is not None:
            return
        estimate = self.estimate_snapshot_memory()
        device = (
            self._bound_shards[0].main_shard.device
            if self._bound_shards
            else self._status.device
        )
        master = torch.empty(estimate.owner_elements, dtype=torch.float32, device=device)
        applied = torch.empty(estimate.owner_elements, dtype=torch.bfloat16, device=device)
        scratch = self._allocate_finish_scratch(device)
        self._master_before_buffer = master
        self._applied_before_buffer = applied
        self._finish_scratch = scratch

    def preflight_snapshot_memory(
        self, *, additional_bytes: int = 0
    ) -> SnapshotMemoryPreflight:
        """Evaluate the complete event peak against allocator and driver limits."""

        if additional_bytes < 0:
            raise ValueError("additional diagnostic bytes must be nonnegative")

        estimate = self.estimate_snapshot_memory()
        requested_bytes = estimate.total_bytes + additional_bytes
        device = (
            self._bound_shards[0].main_shard.device if self._bound_shards else self._status.device
        )
        memory: DeviceMemoryState | None = None
        memory_query_failed = False
        if self.memory_state_provider is not None:
            try:
                memory = self.memory_state_provider(device)
            except Exception:
                memory_query_failed = True
        elif device.type == "cuda":
            try:
                memory = _default_memory_state(device)
            except Exception:
                memory_query_failed = True

        reason = SnapshotMemoryReason.NONE
        available_bytes: int | None = None
        reusable_bytes: int | None = None
        driver_need_bytes: int | None = None
        projected_reserved_bytes: int | None = None
        if memory_query_failed:
            reason = SnapshotMemoryReason.ALLOCATION_FAILED
        elif requested_bytes > self.diagnostic_max_extra_bytes:
            reason = SnapshotMemoryReason.MAX_EXTRA_BYTES
        elif memory is not None:
            reusable_bytes = max(0, memory.reserved_bytes - memory.allocated_bytes)
            driver_need_bytes = max(0, requested_bytes - reusable_bytes)
            projected_reserved_bytes = memory.reserved_bytes + driver_need_bytes
            available_bytes = memory.free_bytes + reusable_bytes
            fraction_limit = int(memory.total_bytes * self.max_memory_fraction)
            if driver_need_bytes > memory.free_bytes:
                reason = SnapshotMemoryReason.DEVICE_HEADROOM
            elif projected_reserved_bytes > fraction_limit:
                reason = SnapshotMemoryReason.DEVICE_MEMORY_FRACTION
        return SnapshotMemoryPreflight(
            estimate=estimate,
            memory=memory,
            max_extra_bytes=self.diagnostic_max_extra_bytes,
            max_memory_fraction=self.max_memory_fraction,
            available_bytes=available_bytes,
            requested_bytes=requested_bytes,
            reusable_bytes=reusable_bytes,
            driver_need_bytes=driver_need_bytes,
            projected_reserved_bytes=projected_reserved_bytes,
            accepted=reason == SnapshotMemoryReason.NONE,
            reason=reason,
        )

    def begin_event(self, *, additional_bytes: int = 0) -> SnapshotMemoryMeasurement | None:
        """Capture pre-state or expose a typed local status with no retained partial state."""

        try:
            return self._begin_event(additional_bytes=additional_bytes)
        except Exception:
            self.abort_event()
            self._set_status(DistributedOptimizerEventStatus.BEGIN_UNEXPECTED_FAILED)
            return None

    def _begin_event(self, *, additional_bytes: int = 0) -> SnapshotMemoryMeasurement | None:
        """Implement begin under the nonthrowing public lifecycle boundary."""

        if self._snapshot is not None:
            self.abort_event()
            self._set_status(DistributedOptimizerEventStatus.BEGIN_ALREADY_ARMED)
            return None
        if self._construction_status != DistributedOptimizerEventStatus.OK:
            self._set_status(self._construction_status)
            return None
        self._set_status(DistributedOptimizerEventStatus.OK)
        self._last_memory_reason = SnapshotMemoryReason.NONE
        try:
            shards = tuple(self.optimizer.iter_model_main_param_shards())
        except Exception:
            self._set_status(DistributedOptimizerEventStatus.BEGIN_ITERATOR_FAILED)
            return None
        if len(shards) != len(self._bound_shards) or any(
            self._shard_identity(bound) != self._shard_identity(current)
            for bound, current in zip(self._bound_shards, shards)
        ):
            self._set_status(DistributedOptimizerEventStatus.BEGIN_IDENTITY_CHANGED)
            return None
        devices = {shard.main_shard.device for shard in shards} | {
            shard.model_shard.device for shard in shards
        }
        if len(devices) > 1:
            self._set_status(DistributedOptimizerEventStatus.BEGIN_DEVICE_MISMATCH)
            return None
        estimate = self.estimate_snapshot_memory()
        if self._master_before_buffer is None or self._applied_before_buffer is None:
            preflight = self.preflight_snapshot_memory(additional_bytes=additional_bytes)
            if not preflight.accepted:
                self._last_memory_reason = preflight.reason
                self._set_status(DistributedOptimizerEventStatus.BEGIN_PREFLIGHT_REJECTED)
                return None
            estimate = preflight.estimate

        device = next(iter(devices), self._status.device)
        allocator_ok, allocator_before = self._sample_allocator_bytes(device)
        if not allocator_ok:
            return None
        master_before = self._master_before_buffer
        applied_before = self._applied_before_buffer
        if master_before is None:
            try:
                master_before = torch.empty(
                    estimate.owner_elements, dtype=torch.float32, device=device
                )
            except Exception:
                self._last_memory_reason = SnapshotMemoryReason.ALLOCATION_FAILED
                self._set_status(DistributedOptimizerEventStatus.BEGIN_FIRST_ALLOCATION_FAILED)
                return None
        if applied_before is None:
            try:
                applied_before = torch.empty(
                    estimate.owner_elements, dtype=torch.bfloat16, device=device
                )
            except Exception:
                master_before = None
                self._last_memory_reason = SnapshotMemoryReason.ALLOCATION_FAILED
                self._set_status(DistributedOptimizerEventStatus.BEGIN_SECOND_ALLOCATION_FAILED)
                return None

        try:
            offsets, unique_capture_indices = self._capture_layout(shards)
            if (
                offsets != self._bound_offsets
                or unique_capture_indices != self._unique_capture_indices
            ):
                raise ValueError("distributed-optimizer capture alias layout changed")
            for index in unique_capture_indices:
                start, end = offsets[index]
                shard = shards[index]
                master_before[start:end].copy_(shard.main_shard.view(-1))
                applied_before[start:end].copy_(shard.model_shard.view(-1))
        except Exception:
            master_before = None
            applied_before = None
            self._set_status(DistributedOptimizerEventStatus.BEGIN_COPY_FAILED)
            return None

        allocator_ok, allocator_after = self._sample_allocator_bytes(device)
        if not allocator_ok:
            master_before = None
            applied_before = None
            return None
        allocator_delta = self._allocator_delta(allocator_before, allocator_after)
        measurement = SnapshotMemoryMeasurement(
            estimated_bytes=estimate.total_bytes,
            payload_bytes=master_before.nbytes + applied_before.nbytes,
            finish_scratch_bytes=estimate.finish_scratch_bytes,
            allocator_delta_bytes=allocator_delta,
            allocator_peak_delta_bytes=allocator_delta,
        )
        self._snapshot = _SnapshotState(
            shards=shards,
            offsets=tuple(offsets),
            master_before=master_before,
            applied_before=applied_before,
            measurement=measurement,
            allocator_before=allocator_before,
            unique_capture_indices=unique_capture_indices,
        )
        return measurement

    def measure_snapshot_memory(self) -> SnapshotMemoryMeasurement | None:
        """Return retained payload and peak measurements for the active event."""

        return None if self._snapshot is None else self._snapshot.measurement

    @torch.no_grad()
    def commit_secant_delta(
        self, accumulator: PackedSufficientStatistics | None = None, *, update_successful: bool
    ) -> SnapshotMemoryMeasurement | None:
        """Turn the shared FP32 pre snapshot into delta without changing live masters.

        The optional accumulator receives the ordinary Tier-0 applied-update moments
        before the pre snapshot is overwritten. Runtime failures remain encoded in
        :attr:`local_status`; this local half never launches a collective.

        Args:
            accumulator: Canonical registry accumulator shared with Tier 0, if due.
            update_successful: Globally agreed optimizer-step result supplied by integration.

        Returns:
            Updated memory measurement, or ``None`` when local preparation failed.
        """

        snapshot = self._snapshot
        if snapshot is None:
            self._set_status(DistributedOptimizerEventStatus.SECANT_DELTA_NOT_ARMED)
            return None
        if snapshot.phase in (
            _SecantAdapterPhase.DELTA_COMMITTED,
            _SecantAdapterPhase.MIDPOINT_INSTALLED,
            _SecantAdapterPhase.RESTORED,
        ):
            return snapshot.commit_measurement
        if snapshot.phase in (_SecantAdapterPhase.DELTA_COMMITTING, _SecantAdapterPhase.FAILED):
            return None
        if snapshot.phase != _SecantAdapterPhase.ARMED:
            self._set_status(DistributedOptimizerEventStatus.SECANT_INVALID_PHASE)
            return None
        if not update_successful:
            self._set_status(DistributedOptimizerEventStatus.UPDATE_SKIPPED)
            self.abort_event()
            return None

        try:
            current_shards = tuple(self.optimizer.iter_model_main_param_shards())
            if len(current_shards) != len(snapshot.shards) or any(
                self._shard_identity(original) != self._shard_identity(current)
                for original, current in zip(snapshot.shards, current_shards)
            ):
                self._set_status(DistributedOptimizerEventStatus.SECANT_DELTA_IDENTITY_CHANGED)
                return None

            scratch = self._allocate_finish_scratch(snapshot.master_before.device)
            try:
                snapshot.phase = _SecantAdapterPhase.DELTA_COMMITTING
                self._verify_materialization(current_shards, scratch)
                torch.eq(
                    self._status,
                    int(DistributedOptimizerEventStatus.OK),
                    out=scratch.materialization_valid,
                )
                if accumulator is not None:
                    self._accumulate_update_moments(
                        accumulator,
                        snapshot,
                        current_shards,
                        scratch,
                        scratch.materialization_valid.squeeze(0),
                    )

                snapshot.post_fingerprints = torch.empty(
                    4, dtype=torch.int64, device=snapshot.master_before.device
                )
                self._fingerprint_shards(
                    current_shards,
                    snapshot.unique_capture_indices,
                    scratch,
                    use_main=True,
                    out=snapshot.post_fingerprints[:2],
                )
                self._fingerprint_shards(
                    current_shards,
                    snapshot.unique_capture_indices,
                    scratch,
                    use_main=False,
                    out=snapshot.post_fingerprints[2:],
                )
                for index in snapshot.unique_capture_indices:
                    start, end = snapshot.offsets[index]
                    snapshot.applied_before[start:end].copy_(
                        current_shards[index].model_shard.detach().view(-1)
                    )
                self._transform_pre_to_delta(snapshot, current_shards, scratch)
                measurement = self._measurement_with_peak(snapshot, snapshot.master_before.device)
                if measurement is None:
                    snapshot.phase = _SecantAdapterPhase.FAILED
                    return None
            finally:
                self._clear_finish_scratch(scratch)
            snapshot.measurement = measurement
            snapshot.commit_measurement = measurement
            snapshot.phase = _SecantAdapterPhase.DELTA_COMMITTED
            return measurement
        except Exception:
            if snapshot.phase == _SecantAdapterPhase.DELTA_COMMITTING:
                snapshot.phase = _SecantAdapterPhase.FAILED
            self._set_status(DistributedOptimizerEventStatus.SECANT_DELTA_FAILED)
            return None

    @torch.no_grad()
    def install_secant_midpoint(self) -> torch.Tensor:
        """Install BF16 owner midpoints through one bounded FP32 cast workspace.

        Returns:
            Device-resident local status for a later fixed readiness agreement.
        """

        snapshot = self._snapshot
        if snapshot is None:
            self._set_status(DistributedOptimizerEventStatus.SECANT_MIDPOINT_UNAVAILABLE)
            return self._status
        if snapshot.phase in (_SecantAdapterPhase.DELTA_COMMITTING, _SecantAdapterPhase.FAILED):
            return self._status
        if snapshot.phase in (_SecantAdapterPhase.MIDPOINT_INSTALLED, _SecantAdapterPhase.RESTORED):
            return self._status
        if snapshot.phase != _SecantAdapterPhase.DELTA_COMMITTED:
            self._set_status(DistributedOptimizerEventStatus.SECANT_INVALID_PHASE)
            return self._status

        snapshot.phase = _SecantAdapterPhase.MIDPOINT_INSTALLED
        try:
            scratch = self._allocate_finish_scratch(snapshot.master_before.device)
            try:
                for index in snapshot.unique_capture_indices:
                    shard = snapshot.shards[index]
                    offset, _ = snapshot.offsets[index]
                    main = shard.main_shard.detach().view(-1)
                    applied = shard.model_shard.detach().view(-1)
                    for start in range(0, main.numel(), self.finish_chunk_elements):
                        end = min(start + self.finish_chunk_elements, main.numel())
                        size = end - start
                        midpoint = scratch.master_delta_fp32[:size]
                        midpoint.copy_(main[start:end])
                        midpoint.add_(
                            snapshot.master_before[offset + start : offset + end], alpha=-0.5
                        )
                        self._copy_secant_chunk(applied[start:end], midpoint, "midpoint")
            finally:
                self._clear_finish_scratch(scratch)
        except Exception:
            self._set_status(DistributedOptimizerEventStatus.SECANT_MIDPOINT_INSTALL_FAILED)
        return self._status

    @torch.no_grad()
    def restore_secant_post(self) -> torch.Tensor:
        """Best-effort restore every owner view and verify post state independently.

        A failure for one physical owner does not skip later owners. FP32 masters
        are never written; their bounded fingerprints are verified alongside the
        bitwise BF16 cast expected in each optimizer parameter-buffer view.

        Returns:
            Device-resident local status for the integration layer's fatal agreement.
        """

        snapshot = self._snapshot
        if snapshot is None:
            self._set_status(DistributedOptimizerEventStatus.SECANT_MIDPOINT_UNAVAILABLE)
            return self._status
        if snapshot.phase in (_SecantAdapterPhase.DELTA_COMMITTING, _SecantAdapterPhase.FAILED):
            return self._status
        if snapshot.phase == _SecantAdapterPhase.RESTORED:
            return self._status
        if snapshot.phase != _SecantAdapterPhase.MIDPOINT_INSTALLED:
            self._set_status(DistributedOptimizerEventStatus.SECANT_INVALID_PHASE)
            return self._status

        scratch: _FinishScratch | None = None
        copy_failed = False
        try:
            scratch = self._allocate_finish_scratch(snapshot.master_before.device)
            for index in snapshot.unique_capture_indices:
                shard = snapshot.shards[index]
                applied = shard.model_shard.detach().view(-1)
                offset, _ = snapshot.offsets[index]
                try:
                    for start in range(0, applied.numel(), self.finish_chunk_elements):
                        end = min(start + self.finish_chunk_elements, applied.numel())
                        self._copy_secant_chunk(
                            applied[start:end],
                            snapshot.applied_before[offset + start : offset + end],
                            "restore",
                        )
                except Exception:
                    copy_failed = True
                    continue

            if copy_failed:
                self._set_status(DistributedOptimizerEventStatus.SECANT_RESTORE_COPY_FAILED)
            self._verify_secant_post(snapshot, scratch)
        except Exception:
            self._set_status(DistributedOptimizerEventStatus.SECANT_RESTORE_VERIFY_FAILED)
        finally:
            if scratch is not None:
                self._clear_finish_scratch(scratch)
            snapshot.phase = _SecantAdapterPhase.RESTORED
        return self._status

    def release_secant_event(self) -> None:
        """Release retained secant buffers after global completion or central fatal handling."""

        self.abort_event()

    @torch.no_grad()
    def finish_event(
        self, accumulator: PackedSufficientStatistics, *, update_successful: bool
    ) -> SnapshotMemoryMeasurement | None:
        """Verify successful materialization, then accumulate bounded update moments.

        ``update_successful`` is mandatory: an early or skipped finish can never
        reinterpret stale applied values as a successful optimizer update.

        Args:
            accumulator: Existing registry-bound Tier-0 accumulator.
            update_successful: Result of the training update-success consensus.

        Returns:
            Completed event memory measurement, or ``None`` on local failure/skip.
        """

        if self._snapshot is None:
            self._set_status(DistributedOptimizerEventStatus.FINISH_NOT_ARMED)
            return None

        try:
            return self._finish_event(accumulator, update_successful=update_successful)
        except Exception:
            self._invalidate_bound_metrics(accumulator)
            self._set_status(DistributedOptimizerEventStatus.FINISH_UNEXPECTED_FAILED)
            return None
        finally:
            self.abort_event()

    def _finish_event(
        self, accumulator: PackedSufficientStatistics, *, update_successful: bool
    ) -> SnapshotMemoryMeasurement | None:
        """Implement finish under the nonthrowing public lifecycle boundary."""

        snapshot = self._snapshot
        assert snapshot is not None
        if snapshot.delta_ready:
            self._set_status(DistributedOptimizerEventStatus.FINISH_UNEXPECTED_FAILED)
            return None
        if not update_successful:
            self._set_status(DistributedOptimizerEventStatus.UPDATE_SKIPPED)
            return None

        try:
            current_shards = tuple(self.optimizer.iter_model_main_param_shards())
        except Exception:
            self._set_status(DistributedOptimizerEventStatus.FINISH_ITERATOR_FAILED)
            return None
        if len(current_shards) != len(snapshot.shards):
            self._set_status(DistributedOptimizerEventStatus.FINISH_SHARD_COUNT_CHANGED)
            return None
        if any(
            self._shard_identity(original) != self._shard_identity(current)
            for original, current in zip(snapshot.shards, current_shards)
        ):
            self._set_status(DistributedOptimizerEventStatus.FINISH_SHARD_IDENTITY_CHANGED)
            return None

        device = snapshot.master_before.device
        scratch = self._finish_scratch
        try:
            if scratch is None:
                try:
                    scratch = self._allocate_finish_scratch(device)
                except Exception:
                    self._last_memory_reason = SnapshotMemoryReason.ALLOCATION_FAILED
                    self._set_status(
                        DistributedOptimizerEventStatus.FINISH_SCRATCH_ALLOCATION_FAILED
                    )
                    return None

            measurement = self._measurement_with_peak(snapshot, device)
            if measurement is None:
                return None
            snapshot.measurement = measurement

            try:
                self._verify_materialization(current_shards, scratch)
                torch.eq(
                    self._status,
                    int(DistributedOptimizerEventStatus.OK),
                    out=scratch.materialization_valid,
                )
                self._accumulate_update_moments(
                    accumulator,
                    snapshot,
                    current_shards,
                    scratch,
                    scratch.materialization_valid.squeeze(0),
                )
            except Exception:
                self._invalidate_bound_metrics(accumulator)
                self._set_status(DistributedOptimizerEventStatus.ACCUMULATION_FAILED)
                return None

            measurement = self._measurement_with_peak(snapshot, device)
            if measurement is None:
                self._invalidate_bound_metrics(accumulator)
            return measurement
        finally:
            if scratch is not None and scratch is not self._finish_scratch:
                self._clear_finish_scratch(scratch)

    def abort_event(self) -> None:
        """Release all retained pre-state without launching a collective."""

        self._snapshot = None

    def _record_construction_failure(self, status: DistributedOptimizerEventStatus) -> None:
        self._construction_status = status
        self._set_status(status)
        self._bound_shards = ()

    def _set_status(self, status: DistributedOptimizerEventStatus) -> None:
        self._status.fill_(int(status))

    def _allocate_finish_scratch(self, device: torch.device) -> _FinishScratch:
        estimate = self.estimate_snapshot_memory()
        layout = _scratch_layout(estimate.finish_chunk_elements)
        storage = torch.empty(layout.total_bytes, dtype=torch.uint8, device=device)
        return _FinishScratch(
            storage=storage,
            expected_bf16=_typed_view(storage, layout.expected_bf16, torch.bfloat16),
            applied_before_fp32=_typed_view(storage, layout.applied_before_fp32, torch.float32),
            applied_after_fp32=_typed_view(storage, layout.applied_after_fp32, torch.float32),
            master_delta_fp32=_typed_view(storage, layout.master_delta_fp32, torch.float32),
            applied_delta_fp32=_typed_view(storage, layout.applied_delta_fp32, torch.float32),
            work_fp64=_typed_view(storage, layout.work_fp64, torch.float64),
            hash_position_int64=_typed_view(storage, layout.hash_position_int64, torch.int64),
            finite=_typed_view(storage, layout.finite, torch.bool),
            auxiliary=_typed_view(storage, layout.auxiliary, torch.bool),
            moment_fp64=_typed_view(storage, layout.moment_fp64, torch.float64),
            moment_fp32=_typed_view(storage, layout.moment_fp32, torch.float32),
            materialization_valid=_typed_view(storage, layout.materialization_valid, torch.bool),
            scalar_bool=_typed_view(storage, layout.scalar_bool, torch.bool),
            status_int64=_typed_view(storage, layout.status_int64, torch.int64),
        )

    @staticmethod
    def _copy_secant_chunk(destination: torch.Tensor, source: torch.Tensor, phase: str) -> None:
        """Copy one bounded owner chunk; ``phase`` is an injection seam for tests."""

        del phase
        destination.copy_(source)

    def _transform_pre_to_delta(
        self,
        snapshot: _SnapshotState,
        shards: tuple[ModelMainParamShard, ...],
        scratch: _FinishScratch,
    ) -> None:
        for index in snapshot.unique_capture_indices:
            shard = shards[index]
            offset, _ = snapshot.offsets[index]
            main = shard.main_shard.detach().view(-1)
            for start in range(0, main.numel(), self.finish_chunk_elements):
                end = min(start + self.finish_chunk_elements, main.numel())
                delta = snapshot.master_before[offset + start : offset + end]
                torch.sub(main[start:end], delta, out=delta)
                finite = scratch.finite[: end - start]
                auxiliary = scratch.auxiliary[: end - start]
                self._initialize_finite_mask(delta, finite, auxiliary)
                torch.all(finite, dim=(0,), out=scratch.scalar_bool.squeeze(0))
                torch.logical_not(scratch.scalar_bool, out=scratch.scalar_bool)
                scratch.status_int64.copy_(scratch.scalar_bool)
                scratch.status_int64.mul_(
                    int(DistributedOptimizerEventStatus.SECANT_DELTA_NONFINITE)
                )
                torch.maximum(self._status, scratch.status_int64, out=self._status)

    def _fingerprint_shards(
        self,
        shards: tuple[ModelMainParamShard, ...],
        indices: tuple[int, ...],
        scratch: _FinishScratch,
        *,
        use_main: bool,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if out.dtype != torch.int64 or out.numel() != 2:
            raise ValueError("post fingerprints require exactly two int64 hash states")
        seeds = (2611923443488327891, 7046029254386353131)
        metadata = tuple(
            token
            for sequence, index in enumerate(indices)
            for token in self._fingerprint_metadata(shards, index, sequence)
        )
        for hash_index, seed in enumerate(seeds):
            out[hash_index].fill_(self._host_metadata_hash(metadata, seed))

        element_position = 0
        for index in indices:
            shard = shards[index]
            values = (shard.main_shard if use_main else shard.model_shard).detach().view(-1)
            for start in range(0, values.numel(), self.finish_chunk_elements):
                end = min(start + self.finish_chunk_elements, values.numel())
                size = end - start
                raw_dtype = torch.int32 if values.dtype == torch.float32 else torch.int16
                raw = values.view(raw_dtype)[start:end]
                work = scratch.work_fp64[:size].view(torch.int64)
                temporary = scratch.hash_position_int64[:size]
                for hash_index, seed in enumerate(seeds):
                    work.copy_(raw)
                    torch.arange(element_position + start, element_position + end, out=temporary)
                    temporary.mul_(1442695040888963407)
                    work.bitwise_xor_(temporary)
                    work.bitwise_xor_(seed)
                    work.mul_(6364136223846793005)
                    torch.bitwise_right_shift(work, 29, out=temporary)
                    work.bitwise_xor_(temporary)
                    work.mul_(1442695040888963407)
                    torch.bitwise_right_shift(work, 32, out=temporary)
                    work.bitwise_xor_(temporary)
                    torch.sum(
                        work, dim=(0,), dtype=torch.int64, out=scratch.status_int64.squeeze(0)
                    )
                    out[hash_index].add_(scratch.status_int64.squeeze(0))
            element_position += values.numel()
        return out

    @classmethod
    def _fingerprint_metadata(
        cls, shards: tuple[ModelMainParamShard, ...], index: int, sequence: int
    ) -> tuple[int, ...]:
        shard = shards[index]
        pair_key = cls._paired_view_key(shard)
        aliases = tuple(
            candidate for candidate in shards if cls._paired_view_key(candidate) == pair_key
        )
        tokens = [
            sequence,
            len(aliases),
            sum(candidate.logical_owner for candidate in aliases),
            sum(candidate.shared for candidate in aliases),
            sum(candidate.tied for candidate in aliases),
            sum(candidate.tied_owner for candidate in aliases),
        ]
        for tensor in (shard.main_shard, shard.model_shard):
            dtype_code = {torch.float32: 1, torch.bfloat16: 2}.get(tensor.dtype)
            if dtype_code is None:
                raise ValueError("unsupported post-fingerprint dtype")
            tokens.extend(
                (
                    dtype_code,
                    tensor.ndim,
                    tensor.storage_offset(),
                    tensor.numel(),
                    -1 if tensor.device.index is None else tensor.device.index,
                    1 if tensor.device.type == "cuda" else 0,
                    *tensor.shape,
                    *tensor.stride(),
                )
            )
        return tuple(tokens)

    @staticmethod
    def _host_metadata_hash(tokens: tuple[int, ...], seed: int) -> int:
        mask = (1 << 64) - 1
        value = seed & mask
        for token in tokens:
            value ^= token & mask
            value = (value * 1099511628211) & mask
            value ^= value >> 32
        return value if value < (1 << 63) else value - (1 << 64)

    def _verify_secant_post(self, snapshot: _SnapshotState, scratch: _FinishScratch) -> None:
        mismatch = scratch.scalar_bool.squeeze(0)
        for index in snapshot.unique_capture_indices:
            shard = snapshot.shards[index]
            applied = shard.model_shard.detach().view(-1)
            offset, _ = snapshot.offsets[index]
            for start in range(0, applied.numel(), self.finish_chunk_elements):
                end = min(start + self.finish_chunk_elements, applied.numel())
                expected = snapshot.applied_before[offset + start : offset + end]
                current_bits = applied[start:end].view(torch.int16)
                expected_bits = expected.view(torch.int16)
                torch.ne(current_bits, expected_bits, out=scratch.auxiliary[: end - start])
                torch.any(scratch.auxiliary[: end - start], dim=(0,), out=mismatch)
                scratch.status_int64.copy_(mismatch)
                scratch.status_int64.mul_(
                    int(DistributedOptimizerEventStatus.SECANT_RESTORE_VERIFY_FAILED)
                )
                torch.maximum(self._status, scratch.status_int64, out=self._status)

        current_master = self._fingerprint_shards(
            snapshot.shards,
            snapshot.unique_capture_indices,
            scratch,
            use_main=True,
            out=scratch.moment_fp64[:2].view(torch.int64),
        )
        current_applied = self._fingerprint_shards(
            snapshot.shards,
            snapshot.unique_capture_indices,
            scratch,
            use_main=False,
            out=scratch.moment_fp64[2:4].view(torch.int64),
        )
        post_fingerprints = snapshot.post_fingerprints
        if post_fingerprints is None:
            self._set_status(DistributedOptimizerEventStatus.SECANT_RESTORE_VERIFY_FAILED)
            return
        for current, expected in (
            (current_master, post_fingerprints[:2]),
            (current_applied, post_fingerprints[2:]),
        ):
            torch.ne(current, expected, out=scratch.auxiliary[:2])
            torch.any(scratch.auxiliary[:2], dim=(0,), out=mismatch)
            scratch.status_int64.copy_(mismatch)
            scratch.status_int64.mul_(
                int(DistributedOptimizerEventStatus.SECANT_RESTORE_VERIFY_FAILED)
            )
            torch.maximum(self._status, scratch.status_int64, out=self._status)

    @staticmethod
    def _clear_finish_scratch(scratch: _FinishScratch) -> None:
        del scratch.expected_bf16
        del scratch.applied_before_fp32
        del scratch.applied_after_fp32
        del scratch.master_delta_fp32
        del scratch.applied_delta_fp32
        del scratch.work_fp64
        del scratch.hash_position_int64
        del scratch.finite
        del scratch.auxiliary
        del scratch.moment_fp64
        del scratch.moment_fp32
        del scratch.materialization_valid
        del scratch.scalar_bool
        del scratch.status_int64
        del scratch.storage

    def _verify_materialization(
        self, shards: tuple[ModelMainParamShard, ...], scratch: _FinishScratch
    ) -> None:
        for shard in shards:
            main = shard.main_shard.view(-1)
            applied = shard.model_shard.view(-1)
            for start in range(0, main.numel(), self.finish_chunk_elements):
                end = min(start + self.finish_chunk_elements, main.numel())
                size = end - start
                expected = scratch.expected_bf16[:size]
                mismatch = scratch.auxiliary[:size]
                expected.copy_(main[start:end])
                torch.ne(expected, applied[start:end], out=mismatch)
                torch.any(mismatch, dim=(0,), out=scratch.scalar_bool.squeeze(0))
                scratch.status_int64.copy_(scratch.scalar_bool)
                scratch.status_int64.mul_(
                    int(DistributedOptimizerEventStatus.MATERIALIZATION_MISMATCH)
                )
                torch.maximum(self._status, scratch.status_int64, out=self._status)

    def _accumulate_update_moments(
        self,
        accumulator: PackedSufficientStatistics,
        snapshot: _SnapshotState,
        shards: tuple[ModelMainParamShard, ...],
        scratch: _FinishScratch,
        materialization_valid: torch.Tensor,
    ) -> None:
        for shard, (snapshot_start, snapshot_end) in zip(shards, snapshot.offsets):
            if not shard.logical_owner:
                continue
            logical_name = self.metric_name_by_parameter[shard.model_param]
            main_before = snapshot.master_before[snapshot_start:snapshot_end]
            applied_before = snapshot.applied_before[snapshot_start:snapshot_end]
            main_after = shard.main_shard.detach().view(-1)
            applied_after = shard.model_shard.detach().view(-1)
            for start in range(0, main_after.numel(), self.finish_chunk_elements):
                end = min(start + self.finish_chunk_elements, main_after.numel())
                moments = self._chunk_moments(
                    scratch,
                    main_before[start:end],
                    main_after[start:end],
                    applied_before[start:end],
                    applied_after[start:end],
                    materialization_valid,
                )
                self.registry.add_applied_update_moments(
                    accumulator,
                    logical_name,
                    **moments,
                    materialization_valid=materialization_valid,
                )

    @staticmethod
    def _chunk_moments(
        scratch: _FinishScratch,
        master_before: torch.Tensor,
        master_after: torch.Tensor,
        applied_before: torch.Tensor,
        applied_after: torch.Tensor,
        materialization_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        size = master_after.numel()
        applied_before_fp32 = scratch.applied_before_fp32[:size]
        applied_after_fp32 = scratch.applied_after_fp32[:size]
        master_delta = scratch.master_delta_fp32[:size]
        applied_delta = scratch.applied_delta_fp32[:size]
        work = scratch.work_fp64[:size]
        finite = scratch.finite[:size]
        auxiliary = scratch.auxiliary[:size]
        moment_fp64 = scratch.moment_fp64
        moment_fp32 = scratch.moment_fp32
        count = moment_fp64[0]
        master_sumsq = moment_fp64[1]
        applied_sumsq = moment_fp64[2]
        applied_pre_sumsq = moment_fp64[3]
        master_nonzero = moment_fp64[4]
        applied_nonzero = moment_fp64[5]
        cast_zero = moment_fp64[6]
        nonfinite = moment_fp64[7]
        arithmetic_error = moment_fp64[8]
        materialization_error = moment_fp64[9]
        maximum = moment_fp32[0]
        minimum = moment_fp32[1]

        applied_before_fp32.copy_(applied_before)
        applied_after_fp32.copy_(applied_after)
        torch.sub(master_after, master_before, out=master_delta)
        torch.sub(applied_after_fp32, applied_before_fp32, out=applied_delta)

        Bf16DistributedOptimizerDiagnosticAdapter._initialize_finite_mask(
            master_before, finite, auxiliary
        )
        Bf16DistributedOptimizerDiagnosticAdapter._and_finite_mask(master_after, finite, auxiliary)
        Bf16DistributedOptimizerDiagnosticAdapter._and_finite_mask(
            applied_before_fp32, finite, auxiliary
        )
        Bf16DistributedOptimizerDiagnosticAdapter._and_finite_mask(
            applied_after_fp32, finite, auxiliary
        )
        torch.sum(finite, dim=(0,), dtype=torch.float64, out=count)
        nonfinite.fill_(size).sub_(count)
        torch.logical_not(finite, out=auxiliary)

        work.copy_(master_delta)
        work.masked_fill_(auxiliary, 0)
        work.square_()
        torch.sum(work, dim=(0,), out=master_sumsq)
        work.copy_(applied_delta)
        work.masked_fill_(auxiliary, 0)
        work.square_()
        torch.sum(work, dim=(0,), out=applied_sumsq)
        work.copy_(applied_before_fp32)
        work.masked_fill_(auxiliary, 0)
        work.square_()
        torch.sum(work, dim=(0,), out=applied_pre_sumsq)

        applied_delta.masked_fill_(auxiliary, -torch.inf)
        torch.amax(applied_delta, dim=(0,), out=maximum)
        applied_delta.masked_fill_(auxiliary, torch.inf)
        torch.amin(applied_delta, dim=(0,), out=minimum)

        torch.ne(master_delta, 0, out=auxiliary)
        torch.logical_and(auxiliary, finite, out=auxiliary)
        torch.sum(auxiliary, dim=(0,), dtype=torch.float64, out=master_nonzero)
        torch.eq(applied_delta, 0, out=finite)
        torch.logical_and(finite, auxiliary, out=finite)
        torch.sum(finite, dim=(0,), dtype=torch.float64, out=cast_zero)
        applied_nonzero.copy_(master_nonzero).sub_(cast_zero)

        numeric = moment_fp64[:8]
        arithmetic_valid = finite[0]
        scalar_auxiliary = scratch.scalar_bool.squeeze(0)
        Bf16DistributedOptimizerDiagnosticAdapter._initialize_finite_mask(
            numeric[0], arithmetic_valid, scalar_auxiliary
        )
        for contribution in numeric[1:]:
            Bf16DistributedOptimizerDiagnosticAdapter._and_finite_mask(
                contribution, arithmetic_valid, scalar_auxiliary
            )
        arithmetic_error.copy_(arithmetic_valid).mul_(-1).add_(1)
        torch.logical_not(arithmetic_valid, out=scalar_auxiliary)
        numeric.masked_fill_(scalar_auxiliary, 0)

        materialization_error.copy_(materialization_valid).mul_(-1).add_(1)
        torch.logical_not(materialization_valid, out=scalar_auxiliary)
        numeric.masked_fill_(scalar_auxiliary, 0)
        maximum.masked_fill_(scalar_auxiliary, -torch.inf)
        minimum.masked_fill_(scalar_auxiliary, torch.inf)
        return {
            "count": count,
            "master_sumsq": master_sumsq,
            "applied_sumsq": applied_sumsq,
            "applied_pre_sumsq": applied_pre_sumsq,
            "master_nonzero": master_nonzero,
            "applied_nonzero": applied_nonzero,
            "cast_zero": cast_zero,
            "nonfinite": nonfinite,
            "maximum": maximum,
            "minimum": minimum,
            "arithmetic_error": arithmetic_error,
            "materialization_error": materialization_error,
        }

    @staticmethod
    def _initialize_finite_mask(
        values: torch.Tensor, finite: torch.Tensor, auxiliary: torch.Tensor
    ) -> None:
        torch.eq(values, values, out=finite)
        Bf16DistributedOptimizerDiagnosticAdapter._and_not_infinite(values, finite, auxiliary)

    @staticmethod
    def _and_finite_mask(
        values: torch.Tensor, finite: torch.Tensor, auxiliary: torch.Tensor
    ) -> None:
        torch.eq(values, values, out=auxiliary)
        torch.logical_and(finite, auxiliary, out=finite)
        Bf16DistributedOptimizerDiagnosticAdapter._and_not_infinite(values, finite, auxiliary)

    @staticmethod
    def _and_not_infinite(
        values: torch.Tensor, finite: torch.Tensor, auxiliary: torch.Tensor
    ) -> None:
        torch.ne(values, torch.inf, out=auxiliary)
        torch.logical_and(finite, auxiliary, out=finite)
        torch.ne(values, -torch.inf, out=auxiliary)
        torch.logical_and(finite, auxiliary, out=finite)

    def _invalidate_bound_metrics(self, accumulator: PackedSufficientStatistics) -> None:
        for shard in self._bound_shards:
            if not shard.logical_owner:
                continue
            logical_name = self.metric_name_by_parameter.get(shard.model_param)
            if logical_name is None:
                continue
            try:
                accumulator.invalidate(logical_name)
            except Exception:
                return

    def _measurement_with_peak(
        self, snapshot: _SnapshotState, device: torch.device
    ) -> SnapshotMemoryMeasurement | None:
        allocator_ok, allocator_now = self._sample_allocator_bytes(device)
        if not allocator_ok:
            return None
        allocator_peak = self._allocator_delta(snapshot.allocator_before, allocator_now)
        previous_peak = snapshot.measurement.allocator_peak_delta_bytes
        if allocator_peak is not None and previous_peak is not None:
            allocator_peak = max(allocator_peak, previous_peak)
        elif allocator_peak is None:
            allocator_peak = previous_peak
        return replace(
            snapshot.measurement,
            payload_bytes=(
                snapshot.master_before.nbytes
                + snapshot.applied_before.nbytes
                + (0 if snapshot.post_fingerprints is None else snapshot.post_fingerprints.nbytes)
                + snapshot.measurement.finish_scratch_bytes
            ),
            allocator_peak_delta_bytes=allocator_peak,
        )

    def _sample_allocator_bytes(self, device: torch.device) -> tuple[bool, int | None]:
        """Sample optional allocator telemetry without crossing the event boundary."""

        try:
            return True, self._allocator_bytes(device)
        except Exception:
            self._set_status(DistributedOptimizerEventStatus.ALLOCATOR_QUERY_FAILED)
            return False, None

    @staticmethod
    def _allocator_bytes(device: torch.device) -> int | None:
        if device.type != "cuda":
            return None
        return torch.cuda.memory_allocated(device)

    @staticmethod
    def _allocator_delta(before: int | None, after: int | None) -> int | None:
        if before is None or after is None:
            return None
        return max(0, after - before)

    @staticmethod
    def _shard_identity(shard: ModelMainParamShard) -> tuple[object, ...]:
        return (
            shard.model_param,
            shard.optimizer_group_index,
            shard.group_parameter_index,
            shard.model_chunk_index,
            shard.buffer_index,
            shard.bucket_index,
            shard.param_range,
            shard.gbuf_world_range,
            shard.bucket_range,
            shard.local_buffer_range,
            shard.main_shard.untyped_storage().data_ptr(),
            shard.main_shard.storage_offset(),
            shard.main_shard.shape,
            shard.main_shard.stride(),
            shard.model_shard.untyped_storage().data_ptr(),
            shard.model_shard.storage_offset(),
            shard.model_shard.shape,
            shard.model_shard.stride(),
            shard.main_shard.numel(),
            shard.model_shard.numel(),
            shard.main_shard.dtype,
            shard.model_shard.dtype,
        )

    @staticmethod
    def _tensor_view_key(tensor: torch.Tensor) -> tuple[object, ...]:
        return (
            tensor.device,
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tensor.numel(),
            tensor.dtype,
            tuple(tensor.shape),
            tensor.stride(),
        )

    @classmethod
    def _paired_view_key(cls, shard: ModelMainParamShard) -> tuple[object, ...]:
        return cls._tensor_view_key(shard.main_shard), cls._tensor_view_key(shard.model_shard)

    @staticmethod
    def _physical_interval(tensor: torch.Tensor) -> tuple[tuple[torch.device, int], int, int]:
        element_size = tensor.element_size()
        start = tensor.storage_offset() * element_size
        return (
            (tensor.device, tensor.untyped_storage().data_ptr()),
            start,
            start + tensor.numel() * element_size,
        )

    @staticmethod
    def _capture_layout(
        shards: tuple[ModelMainParamShard, ...]
    ) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
        """Canonicalize exact paired aliases and reject every other physical overlap."""

        for shard in shards:
            if shard.main_shard.numel() != shard.model_shard.numel():
                raise ValueError("paired main/model optimizer views require equal element counts")
            if not shard.main_shard.is_contiguous() or not shard.model_shard.is_contiguous():
                raise ValueError("optimizer owner views must be contiguous")

        for tensors in (
            tuple(shard.main_shard for shard in shards),
            tuple(shard.model_shard for shard in shards),
        ):
            intervals: dict[tuple[torch.device, int], list[tuple[int, int, tuple[object, ...]]]] = (
                {}
            )
            for tensor in tensors:
                storage, start, end = Bf16DistributedOptimizerDiagnosticAdapter._physical_interval(
                    tensor
                )
                key = Bf16DistributedOptimizerDiagnosticAdapter._tensor_view_key(tensor)
                for other_start, other_end, other_key in intervals.setdefault(storage, []):
                    if start < other_end and other_start < end and key != other_key:
                        raise ValueError("non-exact optimizer tensor views physically overlap")
                intervals[storage].append((start, end, key))

        main_to_model: dict[tuple[object, ...], set[tuple[object, ...]]] = {}
        model_to_main: dict[tuple[object, ...], set[tuple[object, ...]]] = {}
        groups: dict[tuple[object, ...], list[int]] = {}
        for index, shard in enumerate(shards):
            main_key, model_key = Bf16DistributedOptimizerDiagnosticAdapter._paired_view_key(shard)
            main_to_model.setdefault(main_key, set()).add(model_key)
            model_to_main.setdefault(model_key, set()).add(main_key)
            groups.setdefault((main_key, model_key), []).append(index)
        if any(len(models) != 1 for models in main_to_model.values()) or any(
            len(mains) != 1 for mains in model_to_main.values()
        ):
            raise ValueError("main/model aliases do not form exact paired views")

        offsets: list[tuple[int, int] | None] = [None] * len(shards)
        unique_indices: list[int] = []
        offset = 0
        for _, indices in sorted(groups.items(), key=lambda item: repr(item[0])):
            representative = min(indices)
            end = offset + shards[representative].main_shard.numel()
            current = (offset, end)
            for index in indices:
                offsets[index] = current
            unique_indices.append(representative)
            offset = end
        return tuple(value for value in offsets if value is not None), tuple(unique_indices)

    @staticmethod
    def _validate_unique_local_ownership(shards: tuple[ModelMainParamShard, ...]) -> None:
        Bf16DistributedOptimizerDiagnosticAdapter._capture_layout(shards)
