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
    finish_scratch_bytes: int
    snapshot_bytes: int
    total_bytes: int
    owner_elements: int
    finish_chunk_elements: int


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


@dataclass(frozen=True)
class _FinishScratchLayout:
    expected_bf16: tuple[int, int]
    applied_before_fp32: tuple[int, int]
    applied_after_fp32: tuple[int, int]
    master_delta_fp32: tuple[int, int]
    applied_delta_fp32: tuple[int, int]
    work_fp64: tuple[int, int]
    finite: tuple[int, int]
    auxiliary: tuple[int, int]
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
    finite: torch.Tensor
    auxiliary: torch.Tensor


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

    def reserve(bytes_per_element: int, alignment: int) -> tuple[int, int]:
        nonlocal offset
        offset = _align(offset, alignment)
        start = offset
        offset += elements * bytes_per_element
        return start, offset

    expected_bf16 = reserve(2, 2)
    applied_before_fp32 = reserve(4, 4)
    applied_after_fp32 = reserve(4, 4)
    master_delta_fp32 = reserve(4, 4)
    applied_delta_fp32 = reserve(4, 4)
    work_fp64 = reserve(8, 8)
    finite = reserve(1, 1)
    auxiliary = reserve(1, 1)
    return _FinishScratchLayout(
        expected_bf16=expected_bf16,
        applied_before_fp32=applied_before_fp32,
        applied_after_fp32=applied_after_fp32,
        master_delta_fp32=master_delta_fp32,
        applied_delta_fp32=applied_delta_fp32,
        work_fp64=work_fp64,
        finite=finite,
        auxiliary=auxiliary,
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
        self.metric_name_by_parameter = dict(metric_name_by_parameter)
        self.diagnostic_max_extra_bytes = diagnostic_max_extra_bytes
        self.max_memory_fraction = max_memory_fraction
        self.finish_chunk_elements = finish_chunk_elements
        self.capability_report = report
        self.memory_state_provider = memory_state_provider
        self._snapshot: _SnapshotState | None = None
        self._last_memory_reason = SnapshotMemoryReason.NONE
        self._status = torch.zeros(
            1, dtype=torch.int64, device=_negotiation_device(distributed_optimizer, process_group)
        )
        self._construction_status = DistributedOptimizerEventStatus.OK
        self._bound_shards: tuple[ModelMainParamShard, ...] = ()

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
                shard.logical_owner and shard.model_param not in self.metric_name_by_parameter
                for shard in shards
            ):
                raise ValueError("missing typed owner binding")
            if any(
                parameter not in shard_parameters
                or descriptor_kinds.get(logical_name) != StatisticKind.UPDATE
                for parameter, logical_name in self.metric_name_by_parameter.items()
            ):
                raise ValueError("invalid typed owner binding")
        except Exception:
            self._record_construction_failure(
                DistributedOptimizerEventStatus.CONSTRUCTOR_BINDING_FAILED
            )
            return
        try:
            self._validate_unique_local_ownership(shards)
        except Exception:
            self._record_construction_failure(
                DistributedOptimizerEventStatus.CONSTRUCTOR_OWNERSHIP_FAILED
            )
            return
        self._bound_shards = shards

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

    def estimate_snapshot_memory(self) -> SnapshotMemoryEstimate:
        """Return exact retained and bounded finish-scratch bytes."""

        owner_elements = sum(shard.main_shard.numel() for shard in self._bound_shards)
        chunk_elements = min(owner_elements, self.finish_chunk_elements)
        fp32_master_bytes = owner_elements * 4
        bf16_applied_bytes = owner_elements * 2
        finish_scratch_bytes = _scratch_layout(chunk_elements).total_bytes
        snapshot_bytes = fp32_master_bytes + bf16_applied_bytes
        return SnapshotMemoryEstimate(
            fp32_master_bytes=fp32_master_bytes,
            bf16_applied_bytes=bf16_applied_bytes,
            finish_scratch_bytes=finish_scratch_bytes,
            snapshot_bytes=snapshot_bytes,
            total_bytes=snapshot_bytes + finish_scratch_bytes,
            owner_elements=owner_elements,
            finish_chunk_elements=chunk_elements,
        )

    def preflight_snapshot_memory(self) -> SnapshotMemoryPreflight:
        """Evaluate the true event peak through finish against HBM limits."""

        estimate = self.estimate_snapshot_memory()
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
        if memory_query_failed:
            reason = SnapshotMemoryReason.ALLOCATION_FAILED
        elif estimate.total_bytes > self.diagnostic_max_extra_bytes:
            reason = SnapshotMemoryReason.MAX_EXTRA_BYTES
        elif memory is not None:
            available_bytes = memory.free_bytes + max(
                0, memory.reserved_bytes - memory.allocated_bytes
            )
            fraction_limit = int(memory.total_bytes * self.max_memory_fraction)
            if memory.allocated_bytes + estimate.total_bytes > fraction_limit:
                reason = SnapshotMemoryReason.DEVICE_MEMORY_FRACTION
            elif estimate.total_bytes > available_bytes:
                reason = SnapshotMemoryReason.DEVICE_HEADROOM
        return SnapshotMemoryPreflight(
            estimate=estimate,
            memory=memory,
            max_extra_bytes=self.diagnostic_max_extra_bytes,
            max_memory_fraction=self.max_memory_fraction,
            available_bytes=available_bytes,
            accepted=reason == SnapshotMemoryReason.NONE,
            reason=reason,
        )

    def begin_event(self) -> SnapshotMemoryMeasurement | None:
        """Capture pre-state or expose a typed local status with no retained partial state."""

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
        if not shards:
            self._set_status(DistributedOptimizerEventStatus.BEGIN_NO_SHARDS)
            return None
        devices = {shard.main_shard.device for shard in shards} | {
            shard.model_shard.device for shard in shards
        }
        if len(devices) != 1:
            self._set_status(DistributedOptimizerEventStatus.BEGIN_DEVICE_MISMATCH)
            return None
        preflight = self.preflight_snapshot_memory()
        if not preflight.accepted:
            self._last_memory_reason = preflight.reason
            self._set_status(DistributedOptimizerEventStatus.BEGIN_PREFLIGHT_REJECTED)
            return None

        device = next(iter(devices))
        allocator_before = self._allocator_bytes(device)
        master_before: torch.Tensor | None = None
        applied_before: torch.Tensor | None = None
        try:
            master_before = torch.empty(
                preflight.estimate.owner_elements, dtype=torch.float32, device=device
            )
        except Exception:
            master_before = None
            self._last_memory_reason = SnapshotMemoryReason.ALLOCATION_FAILED
            self._set_status(DistributedOptimizerEventStatus.BEGIN_FIRST_ALLOCATION_FAILED)
            return None
        try:
            applied_before = torch.empty(
                preflight.estimate.owner_elements, dtype=torch.bfloat16, device=device
            )
        except Exception:
            applied_before = None
            master_before = None
            self._last_memory_reason = SnapshotMemoryReason.ALLOCATION_FAILED
            self._set_status(DistributedOptimizerEventStatus.BEGIN_SECOND_ALLOCATION_FAILED)
            return None

        offsets: list[tuple[int, int]] = []
        try:
            offset = 0
            for shard in shards:
                end = offset + shard.main_shard.numel()
                master_before[offset:end].copy_(shard.main_shard.view(-1))
                applied_before[offset:end].copy_(shard.model_shard.view(-1))
                offsets.append((offset, end))
                offset = end
        except Exception:
            master_before = None
            applied_before = None
            offsets.clear()
            self._set_status(DistributedOptimizerEventStatus.BEGIN_COPY_FAILED)
            return None

        allocator_after = self._allocator_bytes(device)
        allocator_delta = self._allocator_delta(allocator_before, allocator_after)
        measurement = SnapshotMemoryMeasurement(
            estimated_bytes=preflight.estimate.total_bytes,
            payload_bytes=preflight.estimate.snapshot_bytes,
            finish_scratch_bytes=preflight.estimate.finish_scratch_bytes,
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
        )
        return measurement

    def measure_snapshot_memory(self) -> SnapshotMemoryMeasurement | None:
        """Return retained payload and peak measurements for the active event."""

        return None if self._snapshot is None else self._snapshot.measurement

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
        snapshot = self._snapshot
        if not update_successful:
            self.abort_event()
            self._set_status(DistributedOptimizerEventStatus.UPDATE_SKIPPED)
            return None

        try:
            current_shards = tuple(self.optimizer.iter_model_main_param_shards())
        except Exception:
            self.abort_event()
            self._set_status(DistributedOptimizerEventStatus.FINISH_ITERATOR_FAILED)
            return None
        if len(current_shards) != len(snapshot.shards):
            self.abort_event()
            self._set_status(DistributedOptimizerEventStatus.FINISH_SHARD_COUNT_CHANGED)
            return None
        if any(
            self._shard_identity(original) != self._shard_identity(current)
            for original, current in zip(snapshot.shards, current_shards)
        ):
            self.abort_event()
            self._set_status(DistributedOptimizerEventStatus.FINISH_SHARD_IDENTITY_CHANGED)
            return None

        device = snapshot.master_before.device
        try:
            scratch = self._allocate_finish_scratch(device)
        except Exception:
            self.abort_event()
            self._last_memory_reason = SnapshotMemoryReason.ALLOCATION_FAILED
            self._set_status(DistributedOptimizerEventStatus.FINISH_SCRATCH_ALLOCATION_FAILED)
            return None
        measurement = self._measurement_with_peak(snapshot, device)
        snapshot.measurement = measurement

        try:
            self._verify_materialization(current_shards, scratch)
            materialization_valid = (self._status == DistributedOptimizerEventStatus.OK).squeeze(0)
            self._accumulate_update_moments(
                accumulator, snapshot, current_shards, scratch, materialization_valid
            )
            measurement = self._measurement_with_peak(snapshot, device)
        except Exception:
            self._invalidate_bound_metrics(accumulator)
            self._set_status(DistributedOptimizerEventStatus.ACCUMULATION_FAILED)
            measurement = self._measurement_with_peak(snapshot, device)
            self._clear_finish_scratch(scratch)
            self.abort_event()
            return None

        self._clear_finish_scratch(scratch)
        self.abort_event()
        return measurement

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
            finite=_typed_view(storage, layout.finite, torch.bool),
            auxiliary=_typed_view(storage, layout.auxiliary, torch.bool),
        )

    @staticmethod
    def _clear_finish_scratch(scratch: _FinishScratch) -> None:
        empty = torch.empty(0, dtype=torch.uint8, device="cpu")
        scratch.expected_bf16 = empty
        scratch.applied_before_fp32 = empty
        scratch.applied_after_fp32 = empty
        scratch.master_delta_fp32 = empty
        scratch.applied_delta_fp32 = empty
        scratch.work_fp64 = empty
        scratch.finite = empty
        scratch.auxiliary = empty
        scratch.storage = empty

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
                mismatch_status = torch.any(mismatch).to(dtype=self._status.dtype)
                mismatch_status.mul_(int(DistributedOptimizerEventStatus.MATERIALIZATION_MISMATCH))
                torch.maximum(self._status, mismatch_status.view_as(self._status), out=self._status)

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
    ) -> dict[str, torch.Tensor]:
        size = master_after.numel()
        applied_before_fp32 = scratch.applied_before_fp32[:size]
        applied_after_fp32 = scratch.applied_after_fp32[:size]
        master_delta = scratch.master_delta_fp32[:size]
        applied_delta = scratch.applied_delta_fp32[:size]
        work = scratch.work_fp64[:size]
        finite = scratch.finite[:size]
        auxiliary = scratch.auxiliary[:size]

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
        count = finite.sum(dtype=torch.float64)
        nonfinite = torch.full_like(count, size) - count
        torch.logical_not(finite, out=auxiliary)

        work.copy_(master_delta)
        work.masked_fill_(auxiliary, 0)
        work.square_()
        master_sumsq = work.sum()
        work.copy_(applied_delta)
        work.masked_fill_(auxiliary, 0)
        work.square_()
        applied_sumsq = work.sum()
        work.copy_(applied_before_fp32)
        work.masked_fill_(auxiliary, 0)
        work.square_()
        applied_pre_sumsq = work.sum()

        applied_delta.masked_fill_(auxiliary, -torch.inf)
        maximum = applied_delta.amax()
        applied_delta.masked_fill_(auxiliary, torch.inf)
        minimum = applied_delta.amin()

        torch.ne(master_delta, 0, out=auxiliary)
        torch.logical_and(auxiliary, finite, out=auxiliary)
        master_nonzero = auxiliary.sum(dtype=torch.float64)
        torch.eq(applied_delta, 0, out=finite)
        torch.logical_and(finite, auxiliary, out=finite)
        cast_zero = finite.sum(dtype=torch.float64)
        applied_nonzero = master_nonzero - cast_zero
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
    ) -> SnapshotMemoryMeasurement:
        allocator_now = self._allocator_bytes(device)
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
                + snapshot.measurement.finish_scratch_bytes
            ),
            allocator_peak_delta_bytes=allocator_peak,
        )

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
            shard.model_shard.untyped_storage().data_ptr(),
            shard.model_shard.storage_offset(),
            shard.main_shard.numel(),
            shard.main_shard.dtype,
            shard.model_shard.dtype,
        )

    @staticmethod
    def _validate_unique_local_ownership(shards: tuple[ModelMainParamShard, ...]) -> None:
        intervals: dict[tuple[torch.device, int], list[tuple[int, int]]] = {}
        for shard in shards:
            if not shard.logical_owner:
                continue
            storage_identity = (
                shard.main_shard.device,
                shard.main_shard.untyped_storage().data_ptr(),
            )
            start = shard.main_shard.storage_offset()
            end = start + shard.main_shard.numel()
            existing = intervals.setdefault(storage_identity, [])
            if any(start < other_end and other_start < end for other_start, other_end in existing):
                raise ValueError("logical optimizer owner shards overlap")
            existing.append((start, end))
