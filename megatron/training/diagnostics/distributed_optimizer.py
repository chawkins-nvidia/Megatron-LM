# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Authoritative BF16 distributed-optimizer update diagnostics."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
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


class SnapshotMemoryReason(IntEnum):
    """Stable reasons why exact pre-state snapshot allocation is rejected."""

    NONE = 0
    MAX_EXTRA_BYTES = 1
    DEVICE_MEMORY_FRACTION = 2
    DEVICE_HEADROOM = 3
    ALLOCATION_FAILED = 4


@dataclass(frozen=True)
class DistributedOptimizerCapabilityReport:
    """Collectively negotiated support result.

    Attributes:
        supported: Whether every rank supports the same first-backend contract.
        reasons: Sorted global fail-closed reasons.
        local_reasons: Sorted reasons detected on this rank before negotiation.
        world_size: Number of ranks participating in negotiation.
    """

    supported: bool
    reasons: tuple[DistributedOptimizerDiagnosticReason, ...]
    local_reasons: tuple[DistributedOptimizerDiagnosticReason, ...]
    world_size: int


@dataclass(frozen=True)
class SnapshotMemoryEstimate:
    """Exact logical bytes retained by an armed optimizer event."""

    fp32_master_bytes: int
    bf16_applied_bytes: int
    total_bytes: int
    owner_elements: int


@dataclass(frozen=True)
class DeviceMemoryState:
    """Device allocator and driver memory state used by snapshot preflight."""

    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class SnapshotMemoryPreflight:
    """Exact snapshot estimate evaluated against configured HBM limits."""

    estimate: SnapshotMemoryEstimate
    memory: DeviceMemoryState | None
    max_extra_bytes: int
    max_memory_fraction: float
    available_bytes: int | None
    accepted: bool
    reason: SnapshotMemoryReason


@dataclass(frozen=True)
class SnapshotMemoryMeasurement:
    """Measured payload and allocator delta for the active snapshot."""

    estimated_bytes: int
    payload_bytes: int
    allocator_delta_bytes: int | None


class DistributedOptimizerDiagnosticUnsupportedError(RuntimeError):
    """Raised collectively when the optimizer backend is unsupported."""


class SnapshotMemoryError(RuntimeError):
    """Raised when exact snapshot HBM preflight or allocation fails."""

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


MemoryStateProvider = Callable[[torch.device], DeviceMemoryState]


def _unwrap_distributed_optimizer(
    optimizer: object,
) -> tuple[
    MegatronDistributedOptimizer | None,
    tuple[DistributedOptimizerDiagnosticReason, ...],
]:
    if isinstance(optimizer, MegatronDistributedOptimizer):
        return optimizer, ()
    if isinstance(optimizer, ChainedOptimizer):
        if len(optimizer.chained_optimizers) != 1:
            return None, (DistributedOptimizerDiagnosticReason.CHAIN_ARITY,)
        child = optimizer.chained_optimizers[0]
        if isinstance(child, MegatronDistributedOptimizer):
            return child, ()
    return None, (DistributedOptimizerDiagnosticReason.OPTIMIZER_TYPE,)


def _model_parameters(
    optimizer: MegatronDistributedOptimizer,
) -> tuple[torch.nn.Parameter, ...]:
    groups = getattr(optimizer, "model_float16_groups", ())
    return tuple(parameter for group in groups for parameter in group)


def _local_capability_reasons(
    optimizer: object,
) -> tuple[
    MegatronDistributedOptimizer | None,
    tuple[DistributedOptimizerDiagnosticReason, ...],
]:
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
        parameter.dtype != torch.bfloat16
        for group in model_groups
        for parameter in group
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


class Bf16DistributedOptimizerDiagnosticAdapter:
    """Capture actual BF16 applied updates from one authoritative optimizer.

    The adapter snapshots only local DP/CP owner shards, accumulates all moments
    into an existing owner-aware registry, and leaves collective reduction and
    metric derivation to that registry. Parameter-to-family bindings are supplied
    by the typed model registry; this adapter never infers families from names.
    """

    def __init__(
        self,
        optimizer: object,
        registry: MetricRegistry,
        metric_name_by_parameter: Mapping[torch.nn.Parameter, str],
        *,
        diagnostic_max_extra_bytes: int,
        max_memory_fraction: float = 0.9,
        process_group: object | None = None,
        capability_reducer: PackedReducer | None = None,
        capability_world_size: int | None = None,
        memory_state_provider: MemoryStateProvider | None = None,
    ) -> None:
        """Negotiate support and bind model parameters to update descriptors.

        Args:
            optimizer: One real distributed optimizer or a trivial one-child chain.
            registry: Existing owner-aware metric registry.
            metric_name_by_parameter: Typed parameter-to-update-descriptor bindings.
            diagnostic_max_extra_bytes: Hard event snapshot payload limit.
            max_memory_fraction: Maximum post-allocation device-memory fraction.
            process_group: Collective capability-negotiation group, defaulting to world.
            capability_reducer: Optional injected SUM reducer for tests.
            capability_world_size: Required injected world size when a reducer is supplied.
            memory_state_provider: Optional exact device-memory provider for tests.

        Raises:
            DistributedOptimizerDiagnosticUnsupportedError: If any rank is unsupported.
            ValueError: If memory limits or parameter bindings are invalid.
        """

        if diagnostic_max_extra_bytes < 0:
            raise ValueError("diagnostic_max_extra_bytes must be nonnegative")
        if not 0 < max_memory_fraction <= 1:
            raise ValueError("max_memory_fraction must be in (0, 1]")
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
        self.capability_report = report
        self.memory_state_provider = memory_state_provider
        self._snapshot: _SnapshotState | None = None

        shards = tuple(self.optimizer.iter_model_main_param_shards())
        descriptor_kinds = {
            descriptor.logical_name: descriptor.statistic_kind
            for descriptor in registry.descriptors
        }
        missing = [
            shard.model_param
            for shard in shards
            if shard.logical_owner
            and shard.model_param not in self.metric_name_by_parameter
        ]
        if missing:
            raise ValueError(
                "every logical owner shard requires a typed metric binding"
            )
        for parameter, logical_name in self.metric_name_by_parameter.items():
            if parameter not in {shard.model_param for shard in shards}:
                raise ValueError(
                    "metric binding references a parameter outside this optimizer"
                )
            if descriptor_kinds.get(logical_name) != StatisticKind.UPDATE:
                raise ValueError(
                    f"{logical_name} is not a registered update descriptor"
                )
        self._validate_unique_local_ownership(shards)

    @staticmethod
    def negotiate_capabilities(
        optimizer: object,
        *,
        process_group: object | None = None,
        reducer: PackedReducer | None = None,
        world_size: int | None = None,
    ) -> DistributedOptimizerCapabilityReport:
        """Collectively negotiate the narrow first-backend capability contract.

        Args:
            optimizer: Candidate optimizer or trivial chain.
            process_group: Negotiation process group, defaulting to world.
            reducer: Optional injected SUM collective used by tests.
            world_size: Injected collective size; required with ``reducer``.

        Returns:
            A rank-consistent capability report.

        Raises:
            ValueError: If an injected reducer lacks its world size.
        """

        distributed_optimizer, local_reasons = _local_capability_reasons(optimizer)
        if reducer is not None:
            if world_size is None or world_size <= 0:
                raise ValueError(
                    "an injected capability reducer requires a positive world size"
                )
            negotiated_world_size = world_size
        elif dist.is_available() and dist.is_initialized():
            negotiated_world_size = dist.get_world_size(group=process_group)
        else:
            negotiated_world_size = 1

        reason_count = (
            max(reason.value for reason in DistributedOptimizerDiagnosticReason) + 1
        )
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

        counts = flags.tolist()
        global_reasons = [
            reason
            for reason in DistributedOptimizerDiagnosticReason
            if reason != DistributedOptimizerDiagnosticReason.NONE
            and counts[reason.value] > 0
        ]
        if any(0 < count < negotiated_world_size for count in counts):
            global_reasons.append(
                DistributedOptimizerDiagnosticReason.RANK_INCONSISTENT
            )
        reasons = tuple(sorted(set(global_reasons), key=int))
        return DistributedOptimizerCapabilityReport(
            supported=not reasons,
            reasons=reasons,
            local_reasons=local_reasons,
            world_size=negotiated_world_size,
        )

    @property
    def armed(self) -> bool:
        """Return whether pre-step owner snapshots are currently retained."""

        return self._snapshot is not None

    def iter_owner_shards(self) -> tuple[ModelMainParamShard, ...]:
        """Return the current stable owner-shard sequence."""

        return tuple(self.optimizer.iter_model_main_param_shards())

    def estimate_snapshot_memory(self) -> SnapshotMemoryEstimate:
        """Return exact logical bytes required by FP32 and BF16 pre-state buffers."""

        shards = self.iter_owner_shards()
        owner_elements = sum(shard.main_shard.numel() for shard in shards)
        fp32_master_bytes = owner_elements * (torch.finfo(torch.float32).bits // 8)
        bf16_applied_bytes = owner_elements * (torch.finfo(torch.bfloat16).bits // 8)
        return SnapshotMemoryEstimate(
            fp32_master_bytes=fp32_master_bytes,
            bf16_applied_bytes=bf16_applied_bytes,
            total_bytes=fp32_master_bytes + bf16_applied_bytes,
            owner_elements=owner_elements,
        )

    def preflight_snapshot_memory(self) -> SnapshotMemoryPreflight:
        """Evaluate exact snapshot bytes against the extra-byte and HBM ceilings."""

        estimate = self.estimate_snapshot_memory()
        shards = self.iter_owner_shards()
        device = shards[0].main_shard.device if shards else torch.device("cpu")
        memory: DeviceMemoryState | None = None
        if self.memory_state_provider is not None:
            memory = self.memory_state_provider(device)
        elif device.type == "cuda":
            memory = _default_memory_state(device)

        reason = SnapshotMemoryReason.NONE
        available_bytes: int | None = None
        if estimate.total_bytes > self.diagnostic_max_extra_bytes:
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

    def begin_event(self) -> SnapshotMemoryMeasurement:
        """Snapshot only FP32 master and BF16 applied pre-shards after HBM preflight."""

        if self._snapshot is not None:
            raise RuntimeError(
                "a distributed-optimizer diagnostic event is already armed"
            )
        preflight = self.preflight_snapshot_memory()
        if not preflight.accepted:
            raise SnapshotMemoryError(
                preflight.reason,
                f"diagnostic owner snapshot rejected: {preflight.reason.name}",
            )

        shards = self.iter_owner_shards()
        if not shards:
            raise RuntimeError("the distributed optimizer has no BF16 owner shards")
        devices = {shard.main_shard.device for shard in shards} | {
            shard.model_shard.device for shard in shards
        }
        if len(devices) != 1:
            raise RuntimeError("diagnostic owner shards must reside on one device")
        device = next(iter(devices))
        allocator_before = (
            torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        )
        owner_elements = preflight.estimate.owner_elements
        try:
            master_before = torch.empty(
                owner_elements, dtype=torch.float32, device=device
            )
            applied_before = torch.empty(
                owner_elements, dtype=torch.bfloat16, device=device
            )
            offsets: list[tuple[int, int]] = []
            offset = 0
            for shard in shards:
                end = offset + shard.main_shard.numel()
                master_before[offset:end].copy_(shard.main_shard.view(-1))
                applied_before[offset:end].copy_(shard.model_shard.view(-1))
                offsets.append((offset, end))
                offset = end
        except (RuntimeError, torch.OutOfMemoryError) as error:
            raise SnapshotMemoryError(
                SnapshotMemoryReason.ALLOCATION_FAILED,
                "diagnostic owner snapshot allocation failed",
            ) from error
        allocator_after = (
            torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        )
        payload_bytes = master_before.nbytes + applied_before.nbytes
        measurement = SnapshotMemoryMeasurement(
            estimated_bytes=preflight.estimate.total_bytes,
            payload_bytes=payload_bytes,
            allocator_delta_bytes=(
                None
                if allocator_before is None or allocator_after is None
                else allocator_after - allocator_before
            ),
        )
        self._snapshot = _SnapshotState(
            shards=shards,
            offsets=tuple(offsets),
            master_before=master_before,
            applied_before=applied_before,
            measurement=measurement,
        )
        return measurement

    def measure_snapshot_memory(self) -> SnapshotMemoryMeasurement | None:
        """Return exact payload and allocator measurements for the active event."""

        return None if self._snapshot is None else self._snapshot.measurement

    def finish_event(
        self,
        accumulator: PackedSufficientStatistics,
        *,
        update_successful: bool = True,
    ) -> SnapshotMemoryMeasurement | None:
        """Accumulate post-materialization update moments or discard a skipped update.

        Args:
            accumulator: Existing registry-bound Tier-0 accumulator.
            update_successful: Whether the synchronous optimizer update succeeded.

        Returns:
            Snapshot memory measurement, or ``None`` for a skipped update.

        Raises:
            RuntimeError: If no event is armed or owner identities changed.
        """

        if self._snapshot is None:
            raise RuntimeError("no distributed-optimizer diagnostic event is armed")
        if not update_successful:
            self.abort_event()
            return None

        snapshot = self._snapshot
        current_shards = self.iter_owner_shards()
        if len(current_shards) != len(snapshot.shards):
            self.abort_event()
            raise RuntimeError(
                "distributed-optimizer owner shard count changed during the event"
            )
        try:
            for original, current, (start, end) in zip(
                snapshot.shards, current_shards, snapshot.offsets
            ):
                if self._shard_identity(original) != self._shard_identity(current):
                    raise RuntimeError(
                        "distributed-optimizer owner shard identity changed during the event"
                    )
                if not current.logical_owner:
                    continue
                logical_name = self.metric_name_by_parameter[current.model_param]
                self.registry.add_applied_update(
                    accumulator,
                    logical_name,
                    snapshot.master_before[start:end],
                    current.main_shard.view(-1),
                    snapshot.applied_before[start:end],
                    current.model_shard.view(-1),
                )
        except Exception:
            self.abort_event()
            raise
        measurement = snapshot.measurement
        self.abort_event()
        return measurement

    def abort_event(self) -> None:
        """Release all retained pre-state without accumulating or emitting metrics."""

        self._snapshot = None

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
            shard.bucket_range,
            shard.main_shard.numel(),
            shard.main_shard.dtype,
            shard.model_shard.dtype,
        )

    @staticmethod
    def _validate_unique_local_ownership(
        shards: tuple[ModelMainParamShard, ...],
    ) -> None:
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
            if any(
                start < other_end and other_start < end
                for other_start, other_end in existing
            ):
                raise ValueError("logical optimizer owner shards overlap")
            existing.append((start, end))
