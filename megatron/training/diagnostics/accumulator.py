# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Device-resident packed sufficient statistics for scalable diagnostics."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Iterator, Protocol

import torch
import torch.distributed as dist

from .schema import SCHEMA_PREFIX, Tier0Reason

_SUM_FIELDS = (
    "sum",
    "count",
    "sumsq",
    "dot",
    "lhs_sumsq",
    "rhs_sumsq",
    "zero",
    "nonfinite",
    "mask_error",
    "nonfinite_arithmetic",
    "observation_error",
)
_SUM_FIELD_COUNT = len(_SUM_FIELDS)
DEFAULT_MOMENT_SCRATCH_ELEMENT_CAPACITY = 1024 * 1024
"""Default fixed moment-workspace capacity (1,048,576 logical elements)."""

_MAX_SCRATCH_BYTES_PER_ELEMENT = 96


@dataclass(frozen=True)
class _MomentWorkspace:
    """Typed views over the shared startup-reserved observation workspace."""

    fp32: torch.Tensor
    fp64: torch.Tensor
    boolean: torch.Tensor
    fp64_scalars: torch.Tensor
    fp32_scalars: torch.Tensor
    bool_scalars: torch.Tensor


class PackedReducer(Protocol):
    """Define an injectable packed-collective callable."""

    def __call__(
        self, tensor: torch.Tensor, *, op: object, group: object | None
    ) -> object:
        """Reduce one packed tensor.

        Args:
            tensor: Packed tensor to reduce in place.
            op: Distributed reduction operation.
            group: Process group that owns the collective.

        Returns:
            Backend-specific collective result.
        """

        ...


class ProcessGroupIdentity(StrEnum):
    """Known logical process-group identities for diagnostic reduction."""

    WORLD = "world"
    DATA_PARALLEL = "data_parallel"


class ReductionKind(StrEnum):
    """Fixed collective operations used by an accumulator."""

    PACKED_SUM_MAX_MIN = "packed_sum_max_min"
    HIERARCHICAL_PACKED_SUM_MAX_MIN = "hierarchical_packed_sum_max_min"


@dataclass(frozen=True)
class ReductionBinding:
    """Bind a declared reduction contract to its runtime collective.

    Attributes:
        process_group_identity: Logical identity of the bound process group.
        reduction_kind: Packed collective algorithm implemented by the binding.
        group: Runtime process group, or ``None`` for the default world group.
        reducer: Optional injected collective callable used by tests.
    """

    process_group_identity: ProcessGroupIdentity
    reduction_kind: ReductionKind
    group: object | None
    reducer: PackedReducer | None = None

    def __post_init__(self) -> None:
        """Require typed identities before this binding reaches an accumulator.

        Raises:
            ValueError: If either identity is not its declared enum type.
        """

        if not isinstance(self.process_group_identity, ProcessGroupIdentity):
            raise ValueError(
                "reduction bindings require a typed process-group identity"
            )
        if not isinstance(self.reduction_kind, ReductionKind):
            raise ValueError("reduction bindings require a typed reduction kind")

    @classmethod
    def flat_world(
        cls, group: object | None, *, reducer: PackedReducer | None = None
    ) -> "ReductionBinding":
        """Create the supported flat world SUM/MAX/MIN binding.

        Args:
            group: Runtime world process group, or ``None`` for the default world group.
            reducer: Optional injected collective callable used by tests.

        Returns:
            A typed binding for the flat packed world reduction.
        """

        return cls(
            process_group_identity=ProcessGroupIdentity.WORLD,
            reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
            group=group,
            reducer=reducer,
        )


@dataclass(frozen=True)
class PackedSlots:
    """Store offsets occupied by one metric in the fixed packed buffers.

    Attributes:
        sum: Offset for the weighted sum.
        count: Offset for the weighted finite-element count.
        sumsq: Offset for the weighted square sum.
        dot: Offset for the weighted pair dot product.
        lhs_sumsq: Offset for the weighted left-hand square sum.
        rhs_sumsq: Offset for the weighted right-hand square sum.
        zero: Offset for the weighted zero count.
        nonfinite: Offset for the weighted nonfinite-input count.
        mask_error: Offset for device-resident mask contract failures.
        nonfinite_arithmetic: Offset for device-resident arithmetic failures.
        observation_error: Offset for missing or duplicate semantic observations.
        maximum: Offset in the FP32 MAX pack.
        minimum: Offset in the FP32 MIN pack.
    """

    sum: int
    count: int
    sumsq: int
    dot: int
    lhs_sumsq: int
    rhs_sumsq: int
    zero: int
    nonfinite: int
    mask_error: int
    nonfinite_arithmetic: int
    observation_error: int
    maximum: int
    minimum: int

    @classmethod
    def for_index(cls, index: int) -> "PackedSlots":
        """Return canonical offsets for a metric index.

        Args:
            index: Zero-based metric index in registry order.

        Returns:
            Canonical SUM, MAX, and MIN offsets.

        Raises:
            ValueError: If ``index`` is negative.
        """

        if index < 0:
            raise ValueError("packed statistic indices must be nonnegative")
        base = index * _SUM_FIELD_COUNT
        return cls(
            sum=base,
            count=base + 1,
            sumsq=base + 2,
            dot=base + 3,
            lhs_sumsq=base + 4,
            rhs_sumsq=base + 5,
            zero=base + 6,
            nonfinite=base + 7,
            mask_error=base + 8,
            nonfinite_arithmetic=base + 9,
            observation_error=base + 10,
            maximum=index,
            minimum=index,
        )


@dataclass(frozen=True)
class DerivedStatistic:
    """Store a derived scalar and its device-resident validity state.

    Attributes:
        value: Derived scalar, or NaN when invalid.
        valid: Boolean scalar indicating whether ``value`` is usable.
        reason: Stable :class:`Tier0Reason` integer scalar.
    """

    value: torch.Tensor
    valid: torch.Tensor
    reason: torch.Tensor


def _torch_all_reduce(
    tensor: torch.Tensor, *, op: object, group: object | None
) -> object:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized, or a reducer must be injected"
        )
    return dist.all_reduce(tensor, op=op, group=group)


class PackedSufficientStatistics:
    """Accumulate and reduce fixed-slot sufficient statistics on one device.

    Inputs are rounded to FP32 before numerical accumulation. Products and
    squares are then evaluated in FP64 so finite FP32 inputs cannot overflow
    FP32 arithmetic. Additive statistics use FP64 slots, extrema use FP32
    MAX/MIN packs, and mask/arithmetic failures remain device resident. Values
    are derived only after :meth:`reduce_` or :meth:`finalize_local_`.
    """

    def __init__(
        self,
        slot_names: tuple[str, ...],
        device: torch.device | str,
        *,
        descriptor_hash: str,
        reduction_binding: ReductionBinding,
        schema_identity: str = SCHEMA_PREFIX.rstrip("/"),
        scratch_element_capacity: int = DEFAULT_MOMENT_SCRATCH_ELEMENT_CAPACITY,
    ) -> None:
        """Allocate neutral packed buffers in a stable slot order.

        Args:
            slot_names: Unique logical metric names in collective slot order.
            device: CPU or CUDA device used for accumulation and reduction.
            descriptor_hash: Rank-independent descriptor hash.
            reduction_binding: Typed runtime binding for the declared collective.
            schema_identity: Versioned diagnostic schema identity.
            scratch_element_capacity: Maximum number of logical elements processed
                by any tensor-sized moment temporary at once.

        Raises:
            ValueError: If slot names, identities, or the reduction binding are invalid.
        """

        if len(slot_names) != len(set(slot_names)):
            raise ValueError("packed statistic slot names must be unique")
        if not descriptor_hash or not schema_identity:
            raise ValueError("packed statistic identities must be nonempty")
        if reduction_binding is None:
            raise ValueError("packed statistics require a reduction binding")
        if scratch_element_capacity <= 0:
            raise ValueError("scratch element capacity must be positive")
        if (
            reduction_binding.process_group_identity != ProcessGroupIdentity.WORLD
            or reduction_binding.reduction_kind != ReductionKind.PACKED_SUM_MAX_MIN
        ):
            raise ValueError(
                "only the flat packed world SUM/MAX/MIN reduction is implemented"
            )
        self.slot_names = slot_names
        self.descriptor_hash = descriptor_hash
        self.schema_identity = schema_identity
        self.reduction_binding = reduction_binding
        self.scratch_element_capacity = scratch_element_capacity
        self._slot_indices = {name: index for index, name in enumerate(slot_names)}
        self.sum_pack = torch.zeros(
            len(slot_names) * _SUM_FIELD_COUNT, dtype=torch.float64, device=device
        )
        self.max_pack = torch.full(
            (len(slot_names),), -torch.inf, dtype=torch.float32, device=device
        )
        self.min_pack = torch.full(
            (len(slot_names),), torch.inf, dtype=torch.float32, device=device
        )
        self._peak_scratch_bytes = 0
        self._reduced = False
        self._workspace: _MomentWorkspace | None = None

    def bind_workspace(self, storage: torch.Tensor) -> None:
        """Bind startup-preallocated byte storage as typed reusable scratch views."""

        if storage.dtype != torch.uint8 or storage.device != self.sum_pack.device:
            raise ValueError(
                "packed-statistics workspace has the wrong device or dtype"
            )
        required = self.maximum_scratch_bytes
        if storage.numel() < required:
            raise ValueError(
                "packed-statistics workspace is smaller than its declared bound"
            )
        capacity = self.scratch_element_capacity
        offset = 0

        def take(byte_count: int, dtype: torch.dtype) -> torch.Tensor:
            nonlocal offset
            alignment = {torch.float64: 8, torch.float32: 4, torch.bool: 1}[dtype]
            offset = (offset + alignment - 1) // alignment * alignment
            result = storage[offset : offset + byte_count].view(dtype)
            offset += byte_count
            return result

        fp32 = take(5 * capacity * 4, torch.float32).view(5, capacity)
        fp64 = take(3 * capacity * 8, torch.float64).view(3, capacity)
        boolean = take(5 * capacity, torch.bool).view(5, capacity)
        fp64_scalars = take(16 * 8, torch.float64)
        fp32_scalars = take(4 * 4, torch.float32)
        bool_scalars = take(8, torch.bool)
        if offset > required:
            raise AssertionError(
                "packed-statistics workspace layout exceeds its reservation"
            )
        self._workspace = _MomentWorkspace(
            fp32, fp64, boolean, fp64_scalars, fp32_scalars, bool_scalars
        )

    @property
    def maximum_scratch_bytes(self) -> int:
        """Return the exact configured ceiling for tensor-moment scratch.

        The bound includes every simultaneously live tensor-sized temporary in
        the pair/update path, which is the largest supported observation. Scalar
        pack updates are persistent accumulator storage and are not scratch.
        """

        return self.scratch_element_capacity * _MAX_SCRATCH_BYTES_PER_ELEMENT

    @property
    def peak_scratch_bytes(self) -> int:
        """Return the largest tensor-moment scratch use observed so far."""

        return self._peak_scratch_bytes

    @staticmethod
    def scratch_bytes_for_capacity(
        scratch_element_capacity: int = DEFAULT_MOMENT_SCRATCH_ELEMENT_CAPACITY,
    ) -> int:
        """Compute the exact HBM-preflight scratch ceiling without allocating it.

        Args:
            scratch_element_capacity: Maximum logical elements per scratch chunk.

        Returns:
            Maximum tensor-sized transient bytes for one observation.

        Raises:
            ValueError: If the capacity is not positive.
        """

        if scratch_element_capacity <= 0:
            raise ValueError("scratch element capacity must be positive")
        return scratch_element_capacity * _MAX_SCRATCH_BYTES_PER_ELEMENT

    @property
    def reduced(self) -> bool:
        """Return whether the packs completed their reduction phase."""

        return self._reduced

    def reset_(self) -> "PackedSufficientStatistics":
        """Reset preallocated event packs to their neutral values."""

        self.sum_pack.zero_()
        self.max_pack.fill_(-torch.inf)
        self.min_pack.fill_(torch.inf)
        self._peak_scratch_bytes = 0
        self._reduced = False
        return self

    @property
    def process_group_identity(self) -> ProcessGroupIdentity:
        """Return the process-group identity carried by the runtime binding."""

        return self.reduction_binding.process_group_identity

    @property
    def reduction_kind(self) -> ReductionKind:
        """Return the packed reduction kind carried by the runtime binding."""

        return self.reduction_binding.reduction_kind

    def slots(self, slot: str | int) -> PackedSlots:
        """Resolve a logical name or integer slot to packed offsets.

        Args:
            slot: Registered logical name or zero-based slot index.

        Returns:
            Canonical offsets for the requested metric.

        Raises:
            KeyError: If a logical name is unknown.
            IndexError: If an integer slot is out of range.
        """

        index = self._resolve_index(slot)
        return PackedSlots.for_index(index)

    def invalidate(self, slot: str | int) -> None:
        """Mark one statistic invalid while preserving fixed reduction participation.

        Args:
            slot: Registered logical name or zero-based slot index.

        Raises:
            RuntimeError: If accumulation already completed.
            KeyError: If a logical name is unknown.
            IndexError: If an integer slot is out of range.
        """

        self._ensure_accumulating()
        self.sum_pack[self.slots(slot).mask_error].add_(1)

    def add_masked_tensor(
        self,
        slot: str | int,
        values: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        replication_multiplicity: int = 1,
        require_mask: bool = False,
    ) -> None:
        """Add masked tensor moments without deriving a local metric.

        Invalid masks contribute neutral statistics and set the packed mask
        error field. No tensor-to-host synchronization or rank-local exception
        is used for device, shape, finite-value, or nonnegative-value failures.

        Args:
            slot: Registered logical name or zero-based slot index.
            values: Tensor observation, rounded to FP32 before accumulation.
            mask: Optional same-device broadcastable nonnegative finite weights.
            replication_multiplicity: Number of identical logical replicas.
            require_mask: Whether a missing mask is a packed mask-contract failure.

        Raises:
            RuntimeError: If accumulation already completed.
            ValueError: If replication multiplicity is not positive.
            KeyError: If a logical name is unknown.
            IndexError: If an integer slot is out of range.
        """

        self._ensure_accumulating()
        self._validate_multiplicity(replication_multiplicity)
        slots = self.slots(slot)
        detached = values.detach()
        if self._workspace is None:
            self._add_masked_tensor_legacy(
                slots,
                detached,
                mask,
                replication_multiplicity=replication_multiplicity,
                require_mask=require_mask,
            )
            return
        weights, mask_valid = self._validated_broadcast_mask(
            detached, mask, slots, require_mask=require_mask
        )
        if detached.numel() == 0:
            return
        workspace = self._workspace
        scale = replication_multiplicity
        contributions = workspace.fp64_scalars[:5]
        contributions.zero_()
        candidate_max = workspace.fp32_scalars[0]
        candidate_min = workspace.fp32_scalars[1]
        candidate_max.fill_(-torch.inf)
        candidate_min.fill_(torch.inf)
        tensors = (detached,) if weights is None else (detached, weights)
        for chunks in self._bounded_chunks(*tensors):
            count = chunks[0].numel()
            values_fp32, chunk_weights, clean_fp32, extrema = (
                workspace.fp32[index, :count] for index in range(4)
            )
            selected, finite, finite_selected, auxiliary = (
                workspace.boolean[index, :count] for index in range(4)
            )
            clean, finite_weights, work = (
                workspace.fp64[index, :count] for index in range(3)
            )
            values_fp32.copy_(chunks[0])
            if weights is None:
                chunk_weights.fill_(1)
            else:
                chunk_weights.copy_(chunks[1])
            self._isfinite_out(chunk_weights, selected, auxiliary)
            torch.ge(chunk_weights, 0, out=auxiliary)
            torch.logical_and(selected, auxiliary, out=selected)
            torch.logical_not(selected, out=auxiliary)
            chunk_weights.masked_fill_(auxiliary, 0)
            torch.ne(chunk_weights, 0, out=selected)
            self._isfinite_out(values_fp32, finite, auxiliary)
            torch.logical_and(selected, finite, out=finite_selected)
            clean_fp32.copy_(values_fp32)
            torch.logical_not(finite, out=auxiliary)
            clean_fp32.masked_fill_(auxiliary, 0)
            clean.copy_(clean_fp32)
            clean_fp32.copy_(chunk_weights)
            clean_fp32.masked_fill_(auxiliary, 0)
            finite_weights.copy_(clean_fp32)

            torch.mul(clean, finite_weights, out=work)
            torch.sum(work, dim=(0,), out=workspace.fp64_scalars[8])
            contributions[0].add_(workspace.fp64_scalars[8], alpha=1.0 / scale)
            torch.sum(finite_weights, dim=(0,), out=workspace.fp64_scalars[8])
            contributions[1].add_(workspace.fp64_scalars[8], alpha=1.0 / scale)
            torch.square(clean, out=work)
            work.mul_(finite_weights)
            torch.sum(work, dim=(0,), out=workspace.fp64_scalars[8])
            contributions[2].add_(workspace.fp64_scalars[8], alpha=1.0 / scale)

            torch.eq(values_fp32, 0, out=auxiliary)
            torch.logical_and(finite_selected, auxiliary, out=auxiliary)
            clean_fp32.copy_(chunk_weights)
            torch.logical_not(auxiliary, out=workspace.boolean[4, :count])
            clean_fp32.masked_fill_(workspace.boolean[4, :count], 0)
            torch.sum(
                clean_fp32,
                dim=(0,),
                dtype=torch.float64,
                out=workspace.fp64_scalars[8],
            )
            contributions[3].add_(workspace.fp64_scalars[8], alpha=1.0 / scale)
            torch.logical_not(finite, out=auxiliary)
            torch.logical_and(selected, auxiliary, out=auxiliary)
            clean_fp32.copy_(chunk_weights)
            torch.logical_not(auxiliary, out=workspace.boolean[4, :count])
            clean_fp32.masked_fill_(workspace.boolean[4, :count], 0)
            torch.sum(
                clean_fp32,
                dim=(0,),
                dtype=torch.float64,
                out=workspace.fp64_scalars[8],
            )
            contributions[4].add_(workspace.fp64_scalars[8], alpha=1.0 / scale)

            extrema.copy_(values_fp32)
            torch.logical_not(finite_selected, out=auxiliary)
            extrema.masked_fill_(auxiliary, -torch.inf)
            torch.amax(extrema, out=workspace.fp32_scalars[2])
            torch.maximum(candidate_max, workspace.fp32_scalars[2], out=candidate_max)
            extrema.copy_(values_fp32)
            extrema.masked_fill_(auxiliary, torch.inf)
            torch.amin(extrema, out=workspace.fp32_scalars[2])
            torch.minimum(candidate_min, workspace.fp32_scalars[2], out=candidate_min)
        contributions.mul_(mask_valid)
        weighted_sum, count, weighted_sumsq, zero, nonfinite = contributions.unbind()
        arithmetic_finite = workspace.bool_scalars[1]
        arithmetic_finite.fill_(True)
        for contribution in contributions:
            self._isfinite_out(
                contribution, workspace.bool_scalars[2], workspace.bool_scalars[3]
            )
            arithmetic_finite.logical_and_(workspace.bool_scalars[2])
        torch.logical_not(arithmetic_finite, out=workspace.bool_scalars[2])
        self.sum_pack[slots.nonfinite_arithmetic].add_(workspace.bool_scalars[2])
        for contribution in contributions:
            contribution.mul_(arithmetic_finite)

        self.sum_pack[slots.sum].add_(weighted_sum)
        self.sum_pack[slots.count].add_(count)
        self.sum_pack[slots.sumsq].add_(weighted_sumsq)
        self.sum_pack[slots.lhs_sumsq].add_(weighted_sumsq)
        self.sum_pack[slots.zero].add_(zero)
        self.sum_pack[slots.nonfinite].add_(nonfinite)

        torch.logical_not(mask_valid, out=workspace.bool_scalars[2])
        candidate_max.masked_fill_(workspace.bool_scalars[2], -torch.inf)
        candidate_min.masked_fill_(workspace.bool_scalars[2], torch.inf)
        torch.maximum(
            self.max_pack[slots.maximum],
            candidate_max,
            out=self.max_pack[slots.maximum],
        )
        torch.minimum(
            self.min_pack[slots.minimum],
            candidate_min,
            out=self.min_pack[slots.minimum],
        )

    def _add_masked_tensor_legacy(
        self,
        slots: PackedSlots,
        detached: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        replication_multiplicity: int,
        require_mask: bool,
    ) -> None:
        """Retain the general CPU/test path when no production workspace is bound."""

        weights, mask_valid = self._validated_broadcast_mask(
            detached, mask, slots, require_mask=require_mask
        )
        if detached.numel() == 0:
            return
        scale = replication_multiplicity
        contributions = torch.zeros(5, dtype=torch.float64, device=detached.device)
        candidate_max = torch.full(
            (), -torch.inf, dtype=torch.float32, device=detached.device
        )
        candidate_min = torch.full(
            (), torch.inf, dtype=torch.float32, device=detached.device
        )
        tensors = (detached,) if weights is None else (detached, weights)
        for chunks in self._bounded_chunks(*tensors):
            values_fp32 = chunks[0].to(dtype=torch.float32)
            chunk_weights = (
                torch.ones_like(values_fp32) if weights is None else chunks[1].float()
            )
            valid_weights = torch.isfinite(chunk_weights) & (chunk_weights >= 0)
            chunk_weights = torch.where(
                valid_weights, chunk_weights, torch.zeros_like(chunk_weights)
            )
            selected = chunk_weights != 0
            finite = torch.isfinite(values_fp32)
            finite_selected = selected & finite
            clean_fp32 = torch.where(finite, values_fp32, torch.zeros_like(values_fp32))
            finite_weights_fp32 = torch.where(
                finite, chunk_weights, torch.zeros_like(chunk_weights)
            )
            clean = clean_fp32.double()
            finite_weights = finite_weights_fp32.double()
            contributions[0].add_((clean * finite_weights).sum() / scale)
            contributions[1].add_(finite_weights.sum() / scale)
            contributions[2].add_((clean.square() * finite_weights).sum() / scale)
            contributions[3].add_(
                torch.where(finite_selected & (values_fp32 == 0), chunk_weights, 0).sum(
                    dtype=torch.float64
                )
                / scale
            )
            contributions[4].add_(
                torch.where(selected & ~finite, chunk_weights, 0).sum(
                    dtype=torch.float64
                )
                / scale
            )
            candidate_max = torch.maximum(
                candidate_max,
                torch.where(finite_selected, values_fp32, -torch.inf).amax(),
            )
            candidate_min = torch.minimum(
                candidate_min,
                torch.where(finite_selected, values_fp32, torch.inf).amin(),
            )
        contributions.mul_(mask_valid.to(dtype=contributions.dtype))
        weighted_sum, count, weighted_sumsq, zero, nonfinite = (
            self._finite_contributions(slots, *contributions.unbind())
        )
        self.sum_pack[slots.sum].add_(weighted_sum)
        self.sum_pack[slots.count].add_(count)
        self.sum_pack[slots.sumsq].add_(weighted_sumsq)
        self.sum_pack[slots.lhs_sumsq].add_(weighted_sumsq)
        self.sum_pack[slots.zero].add_(zero)
        self.sum_pack[slots.nonfinite].add_(nonfinite)
        candidate_max = torch.where(mask_valid, candidate_max, -torch.inf)
        candidate_min = torch.where(mask_valid, candidate_min, torch.inf)
        torch.maximum(
            self.max_pack[slots.maximum],
            candidate_max,
            out=self.max_pack[slots.maximum],
        )
        torch.minimum(
            self.min_pack[slots.minimum],
            candidate_min,
            out=self.min_pack[slots.minimum],
        )

    def add_masked_pair(
        self,
        slot: str | int,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        replication_multiplicity: int = 1,
        require_mask: bool = False,
    ) -> None:
        """Add pooled dot, cosine, and relative-norm moments for a pair.

        Args:
            slot: Registered logical name or zero-based slot index.
            lhs: Numerator-side tensor observation.
            rhs: Denominator-side tensor observation.
            mask: Optional same-device broadcastable nonnegative finite weights.
            replication_multiplicity: Number of identical logical replicas.
            require_mask: Whether a missing mask is a packed mask-contract failure.

        Raises:
            RuntimeError: If accumulation already completed.
            ValueError: If replication multiplicity is not positive.
            KeyError: If a logical name is unknown.
            IndexError: If an integer slot is out of range.
        """

        self._add_masked_pair(
            slot,
            lhs,
            rhs,
            mask=mask,
            replication_multiplicity=replication_multiplicity,
            require_mask=require_mask,
            difference_lhs=False,
        )

    def _add_masked_pair(
        self,
        slot: str | int,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        replication_multiplicity: int,
        require_mask: bool,
        difference_lhs: bool,
    ) -> None:
        self._ensure_accumulating()
        self._validate_multiplicity(replication_multiplicity)
        slots = self.slots(slot)
        lhs_values, rhs_values = torch.broadcast_tensors(lhs.detach(), rhs.detach())
        weights, mask_valid = self._validated_broadcast_mask(
            lhs_values, mask, slots, require_mask=require_mask
        )
        if lhs_values.numel() == 0:
            return
        scale = replication_multiplicity
        contributions = torch.zeros(7, dtype=torch.float64, device=lhs_values.device)
        candidate_max = torch.full(
            (), -torch.inf, dtype=torch.float32, device=lhs_values.device
        )
        candidate_min = torch.full(
            (), torch.inf, dtype=torch.float32, device=lhs_values.device
        )
        tensors = (
            (lhs_values, rhs_values)
            if weights is None
            else (lhs_values, rhs_values, weights)
        )
        for chunks in self._bounded_chunks(*tensors):
            lhs_fp32 = chunks[0].to(dtype=torch.float32)
            rhs_fp32 = chunks[1].to(dtype=torch.float32)
            if difference_lhs:
                lhs_fp32 = lhs_fp32 - rhs_fp32
            chunk_weights = (
                torch.ones_like(lhs_fp32)
                if weights is None
                else chunks[2].to(dtype=torch.float32)
            )
            valid_weights = torch.isfinite(chunk_weights) & (chunk_weights >= 0)
            chunk_weights = torch.where(
                valid_weights, chunk_weights, torch.zeros_like(chunk_weights)
            )
            selected = chunk_weights != 0
            finite = torch.isfinite(lhs_fp32) & torch.isfinite(rhs_fp32)
            finite_selected = selected & finite
            clean_lhs_fp32 = torch.where(finite, lhs_fp32, torch.zeros_like(lhs_fp32))
            clean_rhs_fp32 = torch.where(finite, rhs_fp32, torch.zeros_like(rhs_fp32))
            finite_weights_fp32 = torch.where(
                finite, chunk_weights, torch.zeros_like(chunk_weights)
            )
            clean_lhs = clean_lhs_fp32.to(dtype=torch.float64)
            clean_rhs = clean_rhs_fp32.to(dtype=torch.float64)
            finite_weights = finite_weights_fp32.to(dtype=torch.float64)
            contributions[0].add_((clean_lhs * finite_weights).sum() / scale)
            contributions[1].add_(finite_weights.sum() / scale)
            contributions[2].add_((clean_lhs.square() * finite_weights).sum() / scale)
            contributions[3].add_((clean_rhs.square() * finite_weights).sum() / scale)
            contributions[4].add_(
                (clean_lhs * clean_rhs * finite_weights).sum() / scale
            )
            contributions[5].add_(
                torch.where(
                    finite_selected & (lhs_fp32 == 0),
                    chunk_weights,
                    torch.zeros_like(chunk_weights),
                ).sum(dtype=torch.float64)
                / scale
            )
            contributions[6].add_(
                torch.where(
                    selected & ~finite, chunk_weights, torch.zeros_like(chunk_weights)
                ).sum(dtype=torch.float64)
                / scale
            )
            candidate_max = torch.maximum(
                candidate_max,
                torch.where(
                    finite_selected, lhs_fp32, torch.full_like(lhs_fp32, -torch.inf)
                ).amax(),
            )
            candidate_min = torch.minimum(
                candidate_min,
                torch.where(
                    finite_selected, lhs_fp32, torch.full_like(lhs_fp32, torch.inf)
                ).amin(),
            )
        contributions.mul_(mask_valid.to(dtype=contributions.dtype))
        weighted_sum, count, lhs_sumsq, rhs_sumsq, dot, zero, nonfinite = (
            contributions.unbind()
        )
        (weighted_sum, count, lhs_sumsq, rhs_sumsq, dot, zero, nonfinite) = (
            self._finite_contributions(
                slots, weighted_sum, count, lhs_sumsq, rhs_sumsq, dot, zero, nonfinite
            )
        )

        self.sum_pack[slots.sum].add_(weighted_sum)
        self.sum_pack[slots.count].add_(count)
        self.sum_pack[slots.sumsq].add_(lhs_sumsq)
        self.sum_pack[slots.dot].add_(dot)
        self.sum_pack[slots.lhs_sumsq].add_(lhs_sumsq)
        self.sum_pack[slots.rhs_sumsq].add_(rhs_sumsq)
        self.sum_pack[slots.zero].add_(zero)
        self.sum_pack[slots.nonfinite].add_(nonfinite)

        candidate_max = torch.where(
            mask_valid, candidate_max, torch.full_like(candidate_max, -torch.inf)
        )
        candidate_min = torch.where(
            mask_valid, candidate_min, torch.full_like(candidate_min, torch.inf)
        )
        self.max_pack[slots.maximum] = torch.maximum(
            self.max_pack[slots.maximum], candidate_max
        )
        self.min_pack[slots.minimum] = torch.minimum(
            self.min_pack[slots.minimum], candidate_min
        )

    def add_update(
        self,
        slot: str | int,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        replication_multiplicity: int = 1,
        require_mask: bool = False,
    ) -> None:
        """Add update-delta moments with the pre-update tensor as denominator.

        Args:
            slot: Registered logical name or zero-based slot index.
            before: Pre-update tensor.
            after: Post-update tensor.
            mask: Optional same-device broadcastable nonnegative finite weights.
            replication_multiplicity: Number of identical logical replicas.
            require_mask: Whether a missing mask is a packed mask-contract failure.

        Raises:
            RuntimeError: If accumulation already completed.
            ValueError: If replication multiplicity is not positive.
        """

        self._add_masked_pair(
            slot,
            after,
            before,
            mask=mask,
            replication_multiplicity=replication_multiplicity,
            require_mask=require_mask,
            difference_lhs=True,
        )

    def mark_mask_error(
        self, slot: str | int, error: torch.Tensor | None = None
    ) -> None:
        """Record a device-resident mask contract failure for one slot.

        Args:
            slot: Registered logical name or zero-based slot index.
            error: Optional boolean or numeric scalar. A missing, malformed, or
                wrong-device scalar records an error without moving data.

        Raises:
            RuntimeError: If accumulation already completed.
        """

        self._ensure_accumulating()
        slots = self.slots(slot)
        if error is None or error.numel() != 1 or error.device != self.sum_pack.device:
            self.sum_pack[slots.mask_error].add_(1)
            return
        self.sum_pack[slots.mask_error].add_(error.detach().reshape(()))

    def mark_arithmetic_error(
        self, slot: str | int, error: torch.Tensor | None = None
    ) -> None:
        """Record a device-resident arithmetic contract failure for one slot.

        Args:
            slot: Registered logical name or zero-based slot index.
            error: Optional boolean or numeric scalar. A missing, malformed, or
                wrong-device scalar records an error without moving data.

        Raises:
            RuntimeError: If accumulation already completed.
        """

        self._ensure_accumulating()
        slots = self.slots(slot)
        if error is None or error.numel() != 1 or error.device != self.sum_pack.device:
            self.sum_pack[slots.nonfinite_arithmetic].add_(1)
            return
        self.sum_pack[slots.nonfinite_arithmetic].add_(error.detach().reshape(()))

    def mark_observation_error(
        self, slot: str | int, error: torch.Tensor | None = None
    ) -> None:
        """Record missing or duplicate callback completion for one slot.

        Args:
            slot: Registered logical name or zero-based slot index.
            error: Optional boolean or numeric scalar. A missing, malformed, or
                wrong-device scalar records one error without moving data.

        Raises:
            RuntimeError: If accumulation already completed.
        """

        self._ensure_accumulating()
        slots = self.slots(slot)
        if error is None or error.numel() != 1 or error.device != self.sum_pack.device:
            self.sum_pack[slots.observation_error].add_(1)
            return
        self.sum_pack[slots.observation_error].add_(error.detach().reshape(()))

    def add_applied_update(
        self,
        slot: str | int,
        master_before: torch.Tensor,
        master_after: torch.Tensor,
        applied_before: torch.Tensor,
        applied_after: torch.Tensor,
        *,
        replication_multiplicity: int = 1,
    ) -> None:
        """Add authoritative and applied-update retention moments.

        This specialized update layout retains all quantities needed to derive
        metrics only after global pooling: ``sumsq`` stores master-delta square
        sum, ``lhs_sumsq`` applied-delta square sum, ``rhs_sumsq`` applied-pre
        square sum, ``sum`` master-delta nonzero count, ``dot`` applied-delta
        nonzero count among nonzero master deltas, and ``zero`` cast-to-zero
        count among nonzero master deltas.

        Args:
            slot: Registered update slot.
            master_before: Pre-step FP32 authoritative owner shard.
            master_after: Post-step FP32 authoritative owner shard.
            applied_before: Pre-step BF16 forward owner shard.
            applied_after: Post-materialization BF16 forward owner shard.
            replication_multiplicity: Number of identical logical replicas.

        Raises:
            RuntimeError: If accumulation already completed.
            ValueError: If replication multiplicity is not positive.
        """

        self._ensure_accumulating()
        self._validate_multiplicity(replication_multiplicity)
        slots = self.slots(slot)
        (
            master_before_fp32,
            master_after_fp32,
            applied_before_fp32,
            applied_after_fp32,
        ) = torch.broadcast_tensors(
            master_before.detach().to(dtype=torch.float32),
            master_after.detach().to(dtype=torch.float32),
            applied_before.detach().to(dtype=torch.float32),
            applied_after.detach().to(dtype=torch.float32),
        )
        if master_before_fp32.numel() == 0:
            return

        finite = (
            torch.isfinite(master_before_fp32)
            & torch.isfinite(master_after_fp32)
            & torch.isfinite(applied_before_fp32)
            & torch.isfinite(applied_after_fp32)
        )
        master_delta_fp32 = master_after_fp32 - master_before_fp32
        applied_delta_fp32 = applied_after_fp32 - applied_before_fp32
        clean_master_delta = torch.where(
            finite, master_delta_fp32, torch.zeros_like(master_delta_fp32)
        ).to(dtype=torch.float64)
        clean_applied_delta = torch.where(
            finite, applied_delta_fp32, torch.zeros_like(applied_delta_fp32)
        ).to(dtype=torch.float64)
        clean_applied_before = torch.where(
            finite, applied_before_fp32, torch.zeros_like(applied_before_fp32)
        ).to(dtype=torch.float64)
        scale = replication_multiplicity

        count = finite.sum(dtype=torch.float64) / scale
        master_sumsq = clean_master_delta.square().sum() / scale
        applied_sumsq = clean_applied_delta.square().sum() / scale
        applied_pre_sumsq = clean_applied_before.square().sum() / scale
        master_nonzero = (finite & (master_delta_fp32 != 0)).sum(
            dtype=torch.float64
        ) / scale
        applied_nonzero = (
            finite & (master_delta_fp32 != 0) & (applied_delta_fp32 != 0)
        ).sum(dtype=torch.float64) / scale
        cast_zero = (finite & (master_delta_fp32 != 0) & (applied_delta_fp32 == 0)).sum(
            dtype=torch.float64
        ) / scale
        nonfinite = (~finite).sum(dtype=torch.float64) / scale
        (
            count,
            master_sumsq,
            applied_sumsq,
            applied_pre_sumsq,
            master_nonzero,
            applied_nonzero,
            cast_zero,
            nonfinite,
        ) = self._finite_contributions(
            slots,
            count,
            master_sumsq,
            applied_sumsq,
            applied_pre_sumsq,
            master_nonzero,
            applied_nonzero,
            cast_zero,
            nonfinite,
        )

        self.sum_pack[slots.count].add_(count)
        self.sum_pack[slots.sumsq].add_(master_sumsq)
        self.sum_pack[slots.lhs_sumsq].add_(applied_sumsq)
        self.sum_pack[slots.rhs_sumsq].add_(applied_pre_sumsq)
        self.sum_pack[slots.sum].add_(master_nonzero)
        self.sum_pack[slots.dot].add_(applied_nonzero)
        self.sum_pack[slots.zero].add_(cast_zero)
        self.sum_pack[slots.nonfinite].add_(nonfinite)

        candidate_max = torch.where(
            finite, applied_delta_fp32, torch.full_like(applied_delta_fp32, -torch.inf)
        ).amax()
        candidate_min = torch.where(
            finite, applied_delta_fp32, torch.full_like(applied_delta_fp32, torch.inf)
        ).amin()
        self.max_pack[slots.maximum] = torch.maximum(
            self.max_pack[slots.maximum], candidate_max
        )
        self.min_pack[slots.minimum] = torch.minimum(
            self.min_pack[slots.minimum], candidate_min
        )

    def add_applied_update_moments(
        self,
        slot: str | int,
        *,
        count: torch.Tensor,
        master_sumsq: torch.Tensor,
        applied_sumsq: torch.Tensor,
        applied_pre_sumsq: torch.Tensor,
        master_nonzero: torch.Tensor,
        applied_nonzero: torch.Tensor,
        cast_zero: torch.Tensor,
        nonfinite: torch.Tensor,
        maximum: torch.Tensor,
        minimum: torch.Tensor,
        arithmetic_error: torch.Tensor,
        materialization_error: torch.Tensor,
        materialization_valid: torch.Tensor,
        replication_multiplicity: int = 1,
    ) -> None:
        """Add bounded-chunk update moments without allocating elementwise temporaries.

        The distributed-optimizer adapter uses preallocated device scratch to
        calculate these scalar sufficient statistics. ``materialization_valid``
        is a device scalar covering the complete local owner-shard sequence. A
        failed post-step materialization therefore contributes no numeric
        moments and marks the slot invalid before the later fixed event
        consensus.

        Args:
            slot: Registered update slot.
            count: Finite element count in this chunk.
            master_sumsq: FP32-master delta square sum.
            applied_sumsq: Applied BF16 delta square sum.
            applied_pre_sumsq: Pre-update applied BF16 square sum.
            master_nonzero: Nonzero FP32-master delta count.
            applied_nonzero: Surviving applied-delta count.
            cast_zero: Nonzero master deltas lost by BF16 materialization.
            nonfinite: Nonfinite input count.
            maximum: Maximum finite applied delta.
            minimum: Minimum finite applied delta.
            arithmetic_error: Precomputed nonfinite-arithmetic indicator.
            materialization_error: Precomputed materialization-failure indicator.
            materialization_valid: Device boolean for exact post-step materialization.
            replication_multiplicity: Number of identical logical replicas.

        Raises:
            RuntimeError: If accumulation already completed.
            ValueError: If replication multiplicity is not positive or inputs are not scalars.
        """

        self._ensure_accumulating()
        self._validate_multiplicity(replication_multiplicity)
        slots = self.slots(slot)
        contributions = (
            count,
            master_sumsq,
            applied_sumsq,
            applied_pre_sumsq,
            master_nonzero,
            applied_nonzero,
            cast_zero,
            nonfinite,
            maximum,
            minimum,
            arithmetic_error,
            materialization_error,
            materialization_valid,
        )
        if any(contribution.numel() != 1 for contribution in contributions):
            raise ValueError("precomputed applied-update moments must be scalars")
        if materialization_valid.dtype != torch.bool:
            raise ValueError("precomputed materialization validity must be boolean")
        numeric = contributions[:8]
        if any(contribution.dtype != torch.float64 for contribution in numeric):
            raise ValueError(
                "precomputed numeric applied-update moments must be float64"
            )
        if maximum.dtype != self.max_pack.dtype or minimum.dtype != self.min_pack.dtype:
            raise ValueError("precomputed applied-update extrema have the wrong dtype")
        scale = 1.0 / replication_multiplicity

        self.sum_pack[slots.count].add_(count, alpha=scale)
        self.sum_pack[slots.sumsq].add_(master_sumsq, alpha=scale)
        self.sum_pack[slots.lhs_sumsq].add_(applied_sumsq, alpha=scale)
        self.sum_pack[slots.rhs_sumsq].add_(applied_pre_sumsq, alpha=scale)
        self.sum_pack[slots.sum].add_(master_nonzero, alpha=scale)
        self.sum_pack[slots.dot].add_(applied_nonzero, alpha=scale)
        self.sum_pack[slots.zero].add_(cast_zero, alpha=scale)
        self.sum_pack[slots.nonfinite].add_(nonfinite, alpha=scale)
        self.sum_pack[slots.nonfinite_arithmetic].add_(arithmetic_error)
        self.sum_pack[slots.mask_error].add_(materialization_error)
        torch.maximum(
            self.max_pack[slots.maximum], maximum, out=self.max_pack[slots.maximum]
        )
        torch.minimum(
            self.min_pack[slots.minimum], minimum, out=self.min_pack[slots.minimum]
        )

    def reduce_(self) -> "PackedSufficientStatistics":
        """Reduce the three fixed packs with SUM, MAX, and MIN.

        Returns:
            This accumulator after in-place reduction.

        Raises:
            RuntimeError: If already reduced or distributed is unavailable.
        """

        self._ensure_accumulating()
        binding = self.reduction_binding
        reduce_call = _torch_all_reduce if binding.reducer is None else binding.reducer
        reduce_call(self.sum_pack, op=dist.ReduceOp.SUM, group=binding.group)
        reduce_call(self.max_pack, op=dist.ReduceOp.MAX, group=binding.group)
        reduce_call(self.min_pack, op=dist.ReduceOp.MIN, group=binding.group)
        self._reduced = True
        return self

    @staticmethod
    def reduction_arena_bytes(
        accumulators: Iterable["PackedSufficientStatistics"],
    ) -> int:
        """Return exact temporary bytes used by :meth:`reduce_many_`."""

        return sum(
            accumulator.sum_pack.nbytes
            + accumulator.max_pack.nbytes
            + accumulator.min_pack.nbytes
            for accumulator in accumulators
        )

    @staticmethod
    def allocate_reduction_arenas(
        accumulators: Iterable["PackedSufficientStatistics"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate the fixed combined SUM/MAX/MIN arenas once."""

        packed = tuple(accumulators)
        if not packed:
            raise ValueError("packed event reduction requires at least one accumulator")
        device = packed[0].sum_pack.device
        return (
            torch.empty(
                sum(item.sum_pack.numel() for item in packed),
                dtype=torch.float64,
                device=device,
            ),
            torch.empty(
                sum(item.max_pack.numel() for item in packed),
                dtype=torch.float32,
                device=device,
            ),
            torch.empty(
                sum(item.min_pack.numel() for item in packed),
                dtype=torch.float32,
                device=device,
            ),
        )

    @staticmethod
    def reduce_many_(
        accumulators: Iterable["PackedSufficientStatistics"],
        arenas: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple["PackedSufficientStatistics", ...]:
        """Reduce multiple registries with one packed SUM/MAX/MIN sequence."""

        packed = tuple(accumulators)
        if not packed:
            raise ValueError("packed event reduction requires at least one accumulator")
        first = packed[0]
        for accumulator in packed:
            accumulator._ensure_accumulating()
            if accumulator.sum_pack.device != first.sum_pack.device:
                raise ValueError("packed event accumulators must share one device")
            if accumulator.reduction_binding is not first.reduction_binding:
                raise ValueError(
                    "packed event accumulators must share one reduction binding"
                )

        if arenas is None:
            sum_arena, max_arena, min_arena = (
                PackedSufficientStatistics.allocate_reduction_arenas(packed)
            )
        else:
            sum_arena, max_arena, min_arena = arenas
        if (
            sum_arena.numel() != sum(item.sum_pack.numel() for item in packed)
            or max_arena.numel() != sum(item.max_pack.numel() for item in packed)
            or min_arena.numel() != sum(item.min_pack.numel() for item in packed)
        ):
            raise ValueError("packed event reduction arenas have the wrong fixed size")

        sum_offset = max_offset = min_offset = 0
        for accumulator in packed:
            next_sum = sum_offset + accumulator.sum_pack.numel()
            next_max = max_offset + accumulator.max_pack.numel()
            next_min = min_offset + accumulator.min_pack.numel()
            sum_arena[sum_offset:next_sum].copy_(accumulator.sum_pack)
            max_arena[max_offset:next_max].copy_(accumulator.max_pack)
            min_arena[min_offset:next_min].copy_(accumulator.min_pack)
            sum_offset, max_offset, min_offset = next_sum, next_max, next_min
        binding = first.reduction_binding
        reduce_call = _torch_all_reduce if binding.reducer is None else binding.reducer
        reduce_call(sum_arena, op=dist.ReduceOp.SUM, group=binding.group)
        reduce_call(max_arena, op=dist.ReduceOp.MAX, group=binding.group)
        reduce_call(min_arena, op=dist.ReduceOp.MIN, group=binding.group)

        sum_offset = max_offset = min_offset = 0
        for accumulator in packed:
            next_sum = sum_offset + accumulator.sum_pack.numel()
            next_max = max_offset + accumulator.max_pack.numel()
            next_min = min_offset + accumulator.min_pack.numel()
            accumulator.sum_pack.copy_(sum_arena[sum_offset:next_sum])
            accumulator.max_pack.copy_(max_arena[max_offset:next_max])
            accumulator.min_pack.copy_(min_arena[min_offset:next_min])
            accumulator._reduced = True
            sum_offset, max_offset, min_offset = next_sum, next_max, next_min
        return packed

    def finalize_local_(self) -> "PackedSufficientStatistics":
        """Finish deliberate single-contributor accumulation without collectives.

        Returns:
            This accumulator after marking it ready for derivation.

        Raises:
            RuntimeError: If already finalized or reduced.
        """

        self._ensure_accumulating()
        self._reduced = True
        return self

    def mean(self, slot: str | int) -> DerivedStatistic:
        """Derive a pooled mean after reduction.

        Args:
            slot: Registered logical name or zero-based slot index.

        Returns:
            Derived mean and device-resident validity state.
        """

        slots = self._derived_slots(slot)
        return self._ratio(self.sum_pack[slots.sum], self.sum_pack[slots.count], slots)

    def rms(self, slot: str | int) -> DerivedStatistic:
        """Derive a pooled root-mean-square after reduction.

        Args:
            slot: Registered logical name or zero-based slot index.

        Returns:
            Derived RMS and device-resident validity state.
        """

        slots = self._derived_slots(slot)
        ratio = self._ratio(
            self.sum_pack[slots.sumsq], self.sum_pack[slots.count], slots
        )
        return self._safe_sqrt(ratio)

    def cosine(self, slot: str | int) -> DerivedStatistic:
        """Derive a pooled cosine from reduced dot and norm moments.

        Args:
            slot: Registered logical name or zero-based slot index.

        Returns:
            Derived cosine and device-resident validity state.
        """

        slots = self._derived_slots(slot)
        lhs_norm = torch.sqrt(self.sum_pack[slots.lhs_sumsq])
        rhs_norm = torch.sqrt(self.sum_pack[slots.rhs_sumsq])
        return self._ratio(self.sum_pack[slots.dot], lhs_norm * rhs_norm, slots)

    def relative_rms(self, slot: str | int) -> DerivedStatistic:
        """Derive ``sqrt(sum(lhs^2) / sum(rhs^2))`` after reduction.

        Args:
            slot: Registered logical name or zero-based slot index.

        Returns:
            Derived relative RMS and device-resident validity state.
        """

        slots = self._derived_slots(slot)
        ratio = self._ratio(
            self.sum_pack[slots.lhs_sumsq], self.sum_pack[slots.rhs_sumsq], slots
        )
        return self._safe_sqrt(ratio)

    def master_delta_rms(self, slot: str | int) -> DerivedStatistic:
        """Derive authoritative master-delta RMS after global pooling."""

        slots = self._derived_slots(slot)
        return self._safe_sqrt(
            self._ratio(self.sum_pack[slots.sumsq], self.sum_pack[slots.count], slots)
        )

    def applied_delta_rms(self, slot: str | int) -> DerivedStatistic:
        """Derive applied BF16 delta RMS after global pooling."""

        slots = self._derived_slots(slot)
        return self._safe_sqrt(
            self._ratio(
                self.sum_pack[slots.lhs_sumsq], self.sum_pack[slots.count], slots
            )
        )

    def norm_retention(self, slot: str | int) -> DerivedStatistic:
        """Derive ``sqrt(sum(applied_delta^2) / sum(master_delta^2))``."""

        slots = self._derived_slots(slot)
        return self._safe_sqrt(
            self._ratio(
                self.sum_pack[slots.lhs_sumsq], self.sum_pack[slots.sumsq], slots
            )
        )

    def master_nonzero_fraction(self, slot: str | int) -> DerivedStatistic:
        """Derive the fraction of coordinates with a nonzero master delta."""

        slots = self._derived_slots(slot)
        return self._ratio(self.sum_pack[slots.sum], self.sum_pack[slots.count], slots)

    def applied_nonzero_fraction(self, slot: str | int) -> DerivedStatistic:
        """Derive applied-update survival among nonzero master-delta coordinates."""

        slots = self._derived_slots(slot)
        return self._ratio(self.sum_pack[slots.dot], self.sum_pack[slots.sum], slots)

    def cast_zero_fraction(self, slot: str | int) -> DerivedStatistic:
        """Derive cast-to-zero rate among coordinates with nonzero master delta."""

        slots = self._derived_slots(slot)
        return self._ratio(self.sum_pack[slots.zero], self.sum_pack[slots.sum], slots)

    def zero_fraction(self, slot: str | int) -> DerivedStatistic:
        """Derive the zero fraction among selected finite elements.

        Args:
            slot: Registered logical name or zero-based slot index.

        Returns:
            Derived zero fraction and device-resident validity state.
        """

        slots = self._derived_slots(slot)
        return self._ratio(self.sum_pack[slots.zero], self.sum_pack[slots.count], slots)

    def nonfinite_fraction(self, slot: str | int) -> DerivedStatistic:
        """Derive the nonfinite fraction among all selected elements.

        Args:
            slot: Registered logical name or zero-based slot index.

        Returns:
            Measurable nonfinite fraction and device-resident validity state.
        """

        slots = self._derived_slots(slot)
        total = self.sum_pack[slots.count] + self.sum_pack[slots.nonfinite]
        quotient = self.sum_pack[slots.nonfinite] / total
        has_contributors = total > 0
        mask_valid = self.sum_pack[slots.mask_error] == 0
        arithmetic_finite = (
            (self.sum_pack[slots.nonfinite_arithmetic] == 0)
            & torch.isfinite(total)
            & (~has_contributors | torch.isfinite(quotient))
        )
        valid = has_contributors & mask_valid & arithmetic_finite
        value = torch.where(valid, quotient, torch.full_like(total, torch.nan))
        reason = self._reason(
            has_contributors=has_contributors,
            finite_input=torch.ones_like(valid),
            positive_denominator=total > 0,
            mask_valid=mask_valid,
            arithmetic_finite=arithmetic_finite,
            reference=total,
        )
        return DerivedStatistic(value, valid, reason)

    def maximum(self, slot: str | int) -> DerivedStatistic:
        """Return the reduced maximum, invalidating unusable observations.

        Args:
            slot: Registered logical name or zero-based slot index.

        Returns:
            Reduced maximum and device-resident validity state.
        """

        slots = self._derived_slots(slot)
        return self._extremum(self.max_pack[slots.maximum], slots)

    def minimum(self, slot: str | int) -> DerivedStatistic:
        """Return the reduced minimum, invalidating unusable observations.

        Args:
            slot: Registered logical name or zero-based slot index.

        Returns:
            Reduced minimum and device-resident validity state.
        """

        slots = self._derived_slots(slot)
        return self._extremum(self.min_pack[slots.minimum], slots)

    def _ratio(
        self, numerator: torch.Tensor, denominator: torch.Tensor, slots: PackedSlots
    ) -> DerivedStatistic:
        quotient = numerator / denominator
        has_contributors = self.sum_pack[slots.count] > 0
        finite_input = self.sum_pack[slots.nonfinite] == 0
        positive_denominator = denominator > 0
        mask_valid = self.sum_pack[slots.mask_error] == 0
        arithmetic_finite = (
            (self.sum_pack[slots.nonfinite_arithmetic] == 0)
            & torch.isfinite(numerator)
            & torch.isfinite(denominator)
            & (~positive_denominator | torch.isfinite(quotient))
        )
        valid = (
            has_contributors
            & finite_input
            & positive_denominator
            & mask_valid
            & arithmetic_finite
        )
        value = torch.where(valid, quotient, torch.full_like(numerator, torch.nan))
        reason = self._reason(
            has_contributors=has_contributors,
            finite_input=finite_input,
            positive_denominator=positive_denominator,
            mask_valid=mask_valid,
            arithmetic_finite=arithmetic_finite,
            reference=numerator,
        )
        return DerivedStatistic(value, valid, reason)

    def _safe_sqrt(self, statistic: DerivedStatistic) -> DerivedStatistic:
        rooted = torch.sqrt(statistic.value)
        arithmetic_finite = torch.isfinite(rooted)
        valid = statistic.valid & arithmetic_finite
        value = torch.where(valid, rooted, torch.full_like(rooted, torch.nan))
        reason = torch.where(
            statistic.valid & ~arithmetic_finite,
            torch.full_like(statistic.reason, Tier0Reason.NONFINITE_ARITHMETIC),
            statistic.reason,
        )
        return DerivedStatistic(value, valid, reason)

    def _extremum(self, value: torch.Tensor, slots: PackedSlots) -> DerivedStatistic:
        has_contributors = self.sum_pack[slots.count] > 0
        finite_input = self.sum_pack[slots.nonfinite] == 0
        mask_valid = self.sum_pack[slots.mask_error] == 0
        arithmetic_finite = (self.sum_pack[slots.nonfinite_arithmetic] == 0) & (
            ~has_contributors | torch.isfinite(value)
        )
        valid = has_contributors & finite_input & mask_valid & arithmetic_finite
        derived = torch.where(valid, value, torch.full_like(value, torch.nan))
        reason = self._reason(
            has_contributors=has_contributors,
            finite_input=finite_input,
            positive_denominator=torch.ones_like(valid),
            mask_valid=mask_valid,
            arithmetic_finite=arithmetic_finite,
            reference=value,
        )
        return DerivedStatistic(derived, valid, reason)

    @staticmethod
    def _reason(
        *,
        has_contributors: torch.Tensor,
        finite_input: torch.Tensor,
        positive_denominator: torch.Tensor,
        mask_valid: torch.Tensor,
        arithmetic_finite: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        reason = torch.full_like(reference, Tier0Reason.NONE, dtype=torch.int64)
        reason = torch.where(
            ~positive_denominator,
            torch.full_like(reason, Tier0Reason.ZERO_DENOMINATOR),
            reason,
        )
        reason = torch.where(
            ~has_contributors,
            torch.full_like(reason, Tier0Reason.NO_CONTRIBUTORS),
            reason,
        )
        reason = torch.where(
            ~finite_input, torch.full_like(reason, Tier0Reason.NONFINITE_INPUT), reason
        )
        reason = torch.where(
            ~arithmetic_finite,
            torch.full_like(reason, Tier0Reason.NONFINITE_ARITHMETIC),
            reason,
        )
        return torch.where(
            ~mask_valid, torch.full_like(reason, Tier0Reason.MASK_MISMATCH), reason
        )

    def _derived_slots(self, slot: str | int) -> PackedSlots:
        if not self._reduced:
            raise RuntimeError("statistics must be reduced before values are derived")
        return self.slots(slot)

    def _resolve_index(self, slot: str | int) -> int:
        if isinstance(slot, str):
            try:
                return self._slot_indices[slot]
            except KeyError as error:
                raise KeyError(f"unknown packed statistic slot: {slot}") from error
        if slot < 0 or slot >= len(self.slot_names):
            raise IndexError(f"packed statistic slot is out of range: {slot}")
        return slot

    def _validated_broadcast_mask(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None,
        slots: PackedSlots,
        *,
        require_mask: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self._workspace is None:
            mask_valid = torch.ones((), dtype=torch.bool, device=values.device)
        else:
            mask_valid = self._workspace.bool_scalars[0]
            mask_valid.fill_(True)
        if mask is None:
            if require_mask:
                self.sum_pack[slots.mask_error].add_(1)
                mask_valid.zero_()
            return None, mask_valid
        if mask.device != values.device or mask.is_complex():
            self.sum_pack[slots.mask_error].add_(1)
            mask_valid.zero_()
            return None, mask_valid
        try:
            weights = torch.broadcast_to(mask.detach(), values.shape)
        except RuntimeError:
            self.sum_pack[slots.mask_error].add_(1)
            mask_valid.zero_()
            return None, mask_valid

        for (weight_chunk,) in self._bounded_chunks(weights):
            if self._workspace is None:
                weights_fp32 = weight_chunk.to(dtype=torch.float32)
                mask_valid.logical_and_(
                    (torch.isfinite(weights_fp32) & (weights_fp32 >= 0)).all()
                )
            else:
                count = weight_chunk.numel()
                weights_fp32 = self._workspace.fp32[4, :count]
                finite = self._workspace.boolean[4, :count]
                nonnegative = self._workspace.boolean[3, :count]
                weights_fp32.copy_(weight_chunk)
                self._isfinite_out(weights_fp32, finite, nonnegative)
                torch.ge(weights_fp32, 0, out=nonnegative)
                finite.logical_and_(nonnegative)
                torch.all(finite, out=self._workspace.bool_scalars[1])
                mask_valid.logical_and_(self._workspace.bool_scalars[1])
        if self._workspace is None:
            self.sum_pack[slots.mask_error].add_(
                (~mask_valid).to(dtype=self.sum_pack.dtype)
            )
        else:
            torch.logical_not(mask_valid, out=self._workspace.bool_scalars[1])
            self.sum_pack[slots.mask_error].add_(self._workspace.bool_scalars[1])
        return weights, mask_valid

    @staticmethod
    def _isfinite_out(
        values: torch.Tensor, finite: torch.Tensor, auxiliary: torch.Tensor
    ) -> None:
        """Compute finite flags with primitives that support caller-provided output."""

        torch.eq(values, values, out=finite)
        torch.ne(values, torch.inf, out=auxiliary)
        finite.logical_and_(auxiliary)
        torch.ne(values, -torch.inf, out=auxiliary)
        finite.logical_and_(auxiliary)

    def _bounded_chunks(
        self, *tensors: torch.Tensor
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        pending = [tensors]
        while pending:
            chunks = pending.pop()
            reference = chunks[0]
            if reference.numel() <= self.scratch_element_capacity:
                self._peak_scratch_bytes = max(
                    self._peak_scratch_bytes,
                    reference.numel() * _MAX_SCRATCH_BYTES_PER_ELEMENT,
                )
                yield chunks
                continue
            split_dimension = next(
                dimension
                for dimension, length in enumerate(reference.shape)
                if length > 1
            )
            split = reference.shape[split_dimension] // 2
            first = tuple(tensor.narrow(split_dimension, 0, split) for tensor in chunks)
            second = tuple(
                tensor.narrow(
                    split_dimension, split, reference.shape[split_dimension] - split
                )
                for tensor in chunks
            )
            pending.append(second)
            pending.append(first)

    def _finite_contributions(
        self, slots: PackedSlots, *contributions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        arithmetic_finite = torch.stack(
            tuple(torch.isfinite(contribution) for contribution in contributions)
        ).all()
        self.sum_pack[slots.nonfinite_arithmetic].add_(
            (~arithmetic_finite).to(dtype=self.sum_pack.dtype)
        )
        return tuple(
            torch.where(arithmetic_finite, contribution, torch.zeros_like(contribution))
            for contribution in contributions
        )

    @staticmethod
    def _validate_multiplicity(replication_multiplicity: int) -> None:
        if replication_multiplicity <= 0:
            raise ValueError("replication multiplicity must be positive")

    def _ensure_accumulating(self) -> None:
        if self._reduced:
            raise RuntimeError("cannot accumulate or reduce after the reduction phase")
