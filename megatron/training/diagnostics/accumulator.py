# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Device-resident packed sufficient statistics for scalable diagnostics."""

from dataclasses import dataclass
from typing import Protocol

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
)
_SUM_FIELD_COUNT = len(_SUM_FIELDS)


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
        schema_identity: str = SCHEMA_PREFIX.rstrip("/"),
        process_group_identity: str = "unbound",
        reduction_identity: str = "packed_sum_max_min",
    ) -> None:
        """Allocate neutral packed buffers in a stable slot order.

        Args:
            slot_names: Unique logical metric names in collective slot order.
            device: CPU or CUDA device used for accumulation and reduction.
            descriptor_hash: Rank-independent descriptor hash.
            schema_identity: Versioned diagnostic schema identity.
            process_group_identity: Stable name of the intended process group.
            reduction_identity: Stable name of the packed reduction contract.

        Raises:
            ValueError: If slot names or identities are invalid.
        """

        if len(slot_names) != len(set(slot_names)):
            raise ValueError("packed statistic slot names must be unique")
        if (
            not descriptor_hash
            or not schema_identity
            or not process_group_identity
            or not reduction_identity
        ):
            raise ValueError("packed statistic identities must be nonempty")
        self.slot_names = slot_names
        self.descriptor_hash = descriptor_hash
        self.schema_identity = schema_identity
        self.process_group_identity = process_group_identity
        self.reduction_identity = reduction_identity
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
        self._reduced = False

    @property
    def reduced(self) -> bool:
        """Return whether the packs completed their reduction phase."""

        return self._reduced

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

    def add_masked_tensor(
        self,
        slot: str | int,
        values: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        replication_multiplicity: int = 1,
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

        Raises:
            RuntimeError: If accumulation already completed.
            ValueError: If replication multiplicity is not positive.
            KeyError: If a logical name is unknown.
            IndexError: If an integer slot is out of range.
        """

        self._ensure_accumulating()
        self._validate_multiplicity(replication_multiplicity)
        slots = self.slots(slot)
        values_fp32 = values.detach().to(dtype=torch.float32)
        weights = self._weights_like(values_fp32, mask, slots)
        if values_fp32.numel() == 0:
            return

        selected = weights != 0
        finite = torch.isfinite(values_fp32)
        finite_selected = selected & finite
        clean_fp32 = torch.where(finite, values_fp32, torch.zeros_like(values_fp32))
        finite_weights_fp32 = torch.where(finite, weights, torch.zeros_like(weights))
        clean = clean_fp32.to(dtype=torch.float64)
        finite_weights = finite_weights_fp32.to(dtype=torch.float64)
        scale = replication_multiplicity

        weighted_sum = (clean * finite_weights).sum() / scale
        count = finite_weights.sum() / scale
        weighted_sumsq = (clean.square() * finite_weights).sum() / scale
        zero = (
            torch.where(
                finite_selected & (values_fp32 == 0), weights, torch.zeros_like(weights)
            ).sum(dtype=torch.float64)
            / scale
        )
        nonfinite = (
            torch.where(selected & ~finite, weights, torch.zeros_like(weights)).sum(
                dtype=torch.float64
            )
            / scale
        )
        weighted_sum, count, weighted_sumsq, zero, nonfinite = (
            self._finite_contributions(
                slots, weighted_sum, count, weighted_sumsq, zero, nonfinite
            )
        )

        self.sum_pack[slots.sum].add_(weighted_sum)
        self.sum_pack[slots.count].add_(count)
        self.sum_pack[slots.sumsq].add_(weighted_sumsq)
        self.sum_pack[slots.lhs_sumsq].add_(weighted_sumsq)
        self.sum_pack[slots.zero].add_(zero)
        self.sum_pack[slots.nonfinite].add_(nonfinite)

        candidate_max = torch.where(
            finite_selected, values_fp32, torch.full_like(values_fp32, -torch.inf)
        ).amax()
        candidate_min = torch.where(
            finite_selected, values_fp32, torch.full_like(values_fp32, torch.inf)
        ).amin()
        self.max_pack[slots.maximum] = torch.maximum(
            self.max_pack[slots.maximum], candidate_max
        )
        self.min_pack[slots.minimum] = torch.minimum(
            self.min_pack[slots.minimum], candidate_min
        )

    def add_masked_pair(
        self,
        slot: str | int,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        replication_multiplicity: int = 1,
    ) -> None:
        """Add pooled dot, cosine, and relative-norm moments for a pair.

        Args:
            slot: Registered logical name or zero-based slot index.
            lhs: Numerator-side tensor observation.
            rhs: Denominator-side tensor observation.
            mask: Optional same-device broadcastable nonnegative finite weights.
            replication_multiplicity: Number of identical logical replicas.

        Raises:
            RuntimeError: If accumulation already completed.
            ValueError: If replication multiplicity is not positive.
            KeyError: If a logical name is unknown.
            IndexError: If an integer slot is out of range.
        """

        self._ensure_accumulating()
        self._validate_multiplicity(replication_multiplicity)
        slots = self.slots(slot)
        lhs_fp32, rhs_fp32 = torch.broadcast_tensors(
            lhs.detach().to(dtype=torch.float32), rhs.detach().to(dtype=torch.float32)
        )
        weights = self._weights_like(lhs_fp32, mask, slots)
        if lhs_fp32.numel() == 0:
            return

        selected = weights != 0
        finite = torch.isfinite(lhs_fp32) & torch.isfinite(rhs_fp32)
        finite_selected = selected & finite
        clean_lhs_fp32 = torch.where(finite, lhs_fp32, torch.zeros_like(lhs_fp32))
        clean_rhs_fp32 = torch.where(finite, rhs_fp32, torch.zeros_like(rhs_fp32))
        finite_weights_fp32 = torch.where(finite, weights, torch.zeros_like(weights))
        clean_lhs = clean_lhs_fp32.to(dtype=torch.float64)
        clean_rhs = clean_rhs_fp32.to(dtype=torch.float64)
        finite_weights = finite_weights_fp32.to(dtype=torch.float64)
        scale = replication_multiplicity

        weighted_sum = (clean_lhs * finite_weights).sum() / scale
        count = finite_weights.sum() / scale
        lhs_sumsq = (clean_lhs.square() * finite_weights).sum() / scale
        rhs_sumsq = (clean_rhs.square() * finite_weights).sum() / scale
        dot = (clean_lhs * clean_rhs * finite_weights).sum() / scale
        zero = (
            torch.where(
                finite_selected & (lhs_fp32 == 0), weights, torch.zeros_like(weights)
            ).sum(dtype=torch.float64)
            / scale
        )
        nonfinite = (
            torch.where(selected & ~finite, weights, torch.zeros_like(weights)).sum(
                dtype=torch.float64
            )
            / scale
        )
        (
            weighted_sum,
            count,
            lhs_sumsq,
            rhs_sumsq,
            dot,
            zero,
            nonfinite,
        ) = self._finite_contributions(
            slots,
            weighted_sum,
            count,
            lhs_sumsq,
            rhs_sumsq,
            dot,
            zero,
            nonfinite,
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
            finite_selected, lhs_fp32, torch.full_like(lhs_fp32, -torch.inf)
        ).amax()
        candidate_min = torch.where(
            finite_selected, lhs_fp32, torch.full_like(lhs_fp32, torch.inf)
        ).amin()
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
    ) -> None:
        """Add update-delta moments with the pre-update tensor as denominator.

        Args:
            slot: Registered logical name or zero-based slot index.
            before: Pre-update tensor.
            after: Post-update tensor.
            mask: Optional same-device broadcastable nonnegative finite weights.
            replication_multiplicity: Number of identical logical replicas.

        Raises:
            RuntimeError: If accumulation already completed.
            ValueError: If replication multiplicity is not positive.
        """

        before_fp32 = before.detach().to(dtype=torch.float32)
        after_fp32 = after.detach().to(dtype=torch.float32)
        self.add_masked_pair(
            slot,
            after_fp32 - before_fp32,
            before_fp32,
            mask=mask,
            replication_multiplicity=replication_multiplicity,
        )

    def reduce_(
        self, *, group: object | None = None, reducer: PackedReducer | None = None
    ) -> "PackedSufficientStatistics":
        """Reduce the three fixed packs with SUM, MAX, and MIN.

        Args:
            group: Process group matching ``process_group_identity``.
            reducer: Optional test reducer; defaults to ``torch.distributed.all_reduce``.

        Returns:
            This accumulator after in-place reduction.

        Raises:
            RuntimeError: If already reduced or distributed is unavailable.
        """

        self._ensure_accumulating()
        reduce_call = _torch_all_reduce if reducer is None else reducer
        reduce_call(self.sum_pack, op=dist.ReduceOp.SUM, group=group)
        reduce_call(self.max_pack, op=dist.ReduceOp.MAX, group=group)
        reduce_call(self.min_pack, op=dist.ReduceOp.MIN, group=group)
        self._reduced = True
        return self

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
            ~finite_input,
            torch.full_like(reason, Tier0Reason.NONFINITE_INPUT),
            reason,
        )
        reason = torch.where(
            ~arithmetic_finite,
            torch.full_like(reason, Tier0Reason.NONFINITE_ARITHMETIC),
            reason,
        )
        return torch.where(
            ~mask_valid,
            torch.full_like(reason, Tier0Reason.MASK_MISMATCH),
            reason,
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

    def _weights_like(
        self, values: torch.Tensor, mask: torch.Tensor | None, slots: PackedSlots
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(values, dtype=torch.float32)
        if mask.device != values.device or mask.is_complex():
            self.sum_pack[slots.mask_error].add_(1)
            return torch.zeros_like(values, dtype=torch.float32)
        try:
            weights = torch.broadcast_to(
                mask.detach().to(dtype=torch.float32), values.shape
            )
        except RuntimeError:
            self.sum_pack[slots.mask_error].add_(1)
            return torch.zeros_like(values, dtype=torch.float32)

        valid_elements = torch.isfinite(weights) & (weights >= 0)
        mask_valid = valid_elements.all()
        self.sum_pack[slots.mask_error].add_(
            (~mask_valid).to(dtype=self.sum_pack.dtype)
        )
        return torch.where(mask_valid, weights, torch.zeros_like(weights))

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
