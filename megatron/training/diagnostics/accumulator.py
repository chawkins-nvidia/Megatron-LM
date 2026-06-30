# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU-resident packed sufficient statistics for scalable diagnostics."""

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist

from .schema import Tier0Reason

_SUM_FIELDS = (
    "sum",
    "count",
    "sumsq",
    "dot",
    "lhs_sumsq",
    "rhs_sumsq",
    "zero",
    "nonfinite",
)
_SUM_FIELD_COUNT = len(_SUM_FIELDS)


class PackedReducer(Protocol):
    """Callable contract for injecting collective and process-group ownership."""

    def __call__(
        self, tensor: torch.Tensor, *, op: object, group: object | None
    ) -> object: ...


@dataclass(frozen=True)
class PackedSlots:
    """Offsets occupied by one metric in the fixed packed buffers."""

    sum: int
    count: int
    sumsq: int
    dot: int
    lhs_sumsq: int
    rhs_sumsq: int
    zero: int
    nonfinite: int
    maximum: int
    minimum: int

    @classmethod
    def for_index(cls, index: int) -> "PackedSlots":
        """Return canonical offsets for a metric at ``index``."""

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
            maximum=index,
            minimum=index,
        )


@dataclass(frozen=True)
class DerivedStatistic:
    """A derived scalar, its validity flag, and a stable reason code tensor."""

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
    """Accumulate and reduce fixed-slot sufficient statistics on the input device.

    Inputs are converted to FP32 for arithmetic. Additive statistics are summed
    into FP64 slots, while extrema use FP32 MAX/MIN packs. Values are derived only
    after :meth:`reduce_` or explicit single-contributor :meth:`finalize_local_`.
    """

    def __init__(self, slot_names: tuple[str, ...], device: torch.device | str) -> None:
        """Allocate neutral packed buffers in a stable slot order.

        Args:
            slot_names: Unique logical metric names in collective slot order.
            device: CPU or CUDA device on which accumulation and reduction occur.
        """

        if len(slot_names) != len(set(slot_names)):
            raise ValueError("packed statistic slot names must be unique")
        self.slot_names = slot_names
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
        """Whether the packs have completed their reduction phase."""

        return self._reduced

    def slots(self, slot: str | int) -> PackedSlots:
        """Resolve a logical name or integer slot to its canonical packed offsets."""

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
        """Add masked moments for a tensor without deriving a local metric.

        ``mask`` must be broadcastable to ``values`` and may contain fractional
        nonnegative weights. Replicated values are corrected by dividing every
        additive contribution by ``replication_multiplicity``.
        """

        self._ensure_accumulating()
        self._validate_multiplicity(replication_multiplicity)
        index = self._resolve_index(slot)
        values_fp32 = values.detach().to(dtype=torch.float32)
        weights = self._weights_like(values_fp32, mask)
        if values_fp32.numel() == 0:
            return
        selected = weights != 0
        finite = torch.isfinite(values_fp32)
        finite_selected = selected & finite
        clean = torch.where(finite, values_fp32, torch.zeros_like(values_fp32))
        finite_weights = torch.where(finite, weights, torch.zeros_like(weights))
        scale = replication_multiplicity
        slots = PackedSlots.for_index(index)

        self.sum_pack[slots.sum].add_(
            (clean * finite_weights).sum(dtype=torch.float64) / scale
        )
        self.sum_pack[slots.count].add_(finite_weights.sum(dtype=torch.float64) / scale)
        self.sum_pack[slots.sumsq].add_(
            (clean.square() * finite_weights).sum(dtype=torch.float64) / scale
        )
        self.sum_pack[slots.lhs_sumsq].add_(
            (clean.square() * finite_weights).sum(dtype=torch.float64) / scale
        )
        self.sum_pack[slots.zero].add_(
            torch.where(
                finite_selected & (values_fp32 == 0), weights, torch.zeros_like(weights)
            ).sum(dtype=torch.float64)
            / scale
        )
        self.sum_pack[slots.nonfinite].add_(
            torch.where(selected & ~finite, weights, torch.zeros_like(weights)).sum(
                dtype=torch.float64
            )
            / scale
        )
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
        """Add pooled dot/cosine/relative-norm moments for a pair of tensors."""

        self._ensure_accumulating()
        self._validate_multiplicity(replication_multiplicity)
        index = self._resolve_index(slot)
        lhs_fp32, rhs_fp32 = torch.broadcast_tensors(
            lhs.detach().to(dtype=torch.float32), rhs.detach().to(dtype=torch.float32)
        )
        weights = self._weights_like(lhs_fp32, mask)
        if lhs_fp32.numel() == 0:
            return
        selected = weights != 0
        finite = torch.isfinite(lhs_fp32) & torch.isfinite(rhs_fp32)
        finite_selected = selected & finite
        clean_lhs = torch.where(finite, lhs_fp32, torch.zeros_like(lhs_fp32))
        clean_rhs = torch.where(finite, rhs_fp32, torch.zeros_like(rhs_fp32))
        finite_weights = torch.where(finite, weights, torch.zeros_like(weights))
        scale = replication_multiplicity
        slots = PackedSlots.for_index(index)
        lhs_sumsq = (clean_lhs.square() * finite_weights).sum(
            dtype=torch.float64
        ) / scale
        rhs_sumsq = (clean_rhs.square() * finite_weights).sum(
            dtype=torch.float64
        ) / scale

        self.sum_pack[slots.sum].add_(
            (clean_lhs * finite_weights).sum(dtype=torch.float64) / scale
        )
        self.sum_pack[slots.count].add_(finite_weights.sum(dtype=torch.float64) / scale)
        self.sum_pack[slots.sumsq].add_(lhs_sumsq)
        self.sum_pack[slots.dot].add_(
            (clean_lhs * clean_rhs * finite_weights).sum(dtype=torch.float64) / scale
        )
        self.sum_pack[slots.lhs_sumsq].add_(lhs_sumsq)
        self.sum_pack[slots.rhs_sumsq].add_(rhs_sumsq)
        self.sum_pack[slots.zero].add_(
            torch.where(
                finite_selected & (lhs_fp32 == 0), weights, torch.zeros_like(weights)
            ).sum(dtype=torch.float64)
            / scale
        )
        self.sum_pack[slots.nonfinite].add_(
            torch.where(selected & ~finite, weights, torch.zeros_like(weights)).sum(
                dtype=torch.float64
            )
            / scale
        )
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
        """Add update delta moments with the pre-update tensor as denominator."""

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
        self, *, group: object | None, reducer: PackedReducer | None = None
    ) -> "PackedSufficientStatistics":
        """Reduce all fixed packs with an injected process group and reducer."""

        self._ensure_accumulating()
        reduce_call = _torch_all_reduce if reducer is None else reducer
        reduce_call(self.sum_pack, op=dist.ReduceOp.SUM, group=group)
        reduce_call(self.max_pack, op=dist.ReduceOp.MAX, group=group)
        reduce_call(self.min_pack, op=dist.ReduceOp.MIN, group=group)
        self._reduced = True
        return self

    def finalize_local_(self) -> "PackedSufficientStatistics":
        """Finish a deliberate single-contributor accumulation without a collective."""

        self._ensure_accumulating()
        self._reduced = True
        return self

    def mean(self, slot: str | int) -> DerivedStatistic:
        """Derive the pooled mean after reduction."""

        slots = self._derived_slots(slot)
        return self._ratio(self.sum_pack[slots.sum], self.sum_pack[slots.count], slots)

    def rms(self, slot: str | int) -> DerivedStatistic:
        """Derive pooled root-mean-square after reduction."""

        slots = self._derived_slots(slot)
        ratio = self._ratio(
            self.sum_pack[slots.sumsq], self.sum_pack[slots.count], slots
        )
        return DerivedStatistic(torch.sqrt(ratio.value), ratio.valid, ratio.reason)

    def cosine(self, slot: str | int) -> DerivedStatistic:
        """Derive pooled cosine from globally reduced dot and norm moments."""

        slots = self._derived_slots(slot)
        denominator = torch.sqrt(
            self.sum_pack[slots.lhs_sumsq] * self.sum_pack[slots.rhs_sumsq]
        )
        return self._ratio(self.sum_pack[slots.dot], denominator, slots)

    def relative_rms(self, slot: str | int) -> DerivedStatistic:
        """Derive ``sqrt(sum(lhs^2) / sum(rhs^2))`` after reduction."""

        slots = self._derived_slots(slot)
        ratio = self._ratio(
            self.sum_pack[slots.lhs_sumsq], self.sum_pack[slots.rhs_sumsq], slots
        )
        return DerivedStatistic(torch.sqrt(ratio.value), ratio.valid, ratio.reason)

    def zero_fraction(self, slot: str | int) -> DerivedStatistic:
        """Derive the zero fraction among selected finite elements."""

        slots = self._derived_slots(slot)
        return self._ratio(self.sum_pack[slots.zero], self.sum_pack[slots.count], slots)

    def nonfinite_fraction(self, slot: str | int) -> DerivedStatistic:
        """Derive the measurable nonfinite fraction among all selected elements."""

        slots = self._derived_slots(slot)
        total = self.sum_pack[slots.count] + self.sum_pack[slots.nonfinite]
        valid = total > 0
        value = torch.where(
            valid,
            self.sum_pack[slots.nonfinite] / total,
            torch.full_like(total, torch.nan),
        )
        reason = torch.where(
            valid,
            torch.full_like(total, Tier0Reason.NONE, dtype=torch.int64),
            torch.full_like(total, Tier0Reason.NO_CONTRIBUTORS, dtype=torch.int64),
        )
        return DerivedStatistic(value, valid, reason)

    def maximum(self, slot: str | int) -> DerivedStatistic:
        """Return the reduced maximum, invalidating empty or nonfinite inputs."""

        slots = self._derived_slots(slot)
        return self._extremum(self.max_pack[slots.maximum], slots)

    def minimum(self, slot: str | int) -> DerivedStatistic:
        """Return the reduced minimum, invalidating empty or nonfinite inputs."""

        slots = self._derived_slots(slot)
        return self._extremum(self.min_pack[slots.minimum], slots)

    def _ratio(
        self, numerator: torch.Tensor, denominator: torch.Tensor, slots: PackedSlots
    ) -> DerivedStatistic:
        has_contributors = self.sum_pack[slots.count] > 0
        finite_input = self.sum_pack[slots.nonfinite] == 0
        positive_denominator = denominator > 0
        valid = has_contributors & finite_input & positive_denominator
        value = torch.where(
            valid, numerator / denominator, torch.full_like(numerator, torch.nan)
        )
        reason = self._reason(
            has_contributors, finite_input, positive_denominator, numerator
        )
        return DerivedStatistic(value, valid, reason)

    def _extremum(self, value: torch.Tensor, slots: PackedSlots) -> DerivedStatistic:
        has_contributors = self.sum_pack[slots.count] > 0
        finite_input = self.sum_pack[slots.nonfinite] == 0
        valid = has_contributors & finite_input
        derived = torch.where(valid, value, torch.full_like(value, torch.nan))
        reason = self._reason(
            has_contributors, finite_input, torch.ones_like(valid), value
        )
        return DerivedStatistic(derived, valid, reason)

    @staticmethod
    def _reason(
        has_contributors: torch.Tensor,
        finite_input: torch.Tensor,
        positive_denominator: torch.Tensor,
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
        return torch.where(
            ~finite_input, torch.full_like(reason, Tier0Reason.NONFINITE_INPUT), reason
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

    @staticmethod
    def _weights_like(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(values, dtype=torch.float32)
        weights = mask.detach().to(device=values.device, dtype=torch.float32)
        return torch.broadcast_to(weights, values.shape)

    @staticmethod
    def _validate_multiplicity(replication_multiplicity: int) -> None:
        if replication_multiplicity <= 0:
            raise ValueError("replication multiplicity must be positive")

    def _ensure_accumulating(self) -> None:
        if self._reduced:
            raise RuntimeError("cannot accumulate or reduce after the reduction phase")
