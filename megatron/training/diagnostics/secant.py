# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Collective-free Tier-2 secant math, control, restoration, and memory primitives."""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import torch

from .accumulator import PackedSlots, PackedSufficientStatistics, ReductionBinding
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

_SECANT_SLOT_SUFFIXES = ("response", "error_replay", "pre")
_PACKED_BYTES_PER_SLOT = 11 * 8 + 2 * 4
_CANONICAL_SCRATCH_BYTES_PER_ELEMENT = 96
_SECANT_INPUT_AND_MATH_BYTES_PER_ELEMENT = 16 * 4
_OWNER_FINISH_BYTES_PER_ELEMENT = 2 + 4 * 4 + 2 * 8 + 2
_OWNER_FINISH_SCALAR_BYTES = 104
_MAX_REPRESENTABLE_BYTES = (1 << 63) - 1
_TIER2_OUTPUT_SCALAR_COUNT = 17

TIER2_OUTPUT_KEYS: tuple[str, ...] = (
    "diag/v2/t2/true_response/p10",
    "diag/v2/t2/true_response/p50",
    "diag/v2/t2/true_response/p90",
    "diag/v2/t2/secant_error/p10",
    "diag/v2/t2/secant_error/p50",
    "diag/v2/t2/secant_error/p90",
    "diag/v2/t2/secant_cosine/p10",
    "diag/v2/t2/secant_cosine/p50",
    "diag/v2/t2/secant_cosine/p90",
    "diag/v2/t2/realized_midpoint_fraction/p10",
    "diag/v2/t2/realized_midpoint_fraction/p50",
    "diag/v2/t2/realized_midpoint_fraction/p90",
    "diag/v2/t2/replay_floor/p10",
    "diag/v2/t2/replay_floor/p50",
    "diag/v2/t2/replay_floor/p90",
    "diag/v2/t2/unresolved_fraction",
    "diag/v2/t2/valid",
)


@dataclass(frozen=True)
class SecantCellDescriptor:
    """Describe one globally ordered replay cell and its rank-local ownership."""

    logical_name: str
    family: MetricFamily
    global_layer: int
    local_owner: bool = True

    def __post_init__(self) -> None:
        """Reject identities that cannot participate in a global slot sequence."""

        if not self.logical_name:
            raise ValueError("a secant cell requires a logical name")
        if self.global_layer < 0:
            raise ValueError("a secant cell requires a nonnegative global layer")


@dataclass(frozen=True)
class SecantRegistryBinding:
    """Bind canonical replay cells to the existing metric registry."""

    cells: tuple[SecantCellDescriptor, ...]
    registry: MetricRegistry

    @property
    def observation_names(self) -> tuple[str, ...]:
        """Return the one accepted observation sequence."""

        return tuple(cell.logical_name for cell in self.cells)

    def slot_name(self, cell: SecantCellDescriptor, suffix: str) -> str:
        """Return one typed canonical sufficient-statistic slot name."""

        if suffix not in _SECANT_SLOT_SUFFIXES:
            raise ValueError(f"unknown secant slot suffix: {suffix}")
        return f"secant/{cell.logical_name}/{suffix}"


def canonical_secant_cells(
    cells: Iterable[SecantCellDescriptor],
) -> tuple[SecantCellDescriptor, ...]:
    """Return deterministic rank-independent cell order and reject duplicates."""

    materialized = tuple(cells)
    names = tuple(cell.logical_name for cell in materialized)
    if not materialized:
        raise ValueError("a secant registry requires at least one global cell")
    if len(names) != len(set(names)):
        raise ValueError("secant cell logical names must be unique")
    return tuple(
        sorted(
            materialized, key=lambda cell: (cell.global_layer, cell.family.value, cell.logical_name)
        )
    )


def build_secant_registry(
    cells: Iterable[SecantCellDescriptor], *, reduction_binding: ReductionBinding
) -> SecantRegistryBinding:
    """Build three canonical sufficient-statistic slots per global replay cell."""

    ordered = canonical_secant_cells(cells)
    descriptors: list[MetricDescriptor] = []
    local_owners: list[bool] = []
    for cell in ordered:
        for suffix in _SECANT_SLOT_SUFFIXES:
            statistic_kind = (
                StatisticKind.TENSOR_MOMENTS if suffix == "pre" else StatisticKind.PAIR_MOMENTS
            )
            descriptors.append(
                MetricDescriptor(
                    logical_name=f"secant/{cell.logical_name}/{suffix}",
                    family=cell.family,
                    global_layer=cell.global_layer,
                    partition_axes=(
                        PartitionAxis.DATA_SAMPLE,
                        PartitionAxis.TENSOR_FEATURE,
                        PartitionAxis.PIPELINE_LAYER,
                        PartitionAxis.CONTEXT_SEQUENCE,
                    ),
                    replication_axes=(ReplicationAxis.TENSOR,),
                    replication_multiplicity=1,
                    ownership=Ownership.PIPELINE_STAGE,
                    mask_kind=MaskKind.TOKEN,
                    statistic_kind=statistic_kind,
                    denominator_kind=(
                        DenominatorKind.SELECTED_ELEMENTS
                        if statistic_kind == StatisticKind.TENSOR_MOMENTS
                        else DenominatorKind.RHS_SUMSQ
                    ),
                    normalization_kind=NormalizationKind.NONE,
                    process_group_identity=ProcessGroupIdentity.WORLD,
                    reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
                    tied_owner_identity=None,
                    packed_slots=PackedSlots.for_index(len(descriptors)),
                )
            )
            local_owners.append(cell.local_owner)
    registry = MetricRegistry(
        descriptors, reduction_binding=reduction_binding, local_owners=local_owners
    )
    return SecantRegistryBinding(ordered, registry)


@dataclass(frozen=True)
class SecantObservation:
    """Store four aligned replay observations for one canonical cell."""

    logical_name: str
    pre: torch.Tensor
    post: torch.Tensor
    post_repeat: torch.Tensor
    midpoint: torch.Tensor
    mask: torch.Tensor | None = None


class SecantStatistics:
    """Accumulate secant moments into canonical registry-owned packs."""

    def __init__(
        self,
        binding: SecantRegistryBinding,
        device: torch.device | str,
        *,
        chunk_elements: int = 16 * 1024,
    ) -> None:
        """Allocate canonical packs and retain a fixed arithmetic chunk bound."""

        if chunk_elements <= 0:
            raise ValueError("secant chunk elements must be positive")
        self.binding = binding
        self.accumulator = binding.registry.new_accumulator(device)
        self.chunk_elements = chunk_elements

    def add_observations(self, observations: Sequence[SecantObservation]) -> None:
        """Add exactly one observation per global cell in canonical order.

        Missing, duplicate, or reversed observations fail before any pack is changed.
        Nonowners validate the sequence but leave every canonical slot neutral.
        """

        received = tuple(observation.logical_name for observation in observations)
        if received != self.binding.observation_names:
            raise ValueError(
                "secant observations must exactly match canonical global cell order: "
                f"expected={self.binding.observation_names}, received={received}"
            )
        prepared = tuple(self._validate_observation(observation) for observation in observations)
        for cell, observation, observation_tensors in zip(
            self.binding.cells, observations, prepared
        ):
            response_name = self.binding.slot_name(cell, "response")
            error_name = self.binding.slot_name(cell, "error_replay")
            pre_name = self.binding.slot_name(cell, "pre")
            if not self.binding.registry.owns(response_name):
                continue
            pre, post, repeat, midpoint, mask = observation_tensors
            flattened = tuple(tensor.view(-1) for tensor in (pre, post, repeat, midpoint))
            for start in range(0, pre.numel(), self.chunk_elements):
                end = min(start + self.chunk_elements, pre.numel())
                pre_chunk, post_chunk, repeat_chunk, midpoint_chunk = (
                    tensor[start:end].to(dtype=torch.float32) for tensor in flattened
                )
                true = post_chunk - pre_chunk
                predicted = 2.0 * (midpoint_chunk - pre_chunk)
                error = true - predicted
                repeat_error = repeat_chunk - post_chunk
                mask_chunk = self._mask_chunk(mask, pre.shape, start, end)
                self.binding.registry.add_masked_pair(
                    self.accumulator, response_name, true, predicted, mask=mask_chunk
                )
                self.binding.registry.add_masked_pair(
                    self.accumulator, error_name, error, repeat_error, mask=mask_chunk
                )
                self.binding.registry.add_masked_tensor(
                    self.accumulator, pre_name, pre_chunk, mask=mask_chunk
                )

    @staticmethod
    def _validate_observation(
        observation: SecantObservation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        tensors = tuple(
            tensor.detach()
            for tensor in (
                observation.pre,
                observation.post,
                observation.post_repeat,
                observation.midpoint,
            )
        )
        reference = tensors[0]
        for tensor in tensors:
            if (
                tensor.shape != reference.shape
                or tensor.device != reference.device
                or tensor.dtype != reference.dtype
                or tensor.stride() != reference.stride()
            ):
                raise ValueError(
                    "secant pre/post/repeat/midpoint tensors require identical "
                    "shape, device, dtype, and semantic layout"
                )
            if (
                not tensor.is_contiguous()
                or any(stride == 0 for stride in tensor.stride())
                or tensor.is_conj()
                or tensor.is_neg()
            ):
                raise ValueError(
                    "secant observations require contiguous non-broadcast physical layouts"
                )

        mask = None if observation.mask is None else observation.mask.detach()
        if mask is not None:
            if mask.device != reference.device:
                raise ValueError("secant masks must be on the observation device")
            if mask.is_conj() or mask.is_neg():
                raise ValueError("secant masks cannot use conjugate or negative views")
            mask_shape = (1,) * (reference.ndim - mask.ndim) + tuple(mask.shape)
            if mask.ndim > reference.ndim or any(
                mask_extent not in (1, observation_extent)
                for mask_extent, observation_extent in zip(mask_shape, reference.shape)
            ):
                raise ValueError("secant mask is not broadcast-aligned to the observation")
        return (*tensors, mask)

    @staticmethod
    def _mask_chunk(
        mask: torch.Tensor | None, shape: torch.Size, start: int, end: int
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.numel() == 1:
            return mask.view(())
        if (
            mask.shape == shape
            and mask.is_contiguous()
            and not any(stride == 0 for stride in mask.stride())
        ):
            return mask.view(-1)[start:end]

        size = end - start
        remaining = torch.empty(size, dtype=torch.int64, device=mask.device)
        coordinate = torch.empty_like(remaining)
        storage_offsets = torch.full_like(remaining, mask.storage_offset())
        torch.arange(start, end, out=remaining)
        padded_shape = (1,) * (len(shape) - mask.ndim) + tuple(mask.shape)
        padded_stride = (0,) * (len(shape) - mask.ndim) + mask.stride()
        for observation_extent, mask_extent, mask_stride in reversed(
            tuple(zip(shape, padded_shape, padded_stride))
        ):
            torch.remainder(remaining, observation_extent, out=coordinate)
            if mask_extent != 1:
                storage_offsets.add_(coordinate, alpha=mask_stride)
            torch.floor_divide(remaining, observation_extent, out=remaining)
        storage_elements = mask.untyped_storage().nbytes() // mask.element_size()
        storage = mask.as_strided((storage_elements,), (1,), 0)
        return torch.index_select(storage, 0, storage_offsets)


@dataclass(frozen=True)
class SecantSufficientStatisticsView:
    """Typed scalar views into one cell's canonical reduced packs."""

    true_sq: torch.Tensor
    predicted_sq: torch.Tensor
    error_sq: torch.Tensor
    true_predicted_dot: torch.Tensor
    pre_sq: torch.Tensor
    repeat_error_sq: torch.Tensor
    count: torch.Tensor
    nonfinite: torch.Tensor
    contract_error: torch.Tensor

    @classmethod
    def from_accumulator(
        cls,
        binding: SecantRegistryBinding,
        accumulator: PackedSufficientStatistics,
        cell: SecantCellDescriptor,
    ) -> "SecantSufficientStatisticsView":
        """Bind a cell view only when descriptor hash and slot order match exactly."""

        binding.registry._validate_accumulator(accumulator)
        if cell not in binding.cells:
            raise ValueError("secant cell is not present in the bound global registry")
        if not accumulator.reduced:
            raise RuntimeError("secant statistics must be globally reduced before derivation")

        response = accumulator.slots(binding.slot_name(cell, "response"))
        error_replay = accumulator.slots(binding.slot_name(cell, "error_replay"))
        pre = accumulator.slots(binding.slot_name(cell, "pre"))
        pack = accumulator.sum_pack
        counts = torch.stack((pack[response.count], pack[error_replay.count], pack[pre.count]))
        contract_error = (
            (counts != counts[0]).to(dtype=torch.float64).sum()
            + pack[response.mask_error]
            + pack[error_replay.mask_error]
            + pack[pre.mask_error]
            + pack[response.observation_error]
            + pack[error_replay.observation_error]
            + pack[pre.observation_error]
            + pack[response.nonfinite_arithmetic]
            + pack[error_replay.nonfinite_arithmetic]
            + pack[pre.nonfinite_arithmetic]
        )
        return cls(
            true_sq=pack[response.lhs_sumsq],
            predicted_sq=pack[response.rhs_sumsq],
            error_sq=pack[error_replay.lhs_sumsq],
            true_predicted_dot=pack[response.dot],
            pre_sq=pack[pre.sumsq],
            repeat_error_sq=pack[error_replay.rhs_sumsq],
            count=counts[0],
            nonfinite=(
                pack[response.nonfinite] + pack[error_replay.nonfinite] + pack[pre.nonfinite]
            ),
            contract_error=contract_error,
        )


class SecantMathStatus(IntEnum):
    """Stable fail-closed status for one pooled replay cell."""

    OK = 0
    NO_CONTRIBUTORS = 1
    NONFINITE = 2
    CONTRACT_ERROR = 3
    TINY_DENOMINATOR = 4
    REPLAY_UNRESOLVED = 5
    MIDPOINT_OUT_OF_RANGE = 6
    RESTORE_UNVERIFIED = 7


@dataclass(frozen=True)
class SecantCellMetrics:
    """Five Tier-2 layer metrics plus device-resident validity evidence."""

    true_response: torch.Tensor
    secant_error: torch.Tensor
    secant_cosine: torch.Tensor
    realized_midpoint_fraction: torch.Tensor
    replay_floor: torch.Tensor
    valid: torch.Tensor
    status: torch.Tensor
    true_sq: torch.Tensor
    predicted_sq: torch.Tensor
    error_sq: torch.Tensor
    repeat_error_sq: torch.Tensor
    count: torch.Tensor


def derive_secant_cell(
    statistics: SecantSufficientStatisticsView,
    *,
    midpoint_displacement_sq: torch.Tensor | float,
    full_displacement_sq: torch.Tensor | float,
    restore_verified: torch.Tensor | bool,
    replay_floor_multiplier: float = 10.0,
    midpoint_min: float = 0.45,
    midpoint_max: float = 0.55,
    denominator_epsilon: float = torch.finfo(torch.float64).tiny,
) -> SecantCellMetrics:
    """Derive one pooled cell without host reads or rank-local ratios."""

    if replay_floor_multiplier <= 0:
        raise ValueError("replay floor multiplier must be positive")
    if not 0 <= midpoint_min <= midpoint_max:
        raise ValueError("midpoint bounds are invalid")
    if denominator_epsilon < 0:
        raise ValueError("denominator epsilon must be nonnegative")

    reference = statistics.true_sq
    midpoint_sq = torch.as_tensor(
        midpoint_displacement_sq, dtype=torch.float64, device=reference.device
    )
    full_sq = torch.as_tensor(full_displacement_sq, dtype=torch.float64, device=reference.device)
    restored = torch.as_tensor(restore_verified, dtype=torch.bool, device=reference.device)
    raw = torch.stack(
        (
            statistics.true_sq,
            statistics.predicted_sq,
            statistics.error_sq,
            statistics.true_predicted_dot,
            statistics.pre_sq,
            statistics.repeat_error_sq,
            statistics.count,
            statistics.nonfinite,
            statistics.contract_error,
            midpoint_sq,
            full_sq,
        )
    )
    finite = torch.isfinite(raw).all()
    contributors = statistics.count > 0
    no_nonfinite = statistics.nonfinite == 0
    contract_valid = (
        (statistics.contract_error == 0)
        & (statistics.error_sq >= 0)
        & (statistics.repeat_error_sq >= 0)
        & (midpoint_sq >= 0)
    )
    denominators_valid = (
        (statistics.true_sq > denominator_epsilon)
        & (statistics.predicted_sq > denominator_epsilon)
        & (statistics.pre_sq > denominator_epsilon)
        & (full_sq > denominator_epsilon)
    )

    true_response = torch.sqrt(statistics.true_sq / statistics.pre_sq)
    secant_error = torch.sqrt(statistics.error_sq / statistics.true_sq)
    secant_cosine = statistics.true_predicted_dot / torch.sqrt(
        statistics.true_sq * statistics.predicted_sq
    )
    midpoint_fraction = torch.sqrt(midpoint_sq / full_sq)
    replay_floor = torch.sqrt(statistics.repeat_error_sq / statistics.true_sq)
    resolved = statistics.true_sq >= (
        replay_floor_multiplier * replay_floor_multiplier * statistics.repeat_error_sq
    )
    midpoint_valid = (midpoint_fraction >= midpoint_min) & (midpoint_fraction <= midpoint_max)
    derived_finite = (
        torch.stack((true_response, secant_error, secant_cosine, midpoint_fraction, replay_floor))
        .isfinite()
        .all()
    )
    valid = (
        finite
        & contributors
        & no_nonfinite
        & contract_valid
        & denominators_valid
        & derived_finite
        & resolved
        & midpoint_valid
        & restored
    )

    status = torch.zeros((), dtype=torch.int64, device=reference.device)
    status = torch.where(
        ~contributors, torch.full_like(status, int(SecantMathStatus.NO_CONTRIBUTORS)), status
    )
    status = torch.where(
        ~finite | ~no_nonfinite, torch.full_like(status, int(SecantMathStatus.NONFINITE)), status
    )
    status = torch.where(
        contributors & finite & no_nonfinite & ~contract_valid,
        torch.full_like(status, int(SecantMathStatus.CONTRACT_ERROR)),
        status,
    )
    status = torch.where(
        contributors & finite & no_nonfinite & contract_valid & ~denominators_valid,
        torch.full_like(status, int(SecantMathStatus.TINY_DENOMINATOR)),
        status,
    )
    status = torch.where(
        contributors
        & finite
        & no_nonfinite
        & contract_valid
        & denominators_valid
        & ~derived_finite,
        torch.full_like(status, int(SecantMathStatus.NONFINITE)),
        status,
    )
    status = torch.where(
        contributors & finite & no_nonfinite & contract_valid & denominators_valid & ~resolved,
        torch.full_like(status, int(SecantMathStatus.REPLAY_UNRESOLVED)),
        status,
    )
    status = torch.where(
        contributors
        & finite
        & no_nonfinite
        & contract_valid
        & denominators_valid
        & resolved
        & ~midpoint_valid,
        torch.full_like(status, int(SecantMathStatus.MIDPOINT_OUT_OF_RANGE)),
        status,
    )
    status = torch.where(
        contributors
        & finite
        & no_nonfinite
        & contract_valid
        & denominators_valid
        & resolved
        & midpoint_valid
        & ~restored,
        torch.full_like(status, int(SecantMathStatus.RESTORE_UNVERIFIED)),
        status,
    )
    nan = torch.full_like(reference, torch.nan)

    def checked(value: torch.Tensor) -> torch.Tensor:
        return torch.where(valid, value, nan)

    return SecantCellMetrics(
        true_response=checked(true_response),
        secant_error=checked(secant_error),
        secant_cosine=checked(secant_cosine),
        realized_midpoint_fraction=checked(midpoint_fraction),
        replay_floor=checked(replay_floor),
        valid=valid,
        status=status,
        true_sq=statistics.true_sq,
        predicted_sq=statistics.predicted_sq,
        error_sq=statistics.error_sq,
        repeat_error_sq=statistics.repeat_error_sq,
        count=statistics.count,
    )


def derive_tier2_outputs(metrics: Sequence[SecantCellMetrics]) -> dict[str, torch.Tensor]:
    """Return the exact ordered 17-key Tier-2 payload from pooled cell metrics."""

    if not metrics:
        raise ValueError("Tier-2 output derivation requires at least one cell")
    quantiles = torch.tensor((0.1, 0.5, 0.9), dtype=torch.float64, device=metrics[0].valid.device)
    values: list[torch.Tensor] = []
    for attribute in (
        "true_response",
        "secant_error",
        "secant_cosine",
        "realized_midpoint_fraction",
        "replay_floor",
    ):
        stacked = torch.stack(tuple(getattr(metric, attribute) for metric in metrics))
        values.extend(_bounded_nanquantiles(stacked, quantiles).unbind())
    validity = torch.stack(tuple(metric.valid for metric in metrics))
    statuses = torch.stack(tuple(metric.status for metric in metrics))
    unresolved = statuses == int(SecantMathStatus.REPLAY_UNRESOLVED)
    values.append(unresolved.to(dtype=torch.float64).mean())
    values.append(validity.all().to(dtype=torch.float64))
    return dict(zip(TIER2_OUTPUT_KEYS, values))


def _bounded_nanquantiles(values: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Derive linear quantiles with three explicitly bounded cell-sized buffers."""

    sorted_values = torch.sort(values).values
    finite_count = torch.isfinite(values).sum(dtype=torch.int64)
    rank = quantiles * (finite_count.to(dtype=torch.float64) - 1).clamp_min(0)
    lower = torch.floor(rank).to(dtype=torch.int64)
    upper = torch.ceil(rank).to(dtype=torch.int64)
    lower_value = torch.index_select(sorted_values, 0, lower)
    upper_value = torch.index_select(sorted_values, 0, upper)
    result = lower_value + (upper_value - lower_value) * (rank - lower)
    return torch.where(finite_count > 0, result, torch.full_like(result, torch.nan))


class RestorationKind(IntEnum):
    """Fixed independent restoration order for local secant work."""

    FP32_MASTERS = 0
    BF16_PARAMETERS = 1
    OPTIMIZER_VIEWS = 2
    TIED_ALIASES = 3
    OVERLAP_STATE = 4
    MODEL_STATE = 5
    RNG_HANDLES = 6


@dataclass(frozen=True)
class RestorationStage:
    """Bind one independent restore attempt and device-resident verifier."""

    kind: RestorationKind
    restore: Callable[[], None]
    verify: Callable[[], torch.Tensor]


@dataclass(frozen=True)
class RestorationReport:
    """Store per-stage validity and caught Python failures without short-circuiting."""

    stage_valid: torch.Tensor
    valid: torch.Tensor
    failed_kinds: tuple[RestorationKind, ...]


class IndependentRestorer:
    """Attempt every fixed restoration stage even when earlier stages fail."""

    def __init__(self, stages: Iterable[RestorationStage], device: torch.device | str) -> None:
        """Require every independent state class exactly once and in fixed order."""

        self.stages = tuple(stages)
        expected = tuple(RestorationKind)
        received = tuple(stage.kind for stage in self.stages)
        if received != expected:
            raise ValueError(
                "restoration stages must be complete, unique, and ordered: "
                f"expected={expected}, received={received}"
            )
        self.device = torch.device(device)
        self._stage_valid = torch.ones(len(self.stages), dtype=torch.bool, device=self.device)
        self._valid = torch.ones((), dtype=torch.bool, device=self.device)
        self._verified = torch.ones((), dtype=torch.bool, device=self.device)

    def run(self) -> RestorationReport:
        """Run all restore and verification callbacks without short-circuiting."""

        stage_valid = self._stage_valid
        stage_valid.fill_(True)
        failed: list[RestorationKind] = []
        for index, stage in enumerate(self.stages):
            try:
                stage.restore()
            except Exception:
                stage_valid[index] = False
                failed.append(stage.kind)
            try:
                verified = stage.verify().detach().to(device=self.device, dtype=torch.bool)
                torch.all(verified, dim=tuple(range(verified.ndim)), out=self._verified)
                torch.logical_and(stage_valid[index], self._verified, out=stage_valid[index])
            except Exception:
                stage_valid[index] = False
                if stage.kind not in failed:
                    failed.append(stage.kind)
        torch.all(stage_valid, dim=(0,), out=self._valid)
        return RestorationReport(stage_valid, self._valid, tuple(failed))

    def failure_report(self) -> RestorationReport:
        """Return complete fixed-shape failure evidence after a restorer-level failure."""

        self._stage_valid.fill_(False)
        self._valid.fill_(False)
        return RestorationReport(self._stage_valid, self._valid, tuple(RestorationKind))


class TensorBitwiseSnapshot:
    """Capture, independently restore, and device-verify unique tensor storage views."""

    def __init__(self, tensors: Iterable[torch.Tensor], *, chunk_elements: int = 65_536) -> None:
        """Clone each exact physical view once, preserving alias ownership."""

        if chunk_elements <= 0:
            raise ValueError("snapshot chunk elements must be positive")
        unique: list[torch.Tensor] = []
        identities: set[tuple[object, ...]] = set()
        for tensor in tensors:
            if not tensor.is_contiguous():
                raise ValueError("bitwise restoration snapshots require contiguous tensor views")
            identity = (
                tensor.device,
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tensor.numel(),
                tensor.dtype,
            )
            if identity in identities:
                continue
            identities.add(identity)
            unique.append(tensor)
        self.tensors = tuple(unique)
        self.snapshots = tuple(tensor.detach().clone() for tensor in self.tensors)
        self.chunk_elements = chunk_elements

    @property
    def byte_count(self) -> int:
        """Return exact retained snapshot bytes."""

        return sum(snapshot.nbytes for snapshot in self.snapshots)

    def restore(self) -> None:
        """Restore every unique tensor even when another tensor copy fails."""

        failures = 0
        for tensor, snapshot in zip(self.tensors, self.snapshots):
            try:
                destination = tensor.detach().view(-1)
                source = snapshot.view(-1)
                for start in range(0, source.numel(), self.chunk_elements):
                    end = min(start + self.chunk_elements, source.numel())
                    destination[start:end].copy_(source[start:end])
            except Exception:
                failures += 1
        if failures:
            raise RuntimeError(f"{failures} tensor restoration stages failed")

    def verify(self) -> torch.Tensor:
        """Compare tensor bytes without a host read or ``torch.equal``."""

        device = self.snapshots[0].device if self.snapshots else torch.device("cpu")
        valid = torch.ones((), dtype=torch.bool, device=device)
        for tensor, snapshot in zip(self.tensors, self.snapshots):
            if tensor.shape != snapshot.shape or tensor.dtype != snapshot.dtype:
                valid.fill_(False)
                continue
            current_bytes = tensor.detach().view(torch.uint8).view(-1)
            expected_bytes = snapshot.view(torch.uint8).view(-1)
            for start in range(0, expected_bytes.numel(), self.chunk_elements):
                end = min(start + self.chunk_elements, expected_bytes.numel())
                mismatch = torch.ne(current_bytes[start:end], expected_bytes[start:end]).any()
                torch.logical_and(valid, ~mismatch, out=valid)
        return valid


class SecantTransactionState(IntEnum):
    """Fixed local half of the Tier-2 integration protocol."""

    PRE_READY = 0
    PRE_CAPTURED = 1
    UPDATE_SUCCEEDED = 2
    POST_READY = 3
    POST_REPEAT_READY = 4
    MIDPOINT_INSTALLED = 5
    MIDPOINT_CAPTURED = 6
    RESTORED = 7


class SecantTransactionError(IntEnum):
    """Typed local errors retained for fixed global readiness controls."""

    NONE = 0
    INVALID_TRANSITION = 1
    PRE_CAPTURE_FAILED = 2
    UPDATE_FAILED = 3
    POST_CAPTURE_FAILED = 4
    POST_REPEAT_CAPTURE_FAILED = 5
    MIDPOINT_INSTALL_FAILED = 6
    MIDPOINT_CAPTURE_FAILED = 7
    RESTORE_FAILED = 8


@dataclass(frozen=True)
class SecantTransactionResult:
    """Current monotonic local state plus central-fatal evidence."""

    state: SecantTransactionState
    error: SecantTransactionError | torch.Tensor
    fatal_required: bool | torch.Tensor
    restoration: RestorationReport | None


class SecantLocalTransaction:
    """Advance the local protocol without collectives, schedules, or P2P."""

    _TARGET_ERRORS = {
        SecantTransactionState.PRE_CAPTURED: SecantTransactionError.PRE_CAPTURE_FAILED,
        SecantTransactionState.UPDATE_SUCCEEDED: SecantTransactionError.UPDATE_FAILED,
        SecantTransactionState.POST_READY: SecantTransactionError.POST_CAPTURE_FAILED,
        SecantTransactionState.POST_REPEAT_READY: SecantTransactionError.POST_REPEAT_CAPTURE_FAILED,
        SecantTransactionState.MIDPOINT_INSTALLED: SecantTransactionError.MIDPOINT_INSTALL_FAILED,
        SecantTransactionState.MIDPOINT_CAPTURED: SecantTransactionError.MIDPOINT_CAPTURE_FAILED,
        SecantTransactionState.RESTORED: SecantTransactionError.RESTORE_FAILED,
    }

    def __init__(self, restorer: IndependentRestorer) -> None:
        """Start at PRE_READY with no local error."""

        self.restorer = restorer
        self.state = SecantTransactionState.PRE_READY
        self.error = SecantTransactionError.NONE
        self.fatal_required = False
        self.restoration: RestorationReport | None = None
        self.state_history = [self.state]
        self._work_started = False

    def advance(
        self, target: SecantTransactionState, action: Callable[[], None] | None = None
    ) -> SecantTransactionResult:
        """Run one idempotent next transition and record failures without raising."""

        if target == self.state:
            return self.result()
        if self.state == SecantTransactionState.RESTORED:
            return self.result()
        if self.error != SecantTransactionError.NONE or int(target) != int(self.state) + 1:
            return self.fail(SecantTransactionError.INVALID_TRANSITION)
        if target == SecantTransactionState.PRE_CAPTURED:
            self._work_started = True
        try:
            if action is not None:
                action()
        except Exception:
            return self.fail(self._TARGET_ERRORS[target])
        if target == SecantTransactionState.RESTORED:
            self.restoration = self._run_restorer()
            restore_failed = torch.logical_not(self.restoration.valid)
            self.error = torch.where(
                restore_failed,
                torch.full_like(
                    restore_failed, int(SecantTransactionError.RESTORE_FAILED), dtype=torch.int64
                ),
                torch.full_like(
                    restore_failed, int(SecantTransactionError.NONE), dtype=torch.int64
                ),
            )
            self.fatal_required = restore_failed
        self.state = target
        self.state_history.append(target)
        return self.result()

    def fail(self, error: SecantTransactionError) -> SecantTransactionResult:
        """Record one typed error and restore independently once work has begun."""

        if self.error == SecantTransactionError.NONE:
            self.error = error
        if error == SecantTransactionError.INVALID_TRANSITION:
            self.fatal_required = True
        if self._work_started and self.restoration is None:
            self.restoration = self._run_restorer()
            self.fatal_required = True
            if self.state != SecantTransactionState.RESTORED:
                self.state = SecantTransactionState.RESTORED
                self.state_history.append(self.state)
        return self.result()

    def _run_restorer(self) -> RestorationReport:
        try:
            return self.restorer.run()
        except Exception:
            return self.restorer.failure_report()

    def result(self) -> SecantTransactionResult:
        """Return immutable local control evidence."""

        return SecantTransactionResult(
            self.state, self.error, self.fatal_required, self.restoration
        )


@dataclass(frozen=True)
class SecantOptimizerLayout:
    """Pure fail-closed optimizer layout contract for memory preflight."""

    distributed_optimizer: bool = True
    bf16_parameters: bool = True
    fp32_masters: bool = True
    precision_aware: bool = False
    fsdp: bool = False
    fp8_or_fp4: bool = False
    cpu_offload: bool = False
    chained_optimizer_children: int = 1

    def validate(self) -> None:
        """Reject every layout outside the canonical first backend."""

        if not self.distributed_optimizer:
            raise SecantUnsupportedLayoutError("distributed optimizer is required")
        if not self.bf16_parameters or not self.fp32_masters:
            raise SecantUnsupportedLayoutError("BF16 parameters with FP32 masters are required")
        if self.precision_aware or self.fsdp or self.fp8_or_fp4 or self.cpu_offload:
            raise SecantUnsupportedLayoutError(
                "optimizer layout uses an unsupported representation"
            )
        if self.chained_optimizer_children != 1:
            raise SecantUnsupportedLayoutError(
                "exactly one distributed optimizer child is required"
            )


class SecantUnsupportedLayoutError(RuntimeError):
    """Raised when exact Tier-2 memory accounting is impossible."""


class SecantMemoryEstimateError(ValueError):
    """Raised when a legal integer input exceeds the explicit estimate domain."""


class _AdapterMemoryEstimate(Protocol):
    owner_elements: int
    fp32_master_bytes: int
    bf16_applied_bytes: int
    post_fingerprint_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class SecantMemoryInputs:
    """All pure inputs needed for an exact maximum-rank Tier-2 estimate."""

    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    context_parallel_size: int
    loaded_owner_elements_by_rank: tuple[int, ...]
    tied_alias_elements_by_rank: tuple[int, ...]
    replay_payload_bytes: int
    replay_mask_bytes: int
    replay_state_bytes: int
    replay_cap_bytes: int
    registry_slots: int
    chunk_elements: int
    alignment_bytes: int = 256
    allocator_headroom_fraction: float = 0.1
    minimum_allocator_headroom_bytes: int = 0
    driver_headroom_fraction: float = 0.05
    minimum_driver_headroom_bytes: int = 0


@dataclass(frozen=True)
class SecantMemoryEstimate:
    """Exact aligned retained/workspace upper bound for the maximum-loaded rank."""

    world_size: int
    maximum_loaded_rank: int
    maximum_loaded_owner_elements: int
    tied_alias_elements: int
    unique_owner_elements: int
    fp32_pre_or_delta_bytes: int
    bf16_pre_bytes: int
    post_fingerprint_bytes: int
    replay_payload_bytes: int
    replay_mask_bytes: int
    replay_state_bytes: int
    replay_cap_reserve_bytes: int
    packed_statistics_bytes: int
    reduction_arena_bytes: int
    bounded_accumulation_workspace_bytes: int
    owner_hash_restore_workspace_bytes: int
    quantile_sink_workspace_bytes: int
    chunk_workspace_bytes: int
    retained_bytes: int
    allocator_headroom_bytes: int
    driver_headroom_bytes: int
    peak_bytes: int

    def additional_bytes_for_adapter(self, adapter: _AdapterMemoryEstimate) -> int:
        """Compose this authoritative peak with adapter preflight without double counting."""

        if adapter.owner_elements != self.unique_owner_elements:
            raise ValueError("adapter owner count does not match the maximum-loaded rank")
        if (
            adapter.fp32_master_bytes != 4 * self.unique_owner_elements
            or adapter.bf16_applied_bytes != 2 * self.unique_owner_elements
            or adapter.post_fingerprint_bytes != 4 * 8
        ):
            raise ValueError("adapter shared snapshot schema does not match the secant estimate")
        if adapter.total_bytes > self.peak_bytes:
            raise SecantMemoryEstimateError("adapter lifecycle exceeds the complete secant peak")
        return self.peak_bytes - adapter.total_bytes

    @classmethod
    def calculate(
        cls, inputs: SecantMemoryInputs, *, optimizer_layout: SecantOptimizerLayout
    ) -> "SecantMemoryEstimate":
        """Calculate one aligned upper bound or reject unsupported inputs."""

        optimizer_layout.validate()
        sizes = (
            inputs.data_parallel_size,
            inputs.tensor_parallel_size,
            inputs.pipeline_parallel_size,
            inputs.context_parallel_size,
        )
        if any(not isinstance(size, int) for size in sizes):
            raise ValueError("parallel sizes must be integers")
        if any(size <= 0 for size in sizes):
            raise ValueError("parallel sizes must be positive")
        if any(size > _MAX_REPRESENTABLE_BYTES for size in sizes):
            raise SecantMemoryEstimateError("parallel size exceeds the representable cap")
        world_size = math.prod(sizes)
        if world_size > _MAX_REPRESENTABLE_BYTES:
            raise SecantMemoryEstimateError("world size exceeds the representable cap")
        if len(inputs.loaded_owner_elements_by_rank) != world_size:
            raise ValueError("loaded owner counts must contain exactly one entry per world rank")
        if len(inputs.tied_alias_elements_by_rank) != world_size:
            raise ValueError("tied alias counts must contain exactly one entry per world rank")
        scalar_values = (
            *inputs.loaded_owner_elements_by_rank,
            *inputs.tied_alias_elements_by_rank,
            inputs.replay_payload_bytes,
            inputs.replay_mask_bytes,
            inputs.replay_state_bytes,
            inputs.replay_cap_bytes,
            inputs.registry_slots,
            inputs.minimum_allocator_headroom_bytes,
            inputs.minimum_driver_headroom_bytes,
        )
        if any(not isinstance(value, int) for value in scalar_values):
            raise ValueError("memory inputs must be integers")
        if any(value < 0 for value in scalar_values):
            raise ValueError("memory inputs must be nonnegative")
        if any(value > _MAX_REPRESENTABLE_BYTES for value in scalar_values):
            raise SecantMemoryEstimateError("memory input exceeds the representable cap")
        if not isinstance(inputs.chunk_elements, int):
            raise ValueError("chunk elements must be an integer")
        if inputs.chunk_elements <= 0:
            raise ValueError("chunk elements must be positive")
        if inputs.chunk_elements > _MAX_REPRESENTABLE_BYTES:
            raise SecantMemoryEstimateError("chunk elements exceed the representable cap")
        if not isinstance(inputs.alignment_bytes, int):
            raise ValueError("allocation alignment must be an integer")
        if inputs.alignment_bytes <= 0:
            raise ValueError("allocation alignment must be positive")
        if inputs.alignment_bytes > _MAX_REPRESENTABLE_BYTES:
            raise SecantMemoryEstimateError("allocation alignment exceeds the representable cap")
        if not math.isfinite(inputs.allocator_headroom_fraction) or not (
            0 <= inputs.allocator_headroom_fraction < 1
        ):
            raise ValueError("allocator headroom fraction must be in [0, 1)")
        if not math.isfinite(inputs.driver_headroom_fraction) or not (
            0 <= inputs.driver_headroom_fraction < 1
        ):
            raise ValueError("driver headroom fraction must be in [0, 1)")
        if inputs.registry_slots <= 0 or inputs.registry_slots % len(_SECANT_SLOT_SUFFIXES) != 0:
            raise ValueError("registry slots must contain complete canonical secant cells")
        replay_bytes = (
            inputs.replay_payload_bytes + inputs.replay_mask_bytes + inputs.replay_state_bytes
        )
        if replay_bytes > inputs.replay_cap_bytes:
            raise ValueError("replay payload, masks, and state exceed the hard byte cap")
        unique_by_rank = tuple(
            loaded - tied
            for loaded, tied in zip(
                inputs.loaded_owner_elements_by_rank, inputs.tied_alias_elements_by_rank
            )
        )
        if any(unique < 0 for unique in unique_by_rank):
            raise ValueError("tied alias elements cannot exceed loaded owner elements")
        maximum_rank = max(range(world_size), key=unique_by_rank.__getitem__)
        loaded = inputs.loaded_owner_elements_by_rank[maximum_rank]
        tied = inputs.tied_alias_elements_by_rank[maximum_rank]
        unique = unique_by_rank[maximum_rank]

        def align(value: int) -> int:
            aligned = _align_bytes(value, inputs.alignment_bytes)
            if aligned > _MAX_REPRESENTABLE_BYTES:
                raise SecantMemoryEstimateError("aligned allocation exceeds the representable cap")
            return aligned

        fp32 = align(4 * unique)
        bf16 = align(2 * unique)
        fingerprints = align(4 * 8)
        replay_payload = align(inputs.replay_payload_bytes)
        replay_mask = align(inputs.replay_mask_bytes)
        replay_state = align(inputs.replay_state_bytes)
        replay_cap_reserve = max(
            0, align(inputs.replay_cap_bytes) - replay_payload - replay_mask - replay_state
        )
        packs = align(inputs.registry_slots * _PACKED_BYTES_PER_SLOT)
        reduction_arena = align(inputs.registry_slots * _PACKED_BYTES_PER_SLOT)
        chunk = min(unique, inputs.chunk_elements)
        secant_math_workspace = align(
            chunk
            * (_CANONICAL_SCRATCH_BYTES_PER_ELEMENT + _SECANT_INPUT_AND_MATH_BYTES_PER_ELEMENT)
        )
        owner_finish_workspace = align(
            chunk * _OWNER_FINISH_BYTES_PER_ELEMENT + _OWNER_FINISH_SCALAR_BYTES
        )
        cell_count = inputs.registry_slots // len(_SECANT_SLOT_SUFFIXES)
        quantile_sink_workspace = align(
            cell_count * (3 * 8 + 1) + (_TIER2_OUTPUT_SCALAR_COUNT + 24) * 8
        )
        workspace = max(secant_math_workspace, owner_finish_workspace, quantile_sink_workspace)
        retained = (
            fp32
            + bf16
            + fingerprints
            + replay_payload
            + replay_mask
            + replay_state
            + replay_cap_reserve
            + packs
            + reduction_arena
            + workspace
        )
        if retained > _MAX_REPRESENTABLE_BYTES:
            raise SecantMemoryEstimateError("retained estimate exceeds the representable cap")
        numerator, denominator = inputs.allocator_headroom_fraction.as_integer_ratio()
        fractional_headroom = (retained * numerator + denominator - 1) // denominator
        allocator_headroom = max(inputs.minimum_allocator_headroom_bytes, fractional_headroom)
        allocator_headroom = align(allocator_headroom)
        driver_numerator, driver_denominator = inputs.driver_headroom_fraction.as_integer_ratio()
        fractional_driver_headroom = (
            retained * driver_numerator + driver_denominator - 1
        ) // driver_denominator
        driver_headroom = max(inputs.minimum_driver_headroom_bytes, fractional_driver_headroom)
        driver_headroom = align(driver_headroom)
        if retained + allocator_headroom + driver_headroom > _MAX_REPRESENTABLE_BYTES:
            raise SecantMemoryEstimateError("peak estimate exceeds the representable cap")
        return cls(
            world_size=world_size,
            maximum_loaded_rank=maximum_rank,
            maximum_loaded_owner_elements=loaded,
            tied_alias_elements=tied,
            unique_owner_elements=unique,
            fp32_pre_or_delta_bytes=fp32,
            bf16_pre_bytes=bf16,
            post_fingerprint_bytes=fingerprints,
            replay_payload_bytes=replay_payload,
            replay_mask_bytes=replay_mask,
            replay_state_bytes=replay_state,
            replay_cap_reserve_bytes=replay_cap_reserve,
            packed_statistics_bytes=packs,
            reduction_arena_bytes=reduction_arena,
            bounded_accumulation_workspace_bytes=secant_math_workspace,
            owner_hash_restore_workspace_bytes=owner_finish_workspace,
            quantile_sink_workspace_bytes=quantile_sink_workspace,
            chunk_workspace_bytes=workspace,
            retained_bytes=retained,
            allocator_headroom_bytes=allocator_headroom,
            driver_headroom_bytes=driver_headroom,
            peak_bytes=retained + allocator_headroom + driver_headroom,
        )


def _align_bytes(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
