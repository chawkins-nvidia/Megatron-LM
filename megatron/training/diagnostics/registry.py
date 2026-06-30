# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed metric descriptors and deterministic Tier-0 registry."""

import hashlib
import json
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Iterable, Protocol

import torch

from .accumulator import (
    PackedSlots,
    PackedSufficientStatistics,
    ProcessGroupIdentity,
    ReductionBinding,
    ReductionKind,
)
from .schema import SCHEMA_PREFIX


class MetricFamily(StrEnum):
    """Stable logical tensor families used by Tier-0 diagnostics."""

    RESIDUAL = "residual"
    QKV = "qkv"
    ATTN_OUT = "attn_out"
    FC1 = "fc1"
    FC2 = "fc2"
    NORM = "norm"
    EMBEDDING = "embedding"
    OUTPUT = "output"
    EVENT = "event"


class PartitionAxis(StrEnum):
    """Axes along which a logical observation is uniquely partitioned."""

    DATA_SAMPLE = "data_sample"
    TENSOR_FEATURE = "tensor_feature"
    PIPELINE_LAYER = "pipeline_layer"
    CONTEXT_SEQUENCE = "context_sequence"
    EXPERT = "expert"
    OPTIMIZER_SHARD = "optimizer_shard"


class ReplicationAxis(StrEnum):
    """Axes along which local values may duplicate one logical observation."""

    DATA = "data"
    TENSOR = "tensor"
    PIPELINE_TIED = "pipeline_tied"
    CONTEXT = "context"
    EXPERT = "expert"


class Ownership(StrEnum):
    """Static rule that selects unique contributors for a descriptor."""

    EVERY_RANK = "every_rank"
    PIPELINE_STAGE = "pipeline_stage"
    TENSOR_PARALLEL_RANK_ZERO = "tensor_parallel_rank_zero"
    AUTHORITATIVE_SHARD = "authoritative_shard"
    TIED_PARAMETER_OWNER = "tied_parameter_owner"


class MaskKind(StrEnum):
    """Mask semantics expected by a metric descriptor."""

    NONE = "none"
    TOKEN = "token"
    SEQUENCE_PARALLEL_TOKEN = "sequence_parallel_token"
    PARAMETER = "parameter"


class StatisticKind(StrEnum):
    """Sufficient-statistic operation accepted by a metric descriptor."""

    TENSOR_MOMENTS = "tensor_moments"
    PAIR_MOMENTS = "pair_moments"
    UPDATE = "update"


class DenominatorKind(StrEnum):
    """Canonical denominator represented by a descriptor's packed moments."""

    SELECTED_ELEMENTS = "selected_elements"
    RHS_SUMSQ = "rhs_sumsq"
    PRE_UPDATE_SUMSQ = "pre_update_sumsq"


class NormalizationKind(StrEnum):
    """Normalization applied before or during sufficient-statistic pooling."""

    NONE = "none"
    GLOBAL_VALID_TOKENS = "global_valid_tokens"
    LOSS_SCALE_AND_GLOBAL_VALID_TOKENS = "loss_scale_and_global_valid_tokens"


class MetricNormalizationAdapter(Protocol):
    """Apply one declared normalization to reduced packed moments."""

    kind: NormalizationKind
    identity: str

    def normalize_(
        self, accumulator: PackedSufficientStatistics, slot: int, global_valid_tokens: torch.Tensor
    ) -> None:
        """Normalize one reduced packed slot in place."""

        ...


@dataclass(frozen=True)
class MetricDescriptor:
    """Describe one logical metric's complete, rank-independent semantics.

    Attributes:
        logical_name: Stable logical metric name and packed-slot identity.
        family: Logical model tensor family.
        global_layer: Global layer index, or ``None`` for non-layer metrics.
        partition_axes: Axes that contain distinct logical observations.
        replication_axes: Axes that may repeat logical observations.
        replication_multiplicity: Number of repeated copies corrected during pooling.
        ownership: Static rule used to choose authoritative contributors.
        mask_kind: Expected masking semantics.
        statistic_kind: Accumulation operation permitted for the descriptor.
        denominator_kind: Meaning of the packed denominator.
        normalization_kind: Pre-pooling loss or token normalization contract.
        process_group_identity: Stable name of the collective process group.
        reduction_kind: Collective operations applied to the packed buffers.
        tied_owner_identity: Stable tied-parameter owner identity, when applicable.
        packed_slots: Canonical offsets occupied in the packed buffers.
    """

    logical_name: str
    family: MetricFamily
    global_layer: int | None
    partition_axes: tuple[PartitionAxis, ...]
    replication_axes: tuple[ReplicationAxis, ...]
    replication_multiplicity: int
    ownership: Ownership
    mask_kind: MaskKind
    statistic_kind: StatisticKind
    denominator_kind: DenominatorKind
    normalization_kind: NormalizationKind
    process_group_identity: ProcessGroupIdentity
    reduction_kind: ReductionKind
    tied_owner_identity: str | None
    packed_slots: PackedSlots

    def __post_init__(self) -> None:
        """Validate the static descriptor contract.

        Raises:
            ValueError: If an identity, axis, multiplicity, or statistic contract is invalid.
        """

        if not self.logical_name:
            raise ValueError("a metric descriptor requires a logical name")
        if self.global_layer is not None and self.global_layer < 0:
            raise ValueError("global layer indices must be nonnegative")
        if len(self.partition_axes) != len(set(self.partition_axes)):
            raise ValueError("partition axes must be unique")
        if len(self.replication_axes) != len(set(self.replication_axes)):
            raise ValueError("replication axes must be unique")
        if self.replication_multiplicity <= 0:
            raise ValueError("replication multiplicity must be positive")
        if not self.process_group_identity:
            raise ValueError("a metric descriptor requires a process-group identity")
        if self.ownership == Ownership.TIED_PARAMETER_OWNER and not self.tied_owner_identity:
            raise ValueError("tied-parameter ownership requires a tied-owner identity")
        if self.tied_owner_identity is not None and not self.tied_owner_identity:
            raise ValueError("tied-owner identities must be nonempty")
        if not isinstance(self.process_group_identity, ProcessGroupIdentity):
            raise ValueError("metric descriptors require a typed process-group identity")
        if not isinstance(self.reduction_kind, ReductionKind):
            raise ValueError("metric descriptors require a typed reduction kind")
        if self.mask_kind == MaskKind.PARAMETER:
            raise ValueError("parameter and unmasked metrics must use mask kind none")
        if (
            self.normalization_kind != NormalizationKind.NONE
            and self.statistic_kind != StatisticKind.TENSOR_MOMENTS
        ):
            raise ValueError("normalization adapters only support tensor moments")
        if self.process_group_identity != ProcessGroupIdentity.WORLD:
            raise ValueError("only world diagnostic reduction is implemented")
        if self.reduction_kind != ReductionKind.PACKED_SUM_MAX_MIN:
            raise ValueError("hierarchical diagnostic reduction is not implemented")

        expected_denominator = {
            StatisticKind.TENSOR_MOMENTS: DenominatorKind.SELECTED_ELEMENTS,
            StatisticKind.PAIR_MOMENTS: DenominatorKind.RHS_SUMSQ,
            StatisticKind.UPDATE: DenominatorKind.PRE_UPDATE_SUMSQ,
        }[self.statistic_kind]
        if self.denominator_kind != expected_denominator:
            raise ValueError(
                f"{self.statistic_kind.value} requires denominator {expected_denominator.value}"
            )


class MetricRegistry:
    """Hold a fixed descriptor sequence plus rank-local ownership decisions."""

    def __init__(
        self,
        descriptors: Iterable[MetricDescriptor],
        *,
        reduction_binding: ReductionBinding,
        local_owners: Iterable[bool] | None = None,
        normalization_adapters: Iterable[MetricNormalizationAdapter] = (),
    ) -> None:
        """Validate descriptors and retain neutral slots for non-owning ranks.

        Args:
            descriptors: Rank-independent descriptors in collective slot order.
            reduction_binding: Typed runtime binding for this packed registry.
            local_owners: Optional rank-local contribution decision per descriptor.
            normalization_adapters: Explicit typed adapters for non-NONE normalization kinds.

        Raises:
            ValueError: If descriptors, slots, reductions, or ownership lengths disagree.
        """

        if reduction_binding is None:
            raise ValueError("a metric registry requires a reduction binding")
        self.descriptors = tuple(descriptors)
        names = tuple(descriptor.logical_name for descriptor in self.descriptors)
        if len(names) != len(set(names)):
            raise ValueError("metric descriptor logical names must be unique")
        for index, descriptor in enumerate(self.descriptors):
            if descriptor.packed_slots != PackedSlots.for_index(index):
                raise ValueError("descriptor packed slots must follow canonical registry order")

        process_groups = {descriptor.process_group_identity for descriptor in self.descriptors}
        reductions = {descriptor.reduction_kind for descriptor in self.descriptors}
        if len(process_groups) > 1 or len(reductions) > 1:
            raise ValueError(
                "one packed registry must use one process-group and reduction identity"
            )
        self.process_group_identity = next(iter(process_groups), ProcessGroupIdentity.WORLD)
        self.reduction_kind = next(iter(reductions), ReductionKind.PACKED_SUM_MAX_MIN)
        if (
            self.process_group_identity != ProcessGroupIdentity.WORLD
            or self.reduction_kind != ReductionKind.PACKED_SUM_MAX_MIN
        ):
            raise ValueError("only the flat packed world SUM/MAX/MIN reduction is implemented")
        if (
            reduction_binding.process_group_identity != self.process_group_identity
            or reduction_binding.reduction_kind != self.reduction_kind
        ):
            raise ValueError("reduction binding does not match the metric descriptor contract")
        self.reduction_binding = reduction_binding

        adapters = tuple(normalization_adapters)
        if any(
            not isinstance(adapter.kind, NormalizationKind)
            or adapter.kind == NormalizationKind.NONE
            or not adapter.identity
            for adapter in adapters
        ):
            raise ValueError("normalization adapters require typed non-NONE identities")
        if len({adapter.kind for adapter in adapters}) != len(adapters):
            raise ValueError("normalization adapter kinds must be unique")
        self.normalization_adapters = {adapter.kind: adapter for adapter in adapters}
        required_normalizations = {
            descriptor.normalization_kind
            for descriptor in self.descriptors
            if descriptor.normalization_kind != NormalizationKind.NONE
        }
        missing_normalizations = required_normalizations - self.normalization_adapters.keys()
        if missing_normalizations:
            missing = sorted(kind.value for kind in missing_normalizations)
            raise ValueError(
                f"non-none normalization requires a typed normalization adapter: {missing}"
            )

        owners = (
            tuple(True for _ in self.descriptors) if local_owners is None else tuple(local_owners)
        )
        if len(owners) != len(self.descriptors):
            raise ValueError("local ownership must have one entry per descriptor")
        self.local_owners = owners
        self._indices = {name: index for index, name in enumerate(names)}
        self._slot_names = names
        self._descriptor_hash = self._serialize_descriptor_hash()

    @property
    def slot_names(self) -> tuple[str, ...]:
        """Return the fixed collective slot order."""

        return self._slot_names

    @property
    def descriptor_hash(self) -> str:
        """Return a SHA-256 hash of all static, rank-independent semantics."""

        return self._descriptor_hash

    def _serialize_descriptor_hash(self) -> str:
        records = []
        for descriptor in self.descriptors:
            records.append(
                {
                    "denominator_kind": descriptor.denominator_kind.value,
                    "family": descriptor.family.value,
                    "global_layer": descriptor.global_layer,
                    "logical_name": descriptor.logical_name,
                    "mask_kind": descriptor.mask_kind.value,
                    "normalization_kind": descriptor.normalization_kind.value,
                    "normalization_adapter": (
                        self.normalization_adapters[descriptor.normalization_kind].identity
                        if descriptor.normalization_kind != NormalizationKind.NONE
                        else None
                    ),
                    "ownership": descriptor.ownership.value,
                    "packed_slots": {
                        packed_field.name: getattr(descriptor.packed_slots, packed_field.name)
                        for packed_field in fields(PackedSlots)
                    },
                    "partition_axes": sorted(axis.value for axis in descriptor.partition_axes),
                    "process_group_identity": descriptor.process_group_identity,
                    "reduction_kind": descriptor.reduction_kind.value,
                    "replication_axes": sorted(axis.value for axis in descriptor.replication_axes),
                    "replication_multiplicity": descriptor.replication_multiplicity,
                    "statistic_kind": descriptor.statistic_kind.value,
                    "tied_owner_identity": descriptor.tied_owner_identity,
                }
            )
        encoded = json.dumps(
            {"descriptors": records, "schema_identity": SCHEMA_PREFIX.rstrip("/")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def new_accumulator(self, device: torch.device | str) -> PackedSufficientStatistics:
        """Allocate neutral packs bound to this registry's complete identity.

        Args:
            device: CPU or CUDA device on which accumulation and reduction occur.

        Returns:
            An empty accumulator carrying this registry's descriptor and schema identity.
        """

        return PackedSufficientStatistics(
            self._slot_names,
            device,
            descriptor_hash=self._descriptor_hash,
            reduction_binding=self.reduction_binding,
            schema_identity=SCHEMA_PREFIX.rstrip("/"),
        )

    def owns(self, logical_name: str) -> bool:
        """Return whether this rank contributes to a logical metric.

        Args:
            logical_name: Registered logical metric name.

        Returns:
            Whether the rank-local ownership decision is authoritative.
        """

        return self.local_owners[self._index(logical_name)]

    def add_masked_tensor(
        self,
        accumulator: PackedSufficientStatistics,
        logical_name: str,
        values: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate an owned tensor under its registered semantics.

        Args:
            accumulator: Registry-bound packed accumulator.
            logical_name: Registered tensor-moment descriptor name.
            values: Local tensor observation.
            mask: Optional same-device broadcastable nonnegative weights.

        Raises:
            ValueError: If the accumulator or requested operation conflicts with the descriptor.
        """

        index, descriptor = self._operation(
            accumulator, logical_name, StatisticKind.TENSOR_MOMENTS, mask
        )
        if self.local_owners[index]:
            accumulator.add_masked_tensor(
                index,
                values,
                mask=mask,
                replication_multiplicity=descriptor.replication_multiplicity,
                require_mask=descriptor.mask_kind
                in (MaskKind.TOKEN, MaskKind.SEQUENCE_PARALLEL_TOKEN),
            )

    def add_masked_pair(
        self,
        accumulator: PackedSufficientStatistics,
        logical_name: str,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate an owned pair under its registered semantics.

        Args:
            accumulator: Registry-bound packed accumulator.
            logical_name: Registered pair-moment descriptor name.
            lhs: Numerator-side tensor observation.
            rhs: Denominator-side tensor observation.
            mask: Optional same-device broadcastable nonnegative weights.

        Raises:
            ValueError: If the accumulator or requested operation conflicts with the descriptor.
        """

        index, descriptor = self._operation(
            accumulator, logical_name, StatisticKind.PAIR_MOMENTS, mask
        )
        if self.local_owners[index]:
            accumulator.add_masked_pair(
                index,
                lhs,
                rhs,
                mask=mask,
                replication_multiplicity=descriptor.replication_multiplicity,
                require_mask=descriptor.mask_kind
                in (MaskKind.TOKEN, MaskKind.SEQUENCE_PARALLEL_TOKEN),
            )

    def add_update(
        self,
        accumulator: PackedSufficientStatistics,
        logical_name: str,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate an authoritative update or preserve neutral non-owner slots.

        Args:
            accumulator: Registry-bound packed accumulator.
            logical_name: Registered update descriptor name.
            before: Authoritative pre-update tensor.
            after: Authoritative post-update tensor.
            mask: Optional same-device broadcastable nonnegative weights.

        Raises:
            ValueError: If the accumulator or requested operation conflicts with the descriptor.
        """

        index, descriptor = self._operation(accumulator, logical_name, StatisticKind.UPDATE, mask)
        if self.local_owners[index]:
            accumulator.add_update(
                index,
                before,
                after,
                mask=mask,
                replication_multiplicity=descriptor.replication_multiplicity,
                require_mask=descriptor.mask_kind
                in (MaskKind.TOKEN, MaskKind.SEQUENCE_PARALLEL_TOKEN),
            )

    def add_applied_update(
        self,
        accumulator: PackedSufficientStatistics,
        logical_name: str,
        master_before: torch.Tensor,
        master_after: torch.Tensor,
        applied_before: torch.Tensor,
        applied_after: torch.Tensor,
    ) -> None:
        """Accumulate owned master/applied update-retention sufficient statistics.

        Args:
            accumulator: Registry-bound packed accumulator.
            logical_name: Registered update descriptor name.
            master_before: Pre-step authoritative FP32 owner shard.
            master_after: Post-step authoritative FP32 owner shard.
            applied_before: Pre-step BF16 applied owner shard.
            applied_after: Post-materialization BF16 applied owner shard.

        Raises:
            ValueError: If the accumulator or operation conflicts with the descriptor.
        """

        index, descriptor = self._operation(accumulator, logical_name, StatisticKind.UPDATE, None)
        if self.local_owners[index]:
            accumulator.add_applied_update(
                index,
                master_before,
                master_after,
                applied_before,
                applied_after,
                replication_multiplicity=descriptor.replication_multiplicity,
            )

    def add_applied_update_moments(
        self, accumulator: PackedSufficientStatistics, logical_name: str, **moments: torch.Tensor
    ) -> None:
        """Accumulate precomputed bounded-chunk applied-update moments.

        Args:
            accumulator: Registry-bound packed accumulator.
            logical_name: Registered update descriptor name.
            **moments: Scalar arguments accepted by
                :meth:`PackedSufficientStatistics.add_applied_update_moments`.

        Raises:
            ValueError: If the accumulator or operation conflicts with the descriptor.
        """

        index, descriptor = self._operation(accumulator, logical_name, StatisticKind.UPDATE, None)
        if self.local_owners[index]:
            accumulator.add_applied_update_moments(
                index, **moments, replication_multiplicity=descriptor.replication_multiplicity
            )

    def mark_mask_error(
        self,
        accumulator: PackedSufficientStatistics,
        logical_name: str,
        error: torch.Tensor | None = None,
    ) -> None:
        """Record a packed mask error for an owned descriptor.

        Args:
            accumulator: Registry-bound packed accumulator.
            logical_name: Registered descriptor name.
            error: Optional device-resident scalar error flag.
        """

        index = self._index(logical_name)
        self._validate_accumulator(accumulator)
        if self.local_owners[index]:
            accumulator.mark_mask_error(index, error)

    def mark_observation_error(
        self,
        accumulator: PackedSufficientStatistics,
        logical_name: str,
        error: torch.Tensor | None = None,
    ) -> None:
        """Record packed callback-completeness failure for an owned descriptor.

        Args:
            accumulator: Registry-bound packed accumulator.
            logical_name: Registered descriptor name.
            error: Optional device-resident scalar error flag.
        """

        index = self._index(logical_name)
        self._validate_accumulator(accumulator)
        if self.local_owners[index]:
            accumulator.mark_observation_error(index, error)

    def apply_normalizations_(
        self, accumulator: PackedSufficientStatistics, *, global_valid_tokens: torch.Tensor
    ) -> None:
        """Apply all explicit adapters after the packed world reduction.

        Args:
            accumulator: Reduced registry-bound packed accumulator.
            global_valid_tokens: Globally pooled valid-token scalar.

        Raises:
            RuntimeError: If the accumulator has not completed reduction.
        """

        self._validate_accumulator(accumulator)
        if not accumulator.reduced:
            raise RuntimeError("normalization requires reduced packed statistics")
        for index, descriptor in enumerate(self.descriptors):
            if descriptor.normalization_kind != NormalizationKind.NONE:
                self.normalization_adapters[descriptor.normalization_kind].normalize_(
                    accumulator, index, global_valid_tokens
                )

    def _operation(
        self,
        accumulator: PackedSufficientStatistics,
        logical_name: str,
        statistic_kind: StatisticKind,
        mask: torch.Tensor | None,
    ) -> tuple[int, MetricDescriptor]:
        index = self._index(logical_name)
        self._validate_accumulator(accumulator)
        descriptor = self.descriptors[index]
        if descriptor.statistic_kind != statistic_kind:
            raise ValueError(
                f"{logical_name} requires {descriptor.statistic_kind.value}, "
                f"not {statistic_kind.value}"
            )
        if descriptor.mask_kind == MaskKind.NONE and mask is not None:
            raise ValueError(f"{logical_name} does not accept a mask")
        return index, descriptor

    def _index(self, logical_name: str) -> int:
        try:
            return self._indices[logical_name]
        except KeyError as error:
            raise KeyError(f"unknown metric descriptor: {logical_name}") from error

    def _validate_accumulator(self, accumulator: PackedSufficientStatistics) -> None:
        if accumulator.slot_names is not self._slot_names:
            raise ValueError("accumulator slot order does not match the metric registry")
        if (
            accumulator.schema_identity != SCHEMA_PREFIX.rstrip("/")
            or accumulator.descriptor_hash != self._descriptor_hash
            or accumulator.reduction_binding is not self.reduction_binding
        ):
            raise ValueError(
                "accumulator descriptor/schema identity does not match the metric registry"
            )
