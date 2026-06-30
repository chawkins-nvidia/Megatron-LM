# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
from dataclasses import replace

import pytest
import torch

from megatron.training.diagnostics.accumulator import (
    PackedSlots,
    PackedSufficientStatistics,
)
from megatron.training.diagnostics.registry import (
    DenominatorKind,
    MaskKind,
    MetricDescriptor,
    MetricFamily,
    MetricRegistry,
    NormalizationKind,
    Ownership,
    PartitionAxis,
    ReductionKind,
    ReplicationAxis,
    StatisticKind,
)


def _descriptors() -> tuple[MetricDescriptor, ...]:
    return (
        MetricDescriptor(
            logical_name="activation/residual/layer_0",
            family=MetricFamily.RESIDUAL,
            global_layer=0,
            partition_axes=(
                PartitionAxis.DATA_SAMPLE,
                PartitionAxis.CONTEXT_SEQUENCE,
                PartitionAxis.PIPELINE_LAYER,
            ),
            replication_axes=(ReplicationAxis.TENSOR,),
            replication_multiplicity=2,
            ownership=Ownership.TENSOR_PARALLEL_RANK_ZERO,
            mask_kind=MaskKind.TOKEN,
            statistic_kind=StatisticKind.TENSOR_MOMENTS,
            denominator_kind=DenominatorKind.SELECTED_ELEMENTS,
            normalization_kind=NormalizationKind.NONE,
            process_group_identity="diagnostic_world",
            reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
            tied_owner_identity=None,
            packed_slots=PackedSlots.for_index(0),
        ),
        MetricDescriptor(
            logical_name="update/fc1/layer_7",
            family=MetricFamily.FC1,
            global_layer=7,
            partition_axes=(
                PartitionAxis.OPTIMIZER_SHARD,
                PartitionAxis.TENSOR_FEATURE,
                PartitionAxis.PIPELINE_LAYER,
            ),
            replication_axes=(),
            replication_multiplicity=1,
            ownership=Ownership.AUTHORITATIVE_SHARD,
            mask_kind=MaskKind.PARAMETER,
            statistic_kind=StatisticKind.UPDATE,
            denominator_kind=DenominatorKind.PRE_UPDATE_SUMSQ,
            normalization_kind=NormalizationKind.NONE,
            process_group_identity="diagnostic_world",
            reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
            tied_owner_identity="embedding.first_pipeline_stage",
            packed_slots=PackedSlots.for_index(1),
        ),
    )


def test_registry_preserves_complete_descriptor_semantics() -> None:
    registry = MetricRegistry(_descriptors())
    residual, update = registry.descriptors

    assert residual.global_layer == 0
    assert PartitionAxis.CONTEXT_SEQUENCE in residual.partition_axes
    assert PartitionAxis.PIPELINE_LAYER in residual.partition_axes
    assert residual.replication_axes == (ReplicationAxis.TENSOR,)
    assert residual.replication_multiplicity == 2
    assert residual.statistic_kind == StatisticKind.TENSOR_MOMENTS
    assert residual.denominator_kind == DenominatorKind.SELECTED_ELEMENTS
    assert residual.normalization_kind == NormalizationKind.NONE
    assert residual.process_group_identity == "diagnostic_world"
    assert residual.reduction_kind == ReductionKind.PACKED_SUM_MAX_MIN
    assert update.ownership == Ownership.AUTHORITATIVE_SHARD
    assert update.tied_owner_identity == "embedding.first_pipeline_stage"


def test_registry_enforces_registered_multiplicity_and_operation_kind() -> None:
    registry = MetricRegistry(_descriptors())
    accumulator = registry.new_accumulator("cpu")
    registry.add_masked_tensor(
        accumulator, "activation/residual/layer_0", torch.tensor([2.0, 4.0])
    )
    slots = accumulator.slots("activation/residual/layer_0")

    assert (
        "replication_multiplicity"
        not in inspect.signature(registry.add_masked_tensor).parameters
    )
    torch.testing.assert_close(
        accumulator.sum_pack[slots.count], torch.tensor(1.0, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="requires tensor_moments"):
        registry.add_update(
            accumulator,
            "activation/residual/layer_0",
            torch.ones(1),
            torch.ones(1),
        )


def test_owner_aware_add_update_accumulates_only_authoritative_shards() -> None:
    before = torch.tensor([2.0, 4.0])
    after = torch.tensor([3.0, 2.0])

    authoritative = MetricRegistry(_descriptors(), local_owners=(False, True))
    authoritative_accumulator = authoritative.new_accumulator("cpu")
    authoritative.add_update(
        authoritative_accumulator, "update/fc1/layer_7", before, after
    )
    authoritative_accumulator.finalize_local_()
    torch.testing.assert_close(
        authoritative_accumulator.relative_rms("update/fc1/layer_7").value,
        torch.tensor(0.5, dtype=torch.float64),
    )

    nonowner = MetricRegistry(_descriptors(), local_owners=(False, False))
    nonowner_accumulator = nonowner.new_accumulator("cpu")
    nonowner.add_update(nonowner_accumulator, "update/fc1/layer_7", before, after)
    nonowner_accumulator.finalize_local_()
    statistic = nonowner_accumulator.relative_rms("update/fc1/layer_7")
    assert not statistic.valid
    assert torch.isnan(statistic.value)


def test_nonowning_rank_retains_neutral_fixed_slots() -> None:
    registry = MetricRegistry(_descriptors(), local_owners=(False, True))
    accumulator = registry.new_accumulator("cpu")
    registry.add_masked_tensor(
        accumulator, "activation/residual/layer_0", torch.tensor([9.0])
    )
    registry.add_update(
        accumulator,
        "update/fc1/layer_7",
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 4.0]),
    )
    accumulator.finalize_local_()

    assert accumulator.slot_names == (
        "activation/residual/layer_0",
        "update/fc1/layer_7",
    )
    assert not accumulator.rms("activation/residual/layer_0").valid
    torch.testing.assert_close(
        accumulator.relative_rms("update/fc1/layer_7").value,
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_descriptor_hash_is_deterministic_rank_independent_and_semantically_complete() -> (
    None
):
    first = MetricRegistry(_descriptors(), local_owners=(True, False))
    second = MetricRegistry(_descriptors(), local_owners=(False, True))
    assert first.descriptor_hash == second.descriptor_hash
    assert len(first.descriptor_hash) == 64

    residual, update = _descriptors()
    mutations = (
        (replace(residual, replication_multiplicity=1), update),
        (
            replace(
                residual,
                statistic_kind=StatisticKind.PAIR_MOMENTS,
                denominator_kind=DenominatorKind.RHS_SUMSQ,
            ),
            update,
        ),
        (
            replace(
                residual,
                normalization_kind=NormalizationKind.GLOBAL_VALID_TOKENS,
            ),
            update,
        ),
        (residual, replace(update, tied_owner_identity="output.last_pipeline_stage")),
        (
            replace(residual, process_group_identity="data_parallel"),
            replace(update, process_group_identity="data_parallel"),
        ),
        (
            replace(
                residual,
                reduction_kind=ReductionKind.HIERARCHICAL_PACKED_SUM_MAX_MIN,
            ),
            replace(
                update,
                reduction_kind=ReductionKind.HIERARCHICAL_PACKED_SUM_MAX_MIN,
            ),
        ),
    )
    for descriptors in mutations:
        assert MetricRegistry(descriptors).descriptor_hash != first.descriptor_hash


def test_registry_rejects_same_names_with_different_descriptor_identity() -> None:
    first = MetricRegistry(_descriptors())
    changed = list(_descriptors())
    changed[0] = replace(changed[0], replication_multiplicity=1)
    second = MetricRegistry(changed)
    accumulator = first.new_accumulator("cpu")

    assert first.slot_names == second.slot_names
    with pytest.raises(ValueError, match="descriptor/schema identity"):
        second.add_masked_tensor(
            accumulator, "activation/residual/layer_0", torch.ones(1)
        )


def test_registry_binds_and_enforces_schema_identity() -> None:
    registry = MetricRegistry(_descriptors())
    accumulator = registry.new_accumulator("cpu")
    assert accumulator.descriptor_hash == registry.descriptor_hash
    assert accumulator.schema_identity == "diag/v2"

    foreign_schema = PackedSufficientStatistics(
        registry.slot_names,
        "cpu",
        descriptor_hash=registry.descriptor_hash,
        schema_identity="diag/v3",
        process_group_identity=registry.process_group_identity,
        reduction_identity=registry.reduction_kind.value,
    )
    with pytest.raises(ValueError, match="descriptor/schema identity"):
        registry.add_masked_tensor(
            foreign_schema, "activation/residual/layer_0", torch.ones(1)
        )


def test_packed_slot_order_is_static_and_complete() -> None:
    registry = MetricRegistry(_descriptors())
    accumulator = registry.new_accumulator("cpu")

    assert accumulator.slots(0) == PackedSlots(
        sum=0,
        count=1,
        sumsq=2,
        dot=3,
        lhs_sumsq=4,
        rhs_sumsq=5,
        zero=6,
        nonfinite=7,
        mask_error=8,
        nonfinite_arithmetic=9,
        maximum=0,
        minimum=0,
    )
    assert accumulator.slots(1) == PackedSlots.for_index(1)
