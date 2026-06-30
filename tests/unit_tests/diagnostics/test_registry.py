# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.training.diagnostics.accumulator import PackedSlots
from megatron.training.diagnostics.registry import (
    MaskKind,
    MetricDescriptor,
    MetricFamily,
    MetricRegistry,
    Ownership,
    PartitionAxis,
    ReplicationAxis,
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
            ownership=Ownership.TENSOR_PARALLEL_RANK_ZERO,
            mask_kind=MaskKind.TOKEN,
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
            ownership=Ownership.AUTHORITATIVE_SHARD,
            mask_kind=MaskKind.PARAMETER,
            packed_slots=PackedSlots.for_index(1),
        ),
    )


def test_registry_preserves_cp_pp_and_ownership_metadata() -> None:
    registry = MetricRegistry(_descriptors())
    residual, update = registry.descriptors

    assert residual.global_layer == 0
    assert PartitionAxis.CONTEXT_SEQUENCE in residual.partition_axes
    assert PartitionAxis.PIPELINE_LAYER in residual.partition_axes
    assert residual.replication_axes == (ReplicationAxis.TENSOR,)
    assert residual.ownership == Ownership.TENSOR_PARALLEL_RANK_ZERO
    assert update.global_layer == 7
    assert update.ownership == Ownership.AUTHORITATIVE_SHARD


def test_nonowning_rank_retains_neutral_fixed_slots() -> None:
    registry = MetricRegistry(_descriptors(), local_owners=(False, True))
    accumulator = registry.new_accumulator("cpu")
    registry.add_masked_tensor(
        accumulator, "activation/residual/layer_0", torch.tensor([9.0])
    )
    registry.add_masked_tensor(
        accumulator, "update/fc1/layer_7", torch.tensor([2.0, 4.0])
    )
    accumulator.finalize_local_()

    assert accumulator.slot_names == (
        "activation/residual/layer_0",
        "update/fc1/layer_7",
    )
    assert not accumulator.rms("activation/residual/layer_0").valid
    torch.testing.assert_close(
        accumulator.rms("update/fc1/layer_7").value,
        torch.sqrt(torch.tensor(10.0, dtype=torch.float64)),
    )


def test_descriptor_hash_is_deterministic_and_rank_independent() -> None:
    first = MetricRegistry(_descriptors(), local_owners=(True, False))
    second = MetricRegistry(_descriptors(), local_owners=(False, True))
    assert first.descriptor_hash == second.descriptor_hash
    assert (
        first.descriptor_hash
        == "01a8cdd7b2d744830b9e7218e45071da6d5094d882ea59de6025bef4d5aed460"
    )

    changed_descriptors = list(_descriptors())
    original = changed_descriptors[1]
    changed_descriptors[1] = MetricDescriptor(
        logical_name=original.logical_name,
        family=original.family,
        global_layer=8,
        partition_axes=original.partition_axes,
        replication_axes=original.replication_axes,
        ownership=original.ownership,
        mask_kind=original.mask_kind,
        packed_slots=original.packed_slots,
    )
    assert MetricRegistry(changed_descriptors).descriptor_hash != first.descriptor_hash


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
        maximum=0,
        minimum=0,
    )
    assert accumulator.slots(1) == PackedSlots.for_index(1)
