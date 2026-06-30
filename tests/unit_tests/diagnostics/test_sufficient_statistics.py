# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect

import pytest
import torch
import torch.distributed as dist

from megatron.training.diagnostics.accumulator import PackedSufficientStatistics
from megatron.training.diagnostics.schema import Tier0Reason


class _PeerReducer:
    def __init__(
        self, peer: PackedSufficientStatistics, expected_group: object
    ) -> None:
        self.peer = peer
        self.expected_group = expected_group
        self.operations: list[object] = []

    def __call__(
        self, tensor: torch.Tensor, *, op: object, group: object | None
    ) -> None:
        assert group is self.expected_group
        self.operations.append(op)
        if op == dist.ReduceOp.SUM:
            tensor.add_(self.peer.sum_pack)
        elif op == dist.ReduceOp.MAX:
            torch.maximum(tensor, self.peer.max_pack, out=tensor)
        elif op == dist.ReduceOp.MIN:
            torch.minimum(tensor, self.peer.min_pack, out=tensor)
        else:
            raise AssertionError(f"unexpected reduction operation: {op}")


def _accumulator(*slot_names: str) -> PackedSufficientStatistics:
    return PackedSufficientStatistics(tuple(slot_names), "cpu")


def test_unequal_populations_masks_and_microbatches_match_concatenated_reference() -> (
    None
):
    accumulator = _accumulator("activation")
    microbatches = (
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[1.0], [0.0]])),
        (torch.tensor([[5.0, 6.0]]), torch.tensor([[0.5]])),
        (
            torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            torch.tensor([[0.0], [1.0], [1.0]]),
        ),
    )
    for values, mask in microbatches:
        accumulator.add_masked_tensor("activation", values, mask=mask)
    accumulator.finalize_local_()

    selected_values = torch.tensor([1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0])
    selected_weights = torch.tensor([1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    expected_mean = (selected_values * selected_weights).sum() / selected_weights.sum()
    expected_rms = torch.sqrt(
        (selected_values.square() * selected_weights).sum() / selected_weights.sum()
    )

    torch.testing.assert_close(
        accumulator.mean("activation").value, expected_mean.double()
    )
    torch.testing.assert_close(
        accumulator.rms("activation").value, expected_rms.double()
    )
    torch.testing.assert_close(
        accumulator.maximum("activation").value, torch.tensor(12.0)
    )
    torch.testing.assert_close(
        accumulator.minimum("activation").value, torch.tensor(1.0)
    )


def test_zero_contributor_rank_and_injected_process_group_preserve_fixed_collectives() -> (
    None
):
    local = _accumulator("residual", "qkv")
    local.add_masked_tensor("residual", torch.empty(0))
    peer = _accumulator("residual", "qkv")
    peer.add_masked_tensor("qkv", torch.tensor([2.0, 4.0]))
    group = object()
    reducer = _PeerReducer(peer, group)

    local.reduce_(group=group, reducer=reducer)

    assert reducer.operations == [
        dist.ReduceOp.SUM,
        dist.ReduceOp.MAX,
        dist.ReduceOp.MIN,
    ]
    assert not local.rms("residual").valid
    assert local.rms("residual").reason == Tier0Reason.NO_CONTRIBUTORS
    torch.testing.assert_close(
        local.rms("qkv").value, torch.sqrt(torch.tensor(10.0, dtype=torch.float64))
    )
    torch.testing.assert_close(local.maximum("qkv").value, torch.tensor(4.0))
    torch.testing.assert_close(local.minimum("qkv").value, torch.tensor(2.0))


def test_tp_replicated_multiplicity_counts_each_logical_value_once() -> None:
    replicated = _accumulator("residual")
    values = torch.tensor([1.0, 3.0, 5.0])
    replicated.add_masked_tensor("residual", values, replication_multiplicity=2)
    replicated.add_masked_tensor("residual", values, replication_multiplicity=2)
    replicated.finalize_local_()

    reference = _accumulator("residual")
    reference.add_masked_tensor("residual", values)
    reference.finalize_local_()

    torch.testing.assert_close(replicated.sum_pack, reference.sum_pack)
    torch.testing.assert_close(
        replicated.rms("residual").value, reference.rms("residual").value
    )


def test_packed_sum_count_sumsq_zero_and_extrema_slots() -> None:
    accumulator = _accumulator("moments")
    accumulator.add_masked_tensor("moments", torch.tensor([-2.0, 0.0, 0.0, 4.0]))
    accumulator.finalize_local_()
    slots = accumulator.slots("moments")

    torch.testing.assert_close(
        accumulator.sum_pack[slots.sum], torch.tensor(2.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        accumulator.sum_pack[slots.count], torch.tensor(4.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        accumulator.sum_pack[slots.sumsq], torch.tensor(20.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        accumulator.zero_fraction("moments").value,
        torch.tensor(0.5, dtype=torch.float64),
    )
    torch.testing.assert_close(accumulator.maximum("moments").value, torch.tensor(4.0))
    torch.testing.assert_close(accumulator.minimum("moments").value, torch.tensor(-2.0))


def test_nonfinites_are_counted_but_invalidate_numeric_derivations() -> None:
    accumulator = _accumulator("health")
    accumulator.add_masked_tensor(
        "health", torch.tensor([0.0, torch.nan, torch.inf, 2.0])
    )
    accumulator.finalize_local_()

    rms = accumulator.rms("health")
    assert not rms.valid
    assert torch.isnan(rms.value)
    assert rms.reason == Tier0Reason.NONFINITE_INPUT
    torch.testing.assert_close(
        accumulator.nonfinite_fraction("health").value,
        torch.tensor(0.5, dtype=torch.float64),
    )


def test_zero_denominators_are_invalid_nan_not_zero() -> None:
    accumulator = _accumulator("response", "masked")
    accumulator.add_masked_pair("response", torch.ones(4), torch.zeros(4))
    accumulator.add_masked_tensor("masked", torch.ones(4), mask=torch.zeros(4))
    accumulator.finalize_local_()

    relative = accumulator.relative_rms("response")
    assert not relative.valid
    assert torch.isnan(relative.value)
    assert relative.reason == Tier0Reason.ZERO_DENOMINATOR

    masked = accumulator.rms("masked")
    assert not masked.valid
    assert torch.isnan(masked.value)
    assert masked.reason == Tier0Reason.NO_CONTRIBUTORS


def test_pooled_cosine_avoids_known_mean_of_rank_cosines_sign_flip() -> None:
    rank0 = _accumulator("cosine")
    rank0.add_masked_pair("cosine", torch.ones(10), torch.ones(10))
    rank1 = _accumulator("cosine")
    rank1.add_masked_pair(
        "cosine", torch.tensor([10.0, 0.0]), torch.tensor([-8.0, 6.0])
    )

    rank0_local = _accumulator("cosine")
    rank0_local.sum_pack.copy_(rank0.sum_pack)
    rank0_local.max_pack.copy_(rank0.max_pack)
    rank0_local.min_pack.copy_(rank0.min_pack)
    rank0_local.finalize_local_()
    rank1.finalize_local_()
    mean_of_local_cosines = (
        rank0_local.cosine("cosine").value + rank1.cosine("cosine").value
    ) / 2

    group = object()
    rank0.reduce_(group=group, reducer=_PeerReducer(rank1, group))

    assert mean_of_local_cosines > 0
    assert rank0.cosine("cosine").value < 0


@pytest.mark.parametrize(
    "dtype,tolerance", [(torch.float32, 1e-6), (torch.bfloat16, 1e-3)]
)
def test_fp32_and_bf16_inputs_accumulate_with_expected_tolerance(
    dtype: torch.dtype, tolerance: float
) -> None:
    values = torch.linspace(-2.0, 3.0, 257, dtype=torch.float32).to(dtype)
    accumulator = _accumulator("values")
    accumulator.add_masked_tensor("values", values)
    accumulator.finalize_local_()

    reference = torch.sqrt(values.float().square().mean(dtype=torch.float64))
    torch.testing.assert_close(
        accumulator.rms("values").value, reference, rtol=tolerance, atol=tolerance
    )


def test_update_relative_rms_uses_delta_only_after_accumulation() -> None:
    before = torch.tensor([1.0, 2.0, 4.0])
    after = torch.tensor([2.0, 1.0, 4.0])
    accumulator = _accumulator("update")
    accumulator.add_update("update", before, after)
    with pytest.raises(RuntimeError, match="must be reduced"):
        accumulator.relative_rms("update")
    accumulator.finalize_local_()

    expected = torch.linalg.vector_norm(after - before) / torch.linalg.vector_norm(
        before
    )
    torch.testing.assert_close(
        accumulator.relative_rms("update").value, expected.double()
    )


def test_accumulation_source_has_no_host_scalar_transfers() -> None:
    source = inspect.getsource(PackedSufficientStatistics.add_masked_tensor)
    source += inspect.getsource(PackedSufficientStatistics.add_masked_pair)
    source += inspect.getsource(PackedSufficientStatistics.add_update)
    assert ".item(" not in source
    assert ".cpu(" not in source
    assert "float(" not in source
