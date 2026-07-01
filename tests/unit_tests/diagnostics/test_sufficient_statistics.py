# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ast
import inspect

import pytest
import torch
import torch.distributed as dist

import megatron.training.diagnostics.accumulator as accumulator_module
from megatron.training.diagnostics.accumulator import (
    DEFAULT_MOMENT_SCRATCH_ELEMENT_CAPACITY,
    PackedSufficientStatistics,
    ReductionBinding,
)
from megatron.training.diagnostics.schema import Tier0Reason


class _PeerReducer:
    def __init__(self, peer: PackedSufficientStatistics, expected_group: object) -> None:
        self.peer = peer
        self.expected_group = expected_group
        self.operations: list[object] = []

    def __call__(self, tensor: torch.Tensor, *, op: object, group: object | None) -> None:
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


def _accumulator(
    *slot_names: str, reduction_binding: ReductionBinding | None = None
) -> PackedSufficientStatistics:
    return PackedSufficientStatistics(
        tuple(slot_names),
        "cpu",
        descriptor_hash=f"test:{slot_names!r}",
        reduction_binding=(
            ReductionBinding.flat_world(None) if reduction_binding is None else reduction_binding
        ),
    )


def test_unequal_populations_masks_and_microbatches_match_concatenated_reference() -> None:
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

    torch.testing.assert_close(accumulator.mean("activation").value, expected_mean.double())
    torch.testing.assert_close(accumulator.rms("activation").value, expected_rms.double())
    torch.testing.assert_close(accumulator.maximum("activation").value, torch.tensor(12.0))
    torch.testing.assert_close(accumulator.minimum("activation").value, torch.tensor(1.0))


def test_zero_contributor_rank_and_injected_process_group_preserve_fixed_collectives() -> None:
    peer = _accumulator("residual", "qkv")
    peer.add_masked_tensor("qkv", torch.tensor([2.0, 4.0]))
    group = object()
    reducer = _PeerReducer(peer, group)
    local = _accumulator(
        "residual", "qkv", reduction_binding=ReductionBinding.flat_world(group, reducer=reducer)
    )
    local.add_masked_tensor("residual", torch.empty(0))

    local.reduce_()

    assert reducer.operations == [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN]
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
    torch.testing.assert_close(replicated.rms("residual").value, reference.rms("residual").value)


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
        accumulator.zero_fraction("moments").value, torch.tensor(0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(accumulator.maximum("moments").value, torch.tensor(4.0))
    torch.testing.assert_close(accumulator.minimum("moments").value, torch.tensor(-2.0))


def test_nonfinites_are_counted_but_invalidate_numeric_derivations() -> None:
    accumulator = _accumulator("health")
    accumulator.add_masked_tensor("health", torch.tensor([0.0, torch.nan, torch.inf, 2.0]))
    accumulator.finalize_local_()

    rms = accumulator.rms("health")
    assert not rms.valid
    assert torch.isnan(rms.value)
    assert rms.reason == Tier0Reason.NONFINITE_INPUT
    torch.testing.assert_close(
        accumulator.nonfinite_fraction("health").value, torch.tensor(0.5, dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "mask",
    (
        torch.tensor([-1.0, 1.0]),
        torch.tensor([torch.nan, 1.0]),
        torch.tensor([torch.inf, 1.0]),
        torch.ones(3),
        torch.ones(2, dtype=torch.complex64),
        torch.ones(2, device="meta"),
    ),
    ids=("negative", "nan", "infinite", "shape", "complex", "device"),
)
def test_invalid_masks_are_neutral_and_surface_packed_mask_mismatch(mask: torch.Tensor) -> None:
    accumulator = _accumulator("masked")
    accumulator.add_masked_tensor("masked", torch.tensor([3.0, 4.0]), mask=mask)
    slots = accumulator.slots("masked")

    assert accumulator.sum_pack[slots.mask_error] > 0
    torch.testing.assert_close(
        accumulator.sum_pack[slots.count], torch.tensor(0.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        accumulator.sum_pack[slots.sumsq], torch.tensor(0.0, dtype=torch.float64)
    )
    accumulator.finalize_local_()
    statistic = accumulator.rms("masked")
    assert not statistic.valid
    assert torch.isnan(statistic.value)
    assert statistic.reason == Tier0Reason.MASK_MISMATCH


def test_mask_mismatch_reduces_collectively_without_changing_collective_count() -> None:
    peer = _accumulator("masked")
    peer.add_masked_tensor("masked", torch.tensor([100.0, 200.0]), mask=torch.tensor([-1.0, 1.0]))
    group = object()
    reducer = _PeerReducer(peer, group)
    local = _accumulator(
        "masked", reduction_binding=ReductionBinding.flat_world(group, reducer=reducer)
    )
    local.add_masked_tensor("masked", torch.tensor([3.0, 4.0]))

    local.reduce_()

    assert reducer.operations == [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN]
    statistic = local.rms("masked")
    assert not statistic.valid
    assert torch.isnan(statistic.value)
    assert statistic.reason == Tier0Reason.MASK_MISMATCH


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
    rank1 = _accumulator("cosine")
    rank1.add_masked_pair("cosine", torch.tensor([10.0, 0.0]), torch.tensor([-8.0, 6.0]))
    group = object()
    rank0 = _accumulator(
        "cosine",
        reduction_binding=ReductionBinding.flat_world(group, reducer=_PeerReducer(rank1, group)),
    )
    rank0.add_masked_pair("cosine", torch.ones(10), torch.ones(10))

    rank0_local = _accumulator("cosine")
    rank0_local.sum_pack.copy_(rank0.sum_pack)
    rank0_local.max_pack.copy_(rank0.max_pack)
    rank0_local.min_pack.copy_(rank0.min_pack)
    rank0_local.finalize_local_()
    rank1.finalize_local_()
    mean_of_local_cosines = (rank0_local.cosine("cosine").value + rank1.cosine("cosine").value) / 2

    rank0.reduce_()

    assert mean_of_local_cosines > 0
    assert rank0.cosine("cosine").value < 0


def test_finite_fp32_square_overflow_range_is_safe_for_rms() -> None:
    values = torch.tensor([2.0e19, -2.0e19], dtype=torch.float32)
    assert not torch.isfinite(values.square()).all()
    accumulator = _accumulator("large")
    accumulator.add_masked_tensor("large", values)
    accumulator.finalize_local_()

    statistic = accumulator.rms("large")
    assert statistic.valid
    assert torch.isfinite(statistic.value)
    torch.testing.assert_close(
        statistic.value, torch.tensor(2.0e19, dtype=torch.float64), rtol=1e-6, atol=0
    )


def test_finite_fp32_product_overflow_range_is_safe_for_relative_rms_and_cosine() -> None:
    lhs = torch.tensor([2.0e19, -2.0e19], dtype=torch.float32)
    rhs = torch.tensor([2.0e19, -2.0e19], dtype=torch.float32)
    assert not torch.isfinite(lhs * rhs).all()
    accumulator = _accumulator("large_pair")
    accumulator.add_masked_pair("large_pair", lhs, rhs)
    accumulator.finalize_local_()

    relative = accumulator.relative_rms("large_pair")
    cosine = accumulator.cosine("large_pair")
    assert relative.valid
    assert cosine.valid
    assert torch.isfinite(relative.value)
    assert torch.isfinite(cosine.value)
    torch.testing.assert_close(relative.value, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(cosine.value, torch.tensor(1.0, dtype=torch.float64))


def test_nonfinite_reduced_arithmetic_cannot_be_valid() -> None:
    accumulator = _accumulator("corrupt")
    accumulator.add_masked_tensor("corrupt", torch.ones(2))
    slots = accumulator.slots("corrupt")
    accumulator.sum_pack[slots.sumsq] = torch.inf
    accumulator.finalize_local_()

    statistic = accumulator.rms("corrupt")
    assert not statistic.valid
    assert torch.isnan(statistic.value)
    assert statistic.reason == Tier0Reason.NONFINITE_ARITHMETIC


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 1e-6), (torch.bfloat16, 1e-3)])
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

    expected = torch.linalg.vector_norm(after - before) / torch.linalg.vector_norm(before)
    torch.testing.assert_close(accumulator.relative_rms("update").value, expected.double())


@pytest.mark.parametrize("numel", (1, 17, 257, 4097))
def test_tensor_moment_scratch_is_bounded_independently_of_observation_size(numel: int) -> None:
    capacity = 17
    accumulator = PackedSufficientStatistics(
        ("bounded",),
        "cpu",
        descriptor_hash="bounded-scratch-v1",
        reduction_binding=ReductionBinding.flat_world(None),
        scratch_element_capacity=capacity,
    )
    values = torch.linspace(-2.0, 3.0, numel, dtype=torch.bfloat16)
    accumulator.add_masked_tensor("bounded", values)
    accumulator.finalize_local_()

    assert accumulator.maximum_scratch_bytes == accumulator.scratch_bytes_for_capacity(capacity)
    assert accumulator.peak_scratch_bytes <= accumulator.maximum_scratch_bytes
    if numel > capacity:
        assert accumulator.peak_scratch_bytes == accumulator.maximum_scratch_bytes
    reference = values.float().square().mean(dtype=torch.float64).sqrt()
    torch.testing.assert_close(accumulator.rms("bounded").value, reference)


def test_default_moment_workspace_is_96_mib_and_chunk_equivalent() -> None:
    assert DEFAULT_MOMENT_SCRATCH_ELEMENT_CAPACITY == 1_048_576
    assert PackedSufficientStatistics.scratch_bytes_for_capacity() == 96 * 1024**2

    values = torch.linspace(-3.0, 5.0, 4097, dtype=torch.bfloat16)
    mask = (torch.arange(values.numel()) % 3 != 0).to(dtype=torch.float32)
    before = torch.linspace(-1.0, 2.0, values.numel(), dtype=torch.bfloat16)
    after = before + torch.tensor(0.125, dtype=torch.bfloat16)

    def accumulate(capacity: int) -> PackedSufficientStatistics:
        accumulator = PackedSufficientStatistics(
            ("moments", "update"),
            "cpu",
            descriptor_hash=f"chunk-equivalence-{capacity}",
            reduction_binding=ReductionBinding.flat_world(None),
            scratch_element_capacity=capacity,
        )
        workspace = torch.empty(accumulator.maximum_scratch_bytes, dtype=torch.uint8)
        accumulator.bind_workspace(workspace)
        accumulator.add_masked_tensor("moments", values, mask=mask)
        accumulator.add_update("update", before, after)
        return accumulator.finalize_local_()

    chunked = accumulate(257)
    default = accumulate(DEFAULT_MOMENT_SCRATCH_ELEMENT_CAPACITY)
    torch.testing.assert_close(chunked.sum_pack, default.sum_pack, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(chunked.max_pack, default.max_pack, rtol=0, atol=0)
    torch.testing.assert_close(chunked.min_pack, default.min_pack, rtol=0, atol=0)
    torch.testing.assert_close(chunked.rms("moments").value, default.rms("moments").value)
    torch.testing.assert_close(
        chunked.relative_rms("update").value, default.relative_rms("update").value
    )


@pytest.mark.parametrize("operation", ("tensor", "pair", "update"))
def test_bound_workspace_matches_legacy_for_shaped_bf16_and_broadcast_mask(
    operation: str,
) -> None:
    shape = torch.Size((7, 2, 5))
    capacity = 17
    values = torch.linspace(-3.0, 5.0, shape.numel(), dtype=torch.bfloat16).view(shape)
    other = torch.linspace(2.0, -1.0, shape.numel(), dtype=torch.bfloat16).view(shape)
    mask = (torch.arange(shape[0] * shape[1]).view(shape[0], shape[1], 1) % 3 != 0).float()
    assert torch.broadcast_to(mask, shape).stride(-1) == 0

    def accumulate(*, bind_workspace: bool) -> PackedSufficientStatistics:
        accumulator = PackedSufficientStatistics(
            (operation,),
            "cpu",
            descriptor_hash=f"shaped-workspace-{operation}-{bind_workspace}",
            reduction_binding=ReductionBinding.flat_world(None),
            scratch_element_capacity=capacity,
        )
        if bind_workspace:
            storage = torch.empty(accumulator.maximum_scratch_bytes, dtype=torch.uint8)
            accumulator.bind_workspace(storage)
        if operation == "tensor":
            accumulator.add_masked_tensor(operation, values, mask=mask)
        elif operation == "pair":
            accumulator.add_masked_pair(operation, values, other, mask=mask)
        else:
            accumulator.add_update(operation, values, other, mask=mask)
        return accumulator.finalize_local_()

    legacy = accumulate(bind_workspace=False)
    bound = accumulate(bind_workspace=True)
    torch.testing.assert_close(bound.sum_pack, legacy.sum_pack, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(bound.max_pack, legacy.max_pack, rtol=0, atol=0)
    torch.testing.assert_close(bound.min_pack, legacy.min_pack, rtol=0, atol=0)


def test_update_scratch_bound_covers_chunked_delta_without_full_size_temporary() -> None:
    capacity = 31
    accumulator = PackedSufficientStatistics(
        ("update",),
        "cpu",
        descriptor_hash="bounded-update-scratch-v1",
        reduction_binding=ReductionBinding.flat_world(None),
        scratch_element_capacity=capacity,
    )
    before = torch.linspace(-1.0, 1.0, capacity * 128, dtype=torch.bfloat16)
    after = before + torch.tensor(0.125, dtype=torch.bfloat16)
    accumulator.add_update("update", before, after)
    accumulator.finalize_local_()

    assert accumulator.peak_scratch_bytes == accumulator.maximum_scratch_bytes
    expected = torch.linalg.vector_norm(after.float() - before.float()) / torch.linalg.vector_norm(
        before.float()
    )
    torch.testing.assert_close(accumulator.relative_rms("update").value, expected.double())


def test_full_accumulation_module_has_no_prohibited_host_synchronization() -> None:
    source = inspect.getsource(accumulator_module)
    tree = ast.parse(source)
    prohibited_attributes = {"cpu", "item", "numpy", "tolist"}
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in prohibited_attributes:
                violations.append((node.func.attr, node.lineno))
            if node.func.attr == "to" and (
                node.args or any(keyword.arg != "dtype" for keyword in node.keywords)
            ):
                violations.append(("device-moving to", node.lineno))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"bool", "float", "int"}:
                violations.append((node.func.id, node.lineno))

    assert violations == []
