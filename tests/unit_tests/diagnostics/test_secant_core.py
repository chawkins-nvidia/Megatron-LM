# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure CPU tests for the canonical Tier-2 secant core."""

import inspect

import pytest
import torch

import megatron.training.diagnostics.secant as secant_module
from megatron.training.diagnostics.accumulator import PackedSufficientStatistics, ReductionBinding
from megatron.training.diagnostics.registry import MetricFamily
from megatron.training.diagnostics.secant import (
    TIER2_OUTPUT_KEYS,
    IndependentRestorer,
    RestorationKind,
    RestorationStage,
    SecantCellDescriptor,
    SecantLocalTransaction,
    SecantMathStatus,
    SecantMemoryEstimate,
    SecantMemoryInputs,
    SecantObservation,
    SecantOptimizerLayout,
    SecantStatistics,
    SecantSufficientStatisticsView,
    SecantTransactionError,
    SecantTransactionState,
    SecantUnsupportedLayoutError,
    TensorBitwiseSnapshot,
    build_secant_registry,
    derive_secant_cell,
    derive_tier2_outputs,
)


def _cells(*, owner: bool = True) -> tuple[SecantCellDescriptor, ...]:
    return (
        SecantCellDescriptor("layer_0/residual", MetricFamily.RESIDUAL, 0, owner),
        SecantCellDescriptor("layer_1/fc1", MetricFamily.FC1, 1, owner),
    )


def _binding(*, owner: bool = True):
    return build_secant_registry(
        reversed(_cells(owner=owner)), reduction_binding=ReductionBinding.flat_world(None)
    )


def _observation(name: str, pre, post, repeat, midpoint) -> SecantObservation:
    pre_tensor = torch.as_tensor(pre, dtype=torch.float32)
    return SecantObservation(
        name,
        pre_tensor,
        torch.as_tensor(post, dtype=torch.float32),
        torch.as_tensor(repeat, dtype=torch.float32),
        torch.as_tensor(midpoint, dtype=torch.float32),
        torch.ones_like(pre_tensor),
    )


def _one_cell_metrics(pre, post, repeat, midpoint, *, midpoint_sq=1.0, full_sq=4.0):
    cell = SecantCellDescriptor("layer_0/residual", MetricFamily.RESIDUAL, 0)
    binding = build_secant_registry((cell,), reduction_binding=ReductionBinding.flat_world(None))
    statistics = SecantStatistics(binding, "cpu", chunk_elements=2)
    statistics.add_observations((_observation(cell.logical_name, pre, post, repeat, midpoint),))
    statistics.accumulator.finalize_local_()
    view = SecantSufficientStatisticsView.from_accumulator(
        binding, statistics.accumulator, binding.cells[0]
    )
    return derive_secant_cell(
        view,
        midpoint_displacement_sq=midpoint_sq,
        full_displacement_sq=full_sq,
        restore_verified=True,
    )


def _restorer(*, fail_restore=None, fail_verify=None, calls=None) -> IndependentRestorer:
    trace = [] if calls is None else calls
    stages = []
    for kind in RestorationKind:

        def restore(kind=kind):
            trace.append((kind, "restore"))
            if kind == fail_restore:
                raise RuntimeError("injected restore failure")

        def verify(kind=kind):
            trace.append((kind, "verify"))
            if kind == fail_verify:
                raise RuntimeError("injected verification failure")
            return torch.tensor(True)

        stages.append(RestorationStage(kind, restore, verify))
    return IndependentRestorer(stages, "cpu")


def test_descriptor_hash_and_slot_order_are_global_and_owner_independent() -> None:
    owner = _binding(owner=True)
    nonowner = _binding(owner=False)

    assert owner.cells == _cells()
    assert owner.registry.slot_names == nonowner.registry.slot_names
    assert owner.registry.descriptor_hash == nonowner.registry.descriptor_hash
    assert len(owner.registry.slot_names) == 3 * len(owner.cells)


@pytest.mark.parametrize("indices", ((1, 0), (0,), (0, 0)))
def test_reversed_missing_and_duplicate_observations_are_fatal(indices) -> None:
    binding = _binding()
    observations = tuple(
        _observation(binding.cells[index].logical_name, [1.0], [2.0], [2.0], [1.5])
        for index in indices
    )
    statistics = SecantStatistics(binding, "cpu")

    with pytest.raises(ValueError, match="canonical global cell order"):
        statistics.add_observations(observations)
    assert torch.count_nonzero(statistics.accumulator.sum_pack) == 0


def test_reversed_packed_slots_are_rejected_before_formula_binding() -> None:
    binding = _binding()
    reversed_accumulator = PackedSufficientStatistics(
        tuple(reversed(binding.registry.slot_names)),
        "cpu",
        descriptor_hash=binding.registry.descriptor_hash,
        reduction_binding=binding.registry.reduction_binding,
    )

    with pytest.raises(ValueError, match="slot order"):
        SecantSufficientStatisticsView.from_accumulator(
            binding, reversed_accumulator, binding.cells[0]
        )


def test_nonowner_contributions_leave_all_canonical_packs_neutral() -> None:
    binding = _binding(owner=False)
    statistics = SecantStatistics(binding, "cpu")
    observations = tuple(
        _observation(cell.logical_name, [1.0], [2.0], [2.0], [1.5]) for cell in binding.cells
    )
    statistics.add_observations(observations)

    assert torch.count_nonzero(statistics.accumulator.sum_pack) == 0
    assert torch.all(statistics.accumulator.max_pack == -torch.inf)
    assert torch.all(statistics.accumulator.min_pack == torch.inf)


def test_affine_and_quadratic_formula_references() -> None:
    affine = _one_cell_metrics([1.0, 2.0], [3.0, 6.0], [3.0, 6.0], [2.0, 4.0])
    quadratic = _one_cell_metrics([4.0], [16.0], [16.0], [9.0])

    assert affine.valid
    torch.testing.assert_close(affine.true_response, torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(affine.secant_error, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(affine.secant_cosine, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(
        affine.realized_midpoint_fraction, torch.tensor(0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(quadratic.secant_error, torch.tensor(1.0 / 6.0, dtype=torch.float64))


@pytest.mark.parametrize(
    ("pre", "post", "repeat", "midpoint", "status"),
    (
        ([1.0], [1.0], [1.0], [1.0], SecantMathStatus.TINY_DENOMINATOR),
        ([0.0], [1.0], [1.0], [0.5], SecantMathStatus.TINY_DENOMINATOR),
        ([1.0], [2.0], [2.5], [1.5], SecantMathStatus.REPLAY_UNRESOLVED),
        ([1.0], [float("nan")], [2.0], [1.5], SecantMathStatus.NONFINITE),
    ),
)
def test_zero_tiny_unresolved_and_nonfinite_behavior(pre, post, repeat, midpoint, status) -> None:
    metrics = _one_cell_metrics(pre, post, repeat, midpoint)

    assert not metrics.valid
    assert metrics.status == status
    assert torch.isnan(metrics.secant_error)


def test_exact_17_key_order_and_q3_outputs() -> None:
    expected = (
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
    metrics = _one_cell_metrics([1.0], [2.0], [2.0], [1.5])
    outputs = derive_tier2_outputs((metrics,))

    assert TIER2_OUTPUT_KEYS == expected
    assert tuple(outputs) == expected
    assert len(outputs) == 17
    assert outputs["diag/v2/t2/valid"] == 1


@pytest.mark.parametrize("failed_kind", tuple(RestorationKind))
def test_every_restoration_stage_is_attempted_after_each_injected_failure(failed_kind) -> None:
    calls = []
    report = _restorer(fail_restore=failed_kind, calls=calls).run()

    assert len(calls) == 2 * len(RestorationKind)
    assert {kind for kind, operation in calls if operation == "restore"} == set(RestorationKind)
    assert {kind for kind, operation in calls if operation == "verify"} == set(RestorationKind)
    assert not report.valid
    assert failed_kind in report.failed_kinds


def test_tensor_snapshot_restores_aliases_and_verifies_bits() -> None:
    tensor = torch.tensor([1.0, -0.0, float("nan")], dtype=torch.float32)
    alias = tensor.view_as(tensor)
    snapshot = TensorBitwiseSnapshot((tensor, alias), chunk_elements=2)
    expected = tensor.view(torch.uint8).clone()
    tensor.fill_(7)

    snapshot.restore()

    assert snapshot.byte_count == tensor.nbytes
    assert snapshot.verify()
    assert torch.equal(tensor.view(torch.uint8), expected)


@pytest.mark.parametrize("target", tuple(SecantTransactionState)[1:])
def test_failure_at_every_transaction_stage_restores_then_requires_central_fatal(target) -> None:
    transaction = SecantLocalTransaction(_restorer())
    for state in tuple(SecantTransactionState)[1:]:
        action = (
            (lambda: (_ for _ in ()).throw(RuntimeError("injected"))) if state == target else None
        )
        result = transaction.advance(state, action)
        if state == target:
            break

    assert result.error != SecantTransactionError.NONE
    assert result.fatal_required
    assert result.state == SecantTransactionState.RESTORED
    assert result.restoration is not None
    assert result.restoration.valid


def test_successful_transaction_is_monotonic_and_idempotent() -> None:
    transaction = SecantLocalTransaction(_restorer())
    for target in tuple(SecantTransactionState)[1:]:
        first = transaction.advance(target)
        second = transaction.advance(target)
        assert first == second

    assert transaction.state_history == list(SecantTransactionState)
    assert transaction.result().error == SecantTransactionError.NONE
    assert not transaction.result().fatal_required
    assert transaction.result().restoration is not None
    assert transaction.result().restoration.valid


def test_invalid_transition_is_typed_and_centrally_fatal() -> None:
    transaction = SecantLocalTransaction(_restorer())

    result = transaction.advance(SecantTransactionState.UPDATE_SUCCEEDED)

    assert result.error == SecantTransactionError.INVALID_TRANSITION
    assert result.fatal_required
    assert result.state == SecantTransactionState.PRE_READY


@pytest.mark.parametrize(
    ("dp", "tp", "pp", "cp"), ((1024, 1, 1, 1), (128, 8, 1, 1), (16, 8, 8, 1), (8, 8, 8, 2))
)
def test_world_1024_memory_bound_uses_max_loaded_shard_and_tied_ownership(dp, tp, pp, cp) -> None:
    world_size = dp * tp * pp * cp
    loaded = [1_000] * world_size
    tied = [100] * world_size
    loaded[777] = 10_000
    tied[777] = 2_000
    inputs = SecantMemoryInputs(
        data_parallel_size=dp,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        context_parallel_size=cp,
        loaded_owner_elements_by_rank=tuple(loaded),
        tied_alias_elements_by_rank=tuple(tied),
        replay_payload_bytes=4096,
        replay_mask_bytes=1024,
        replay_state_bytes=2048,
        replay_cap_bytes=8192,
        registry_slots=300,
        chunk_elements=1024,
        alignment_bytes=256,
        allocator_headroom_fraction=0.1,
        minimum_allocator_headroom_bytes=4096,
    )
    estimate = SecantMemoryEstimate.calculate(inputs, optimizer_layout=SecantOptimizerLayout())

    assert estimate.world_size == 1024
    assert estimate.maximum_loaded_rank == 777
    assert estimate.maximum_loaded_owner_elements == 10_000
    assert estimate.tied_alias_elements == 2_000
    assert estimate.unique_owner_elements == 8_000
    assert estimate.fp32_pre_or_delta_bytes == 32_000
    assert estimate.bf16_pre_bytes == 16_128
    assert estimate.packed_statistics_bytes == 28_928
    assert estimate.reduction_arena_bytes == estimate.packed_statistics_bytes
    assert estimate.chunk_workspace_bytes == 131_072
    assert estimate.retained_bytes == (
        estimate.fp32_pre_or_delta_bytes
        + estimate.bf16_pre_bytes
        + estimate.post_fingerprint_bytes
        + estimate.replay_payload_bytes
        + estimate.replay_mask_bytes
        + estimate.replay_state_bytes
        + estimate.packed_statistics_bytes
        + estimate.reduction_arena_bytes
        + estimate.chunk_workspace_bytes
    )
    assert estimate.peak_bytes == estimate.retained_bytes + estimate.allocator_headroom_bytes


def test_memory_estimator_rejects_cap_and_unsupported_layout_fail_closed() -> None:
    inputs = SecantMemoryInputs(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        loaded_owner_elements_by_rank=(1,),
        tied_alias_elements_by_rank=(0,),
        replay_payload_bytes=2,
        replay_mask_bytes=2,
        replay_state_bytes=2,
        replay_cap_bytes=5,
        registry_slots=3,
        chunk_elements=1,
    )
    with pytest.raises(ValueError, match="hard byte cap"):
        SecantMemoryEstimate.calculate(inputs, optimizer_layout=SecantOptimizerLayout())
    with pytest.raises(SecantUnsupportedLayoutError):
        SecantMemoryEstimate.calculate(
            SecantMemoryInputs(**{**inputs.__dict__, "replay_cap_bytes": 6}),
            optimizer_layout=SecantOptimizerLayout(precision_aware=True),
        )


def test_production_secant_core_has_no_host_sync_or_collective_calls() -> None:
    sources = (
        inspect.getsource(secant_module),
        inspect.getsource(secant_module.SecantStatistics),
        inspect.getsource(secant_module.SecantLocalTransaction),
    )
    prohibited = (
        ".item(",
        ".cpu(",
        ".tolist(",
        "all_reduce(",
        "new_group(",
        "barrier(",
        "synchronize(",
        "torch.equal(",
    )
    for source in sources:
        assert all(token not in source for token in prohibited)
