# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure CPU tests for the canonical Tier-2 secant core."""

import inspect
import json
from pathlib import Path

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
    SecantMemoryEstimateError,
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


def _restorer(
    *, fail_restore=None, fail_verify=None, false_verify=None, calls=None
) -> IndependentRestorer:
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
            return torch.tensor(kind != false_verify)

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


@pytest.mark.parametrize("mismatch", ("binding", "schema"))
def test_formula_binding_reuses_complete_canonical_registry_validation(mismatch) -> None:
    binding = _binding()
    accumulator = PackedSufficientStatistics(
        binding.registry.slot_names,
        "cpu",
        descriptor_hash=binding.registry.descriptor_hash,
        reduction_binding=(
            ReductionBinding.flat_world(object())
            if mismatch == "binding"
            else binding.registry.reduction_binding
        ),
        schema_identity="diag/v999" if mismatch == "schema" else "diag/v2",
    )
    accumulator.finalize_local_()

    with pytest.raises(ValueError, match="descriptor/schema identity|slot order"):
        SecantSufficientStatisticsView.from_accumulator(binding, accumulator, binding.cells[0])


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


@pytest.mark.parametrize(
    "observation",
    (
        SecantObservation(
            "layer_0/residual",
            torch.ones(2, 1),
            torch.ones(1, 2),
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.ones(2, 1),
        ),
        SecantObservation(
            "layer_0/residual",
            torch.ones(2, 2).t(),
            torch.ones(2, 2).t(),
            torch.ones(2, 2).t(),
            torch.ones(2, 2).t(),
            torch.ones(2, 2),
        ),
        SecantObservation(
            "layer_0/residual",
            torch.ones(1).expand(4),
            torch.ones(1).expand(4),
            torch.ones(1).expand(4),
            torch.ones(1).expand(4),
            torch.ones(4),
        ),
    ),
)
def test_misaligned_noncontiguous_and_stride_zero_observations_fail_before_accumulation(
    observation,
) -> None:
    cell = SecantCellDescriptor("layer_0/residual", MetricFamily.RESIDUAL, 0)
    binding = build_secant_registry((cell,), reduction_binding=ReductionBinding.flat_world(None))
    statistics = SecantStatistics(binding, "cpu", chunk_elements=2)

    with pytest.raises(ValueError, match="identical|contiguous"):
        statistics.add_observations((observation,))
    assert torch.count_nonzero(statistics.accumulator.sum_pack) == 0


def test_validated_mask_broadcast_is_chunked_without_broadcasting_observations() -> None:
    cell = SecantCellDescriptor("layer_0/residual", MetricFamily.RESIDUAL, 0)
    binding = build_secant_registry((cell,), reduction_binding=ReductionBinding.flat_world(None))
    statistics = SecantStatistics(binding, "cpu", chunk_elements=2)
    pre = torch.ones(2, 3)
    observation = SecantObservation(
        cell.logical_name, pre, pre + 1, pre + 1, pre + 0.5, torch.tensor([[1.0], [0.0]])
    )

    statistics.add_observations((observation,))
    statistics.accumulator.finalize_local_()
    view = SecantSufficientStatisticsView.from_accumulator(binding, statistics.accumulator, cell)

    assert view.count == 3


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
    fixture = Path(__file__).with_name("fixtures") / "tier2_scaling_keys.json"
    expected = tuple(json.loads(fixture.read_text(encoding="utf-8")))
    metrics = _one_cell_metrics([1.0], [2.0], [2.0], [1.5])
    outputs = derive_tier2_outputs((metrics,))

    assert TIER2_OUTPUT_KEYS == expected
    assert tuple(outputs) == expected
    assert len(outputs) == 17
    assert outputs["diag/v2/t2/valid"] == 1


def test_unresolved_fraction_excludes_midpoint_and_other_non_replay_failures() -> None:
    valid = _one_cell_metrics([1.0], [2.0], [2.0], [1.5])
    midpoint_invalid = _one_cell_metrics([1.0], [2.0], [2.0], [1.5], midpoint_sq=3.24, full_sq=4.0)
    unresolved = _one_cell_metrics([1.0], [2.0], [2.5], [1.5])

    outputs = derive_tier2_outputs((valid, midpoint_invalid, unresolved))

    assert midpoint_invalid.status == SecantMathStatus.MIDPOINT_OUT_OF_RANGE
    assert unresolved.status == SecantMathStatus.REPLAY_UNRESOLVED
    torch.testing.assert_close(
        outputs["diag/v2/t2/unresolved_fraction"], torch.tensor(1 / 3, dtype=torch.float64)
    )
    assert outputs["diag/v2/t2/valid"] == 0


@pytest.mark.parametrize("failed_kind", tuple(RestorationKind))
def test_every_restoration_stage_is_attempted_after_each_injected_failure(failed_kind) -> None:
    calls = []
    report = _restorer(fail_restore=failed_kind, calls=calls).run()

    assert len(calls) == 2 * len(RestorationKind)
    assert {kind for kind, operation in calls if operation == "restore"} == set(RestorationKind)
    assert {kind for kind, operation in calls if operation == "verify"} == set(RestorationKind)
    assert not report.valid
    assert failed_kind in report.failed_kinds


@pytest.mark.parametrize("failed_kind", tuple(RestorationKind))
def test_verifier_exception_still_attempts_every_restoration_stage(failed_kind) -> None:
    calls = []
    report = _restorer(fail_verify=failed_kind, calls=calls).run()

    assert len(calls) == 2 * len(RestorationKind)
    assert not report.valid
    assert failed_kind in report.failed_kinds


def test_false_restoration_verifier_runs_all_stages_and_requires_central_fatal() -> None:
    calls = []
    failed_kind = RestorationKind.FP32_MASTERS
    transaction = SecantLocalTransaction(_restorer(false_verify=failed_kind, calls=calls))
    result = None
    for target in tuple(SecantTransactionState)[1:]:
        result = transaction.advance(target)

    assert result is not None
    assert len(calls) == 2 * len(RestorationKind)
    assert result.error == SecantTransactionError.RESTORE_FAILED
    assert result.fatal_required
    assert result.restoration is not None
    assert not result.restoration.valid
    assert not result.restoration.stage_valid[int(failed_kind)]


def test_restorer_level_exception_becomes_complete_typed_fatal_report(monkeypatch) -> None:
    restorer = _restorer()
    monkeypatch.setattr(restorer, "run", lambda: (_ for _ in ()).throw(RuntimeError("OOM")))
    transaction = SecantLocalTransaction(restorer)
    result = None
    for target in tuple(SecantTransactionState)[1:]:
        result = transaction.advance(target)

    assert result is not None
    assert result.error == SecantTransactionError.RESTORE_FAILED
    assert result.fatal_required
    assert result.restoration is not None
    assert result.restoration.failed_kinds == tuple(RestorationKind)
    assert result.restoration.stage_valid.shape == (len(RestorationKind),)
    assert not torch.any(result.restoration.stage_valid)


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
    assert estimate.packed_statistics_bytes == 29_184
    assert estimate.reduction_arena_bytes == estimate.packed_statistics_bytes
    assert estimate.chunk_workspace_bytes == max(
        estimate.reduction_arena_bytes,
        estimate.bounded_accumulation_workspace_bytes,
        estimate.owner_hash_restore_workspace_bytes,
        estimate.quantile_sink_workspace_bytes,
    )
    assert estimate.owner_hash_restore_workspace_bytes > 0
    assert estimate.quantile_sink_workspace_bytes > 0
    assert estimate.retained_bytes == (
        estimate.fp32_pre_or_delta_bytes
        + estimate.bf16_pre_bytes
        + estimate.post_fingerprint_bytes
        + estimate.replay_payload_bytes
        + estimate.replay_mask_bytes
        + estimate.replay_state_bytes
        + estimate.replay_cap_reserve_bytes
        + estimate.packed_statistics_bytes
        + estimate.chunk_workspace_bytes
    )
    assert estimate.peak_bytes == (
        estimate.retained_bytes + estimate.allocator_headroom_bytes + estimate.driver_headroom_bytes
    )


@pytest.mark.parametrize(
    ("alignment", "cell_count", "expected_pack_bytes", "expected_peak_bytes"),
    (
        (256, 1, 1_024, 1_060_864),
        (256, 100, 29_184, 1_274_880),
        (256, 4_252_361_473_883_687, 1_224_680_104_478_502_144, 9_223_372_036_854_775_040),
        (512, 1, 1_536, 1_072_640),
        (512, 100, 29_696, 1_463_296),
        (512, 2_328_546_335_989_322, 670_621_344_764_925_440, 9_223_372_036_854_772_224),
    ),
)
def test_zero_headroom_memory_estimate_aligns_each_pack_allocation_and_composes_live_phases(
    alignment, cell_count, expected_pack_bytes, expected_peak_bytes
) -> None:
    inputs = SecantMemoryInputs(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        loaded_owner_elements_by_rank=(0,),
        tied_alias_elements_by_rank=(0,),
        replay_payload_bytes=0,
        replay_mask_bytes=0,
        replay_state_bytes=0,
        replay_cap_bytes=0,
        registry_slots=3 * cell_count,
        chunk_elements=1,
        alignment_bytes=alignment,
        allocator_headroom_fraction=0.0,
        driver_headroom_fraction=0.0,
    )

    estimate = SecantMemoryEstimate.calculate(inputs, optimizer_layout=SecantOptimizerLayout())

    slots = 3 * cell_count

    def align(value: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    independently_aligned_pack = align(88 * slots) + align(4 * slots) + align(4 * slots)
    assert independently_aligned_pack == expected_pack_bytes
    assert estimate.packed_statistics_bytes == independently_aligned_pack
    assert estimate.reduction_arena_bytes == independently_aligned_pack
    if alignment == 512 and cell_count == 1:
        assert estimate.packed_statistics_bytes == 1_536
        assert estimate.packed_statistics_bytes + estimate.reduction_arena_bytes == 3_072

    phase_workspace = max(
        estimate.reduction_arena_bytes,
        estimate.bounded_accumulation_workspace_bytes,
        estimate.owner_hash_restore_workspace_bytes,
        estimate.quantile_sink_workspace_bytes,
    )
    persistent = (
        estimate.fp32_pre_or_delta_bytes
        + estimate.bf16_pre_bytes
        + estimate.post_fingerprint_bytes
        + estimate.replay_payload_bytes
        + estimate.replay_mask_bytes
        + estimate.replay_state_bytes
        + estimate.replay_cap_reserve_bytes
        + estimate.packed_statistics_bytes
    )
    assert estimate.chunk_workspace_bytes == phase_workspace
    assert estimate.retained_bytes == persistent + phase_workspace
    assert estimate.allocator_headroom_bytes == 0
    assert estimate.driver_headroom_bytes == 0
    assert estimate.peak_bytes == expected_peak_bytes

    maximum_cells = {256: 4_252_361_473_883_687, 512: 2_328_546_335_989_322}[alignment]
    if cell_count == maximum_cells:
        with pytest.raises(SecantMemoryEstimateError):
            SecantMemoryEstimate.calculate(
                SecantMemoryInputs(**{**inputs.__dict__, "registry_slots": 3 * (cell_count + 1)}),
                optimizer_layout=SecantOptimizerLayout(),
            )


def test_memory_estimate_covers_independent_100_cell_derived_and_quantile_live_graph() -> None:
    row = torch.tensor(
        (4.0, 4.0, 0.04, 4.0, 1.0, 1.0e-6, 10.0, 0.0, 0.0, 1.0, 4.0), dtype=torch.float64
    )
    source_pack = row.repeat(100, 1)
    metrics = []
    for cell in source_pack:
        statistics = SecantSufficientStatisticsView(
            true_sq=cell[0],
            predicted_sq=cell[1],
            error_sq=cell[2],
            true_predicted_dot=cell[3],
            pre_sq=cell[4],
            repeat_error_sq=cell[5],
            count=cell[6],
            nonfinite=cell[7],
            contract_error=cell[8],
        )
        metrics.append(
            derive_secant_cell(
                statistics,
                midpoint_displacement_sq=cell[9],
                full_displacement_sq=cell[10],
                restore_verified=True,
            )
        )

    derived_fields = tuple(
        getattr(metric, field)
        for metric in metrics
        for field in (
            "true_response",
            "secant_error",
            "secant_cosine",
            "realized_midpoint_fraction",
            "replay_floor",
            "valid",
            "status",
        )
    )
    derived_storages = {
        (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
        for tensor in derived_fields
    }
    assert len(derived_storages) == 700
    assert sum(size for _, size in derived_storages) == 4_900

    cell_stack = torch.stack(tuple(metric.true_response for metric in metrics))
    sorted_values, sort_indices = torch.sort(cell_stack)
    cell_bool = torch.stack(tuple(metric.valid for metric in metrics))
    outputs = derive_tier2_outputs(metrics)
    live_tensors = (
        *derived_fields,
        cell_stack,
        sorted_values,
        sort_indices,
        cell_bool,
        *outputs.values(),
    )
    live_storages = {
        (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
        for tensor in live_tensors
    }
    review_minimum_live_bytes = sum(size for _, size in live_storages)
    assert review_minimum_live_bytes == 7_536

    estimate = SecantMemoryEstimate.calculate(
        SecantMemoryInputs(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            loaded_owner_elements_by_rank=(0,),
            tied_alias_elements_by_rank=(0,),
            replay_payload_bytes=0,
            replay_mask_bytes=0,
            replay_state_bytes=0,
            replay_cap_bytes=0,
            registry_slots=300,
            chunk_elements=1,
            alignment_bytes=256,
            allocator_headroom_fraction=0.0,
            driver_headroom_fraction=0.0,
        ),
        optimizer_layout=SecantOptimizerLayout(),
    )
    declared_alignment_and_scalar_workspace = (
        estimate.derived_metric_storage_bytes
        + estimate.quantile_cell_workspace_bytes
        + estimate.tier2_output_storage_bytes
        + estimate.quantile_scalar_workspace_bytes
        - review_minimum_live_bytes
    )
    assert estimate.quantile_sink_workspace_bytes == (
        review_minimum_live_bytes
        + declared_alignment_and_scalar_workspace
        + estimate.sort_backend_workspace_bytes
    )
    assert estimate.quantile_sink_workspace_bytes > 2_828
    assert estimate.allocator_headroom_bytes == 0
    assert estimate.driver_headroom_bytes == 0


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


def test_memory_estimator_rejects_huge_legal_integer_with_typed_error() -> None:
    inputs = SecantMemoryInputs(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        loaded_owner_elements_by_rank=(10**400,),
        tied_alias_elements_by_rank=(0,),
        replay_payload_bytes=0,
        replay_mask_bytes=0,
        replay_state_bytes=0,
        replay_cap_bytes=0,
        registry_slots=3,
        chunk_elements=1,
    )

    with pytest.raises(SecantMemoryEstimateError, match="representable cap"):
        SecantMemoryEstimate.calculate(inputs, optimizer_layout=SecantOptimizerLayout())


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
