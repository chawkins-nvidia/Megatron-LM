# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for canonical distributed-optimizer Tier-2 owner operations."""

import inspect
from dataclasses import replace

import pytest
import torch

from megatron.training.diagnostics.distributed_optimizer import (
    Bf16DistributedOptimizerDiagnosticAdapter,
    DistributedOptimizerEventStatus,
)
from megatron.training.diagnostics.registry import MetricFamily
from megatron.training.diagnostics.secant import (
    SecantMemoryEstimate,
    SecantMemoryInputs,
    SecantOptimizerLayout,
)
from tests.unit_tests.optimizer.test_distributed_optimizer_diagnostics import (
    _adapter,
    _fake_optimizer,
    _real_buffer_optimizer,
    _registry,
)


def _apply_step_and_materialize(adapter, optimizer, amount: float = 0.25):
    post_masters = []
    with torch.no_grad():
        for shard in adapter.iter_owner_shards():
            shard.main_shard.add_(amount)
            post_masters.append(shard.main_shard.detach().clone())
        optimizer._copy_main_params_to_model_params()
    return tuple(post_masters)


@pytest.mark.parametrize("factory", (_fake_optimizer, _real_buffer_optimizer))
def test_fake_and_real_cpu_owner_ranges_feed_shared_secant_snapshot(factory) -> None:
    optimizer, parameters = factory()
    adapter, _ = _adapter(optimizer, parameters, finish_chunk_elements=3)
    measurement = adapter.begin_event()

    assert measurement is not None
    expected_elements = sum(shard.main_shard.numel() for shard in adapter.iter_owner_shards())
    assert adapter.estimate_snapshot_memory().owner_elements == expected_elements
    assert adapter.secant_applied_pre_buffer is not None
    assert adapter.secant_applied_pre_buffer.numel() == expected_elements


def test_delta_construction_never_changes_live_fp32_post_masters() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters, finish_chunk_elements=2)
    assert adapter.begin_event() is not None
    pre = adapter.secant_applied_pre_buffer.detach().clone()
    post_masters = _apply_step_and_materialize(adapter, optimizer)

    assert (
        adapter.commit_secant_delta(registry.new_accumulator("cpu"), update_successful=True)
        is not None
    )

    for expected, shard in zip(post_masters, adapter.iter_owner_shards()):
        assert torch.equal(shard.main_shard, expected)
    delta = adapter.secant_delta_buffer
    assert delta is not None
    torch.testing.assert_close(delta, torch.full_like(delta, 0.25))
    assert adapter.secant_applied_pre_buffer is None
    assert not torch.equal(adapter._snapshot.applied_before, pre)


def test_midpoint_install_and_post_restore_are_bitwise_roundtrip() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters, finish_chunk_elements=2)
    assert adapter.begin_event() is not None
    pre_fp32 = torch.cat(
        tuple(shard.main_shard.detach().clone() for shard in adapter.iter_owner_shards())
    )
    post_masters = _apply_step_and_materialize(adapter, optimizer, amount=0.5)
    expected_post = tuple(master.to(torch.bfloat16) for master in post_masters)
    assert adapter.commit_secant_delta(update_successful=True) is not None

    adapter.install_secant_midpoint()
    offset = 0
    for index, shard in enumerate(adapter.iter_owner_shards()):
        end = offset + shard.main_shard.numel()
        expected_midpoint = (pre_fp32[offset:end] + 0.25).to(torch.bfloat16)
        assert torch.equal(shard.model_shard, expected_midpoint)
        assert torch.equal(shard.main_shard, post_masters[index])
        offset = end

    adapter.restore_secant_post()
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.OK
    for shard, expected in zip(adapter.iter_owner_shards(), expected_post):
        assert torch.equal(shard.model_shard, expected)


@pytest.mark.parametrize("alias_first", (False, True))
def test_exact_tied_physical_alias_uses_one_snapshot_and_one_copy(alias_first) -> None:
    optimizer, parameters = _fake_optimizer(parameter_sizes=(5, 6))
    original = tuple(optimizer.iter_model_main_param_shards())[0]
    alias = replace(original, shared=True, tied=True, tied_owner=False, logical_owner=False)
    optimizer.model_float16_groups = [[parameters[0]]]
    optimizer.shard_float16_groups = [[optimizer.shard_float16_groups[0][0]]]
    optimizer.shard_fp32_from_float16_groups = [[optimizer.shard_fp32_from_float16_groups[0][0]]]
    enumerated = (alias, original) if alias_first else (original, alias)
    optimizer.iter_model_main_param_shards = lambda: iter(enumerated)
    registry, names = _registry({parameters[0]: ("update/fc1/0", MetricFamily.FC1)})
    names = {parameters[0]: "update/fc1/0"}
    adapter = Bf16DistributedOptimizerDiagnosticAdapter(
        optimizer, registry, names, diagnostic_max_extra_bytes=1_000_000, finish_chunk_elements=2
    )

    assert len(adapter.iter_owner_shards()) == 2
    assert adapter.estimate_snapshot_memory().owner_elements == original.main_shard.numel()
    assert adapter.begin_event() is not None
    with torch.no_grad():
        original.main_shard.add_(0.5)
        original.model_shard.copy_(original.main_shard)
    expected_post = original.main_shard.to(torch.bfloat16)
    assert adapter.commit_secant_delta(update_successful=True) is not None
    adapter.install_secant_midpoint()
    adapter.restore_secant_post()
    assert torch.equal(original.model_shard, expected_post)
    assert torch.equal(alias.model_shard, expected_post)


@pytest.mark.parametrize("alias_first", (False, True))
def test_partial_nonowner_overlap_fails_closed_in_every_enumeration_order(alias_first) -> None:
    optimizer, parameters = _fake_optimizer(parameter_sizes=(5, 6))
    original = tuple(optimizer.iter_model_main_param_shards())[0]
    partial = replace(
        original,
        main_shard=original.main_shard[1:],
        model_shard=original.model_shard[1:],
        shared=True,
        tied=True,
        tied_owner=False,
        logical_owner=False,
    )
    enumerated = (partial, original) if alias_first else (original, partial)
    optimizer.iter_model_main_param_shards = lambda: iter(enumerated)
    registry, names = _registry({parameters[0]: ("update/fc1/0", MetricFamily.FC1)})
    adapter = Bf16DistributedOptimizerDiagnosticAdapter(
        optimizer, registry, names, diagnostic_max_extra_bytes=1_000_000
    )

    assert (
        adapter.local_status.item() == DistributedOptimizerEventStatus.CONSTRUCTOR_OWNERSHIP_FAILED
    )
    assert adapter.begin_event() is None
    assert not adapter.armed


def test_empty_owner_rank_is_a_neutral_valid_local_participant() -> None:
    optimizer, _ = _fake_optimizer()
    optimizer.model_float16_groups = [[]]
    optimizer.shard_float16_groups = [[]]
    optimizer.shard_fp32_from_float16_groups = [[]]
    optimizer.model_param_gbuf_map = {}
    optimizer.iter_model_main_param_shards = lambda: iter(())
    registry, names = _registry({})
    adapter = Bf16DistributedOptimizerDiagnosticAdapter(
        optimizer, registry, names, diagnostic_max_extra_bytes=1_000_000
    )

    assert adapter.begin_event() is not None
    assert adapter.commit_secant_delta(update_successful=True) is not None
    assert adapter.secant_delta_buffer is not None
    assert adapter.secant_delta_buffer.numel() == 0
    adapter.install_secant_midpoint()
    adapter.restore_secant_post()
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.OK


def test_restore_failure_on_one_owner_does_not_skip_later_owners(monkeypatch) -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters, finish_chunk_elements=2)
    assert adapter.begin_event() is not None
    _apply_step_and_materialize(adapter, optimizer, amount=0.5)
    assert adapter.commit_secant_delta(update_successful=True) is not None
    expected_post = tuple(shard.model_shard.clone() for shard in adapter.iter_owner_shards())
    adapter.install_secant_midpoint()
    original_copy = adapter._copy_secant_chunk
    failed = False
    restore_attempts = 0

    def fail_first_restore(destination, source, phase):
        nonlocal failed, restore_attempts
        if phase == "restore":
            restore_attempts += 1
        if phase == "restore" and not failed:
            failed = True
            raise RuntimeError("injected restore failure")
        original_copy(destination, source, phase)

    monkeypatch.setattr(adapter, "_copy_secant_chunk", fail_first_restore)
    adapter.restore_secant_post()

    assert failed
    assert adapter.local_status.item() in (
        DistributedOptimizerEventStatus.SECANT_RESTORE_COPY_FAILED,
        DistributedOptimizerEventStatus.SECANT_RESTORE_VERIFY_FAILED,
    )
    assert restore_attempts > 1
    assert any(
        torch.equal(shard.model_shard, expected)
        for shard, expected in zip(adapter.iter_owner_shards(), expected_post)
    )


def test_post_hashes_reject_permuted_master_and_restore_original_bf16_bits() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters, finish_chunk_elements=2)
    assert adapter.begin_event() is not None
    _apply_step_and_materialize(adapter, optimizer, amount=0.5)
    expected_post = tuple(
        shard.model_shard.detach().view(torch.int16).clone()
        for shard in adapter.iter_owner_shards()
    )
    assert adapter.commit_secant_delta(update_successful=True) is not None
    adapter.install_secant_midpoint()
    first = adapter.iter_owner_shards()[0].main_shard
    with torch.no_grad():
        swapped = first[[1, 0]].clone()
        first[:2].copy_(swapped)

    adapter.restore_secant_post()

    assert (
        adapter.local_status.item() == DistributedOptimizerEventStatus.SECANT_RESTORE_VERIFY_FAILED
    )
    for shard, expected_bits in zip(adapter.iter_owner_shards(), expected_post):
        assert torch.equal(shard.model_shard.view(torch.int16), expected_bits)


def test_post_hashes_distinguish_signed_zero_and_restore_original_sign_bit() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters, finish_chunk_elements=2)
    assert adapter.begin_event() is not None
    first = adapter.iter_owner_shards()[0]
    with torch.no_grad():
        first.main_shard[0] = 0.0
        optimizer._copy_main_params_to_model_params()
    original_model_bits = first.model_shard.view(torch.int16).clone()
    assert adapter.commit_secant_delta(update_successful=True) is not None
    adapter.install_secant_midpoint()
    with torch.no_grad():
        first.main_shard[0] = -0.0

    adapter.restore_secant_post()

    assert (
        adapter.local_status.item() == DistributedOptimizerEventStatus.SECANT_RESTORE_VERIFY_FAILED
    )
    assert torch.equal(first.model_shard.view(torch.int16), original_model_bits)
    assert first.main_shard.view(torch.int32)[0] != 0


def test_successful_adapter_mutations_are_idempotent_and_phase_bound() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters, finish_chunk_elements=2)
    assert adapter.begin_event() is not None
    _apply_step_and_materialize(adapter, optimizer, amount=0.5)

    first_commit = adapter.commit_secant_delta(update_successful=True)
    second_commit = adapter.commit_secant_delta(update_successful=True)
    assert first_commit is second_commit
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.OK

    adapter.install_secant_midpoint()
    midpoint = tuple(shard.model_shard.clone() for shard in adapter.iter_owner_shards())
    adapter.install_secant_midpoint()
    assert all(
        torch.equal(shard.model_shard, expected)
        for shard, expected in zip(adapter.iter_owner_shards(), midpoint)
    )

    adapter.restore_secant_post()
    restored = tuple(shard.model_shard.clone() for shard in adapter.iter_owner_shards())
    adapter.restore_secant_post()
    adapter.install_secant_midpoint()
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.OK
    assert all(
        torch.equal(shard.model_shard, expected)
        for shard, expected in zip(adapter.iter_owner_shards(), restored)
    )


def test_invalid_adapter_ordering_rejects_before_weight_mutation() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters)
    assert adapter.begin_event() is not None
    before = tuple(shard.model_shard.clone() for shard in adapter.iter_owner_shards())

    adapter.install_secant_midpoint()

    assert adapter.local_status.item() == DistributedOptimizerEventStatus.SECANT_INVALID_PHASE
    assert all(
        torch.equal(shard.model_shard, expected)
        for shard, expected in zip(adapter.iter_owner_shards(), before)
    )


def test_authoritative_secant_estimate_composes_with_adapter_without_double_counting() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters, finish_chunk_elements=2)
    adapter_estimate = adapter.estimate_snapshot_memory()
    estimate = SecantMemoryEstimate.calculate(
        SecantMemoryInputs(
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            loaded_owner_elements_by_rank=(adapter_estimate.owner_elements,),
            tied_alias_elements_by_rank=(0,),
            replay_payload_bytes=128,
            replay_mask_bytes=64,
            replay_state_bytes=64,
            replay_cap_bytes=256,
            registry_slots=6,
            chunk_elements=2,
            alignment_bytes=1,
            allocator_headroom_fraction=0.1,
        ),
        optimizer_layout=SecantOptimizerLayout(),
    )

    additional = estimate.additional_bytes_for_adapter(adapter_estimate)
    preflight = adapter.preflight_snapshot_memory(additional_bytes=additional)

    assert preflight.requested_bytes == estimate.peak_bytes
    assert preflight.accepted


def test_adapter_estimate_covers_enumerated_commit_peak_including_hash_state(monkeypatch) -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters, finish_chunk_elements=2)
    estimate = adapter.estimate_snapshot_memory()
    observed = []
    original_transform = adapter._transform_pre_to_delta

    def enumerate_peak(snapshot, shards, scratch):
        tensors = (
            snapshot.master_before,
            snapshot.applied_before,
            snapshot.post_fingerprints,
            scratch.storage,
        )
        storages = {
            (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            for tensor in tensors
            if tensor is not None
        }
        observed.append(sum(size for _, size in storages))
        original_transform(snapshot, shards, scratch)

    monkeypatch.setattr(adapter, "_transform_pre_to_delta", enumerate_peak)
    assert adapter.begin_event() is not None
    _apply_step_and_materialize(adapter, optimizer)
    assert adapter.commit_secant_delta(update_successful=True) is not None

    assert observed == [estimate.total_bytes]


def test_secant_adapter_runtime_methods_have_no_host_sync_or_collectives() -> None:
    methods = (
        Bf16DistributedOptimizerDiagnosticAdapter.commit_secant_delta,
        Bf16DistributedOptimizerDiagnosticAdapter.install_secant_midpoint,
        Bf16DistributedOptimizerDiagnosticAdapter.restore_secant_post,
        Bf16DistributedOptimizerDiagnosticAdapter._transform_pre_to_delta,
        Bf16DistributedOptimizerDiagnosticAdapter._verify_secant_post,
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
    for method in methods:
        source = inspect.getsource(method)
        assert all(token not in source for token in prohibited)
