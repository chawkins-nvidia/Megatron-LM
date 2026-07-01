# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import math
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training.datasets.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
    RandomSeedDataset,
    SamplerIssuedIndex,
)
from megatron.training.diagnostics.accumulator import ReductionBinding
from megatron.training.diagnostics.diagnostic_replay import (
    DIAGNOSTIC_MASK_FIELD,
    SAMPLE_EPOCH_FIELD,
    SAMPLE_INDEX_FIELD,
    CollectiveBinding,
    FixedPlanCodec,
    NonInterleavedReplaySchedule,
    PopulationCollectiveWorkspace,
    ReadinessConsensus,
    RecordedBatch,
    ReplayBatchRecorder,
    ReplayIterator,
    ReplayMemoryConfig,
    ReplayMemoryPolicy,
    ReplayPlan,
    ReplayPreflightError,
    ReplayRestorationError,
    ReplaySnapshotAllocationError,
    ReplayStateGuard,
    SampleId,
    SamplePopulation,
    StableSampleDataset,
    Tier1ReplayEngine,
    TokenId,
    _systematic_positions,
    broadcast_replay_plan,
    build_distributed_source_plan,
    build_local_replay_plan,
    cp_sequence_columns,
    estimate_replay_memory,
    local_sample_populations,
    select_local_token_ids,
    slice_replay_mask,
)
from megatron.training.diagnostics.function_response import (
    RESPONSE_FAMILIES,
    FunctionResponseProbe,
    ResponseFamily,
    ResponseHookDescriptor,
)


class _MappingDataset(Dataset):
    def __init__(self, size: int, sequence_length: int = 8) -> None:
        self.size = size
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.arange(self.sequence_length, dtype=torch.int64) + index,
            "labels": torch.arange(self.sequence_length, dtype=torch.int64) + index + 1,
            "loss_mask": torch.ones(self.sequence_length, dtype=torch.float32),
            "position_ids": torch.arange(self.sequence_length, dtype=torch.int64),
        }


def _raw_batch(
    indices: tuple[int, ...] = (3, 7), *, epoch: int = 2, sequence_length: int = 8
) -> dict[str, torch.Tensor]:
    batch = len(indices)
    return {
        "tokens": torch.arange(batch * sequence_length, dtype=torch.int64).view(batch, -1),
        "labels": torch.arange(batch * sequence_length, dtype=torch.int64).view(batch, -1) + 1,
        "loss_mask": torch.ones(batch, sequence_length, dtype=torch.float32),
        "position_ids": torch.arange(sequence_length, dtype=torch.int64).expand(batch, -1),
        SAMPLE_INDEX_FIELD: torch.tensor(indices, dtype=torch.int64),
        SAMPLE_EPOCH_FIELD: torch.full((batch,), epoch, dtype=torch.int64),
    }


def _plan(*, target_microbatches: int = 2, sequence_length: int = 8):
    recorded = (RecordedBatch.from_raw(_raw_batch(sequence_length=sequence_length)),)
    populations = local_sample_populations(recorded)
    selected = select_local_token_ids(
        recorded, populations, probe_tokens=2, run_seed=11, event_id=5
    )
    return build_local_replay_plan(
        recorded, selected, micro_batch_size=2, target_microbatches=target_microbatches
    )


def test_sequential_sampler_issues_resume_stable_identities() -> None:
    dataset = StableSampleDataset(_MappingDataset(12))
    sampler = MegatronPretrainingSampler(
        total_samples=12,
        consumed_samples=4,
        micro_batch_size=2,
        data_parallel_rank=1,
        data_parallel_size=2,
        dataset=dataset,
    )

    issued = tuple(tuple(batch) for batch in sampler)

    assert issued == (
        (SamplerIssuedIndex(0, 6), SamplerIssuedIndex(0, 7)),
        (SamplerIssuedIndex(0, 10), SamplerIssuedIndex(0, 11)),
    )
    assert sampler.consumed_samples == 4


def test_ordinary_datasets_keep_integer_sampler_indices() -> None:
    sampler = MegatronPretrainingSampler(
        total_samples=8,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=0,
        data_parallel_size=1,
        dataset=_MappingDataset(8),
    )

    assert all(isinstance(index, int) for batch in sampler for index in batch)


@pytest.mark.parametrize("data_sharding", (False, True))
@pytest.mark.parametrize("wrapper_order", ("stable_outer", "random_outer"))
def test_cyclic_identity_crosses_persistent_worker_epochs_and_resume(
    data_sharding: bool, wrapper_order: str
) -> None:
    base = _MappingDataset(24)
    dataset = (
        StableSampleDataset(RandomSeedDataset(base, seed=123))
        if wrapper_order == "stable_outer"
        else RandomSeedDataset(StableSampleDataset(base), seed=123)
    )
    sampler = MegatronPretrainingRandomSampler(
        dataset,
        total_samples=24,
        consumed_samples=0,
        micro_batch_size=2,
        data_parallel_rank=0,
        data_parallel_size=2,
        data_sharding=data_sharding,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2, persistent_workers=True)

    first = [
        (int(epoch), int(index))
        for batch in loader
        for epoch, index in zip(batch[SAMPLE_EPOCH_FIELD], batch[SAMPLE_INDEX_FIELD], strict=True)
    ]
    second = [
        (int(epoch), int(index))
        for batch in loader
        for epoch, index in zip(batch[SAMPLE_EPOCH_FIELD], batch[SAMPLE_INDEX_FIELD], strict=True)
    ]
    resumed = MegatronPretrainingRandomSampler(
        dataset,
        total_samples=24,
        consumed_samples=48,
        micro_batch_size=2,
        data_parallel_rank=0,
        data_parallel_size=2,
        data_sharding=data_sharding,
    )
    third = [identity for batch in resumed for identity in batch]
    if loader._iterator is not None:
        loader._iterator._shutdown_workers()

    assert {epoch for epoch, _index in first} == {0}
    assert {epoch for epoch, _index in second} == {1}
    assert {identity.epoch for identity in third} == {2}
    assert len(set(first + second + [(item.epoch, item.sampler_index) for item in third])) == 36


def test_replay_does_not_advance_sampler_and_masks_filler_deterministically() -> None:
    sampler = MegatronPretrainingRandomSampler(
        _MappingDataset(16),
        total_samples=16,
        consumed_samples=4,
        micro_batch_size=2,
        data_parallel_rank=0,
        data_parallel_size=1,
        data_sharding=False,
    )
    consumed = sampler.consumed_samples
    recorded = (RecordedBatch.from_raw(_raw_batch()),)
    populations = (
        SamplePopulation(recorded[0].samples[0].sample_id, 8),
        SamplePopulation(recorded[0].samples[1].sample_id, 8),
    )
    selected = select_local_token_ids(recorded, populations, probe_tokens=1, run_seed=3, event_id=9)

    plan = build_local_replay_plan(recorded, selected, micro_batch_size=2, target_microbatches=3)

    assert sampler.consumed_samples == consumed
    assert plan.num_microbatches == 3
    assert plan.microbatches[0].data["loss_mask"][1].count_nonzero() == 0
    assert plan.microbatches[1].data["loss_mask"].count_nonzero() == 0
    assert plan.microbatches[2].data[DIAGNOSTIC_MASK_FIELD].count_nonzero() == 0
    assert plan.microbatches[1].sample_ids == plan.microbatches[2].sample_ids


def test_systematic_grid_is_exactly_uniform_and_seed_replay_is_unbiased() -> None:
    exhaustive = [_systematic_positions(total=7, selected=3, offset=offset) for offset in range(7)]
    assert [sum(position in grid for grid in exhaustive) for position in range(7)] == [3] * 7

    raw = _raw_batch(indices=(10, 20), sequence_length=2)
    raw["loss_mask"] = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    recorded = (RecordedBatch.from_raw(raw),)
    populations = local_sample_populations(recorded)
    counts = {10: 0, 20: 0}
    for seed in range(6000):
        selected = select_local_token_ids(
            recorded, populations, probe_tokens=1, run_seed=seed, event_id=9
        )
        replayed = select_local_token_ids(
            recorded, populations, probe_tokens=1, run_seed=seed, event_id=9
        )
        assert selected == replayed
        counts[selected[0].sample.sampler_index] += 1

    assert counts[10] / 6000 == pytest.approx(1 / 3, abs=0.025)
    assert counts[20] / 6000 == pytest.approx(2 / 3, abs=0.025)


def test_raw_replay_fields_are_fixed_and_attention_masks_fail_closed() -> None:
    raw = _raw_batch()
    raw["attention_mask"] = torch.ones(2, 1, 8, 8, dtype=torch.bool)

    with pytest.raises(ValueError, match="exactly the fixed"):
        RecordedBatch.from_raw(raw)


def test_replay_recorder_rejects_batch_and_byte_overflow_before_retaining() -> None:
    raw = _raw_batch()
    one_batch_bytes = RecordedBatch.from_raw(raw).tensor_bytes
    batch_limited = ReplayBatchRecorder(
        iter((raw, raw)), maximum_batches=1, maximum_host_bytes=2 * one_batch_bytes
    )
    next(batch_limited)
    with pytest.raises(ReplayPreflightError, match="batch cap"):
        next(batch_limited)
    assert len(batch_limited.recorded) == 1

    byte_limited = ReplayBatchRecorder(
        iter((raw,)), maximum_batches=1, maximum_host_bytes=one_batch_bytes - 1
    )
    with pytest.raises(ReplayPreflightError, match="byte cap"):
        next(byte_limited)
    assert byte_limited.recorded == []
    assert byte_limited.host_capture_bytes == 0


def test_fixed_plan_codec_constructs_equal_neutral_non_source_plan() -> None:
    source = _plan()
    codec = FixedPlanCodec(8, 4, 2, 8)
    binding = CollectiveBinding("tp", None, 1)
    readiness = ReadinessConsensus(binding)

    received = broadcast_replay_plan(
        source, codec=codec, binding=binding, source_group_rank=0, readiness=readiness
    )
    integer, mask = codec.allocate("cpu")
    codec.encode(source, integer, mask)
    metadata = codec.decode(integer, mask)
    neutral = replace(received, metadata=metadata, microbatches=[], source_rank=False)

    assert neutral.metadata == source.metadata
    assert neutral.metadata.descriptor_hash == source.metadata.descriptor_hash
    assert neutral.num_microbatches == source.num_microbatches
    assert not neutral.source_rank


def test_bounded_population_workspace_builds_deterministic_source_plan() -> None:
    recorded = (RecordedBatch.from_raw(_raw_batch()),)
    binding = CollectiveBinding("dp", None, 1)
    readiness = ReadinessConsensus(binding)
    workspace = PopulationCollectiveWorkspace(binding, maximum_local_samples=4, device="cpu")

    first = build_distributed_source_plan(
        recorded,
        workspace=workspace,
        readiness=readiness,
        probe_tokens=3,
        run_seed=17,
        event_id=2,
        micro_batch_size=2,
        maximum_microbatches=4,
    )
    second = build_distributed_source_plan(
        recorded,
        workspace=workspace,
        readiness=readiness,
        probe_tokens=3,
        run_seed=17,
        event_id=2,
        micro_batch_size=2,
        maximum_microbatches=4,
    )

    assert first.metadata == second.metadata
    assert len(first.metadata.selected_tokens) == 3


@pytest.mark.parametrize("sequence_length", (128, 1024))
@pytest.mark.parametrize("micro_batch_size", (1, 4))
def test_cp_sp_mask_slicing_matches_layout(sequence_length: int, micro_batch_size: int) -> None:
    mask = torch.arange(micro_batch_size * sequence_length).view(micro_batch_size, -1) % 3 == 0
    cp, sp = slice_replay_mask(
        mask,
        context_parallel_size=4,
        context_parallel_rank=2,
        sequence_parallel=True,
        tensor_parallel_size=2,
        tensor_parallel_rank=1,
    )
    columns = cp_sequence_columns(sequence_length, 4, 2)

    torch.testing.assert_close(cp, mask.index_select(1, columns))
    assert cp.shape == (micro_batch_size, sequence_length // 4)
    assert sp is not None
    torch.testing.assert_close(sp, cp[:, sequence_length // 8 :])


class _Tracker:
    def __init__(self) -> None:
        self.states = {"model-parallel-rng": torch.tensor([1, 2, 3], dtype=torch.uint8)}
        self._current_state_name = "model-parallel-rng"

    def get_states(self):
        return self.states

    def set_states(self, states):
        self.states = states


class _StatefulModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cache", torch.tensor([2.0]))
        self.input_tensor = torch.tensor([4.0])


def _raise_fault() -> None:
    raise RuntimeError("injected restoration fault")


@pytest.mark.parametrize(
    "stage",
    (
        "model_restore",
        "identity_restore",
        "overlap_restore",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "tracker_rng",
        "model_verify",
        "identity_verify",
        "overlap_verify",
    ),
)
def test_every_restore_stage_fault_still_attempts_later_restorations(stage: str) -> None:
    model = _StatefulModel()
    tracker = _Tracker()
    plan = _plan(target_microbatches=1)
    iterator = iter(plan.microbatches)
    torch_before = torch.get_rng_state().clone()
    tracker_before = tracker.states["model-parallel-rng"].clone()
    guard = ReplayStateGuard(
        (model,),
        mutable_buffer_names=("cache",),
        tracker_getter=lambda: tracker,
        fault_injections={stage: _raise_fault},
    )

    with pytest.raises(ReplayRestorationError) as caught:
        with guard:
            model.cache.add_(7)
            model.input_tensor.add_(9)
            model.eval()
            random.random()
            np.random.random()
            torch.rand(4)
            tracker.states["model-parallel-rng"].add_(5)
            next(iterator)

    assert stage in {name for name, _error in caught.value.failures}
    if stage not in ("torch_rng",):
        assert torch.equal(torch.get_rng_state(), torch_before)
    if stage != "tracker_rng":
        assert torch.equal(tracker.states["model-parallel-rng"], tracker_before)
    if stage not in ("model_restore",):
        assert model.training
        assert model.cache.item() == 2
        assert model.input_tensor.item() == 4


def test_nested_state_guards_restore_their_independent_capture_points() -> None:
    model = _StatefulModel()
    tracker = _Tracker()
    with ReplayStateGuard(
        (model,), mutable_buffer_names=("cache",), tracker_getter=lambda: tracker
    ):
        model.cache.fill_(5)
        with ReplayStateGuard(
            (model,), mutable_buffer_names=("cache",), tracker_getter=lambda: tracker
        ):
            model.cache.fill_(9)
        assert model.cache.item() == 5
    assert model.cache.item() == 2


def test_iterator_and_sampler_identity_are_restored_independently() -> None:
    model = _StatefulModel()
    tracker = _Tracker()
    replay_iterator = ReplayIterator(_plan(target_microbatches=2))
    sampler = type("SamplerState", (), {"consumed_samples": 12, "epoch": 3})()

    with ReplayStateGuard(
        (model,),
        mutable_buffer_names=("cache",),
        replay_iterators=(replay_iterator,),
        samplers=(sampler,),
        tracker_getter=lambda: tracker,
    ):
        next(replay_iterator)
        sampler.consumed_samples = 99
        sampler.epoch = 7

    assert replay_iterator.index == 0
    assert sampler.consumed_samples == 12
    assert sampler.epoch == 3


def test_bogus_rng_tracker_is_rejected_before_replay() -> None:
    model = _StatefulModel()
    with pytest.raises(TypeError, match="get_states/set_states"):
        with ReplayStateGuard(
            (model,), mutable_buffer_names=("cache",), tracker_getter=lambda: object()
        ):
            pass


def test_active_overlap_state_is_rejected_before_capture() -> None:
    model = _StatefulModel()
    tracker = _Tracker()
    overlap = type("Overlap", (), {"param_gather_handle": object()})()

    with pytest.raises(ValueError, match="quiescent overlap"):
        with ReplayStateGuard(
            (model,),
            mutable_buffer_names=("cache",),
            overlap_objects=(overlap,),
            tracker_getter=lambda: tracker,
        ):
            pass


def test_quiescent_none_overlap_handles_prepare_restore_and_verify() -> None:
    overlap = SimpleNamespace(param_gather_handle=None)
    guard = ReplayStateGuard(
        (torch.nn.Identity(),), overlap_objects=(overlap,), tracker_getter=lambda: _Tracker()
    )

    guard.prepare()
    with guard:
        overlap.param_gather_handle = object()
        overlap.grad_reduce_handle = object()

    assert overlap.param_gather_handle is None
    assert not hasattr(overlap, "grad_reduce_handle")


def test_supported_immutable_and_tensor_overlap_state_shares_model_snapshot_plan() -> None:
    model = torch.nn.Identity()
    shared = torch.arange(1024, dtype=torch.float32)
    model.shared_state = shared
    overlap = SimpleNamespace(
        param_gather_handle=None,
        phase=("ready", torch.float32, torch.device("cpu")),
        tensor_state=shared,
    )
    guard = ReplayStateGuard(
        (model,),
        overlap_objects=(overlap,),
        tracker_getter=lambda: _Tracker(),
        maximum_model_state_bytes=4096,
    )

    guard.prepare()

    assert guard.model.tensor_bytes == 4096
    assert guard.model.snapshot_plan is not None
    assert len(guard.model.snapshot_plan.storages) == 1
    with guard:
        shared.add_(9)
        overlap.phase = ("running", torch.float64, torch.device("cpu"))
        overlap.tensor_state = shared.clone()

    assert overlap.phase == ("ready", torch.float32, torch.device("cpu"))
    assert overlap.tensor_state is shared
    torch.testing.assert_close(shared, torch.arange(1024, dtype=torch.float32))


def test_overlap_tensor_state_is_rejected_by_shared_snapshot_cap() -> None:
    overlap = SimpleNamespace(param_gather_handle=None, tensor_state=torch.ones(1024))

    with pytest.raises(ValueError, match="snapshot exceeds its declared byte cap"):
        ReplayStateGuard(
            (torch.nn.Identity(),),
            overlap_objects=(overlap,),
            tracker_getter=lambda: _Tracker(),
            maximum_model_state_bytes=4000,
        ).prepare()


def test_undeclared_registered_buffer_is_rejected_before_mutation() -> None:
    model = _StatefulModel()
    tracker = _Tracker()

    with pytest.raises(ValueError, match="registered model buffers"):
        with ReplayStateGuard((model,), tracker_getter=lambda: tracker):
            model.cache.fill_(99)

    assert model.cache.item() == 2


def test_declared_buffer_registry_and_lazily_created_cache_are_restored() -> None:
    model = _StatefulModel()
    tracker = _Tracker()
    original_cache = model.cache

    with ReplayStateGuard(
        (model,), mutable_buffer_names=("cache",), tracker_getter=lambda: tracker
    ):
        model.cache = torch.tensor([99.0])
        model.register_buffer("replay_only", torch.tensor([7.0]))
        model._decoder_hidden_states_cache = torch.tensor([5.0])

    assert model.cache is original_cache
    assert model.cache.item() == 2
    assert "replay_only" not in model._buffers
    assert not hasattr(model, "_decoder_hidden_states_cache")


def test_mutable_bytearray_cache_is_rejected_even_with_zero_snapshot_cap() -> None:
    model = torch.nn.Identity()
    model.custom_cache = bytearray(b"abc")

    with pytest.raises(TypeError, match="unsupported mutable model-state value"):
        ReplayStateGuard(
            (model,), tracker_getter=lambda: _Tracker(), maximum_model_state_bytes=0
        ).prepare()

    assert model.custom_cache == bytearray(b"abc")


def test_recursive_snapshot_clones_32_tensor_aliases_once_under_cap() -> None:
    model = _dense_gpt_stub()
    model.last_child = torch.nn.Identity()
    shared = torch.arange(1024, dtype=torch.float32)
    model.last_child.alias_cache = [shared] * 32
    guard = ReplayStateGuard(
        (model,), tracker_getter=lambda: _Tracker(), maximum_model_state_bytes=10_000
    )

    guard.prepare()

    assert guard.model.tensor_bytes == 4096
    assert guard.model.snapshot_plan is not None
    assert len(guard.model.snapshot_plan.storages) == 1
    with guard:
        shared.add_(7)
        model.last_child.alias_cache[0] = shared.clone()

    assert all(value is shared for value in model.last_child.alias_cache)
    torch.testing.assert_close(shared, torch.arange(1024, dtype=torch.float32))


def test_recursive_snapshot_restores_partial_view_alias_topology_and_contents() -> None:
    model = torch.nn.Identity()
    base = torch.arange(1024, dtype=torch.float32)
    partial = base[17:81]
    strided = base[5:133:2]
    model.view_cache = [base, partial, strided]
    guard = ReplayStateGuard(
        (model,), tracker_getter=lambda: _Tracker(), maximum_model_state_bytes=5000
    )

    guard.prepare()

    assert guard.model.tensor_bytes == 4096
    assert guard.model.snapshot_plan is not None
    assert len(guard.model.snapshot_plan.storages) == 1
    with guard:
        base.fill_(-1)
        partial.set_(torch.zeros(64))
        model.view_cache[:] = [torch.zeros(1)]

    assert len(model.view_cache) == 3
    assert model.view_cache[0] is base
    assert model.view_cache[1] is partial
    assert model.view_cache[2] is strided
    assert partial.untyped_storage()._cdata == base.untyped_storage()._cdata
    assert strided.untyped_storage()._cdata == base.untyped_storage()._cdata
    assert partial.storage_offset() == 17
    assert strided.storage_offset() == 5
    assert strided.stride() == (2,)
    torch.testing.assert_close(base, torch.arange(1024, dtype=torch.float32))


def test_recursive_snapshot_rejects_cycles_before_any_storage_clone(monkeypatch) -> None:
    from megatron.training.diagnostics import diagnostic_replay

    model = torch.nn.Identity()
    tensor = torch.ones(16)
    cycle = [tensor]
    cycle.append(cycle)
    model.cycle_cache = cycle
    clones = 0

    def unexpected_clone(_self):
        nonlocal clones
        clones += 1

    monkeypatch.setattr(diagnostic_replay._StorageSnapshot, "capture", unexpected_clone)

    with pytest.raises(TypeError, match="cyclic mutable model-state"):
        ReplayStateGuard((model,), tracker_getter=lambda: _Tracker()).prepare()

    assert clones == 0


def test_snapshot_allocation_failure_has_typed_pre_readiness_error(monkeypatch) -> None:
    from megatron.training.diagnostics import diagnostic_replay

    model = torch.nn.Identity()
    model.tensor_cache = torch.ones(16)

    def fail_allocation(_self):
        raise RuntimeError("injected allocator failure")

    monkeypatch.setattr(diagnostic_replay._StorageSnapshot, "capture", fail_allocation)

    with pytest.raises(ReplaySnapshotAllocationError, match="allocation failed before readiness"):
        ReplayStateGuard((model,), tracker_getter=lambda: _Tracker()).prepare()


class _Probe:
    def __init__(self, expected: int) -> None:
        self.expected_hook_calls = expected
        self.descriptor_hash = "00" * 32
        self.masks = []

    def set_masks(self, full_mask, *, sequence_parallel_mask=None):
        self.masks.append((full_mask.clone(), sequence_parallel_mask))

    @contextlib.contextmanager
    def capture_pre(self):
        yield

    @contextlib.contextmanager
    def capture_post(self):
        yield

    def finalize(self):
        return self

    def release(self):
        return None


def test_pp2_noninterleaved_schedule_runs_multiple_microbatches_and_accumulation() -> None:
    plan = _plan(target_microbatches=3)
    model = torch.nn.Identity()
    accumulation = []

    def forward_step(data_iterator, replay_model):
        batch = next(data_iterator)
        accumulation.append(batch["tokens"].sum())
        output = replay_model(batch["tokens"].float())
        return output, lambda value: value

    def schedule(**kwargs):
        for _ in range(kwargs["num_microbatches"]):
            output, reducer = kwargs["forward_step_func"](
                kwargs["data_iterator"], kwargs["model"][0]
            )
            reducer(output)

    probe = _Probe(3)
    first_stage = NonInterleavedReplaySchedule(
        forward_backward_func=schedule,
        forward_step_func=forward_step,
        model=(model,),
        sequence_length=8,
        micro_batch_size=2,
        probe_device="cpu",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        pipeline_data_owner=True,
    )
    last_stage_calls = []

    def last_forward(_data_iterator, replay_model):
        last_stage_calls.append(1)
        output = replay_model(torch.ones(2, 8))
        return output, lambda value: value

    last_probe = _Probe(3)
    last_stage = NonInterleavedReplaySchedule(
        forward_backward_func=schedule,
        forward_step_func=last_forward,
        model=(model,),
        sequence_length=8,
        micro_batch_size=2,
        probe_device="cpu",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        pipeline_data_owner=False,
    )

    first_stage(plan, probe, "pre")
    last_stage(plan, last_probe, "pre")

    assert len(accumulation) == 3
    assert len(probe.masks) == 3
    assert len(last_probe.masks) == 3
    assert len(last_stage_calls) == 3
    assert all(mask.shape == (2, 8) for mask, _sp in probe.masks)
    assert first_stage.p2p_started and last_stage.p2p_started


def test_interleaved_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="interleaved"):
        NonInterleavedReplaySchedule(
            forward_backward_func=lambda **kwargs: None,
            forward_step_func=lambda *args: None,
            model=(torch.nn.Identity(), torch.nn.Identity()),
            sequence_length=8,
            micro_batch_size=1,
            probe_device="cpu",
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            virtual_pipeline_size=2,
        )


def _dense_gpt_stub() -> GPTModel:
    model = GPTModel.__new__(GPTModel)
    torch.nn.Module.__init__(model)
    model.config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        transformer_impl="local",
        params_dtype=torch.float32,
    )
    model.model_type = ModelType.encoder_or_decoder
    model.mtp_process = False
    return model


def _dense_engine_fixture(*, gated: bool = False, scratch_capacity: int = 2):
    model = _dense_gpt_stub()
    model.config.gated_linear_unit = gated
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_number = 1
    layer.is_moe_layer = False
    attention = SelfAttention.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    mlp = MLP.__new__(MLP)
    torch.nn.Module.__init__(mlp)
    qkv = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(qkv)
    projection = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(projection)
    fc1 = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(fc1)
    fc2 = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(fc2)
    core_attention = DotProductAttention.__new__(DotProductAttention)
    torch.nn.Module.__init__(core_attention)
    qkv.gather_output = False
    qkv.skip_bias_add = False
    qkv.sequence_parallel = False
    qkv.allreduce_dgrad = False
    projection.skip_bias_add = False
    projection.sequence_parallel = False
    fc1.gather_output = False
    fc1.skip_bias_add = True
    fc1.sequence_parallel = False
    fc1.allreduce_dgrad = False
    fc2.skip_bias_add = True
    fc2.sequence_parallel = False
    attention.linear_qkv = qkv
    attention.linear_proj = projection
    attention.core_attention = core_attention
    mlp.linear_fc1 = fc1
    mlp.linear_fc2 = fc2
    layer.self_attention = attention
    layer.mlp = mlp
    model.decoder_layer = layer
    modules = {
        ResponseFamily.RESIDUAL: layer,
        ResponseFamily.QKV: qkv,
        ResponseFamily.ATTN_OUT: projection,
        ResponseFamily.FC1: fc1,
        ResponseFamily.FC2: fc2,
    }
    descriptors = tuple(
        ResponseHookDescriptor(
            global_layer=0,
            family=family,
            module=modules[family],
            owner=True,
            sequence_sharded=False,
            affine_bias_output=family != ResponseFamily.RESIDUAL,
        )
        for family in RESPONSE_FAMILIES
    )
    plan = _plan()
    probe = FunctionResponseProbe(
        descriptors,
        global_layers=1,
        device="cpu",
        expected_hook_calls=plan.num_microbatches,
        reduction_binding=ReductionBinding.flat_world(None),
        scratch_element_capacity=scratch_capacity,
    )
    schedule = NonInterleavedReplaySchedule(
        forward_backward_func=lambda **kwargs: None,
        forward_step_func=lambda *args: None,
        model=(model,),
        sequence_length=8,
        micro_batch_size=2,
        probe_device="cpu",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
    )
    tracker = _Tracker()
    engine = Tier1ReplayEngine(
        (model,),
        readiness=ReadinessConsensus(CollectiveBinding("world", None, 1)),
        tracker_getter=lambda: tracker,
    )
    return engine, model, plan, probe, schedule


def _prepare_fixture(engine, plan, probe, schedule, *, maximum_extra_bytes: int = 2**40):
    return engine.prepare(
        plan=plan,
        probe=probe,
        schedule=schedule,
        memory_policy=ReplayMemoryPolicy(),
        maximum_extra_bytes=maximum_extra_bytes,
        currently_reserved_bytes=0,
        total_device_bytes=2**41,
    )


def _count_schedule_calls(schedule):
    calls = []

    def forward_backward(**kwargs):
        calls.append(kwargs["num_microbatches"])
        for _ in range(kwargs["num_microbatches"]):
            kwargs["forward_step_func"](kwargs["data_iterator"], kwargs["model"][0])

    def forward_step(_data_iterator, _model):
        return torch.zeros(1), lambda value: value

    schedule.forward_backward_func = forward_backward
    schedule.forward_step_func = forward_step
    return calls


def test_engine_prepare_makes_memory_preflight_mandatory_and_engine_owned() -> None:
    binding = CollectiveBinding("world", None, 1)
    tracker = _Tracker()
    model = _dense_gpt_stub()
    engine = Tier1ReplayEngine(
        (model,), readiness=ReadinessConsensus(binding), tracker_getter=lambda: tracker
    )
    probe = _Probe(2)
    probe.global_layers = 1
    probe.descriptors = (SimpleNamespace(global_layer=0),)
    schedule = NonInterleavedReplaySchedule(
        forward_backward_func=lambda **kwargs: None,
        forward_step_func=lambda *args: None,
        model=(model,),
        sequence_length=8,
        micro_batch_size=2,
        probe_device="cpu",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
    )

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        engine.prepare(
            plan=_plan(),
            probe=probe,
            schedule=schedule,
            memory_policy=ReplayMemoryPolicy(),
            maximum_extra_bytes=0,
            currently_reserved_bytes=0,
            total_device_bytes=2**40,
        )

    with pytest.raises(TypeError, match="memory_estimate"):
        engine.prepare(
            plan=_plan(),
            probe=probe,
            schedule=schedule,
            memory_estimate=ReplayMemoryConfig,  # type: ignore[call-arg]
            maximum_extra_bytes=2**40,
            currently_reserved_bytes=0,
            total_device_bytes=2**40,
        )


def test_engine_rejects_schedule_model_identity_mismatch_before_schedule() -> None:
    engine, _model, plan, probe, schedule = _dense_engine_fixture()
    other = _dense_gpt_stub()
    schedule.model = (other,)

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        _prepare_fixture(engine, plan, probe, schedule)

    assert not schedule.p2p_started


def test_engine_rejects_arbitrary_child_module_before_schedule() -> None:
    engine, model, plan, probe, schedule = _dense_engine_fixture()

    class Bad(torch.nn.Module):
        pass

    model.bad = Bad()

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        _prepare_fixture(engine, plan, probe, schedule)

    assert not schedule.p2p_started


def test_engine_recomputes_selected_tokens_from_fixed_masks() -> None:
    engine, _model, plan, probe, schedule = _dense_engine_fixture()
    forged_masks = tuple(tuple(True for _ in mask) for mask in plan.metadata.diagnostic_masks)
    plan.metadata = replace(plan.metadata, diagnostic_masks=forged_masks)

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        _prepare_fixture(engine, plan, probe, schedule)

    assert not schedule.p2p_started


def test_engine_rejects_self_consistent_count_with_nonexistent_selected_samples() -> None:
    engine, _model, plan, probe, schedule = _dense_engine_fixture()
    forged_ordered = tuple(
        SampleId(999 + index, 999 + index) for index in range(len(plan.metadata.ordered_sample_ids))
    )
    forged_selected = tuple(
        sorted(
            TokenId(
                forged_ordered[
                    batch_index * plan.metadata.micro_batch_size
                    + flat_index // plan.metadata.sequence_length
                ],
                flat_index % plan.metadata.sequence_length,
            )
            for batch_index, mask in enumerate(plan.metadata.diagnostic_masks)
            for flat_index, selected in enumerate(mask)
            if selected
        )
    )
    for batch_index, batch in enumerate(plan.microbatches):
        start = batch_index * plan.metadata.micro_batch_size
        batch.sample_ids = forged_ordered[start : start + plan.metadata.micro_batch_size]
        batch.data[SAMPLE_EPOCH_FIELD].copy_(
            torch.tensor([sample.epoch for sample in batch.sample_ids])
        )
        batch.data[SAMPLE_INDEX_FIELD].copy_(
            torch.tensor([sample.sampler_index for sample in batch.sample_ids])
        )
    forged_metadata = replace(
        plan.metadata, selected_tokens=forged_selected, ordered_sample_ids=forged_ordered
    )
    forged_plan = ReplayPlan(forged_metadata, plan.microbatches, True)

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        _prepare_fixture(engine, forged_plan, probe, schedule)

    assert not schedule.p2p_started


def test_engine_rejects_selected_position_outside_true_loss_mask() -> None:
    engine, _model, plan, probe, schedule = _dense_engine_fixture()
    selected = plan.metadata.selected_tokens[0]
    batch_index = next(
        index
        for index, batch in enumerate(plan.microbatches)
        if selected.sample in batch.sample_ids
    )
    lane = plan.microbatches[batch_index].sample_ids.index(selected.sample)
    plan.microbatches[batch_index].data["loss_mask"][lane, selected.sequence_column] = 0

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        _prepare_fixture(engine, plan, probe, schedule)

    assert not schedule.p2p_started


@pytest.mark.parametrize("mutation", ("unknown", "oversized"))
def test_engine_rejects_changed_fixed_batch_fields_before_schedule(mutation: str) -> None:
    engine, _model, plan, probe, schedule = _dense_engine_fixture()
    if mutation == "unknown":
        plan.microbatches[0].data["caller_field"] = torch.zeros(2, 8)
    else:
        plan.microbatches[0].data["tokens"] = torch.zeros(2, 9, dtype=torch.int64)

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        _prepare_fixture(engine, plan, probe, schedule)

    assert not schedule.p2p_started


def test_engine_binds_gated_fc1_width_and_actual_probe_scratch() -> None:
    engine, _model, plan, probe, schedule = _dense_engine_fixture(
        gated=True, scratch_capacity=1_000_000
    )
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    assert transaction.memory_estimate is not None
    expected_response = (16 + 48 + 16 + 128 + 16) * 2 * 4
    assert transaction.memory_estimate.term("retained_response_rows") == expected_response
    assert (
        transaction.memory_estimate.term("accumulator_scratch")
        == probe.maximum_accumulator_scratch_bytes
        == 96_000_000
    )


def test_engine_revalidates_bound_plan_and_topology_before_schedule() -> None:
    engine, _model, plan, probe, schedule = _dense_engine_fixture()
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    schedule.tp_rank = 1

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        transaction.run_pre()

    assert not schedule.p2p_started


def test_exact_reviewer_pre_execution_attribute_drift_calls_no_schedule() -> None:
    engine, model, plan, probe, schedule = _dense_engine_fixture()
    schedule_calls = _count_schedule_calls(schedule)
    fc1 = model.decoder_layer.mlp.linear_fc1
    fc1.skip_bias_add = True
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    fc1.skip_bias_add = False

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        transaction.run_pre()

    assert schedule_calls == []
    assert not schedule.p2p_started


def test_exact_reviewer_post_attribute_and_mode_drift_calls_no_post_schedule() -> None:
    engine, model, plan, probe, schedule = _dense_engine_fixture()
    schedule_calls = _count_schedule_calls(schedule)
    fc1 = model.decoder_layer.mlp.linear_fc1
    fc1.skip_bias_add = False
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    transaction.run_pre()
    fc1.skip_bias_add = True
    model.eval()

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        transaction.finish(update_succeeded=True)

    assert schedule_calls == [plan.num_microbatches]
    assert fc1.skip_bias_add
    assert not model.training


@pytest.mark.parametrize("phase", ("pre", "post"))
@pytest.mark.parametrize(
    ("module_path", "attribute"),
    (
        (("decoder_layer", "self_attention", "linear_qkv"), "gather_output"),
        (("decoder_layer", "self_attention", "linear_qkv"), "sequence_parallel"),
        (("decoder_layer", "self_attention", "linear_qkv"), "allreduce_dgrad"),
        (("decoder_layer", "self_attention", "linear_proj"), "skip_bias_add"),
        (("decoder_layer", "self_attention", "linear_proj"), "sequence_parallel"),
        (("decoder_layer", "mlp", "linear_fc1"), "skip_bias_add"),
    ),
)
def test_parallel_linear_flag_drift_is_rejected_at_both_final_gates(
    phase: str, module_path: tuple[str, ...], attribute: str
) -> None:
    engine, model, plan, probe, schedule = _dense_engine_fixture()
    schedule_calls = _count_schedule_calls(schedule)
    module = model
    for name in module_path:
        module = getattr(module, name)
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    if phase == "post":
        transaction.run_pre()
    setattr(module, attribute, not getattr(module, attribute))

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        if phase == "pre":
            transaction.run_pre()
        else:
            transaction.finish(update_succeeded=True)

    assert schedule_calls == ([] if phase == "pre" else [plan.num_microbatches])


@pytest.mark.parametrize("phase", ("pre", "post"))
def test_every_module_training_mode_is_rejected_at_both_final_gates(phase: str) -> None:
    engine, model, plan, probe, schedule = _dense_engine_fixture()
    schedule_calls = _count_schedule_calls(schedule)
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    if phase == "post":
        transaction.run_pre()
    for module in model.modules():
        module.training = not module.training

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        if phase == "pre":
            transaction.run_pre()
        else:
            transaction.finish(update_succeeded=True)

    assert schedule_calls == ([] if phase == "pre" else [plan.num_microbatches])


@pytest.mark.parametrize(
    "surface", ("runtime_binding", "unsupported_state", "mutable_tensor_state")
)
def test_unmodeled_module_state_drift_fails_closed_before_schedule(surface: str) -> None:
    engine, model, plan, probe, schedule = _dense_engine_fixture()
    schedule_calls = _count_schedule_calls(schedule)
    fc1 = model.decoder_layer.mlp.linear_fc1
    if surface == "runtime_binding":
        fc1.tp_group = object()
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    if surface == "runtime_binding":
        fc1.tp_group = object()
    else:
        fc1.unmodeled_mutable_state = (
            bytearray(b"drift") if surface == "unsupported_state" else (torch.ones(1),)
        )

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        transaction.run_pre()

    assert schedule_calls == []
    assert not schedule.p2p_started


def test_exact_reviewer_post_preflight_graph_schedule_config_and_batch_drift() -> None:
    engine, model, plan, probe, schedule = _dense_engine_fixture()
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    original_fc1 = model.decoder_layer.mlp.linear_fc1
    replacement = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(replacement)
    model.decoder_layer.mlp.linear_fc1 = replacement
    model.config.sequence_parallel = not model.config.sequence_parallel
    schedule.decoder_sequence_length = 999
    schedule.adjust_tensor_shapes_fn = lambda shapes: shapes
    schedule.pg_collection = object()
    plan.microbatches[0].data["tokens"].fill_(999)

    assert probe.descriptors[3].module is original_fc1
    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        transaction.run_pre()

    assert not schedule.p2p_started


@pytest.mark.parametrize("surface", ("parameter", "buffer", "module_spec"))
def test_parameter_buffer_and_module_spec_graph_drift_is_rejected(surface: str) -> None:
    engine, model, plan, probe, schedule = _dense_engine_fixture()
    if surface == "parameter":
        model.decoder_layer.register_parameter("graph_weight", torch.nn.Parameter(torch.ones(4)))
    elif surface == "buffer":
        model.decoder_layer.register_buffer("graph_cache", torch.ones(4))
        engine.mutable_buffer_names = ("decoder_layer.graph_cache",)
    else:
        model.transformer_layer_spec = ModuleSpec(module=TransformerLayer, params={"marker": [1]})
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    if surface == "parameter":
        model.decoder_layer.graph_weight = torch.nn.Parameter(torch.ones(4))
    elif surface == "buffer":
        model.decoder_layer.graph_cache = torch.ones(4)
    else:
        model.transformer_layer_spec.params["marker"].append(2)

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        transaction.run_pre()

    assert not schedule.p2p_started


def test_execution_graph_is_revalidated_again_immediately_before_post_schedule() -> None:
    engine, model, plan, probe, schedule = _dense_engine_fixture()
    schedule_calls = _count_schedule_calls(schedule)
    transaction = _prepare_fixture(engine, plan, probe, schedule)
    transaction.run_pre()
    model.config.sequence_parallel = not model.config.sequence_parallel

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        transaction.finish(update_succeeded=True)

    assert schedule_calls == [plan.num_microbatches]


def test_engine_rejects_96mb_actual_probe_scratch_under_10mb_cap() -> None:
    engine, _model, plan, probe, schedule = _dense_engine_fixture(scratch_capacity=1_000_000)

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        _prepare_fixture(engine, plan, probe, schedule, maximum_extra_bytes=10_000_000)

    assert not schedule.p2p_started


def _memory_config(**overrides) -> ReplayMemoryConfig:
    values = dict(
        sequence_length=1024,
        micro_batch_size=4,
        global_batch_size=512,
        data_parallel_size=8,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        context_parallel_size=2,
        global_layers=32,
        local_layers=16,
        hidden_size=4096,
        ffn_hidden_size=16384,
        num_attention_heads=32,
        selected_tokens=256,
        replay_microbatches=64,
        element_size=2,
        rng_snapshot_bytes=8192,
        model_state_bytes=32768,
        fixed_workspace_bytes=65536,
        alignment=256,
        headroom_fraction=0.1,
    )
    values.update(overrides)
    if "replay_microbatches" not in overrides:
        values["replay_microbatches"] = math.ceil(
            min(values["global_batch_size"], values["selected_tokens"]) / values["micro_batch_size"]
        )
    return ReplayMemoryConfig(**values)


def test_memory_prediction_bounds_all_fake_device_allocations() -> None:
    estimate = estimate_replay_memory(
        _memory_config(sequence_length=128, hidden_size=256, ffn_hidden_size=1024)
    )
    allocations = [torch.empty(value, dtype=torch.uint8) for _name, value in estimate.terms]
    measured = sum(tensor.numel() * tensor.element_size() for tensor in allocations)

    assert measured == sum(value for _name, value in estimate.terms)
    assert measured <= estimate.predicted_allocated_bytes
    assert estimate.predicted_allocated_bytes <= estimate.predicted_reserved_bytes


def test_memory_model_counts_every_padded_replay_microbatch() -> None:
    one = estimate_replay_memory(
        _memory_config(
            sequence_length=128,
            hidden_size=256,
            ffn_hidden_size=1024,
            selected_tokens=1,
            global_batch_size=1,
            micro_batch_size=1,
            replay_microbatches=1,
        )
    )
    padded = estimate_replay_memory(
        _memory_config(
            sequence_length=128,
            hidden_size=256,
            ffn_hidden_size=1024,
            selected_tokens=1,
            global_batch_size=1,
            micro_batch_size=1,
            replay_microbatches=4,
        )
    )

    assert padded.term("host_replay_inputs") == 4 * one.term("host_replay_inputs")


def test_memory_model_scales_all_topology_dimensions_without_dp_cp_duplication() -> None:
    base = estimate_replay_memory(_memory_config())
    longer = estimate_replay_memory(_memory_config(sequence_length=2048))
    larger_mbs = estimate_replay_memory(_memory_config(micro_batch_size=8))
    larger_gbs = estimate_replay_memory(
        _memory_config(global_batch_size=1024, selected_tokens=1024)
    )
    more_dp = estimate_replay_memory(_memory_config(data_parallel_size=16))
    more_tp = estimate_replay_memory(_memory_config(tensor_parallel_size=8))
    more_pp = estimate_replay_memory(_memory_config(pipeline_parallel_size=4, local_layers=8))
    more_cp = estimate_replay_memory(_memory_config(context_parallel_size=4))

    assert longer.predicted_allocated_bytes > base.predicted_allocated_bytes
    assert larger_mbs.term("device_full_and_cp_inputs") > base.term("device_full_and_cp_inputs")
    assert larger_gbs.term("host_replay_inputs") > base.term("host_replay_inputs")
    assert more_dp.term("retained_response_rows") == base.term("retained_response_rows")
    assert more_tp.term("retained_response_rows") < base.term("retained_response_rows")
    assert more_pp.term("retained_response_rows") < base.term("retained_response_rows")
    assert more_cp.term("device_full_and_cp_inputs") < base.term("device_full_and_cp_inputs")
    assert more_cp.term("retained_response_rows") == base.term("retained_response_rows")


def test_modeled_world_size_1024_has_bounded_cardinality() -> None:
    config = _memory_config(
        data_parallel_size=32,
        tensor_parallel_size=8,
        pipeline_parallel_size=4,
        context_parallel_size=1,
        local_layers=8,
    )
    estimate = estimate_replay_memory(config)

    assert (
        math.prod(
            (
                config.data_parallel_size,
                config.tensor_parallel_size,
                config.pipeline_parallel_size,
                config.context_parallel_size,
            )
        )
        == 1024
    )
    assert len(estimate.terms) == 11
    assert estimate.term("packed_statistics") == estimate.term("reduction_arena")
    assert estimate.term("packed_statistics") == config.global_layers * 8 * 96


def test_memory_bound_covers_tp0_and_dp_cp_concentration() -> None:
    concentrated = estimate_replay_memory(
        _memory_config(
            selected_tokens=1,
            global_batch_size=1,
            data_parallel_size=4,
            context_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            global_layers=1,
            local_layers=1,
            hidden_size=16,
            ffn_hidden_size=64,
            num_attention_heads=4,
            sequence_length=8,
            micro_batch_size=1,
            element_size=4,
        )
    )

    expected_response = (16 + 24 + 16 + 32 + 16) * 4
    assert concentrated.term("retained_response_rows") == expected_response
    assert concentrated.term("retained_response_rows") >= 416
