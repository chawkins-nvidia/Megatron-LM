# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import math
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from megatron.training.datasets.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
    RandomSeedDataset,
    SamplerIssuedIndex,
)
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
    ReplayIterator,
    ReplayMemoryConfig,
    ReplayRestorationError,
    ReplayStateGuard,
    SamplePopulation,
    StableSampleDataset,
    Tier1ReplayEngine,
    broadcast_replay_plan,
    build_distributed_source_plan,
    build_local_replay_plan,
    cp_sequence_columns,
    estimate_replay_memory,
    local_sample_populations,
    select_local_token_ids,
    slice_replay_mask,
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
    indices: tuple[int, ...] = (3, 7),
    *,
    epoch: int = 2,
    sequence_length: int = 8,
) -> dict[str, torch.Tensor]:
    batch = len(indices)
    return {
        "tokens": torch.arange(batch * sequence_length, dtype=torch.int64).view(
            batch, -1
        ),
        "labels": torch.arange(batch * sequence_length, dtype=torch.int64).view(
            batch, -1
        )
        + 1,
        "loss_mask": torch.ones(batch, sequence_length, dtype=torch.float32),
        "position_ids": torch.arange(sequence_length, dtype=torch.int64).expand(
            batch, -1
        ),
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
        recorded,
        selected,
        micro_batch_size=2,
        target_microbatches=target_microbatches,
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
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=2, persistent_workers=True
    )

    first = [
        (int(epoch), int(index))
        for batch in loader
        for epoch, index in zip(
            batch[SAMPLE_EPOCH_FIELD], batch[SAMPLE_INDEX_FIELD], strict=True
        )
    ]
    second = [
        (int(epoch), int(index))
        for batch in loader
        for epoch, index in zip(
            batch[SAMPLE_EPOCH_FIELD], batch[SAMPLE_INDEX_FIELD], strict=True
        )
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
    assert (
        len(set(first + second + [(item.epoch, item.sampler_index) for item in third]))
        == 36
    )


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
    selected = select_local_token_ids(
        recorded, populations, probe_tokens=1, run_seed=3, event_id=9
    )

    plan = build_local_replay_plan(
        recorded, selected, micro_batch_size=2, target_microbatches=3
    )

    assert sampler.consumed_samples == consumed
    assert plan.num_microbatches == 3
    assert plan.microbatches[0].data["loss_mask"][1].count_nonzero() == 0
    assert plan.microbatches[1].data["loss_mask"].count_nonzero() == 0
    assert plan.microbatches[2].data[DIAGNOSTIC_MASK_FIELD].count_nonzero() == 0
    assert plan.microbatches[1].sample_ids == plan.microbatches[2].sample_ids


def test_fixed_plan_codec_constructs_equal_neutral_non_source_plan() -> None:
    source = _plan()
    codec = FixedPlanCodec(8, 4, 2, 8)
    binding = CollectiveBinding("tp", None, 1)
    readiness = ReadinessConsensus(binding)

    received = broadcast_replay_plan(
        source,
        codec=codec,
        binding=binding,
        source_group_rank=0,
        readiness=readiness,
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
    workspace = PopulationCollectiveWorkspace(
        binding, maximum_local_samples=4, device="cpu"
    )

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
def test_cp_sp_mask_slicing_matches_layout(
    sequence_length: int, micro_batch_size: int
) -> None:
    mask = (
        torch.arange(micro_batch_size * sequence_length).view(micro_batch_size, -1) % 3
        == 0
    )
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
def test_every_restore_stage_fault_still_attempts_later_restorations(
    stage: str,
) -> None:
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
        with ReplayStateGuard((model,), tracker_getter=lambda: object()):
            pass


def test_active_overlap_state_is_rejected_before_capture() -> None:
    model = _StatefulModel()
    tracker = _Tracker()
    overlap = type("Overlap", (), {"param_gather_handle": object()})()

    with pytest.raises(ValueError, match="quiescent overlap"):
        with ReplayStateGuard(
            (model,), overlap_objects=(overlap,), tracker_getter=lambda: tracker
        ):
            pass


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


def test_pp2_noninterleaved_schedule_runs_multiple_microbatches_and_accumulation() -> (
    None
):
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


def test_engine_prepare_makes_memory_preflight_mandatory() -> None:
    binding = CollectiveBinding("world", None, 1)
    tracker = _Tracker()
    engine = Tier1ReplayEngine(
        (torch.nn.Identity(),),
        readiness=ReadinessConsensus(binding),
        tracker_getter=lambda: tracker,
    )
    probe = _Probe(2)
    schedule = type("Schedule", (), {"p2p_started": False, "completed": False})()
    schedule.__call__ = lambda plan, response_probe, phase: None
    estimate = estimate_replay_memory(
        _memory_config(sequence_length=128, hidden_size=256, ffn_hidden_size=1024)
    )

    with pytest.raises(ReplayPreflightError, match="failed collectively"):
        engine.prepare(
            plan=_plan(),
            probe=probe,
            schedule=schedule,
            memory_estimate=estimate,
            maximum_extra_bytes=0,
            currently_reserved_bytes=0,
            total_device_bytes=2**40,
        )


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
        selected_tokens=256,
        element_size=2,
        rng_snapshot_bytes=8192,
        model_state_bytes=32768,
        fixed_workspace_bytes=65536,
        alignment=256,
        headroom_fraction=0.1,
    )
    values.update(overrides)
    return ReplayMemoryConfig(**values)


def test_memory_prediction_bounds_all_fake_device_allocations() -> None:
    estimate = estimate_replay_memory(
        _memory_config(sequence_length=128, hidden_size=256, ffn_hidden_size=1024)
    )
    allocations = [
        torch.empty(value, dtype=torch.uint8) for _name, value in estimate.terms
    ]
    measured = sum(tensor.numel() * tensor.element_size() for tensor in allocations)

    assert measured == sum(value for _name, value in estimate.terms)
    assert measured <= estimate.predicted_allocated_bytes
    assert estimate.predicted_allocated_bytes <= estimate.predicted_reserved_bytes


def test_memory_model_scales_all_topology_dimensions_without_dp_cp_duplication() -> (
    None
):
    base = estimate_replay_memory(_memory_config())
    longer = estimate_replay_memory(_memory_config(sequence_length=2048))
    larger_mbs = estimate_replay_memory(_memory_config(micro_batch_size=8))
    larger_gbs = estimate_replay_memory(
        _memory_config(global_batch_size=1024, selected_tokens=1024)
    )
    more_dp = estimate_replay_memory(_memory_config(data_parallel_size=16))
    more_tp = estimate_replay_memory(_memory_config(tensor_parallel_size=8))
    more_pp = estimate_replay_memory(
        _memory_config(pipeline_parallel_size=4, local_layers=8)
    )
    more_cp = estimate_replay_memory(_memory_config(context_parallel_size=4))

    assert longer.predicted_allocated_bytes > base.predicted_allocated_bytes
    assert larger_mbs.term("device_full_and_cp_inputs") > base.term(
        "device_full_and_cp_inputs"
    )
    assert larger_gbs.term("host_replay_inputs") > base.term("host_replay_inputs")
    assert more_dp.term("retained_response_rows") < base.term("retained_response_rows")
    assert more_tp.term("retained_response_rows") < base.term("retained_response_rows")
    assert more_pp.term("retained_response_rows") < base.term("retained_response_rows")
    assert more_cp.term("device_full_and_cp_inputs") < base.term(
        "device_full_and_cp_inputs"
    )


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
    assert estimate.term("packed_statistics") == config.global_layers * 5 * 96
