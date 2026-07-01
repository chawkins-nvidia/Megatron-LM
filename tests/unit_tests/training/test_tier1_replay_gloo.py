# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.training.diagnostics.accumulator import ReductionBinding
from megatron.training.diagnostics.diagnostic_replay import (
    SAMPLE_EPOCH_FIELD,
    SAMPLE_INDEX_FIELD,
    CollectiveBinding,
    FixedPlanCodec,
    PopulationCollectiveWorkspace,
    ReadinessConsensus,
    RecordedBatch,
    ReplayPreflightError,
    Tier1ReplayTransaction,
    _ReplayPlanFacts,
    broadcast_replay_plan,
    build_distributed_source_plan,
    build_local_replay_plan,
    local_sample_populations,
    select_local_token_ids,
    verify_replay_plan_consensus,
)
from megatron.training.diagnostics.function_response import (
    FunctionResponseProbe,
    verify_response_descriptor_consensus,
)


def _raw_batch():
    return {
        "tokens": torch.arange(16, dtype=torch.int64).view(2, 8),
        "labels": torch.arange(16, dtype=torch.int64).view(2, 8) + 1,
        "loss_mask": torch.ones(2, 8, dtype=torch.float32),
        "position_ids": torch.arange(8, dtype=torch.int64).expand(2, -1),
        SAMPLE_INDEX_FIELD: torch.tensor([3, 5], dtype=torch.int64),
        SAMPLE_EPOCH_FIELD: torch.tensor([7, 7], dtype=torch.int64),
    }


def _source_plan():
    recorded = (RecordedBatch.from_raw(_raw_batch()),)
    selected = select_local_token_ids(
        recorded, local_sample_populations(recorded), probe_tokens=3, run_seed=19, event_id=4
    )
    return build_local_replay_plan(recorded, selected, micro_batch_size=2, target_microbatches=2)


def _write_result(directory: str, rank: int, result: str) -> None:
    Path(directory, f"rank-{rank}.txt").write_text(result, encoding="utf-8")


def _init(rank: int, init_method: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)


def _shutdown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _plan_worker(rank: int, init_method: str, directory: str) -> None:
    _init(rank, init_method)
    try:
        binding = CollectiveBinding("tp", None, 2)
        readiness = ReadinessConsensus(binding)
        plan = broadcast_replay_plan(
            _source_plan() if rank == 0 else None,
            codec=FixedPlanCodec(8, 4, 2, 8),
            binding=binding,
            source_group_rank=0,
            readiness=readiness,
        )
        verify_replay_plan_consensus(plan, binding=binding, readiness=readiness, device="cpu")
        digest = torch.tensor(list(bytes.fromhex(plan.metadata.descriptor_hash)), dtype=torch.uint8)
        minimum = digest.clone()
        maximum = digest.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        assert torch.equal(minimum, maximum)
        assert plan.num_microbatches == 2
        assert plan.source_rank == (rank == 0)
        assert len(plan.microbatches) == (2 if rank == 0 else 0)
        _write_result(directory, rank, "ok")
    finally:
        _shutdown()


def _descriptor_worker(rank: int, init_method: str, directory: str) -> None:
    _init(rank, init_method)
    try:
        probe = FunctionResponseProbe(
            (),
            global_layers=rank + 1,
            device="cpu",
            expected_hook_calls=0,
            reduction_binding=ReductionBinding.flat_world(None),
        )
        with pytest.raises(RuntimeError, match="descriptor/slot hash mismatch"):
            verify_response_descriptor_consensus(probe.accumulator)
        _write_result(directory, rank, "rejected")
    finally:
        _shutdown()


def _dp_spanning_plan_worker(rank: int, init_method: str, directory: str) -> None:
    _init(rank, init_method)
    try:
        raw = _raw_batch()
        raw[SAMPLE_INDEX_FIELD] = torch.tensor([100 + 2 * rank, 101 + 2 * rank], dtype=torch.int64)
        recorded = (RecordedBatch.from_raw(raw),)
        binding = CollectiveBinding("dp", None, 2)
        readiness = ReadinessConsensus(binding)
        workspace = PopulationCollectiveWorkspace(binding, maximum_local_samples=2, device="cpu")
        plan = build_distributed_source_plan(
            recorded,
            workspace=workspace,
            readiness=readiness,
            probe_tokens=16,
            run_seed=29,
            event_id=7,
            micro_batch_size=2,
            maximum_microbatches=2,
        )
        facts = _ReplayPlanFacts.observe(plan)
        assert plan.metadata.global_selected_tokens == 16
        assert facts.local_selected_tokens == plan.metadata.local_selected_tokens == 8
        assert sum(map(sum, plan.metadata.diagnostic_masks)) == 8

        codec = FixedPlanCodec(16, 2, 2, 8)
        integer, mask = codec.allocate("cpu")
        codec.encode(plan, integer, mask)
        decoded = codec.decode(integer, mask)
        assert decoded.global_selected_tokens == 16
        assert len(decoded.selected_tokens) == 8

        descriptors: list[str | None] = [None, None]
        dist.all_gather_object(descriptors, plan.metadata.descriptor_hash)
        assert len(set(descriptors)) == 2
        identity = workspace.global_selection_digest.clone()
        minimum = identity.clone()
        maximum = identity.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        assert torch.equal(minimum, maximum)
        _write_result(directory, rank, "local=8,global=16,dp-plans-distinct")
    finally:
        _shutdown()


def _readiness_worker(rank: int, init_method: str, directory: str) -> None:
    _init(rank, init_method)
    try:
        readiness = ReadinessConsensus(CollectiveBinding("world", None, 2))
        local_error = RuntimeError("injected capture fault") if rank == 1 else None
        with pytest.raises(ReplayPreflightError, match="failed collectively"):
            readiness.settle(local_error, "capture")
        _write_result(directory, rank, "settled")
    finally:
        _shutdown()


class _Probe:
    expected_hook_calls = 2
    descriptor_hash = "test"

    def __init__(self, release_fault: bool = False) -> None:
        self.release_fault = release_fault
        self.release_calls = 0

    @contextlib.contextmanager
    def capture_pre(self):
        yield

    @contextlib.contextmanager
    def capture_post(self):
        yield

    def set_masks(self, full_mask, *, sequence_parallel_mask=None):
        return None

    def finalize(self):
        return self

    def release(self):
        self.release_calls += 1
        if self.release_fault:
            raise RuntimeError("injected probe release fault")
        return None


class _FaultSchedule:
    def __init__(self, rank: int, fault: str) -> None:
        self.rank = rank
        self.fault = fault
        self.p2p_started = False
        self.completed = False
        self.calls = 0

    def __call__(self, plan, probe, phase):
        self.calls += 1
        self.p2p_started = True
        if self.fault == "asymmetric" and self.rank == 0:
            raise RuntimeError("injected asymmetric schedule fault")
        dist.barrier()
        if self.fault in ("schedule", "cleanup_schedule") and self.rank == 0:
            raise RuntimeError("injected schedule fault")
        self.completed = True


class _FailOnSecondSetTracker:
    def __init__(self, fail: bool) -> None:
        self.states = {"model-parallel-rng": torch.tensor([1, 2], dtype=torch.uint8)}
        self.set_calls = 0
        self.fail = fail

    def get_states(self):
        return self.states

    def set_states(self, states):
        self.set_calls += 1
        if self.fail and self.set_calls == 2:
            raise RuntimeError("injected tracker restore fault")
        self.states = states


class _FatalRaised(RuntimeError):
    pass


def _transaction_worker(rank: int, init_method: str, directory: str, fault: str) -> None:
    _init(rank, init_method)
    tracker = _FailOnSecondSetTracker(fail=fault == "restore" and rank == 0)
    fatal_errors = []

    def fatal_abort(error: BaseException) -> None:
        fatal_errors.append(error)
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except RuntimeError:
                pass
        raise _FatalRaised(str(error))

    transaction = Tier1ReplayTransaction(
        models=(torch.nn.Identity(),),
        plan=_source_plan(),
        probe=_Probe(release_fault=fault == "cleanup_schedule" and rank == 0),
        schedule=_FaultSchedule(rank, fault),
        readiness=ReadinessConsensus(CollectiveBinding("world", None, 2)),
        mutable_buffer_names=(),
        tracker_getter=lambda: tracker,
        cuda_device=(
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        samplers=(),
        overlap_objects=(),
        fatal_abort=fatal_abort,
    )
    try:
        transaction.run_pre()
    except _FatalRaised:
        _write_result(
            directory,
            rank,
            "fatal" if transaction.schedule.calls == len(fatal_errors) == 1 else "unbounded",
        )
    else:
        _write_result(directory, rank, "unexpected-return")
        if dist.is_initialized():
            dist.destroy_process_group()


def _ordinary_cleanup_worker(rank: int, init_method: str, directory: str, outcome: str) -> None:
    _init(rank, init_method)
    tracker = _FailOnSecondSetTracker(fail=False)
    probe = _Probe(release_fault=rank == 0)

    def fatal_abort(error: BaseException) -> None:
        result = (
            f"fatal:{type(error).__name__}:release_calls={probe.release_calls}:"
            f"state={transaction.state}"
        )
        _write_result(directory, rank, result)
        if dist.is_initialized():
            dist.destroy_process_group()
        raise _FatalRaised(str(error))

    transaction = Tier1ReplayTransaction(
        models=(torch.nn.Identity(),),
        plan=_source_plan(),
        probe=probe,
        schedule=_FaultSchedule(rank, "none"),
        readiness=ReadinessConsensus(CollectiveBinding("world", None, 2)),
        mutable_buffer_names=(),
        tracker_getter=lambda: tracker,
        cuda_device=None,
        samplers=(),
        overlap_objects=(),
        fatal_abort=fatal_abort,
    )
    try:
        transaction.run_pre()
        transaction.finish(update_succeeded=outcome == "post")
    except _FatalRaised:
        pass
    else:
        _write_result(directory, rank, "unexpected-return")
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_rank(tmp_path: Path, worker, *args: str) -> list[str]:
    init_file = tmp_path / "gloo-init"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    mp.spawn(worker, args=(f"file://{init_file}", str(result_dir), *args), nprocs=2, join=True)
    return [(result_dir / f"rank-{rank}.txt").read_text(encoding="utf-8") for rank in range(2)]


def _pre_schedule_fault_worker(rank: int, init_method: str, directory: str, fault: str) -> None:
    _init(rank, init_method)
    tracker: object = _FailOnSecondSetTracker(fail=False)
    model: torch.nn.Module = torch.nn.Identity()
    overlap_objects: tuple[object, ...] = ()
    if rank == 0 and fault == "model":
        model.register_buffer("undeclared_cache", torch.tensor([1.0]))
    if rank == 0 and fault == "overlap":
        overlap_objects = (type("Overlap", (), {"param_gather_handle": object()})(),)
    if rank == 0 and fault == "tracker":
        tracker = object()
    preflight_validator = None
    if fault == "graph":
        preflight_validator = lambda _registrations: (
            (_ for _ in ()).throw(RuntimeError("injected execution graph drift"))
            if rank == 0
            else None
        )
    schedule = _FaultSchedule(rank, "none")
    transaction = Tier1ReplayTransaction(
        models=(model,),
        plan=_source_plan(),
        probe=_Probe(),
        schedule=schedule,
        readiness=ReadinessConsensus(CollectiveBinding("world", None, 2)),
        mutable_buffer_names=(),
        tracker_getter=lambda: tracker,
        cuda_device=None,
        samplers=(),
        overlap_objects=overlap_objects,
        fatal_abort=lambda error: (_ for _ in ()).throw(_FatalRaised(str(error))),
        preflight_validator=preflight_validator,
    )
    try:
        transaction.run_pre()
    except ReplayPreflightError:
        _write_result(
            directory,
            rank,
            "settled-no-schedule" if not schedule.p2p_started else "schedule-called",
        )
    finally:
        if dist.is_initialized():
            _shutdown()


def _post_schedule_drift_worker(rank: int, init_method: str, directory: str) -> None:
    _init(rank, init_method)
    tracker = _FailOnSecondSetTracker(fail=False)
    schedule = _FaultSchedule(rank, "none")
    drifted = False

    def validate(_registrations) -> None:
        if drifted and rank == 0:
            raise RuntimeError("injected post execution graph drift")

    transaction = Tier1ReplayTransaction(
        models=(torch.nn.Identity(),),
        plan=_source_plan(),
        probe=_Probe(),
        schedule=schedule,
        readiness=ReadinessConsensus(CollectiveBinding("world", None, 2)),
        mutable_buffer_names=(),
        tracker_getter=lambda: tracker,
        cuda_device=None,
        samplers=(),
        overlap_objects=(),
        fatal_abort=lambda error: (_ for _ in ()).throw(_FatalRaised(str(error))),
        preflight_validator=validate,
    )
    try:
        transaction.run_pre()
        drifted = True
        transaction.finish(update_succeeded=True)
    except ReplayPreflightError:
        _write_result(
            directory,
            rank,
            "settled-no-post-schedule" if schedule.calls == 1 else "post-schedule-called",
        )
    finally:
        if dist.is_initialized():
            _shutdown()


def test_two_rank_tp_source_non_source_plan_equality(tmp_path: Path) -> None:
    assert _run_two_rank(tmp_path, _plan_worker) == ["ok", "ok"]


def test_two_rank_dp_selection_spans_distinct_local_plans(tmp_path: Path) -> None:
    assert _run_two_rank(tmp_path, _dp_spanning_plan_worker) == [
        "local=8,global=16,dp-plans-distinct",
        "local=8,global=16,dp-plans-distinct",
    ]


def test_two_rank_missing_descriptor_hash_is_rejected(tmp_path: Path) -> None:
    assert _run_two_rank(tmp_path, _descriptor_worker) == ["rejected", "rejected"]


def test_two_rank_one_rank_capture_fault_settles_without_hang(tmp_path: Path) -> None:
    assert _run_two_rank(tmp_path, _readiness_worker) == ["settled", "settled"]


@pytest.mark.parametrize("fault", ("overlap", "model", "tracker", "graph"))
def test_two_rank_guard_fault_settles_before_peer_barrier_schedule(
    tmp_path: Path, fault: str
) -> None:
    assert _run_two_rank(tmp_path, _pre_schedule_fault_worker, fault) == [
        "settled-no-schedule",
        "settled-no-schedule",
    ]


def test_two_rank_one_rank_post_graph_drift_settles_before_second_schedule(tmp_path: Path) -> None:
    assert _run_two_rank(tmp_path, _post_schedule_drift_worker) == [
        "settled-no-post-schedule",
        "settled-no-post-schedule",
    ]


@pytest.mark.parametrize(
    "fault", ("schedule", "restore", "cleanup_schedule", "asymmetric")
)
def test_two_rank_post_p2p_fault_is_fatal_on_every_rank(tmp_path: Path, fault: str) -> None:
    assert _run_two_rank(tmp_path, _transaction_worker, fault) == ["fatal", "fatal"]


@pytest.mark.parametrize("outcome", ("overflow", "post"))
def test_two_rank_one_rank_ordinary_release_fault_is_collectively_fatal(
    tmp_path: Path, outcome: str
) -> None:
    results = _run_two_rank(tmp_path, _ordinary_cleanup_worker, outcome)

    assert all(result.startswith("fatal:ReplayPreflightError:") for result in results)
    assert all("release_calls=1:state=closed" in result for result in results)
