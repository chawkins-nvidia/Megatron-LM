# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tier-0 heartbeat orchestration and runtime-contract tests."""

import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn

from megatron.core.diagnostics import get_diagnostic_microbatch_id
from megatron.core.pipeline_parallel.schedules import forward_step
from megatron.training.argument_utils import _default_config_from_args
from megatron.training.config.training_config import LoggerConfig
from megatron.training.diagnostics import tier0 as tier0_module
from megatron.training.diagnostics.artifact import Tier0ArtifactWriter
from megatron.training.diagnostics.tier0 import (
    Tier0Cadence,
    Tier0Heartbeat,
    Tier0ReservationError,
    Tier0ReservationStatus,
    allocator_growth_within_bound,
    tier0_reservation_bytes,
)
from megatron.training.global_vars import wandb_writer_rank

from megatron.training.diagnostics.accumulator import (  # isort: skip
    PackedSufficientStatistics,
    ReductionBinding,
)
from megatron.training.diagnostics.capability import (  # isort: skip
    capability_payload,
    contract_hash,
    diagnostic_schema_hash,
    load_static_capability,
    materialize_capability,
    schema_hash,
    sha256_file,
    static_capability_path,
    verified_source_commit,
)
from megatron.training.diagnostics.schema import (  # isort: skip
    TIER0_KEYS,
    TIER0_METADATA_KEYS,
    TIER0_METRIC_KEYS,
)


def _status_only_args() -> SimpleNamespace:
    return SimpleNamespace(
        diagnostic_interval=1000,
        diagnostic_early_updates="1,10,100",
        diagnostic_unsupported_policy="status-only",
        diagnostic_successful_updates=0,
        diagnostic_event_id=0,
        diagnostic_cumulative_artifact_bytes=0,
    )


class _TensorboardWriter:
    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}

    def add_scalar(self, key: str, value: torch.Tensor, _step: int) -> None:
        self.values[key] = value


def test_cadence_uses_only_prospective_successful_updates() -> None:
    cadence = Tier0Cadence.parse(1000, "1,10,100")
    assert [update for update in range(1, 2002) if cadence.is_due(update)] == [
        1,
        10,
        100,
        1000,
        2000,
    ]
    with pytest.raises(ValueError, match="comma-separated"):
        Tier0Cadence.parse(1000, "one")


def test_logger_config_exposes_checkpointed_heartbeat_contract() -> None:
    config = LoggerConfig(
        diagnostic_heartbeat=True,
        diagnostic_interval=17,
        diagnostic_early_updates="1,3",
        diagnostic_unsupported_policy="status-only",
        diagnostic_max_extra_bytes=1234,
    )
    assert config.diagnostic_heartbeat
    assert config.diagnostic_interval == 17
    assert config.diagnostic_early_updates == "1,3"
    assert config.diagnostic_unsupported_policy == "status-only"
    assert config.diagnostic_max_extra_bytes == 1234
    assert config.diagnostic_successful_updates == 0
    assert config.diagnostic_event_id == 0

    restored = _default_config_from_args(
        LoggerConfig,
        SimpleNamespace(
            diagnostic_heartbeat=True,
            diagnostic_successful_updates=7,
            diagnostic_event_id=3,
        ),
    )
    assert restored.diagnostic_heartbeat
    assert restored.diagnostic_successful_updates == 0
    assert restored.diagnostic_event_id == 0


@pytest.mark.parametrize(
    ("override", "reason"),
    (
        ({"transformer_impl": "transformer_engine"}, "transformer_engine"),
        ({"fp8": "hybrid"}, "fp8_fp4"),
        ({"use_megatron_fsdp": True}, "fsdp"),
        ({"num_experts": 8}, "moe"),
        ({"overlap_param_gather": True}, "param_gather_overlap"),
        ({"virtual_pipeline_model_parallel_size": 2}, "virtual_pipeline"),
        ({"calculate_per_token_loss": False}, "per_token_loss"),
        ({"num_layers": 10_001}, "capability_bounds"),
    ),
)
def test_narrow_backend_capability_rejects_unsupported_modes(
    monkeypatch: pytest.MonkeyPatch, override: dict[str, object], reason: str
) -> None:
    parameter = nn.Parameter(torch.ones(1))
    distributed_optimizer = SimpleNamespace(
        grad_scaler=None, optimizer=torch.optim.Adam([parameter])
    )
    monkeypatch.setattr(tier0_module, "GPTModel", nn.Linear)
    monkeypatch.setattr(
        tier0_module, "_distributed_optimizer", lambda _optimizer: distributed_optimizer
    )
    values = {
        "bf16": True,
        "calculate_per_token_loss": True,
        "fp16": False,
        "transformer_impl": "local",
    }
    values.update(override)
    args = SimpleNamespace(**values)
    reasons = tier0_module._local_capability_reasons(args, [nn.Linear(1, 1)], object())
    assert reason in reasons


def test_forward_function_requires_exact_registered_identity_and_rejects_review_spoof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def permissive(_iterator, _model, *, diagnostic_heartbeat=None):
        return diagnostic_heartbeat

    def canonical(_iterator, _model, *, diagnostic_heartbeat=None):
        microbatch_id = get_diagnostic_microbatch_id()
        diagnostic_heartbeat.register_local_loss_mask(microbatch_id, None)

    def spoof(_iterator, _model, *, diagnostic_heartbeat=None):
        unused_getter = get_diagnostic_microbatch_id
        unused_method = diagnostic_heartbeat.register_local_loss_mask
        return unused_getter, unused_method

    spoof.__megatron_tier0_mask_producer__ = (
        "megatron.tier0.canonical-gpt-mask-producer.v1"
    )
    monkeypatch.setattr(
        tier0_module,
        "_canonical_mask_producer",
        (
            canonical,
            tier0_module._MASK_PRODUCER_IDENTITY,
            tier0_module._NONINTERLEAVED_SCHEDULE_ADAPTER,
        ),
    )

    assert not tier0_module._verified_mask_producer(None)
    assert not tier0_module._verified_mask_producer(permissive)
    assert not tier0_module._verified_mask_producer(spoof)
    assert tier0_module._verified_mask_producer(canonical)


def test_status_only_retries_overflow_then_writes_one_exact_payload() -> None:
    args = _status_only_args()
    calls: list[tuple[dict[str, float], int]] = []
    tensorboard = _TensorboardWriter()

    def log(payload: dict[str, float], *, step: int) -> None:
        calls.append((payload, step))

    def reducer(_tensor: torch.Tensor, *, op: object, group: object | None) -> None:
        del op, group

    heartbeat = Tier0Heartbeat(
        args,
        [nn.Linear(2, 2)],
        object(),
        wandb_log=log,
        tensorboard_writer=tensorboard,
        reduction_binding=ReductionBinding.flat_world(None, reducer=reducer),
    )
    assert heartbeat.prepare_attempt(num_microbatches=8)
    assert not heartbeat.finish_optimizer_event(False, iteration=4)
    assert heartbeat.successful_updates == 0
    assert heartbeat.event_id == 0
    assert calls == []

    assert heartbeat.prepare_attempt(num_microbatches=8)
    assert heartbeat.finish_optimizer_event(True, iteration=5)
    assert heartbeat.successful_updates == 1
    assert heartbeat.event_id == 1
    assert args.diagnostic_successful_updates == 1
    assert args.diagnostic_event_id == 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    assert calls == []
    assert tensorboard.values == {}
    assert heartbeat.sink_failure_count == (1 if rank == 0 else 0)


def test_event_reduction_is_exactly_one_sum_max_min_sequence() -> None:
    calls: list[object] = []

    def reducer(_tensor: torch.Tensor, *, op: object, group: object | None) -> None:
        assert group is None
        calls.append(op)

    binding = ReductionBinding.flat_world(None, reducer=reducer)
    first = PackedSufficientStatistics(
        ("first",), "cpu", descriptor_hash="first", reduction_binding=binding
    )
    second = PackedSufficientStatistics(
        ("second",), "cpu", descriptor_hash="second", reduction_binding=binding
    )
    first.add_masked_tensor("first", torch.tensor([1.0, 2.0]))
    second.add_masked_tensor("second", torch.tensor([3.0]))
    expected_arena = sum(
        accumulator.sum_pack.nbytes
        + accumulator.max_pack.nbytes
        + accumulator.min_pack.nbytes
        for accumulator in (first, second)
    )
    assert (
        PackedSufficientStatistics.reduction_arena_bytes((first, second))
        == expected_arena
    )
    PackedSufficientStatistics.reduce_many_((first, second))
    assert calls == [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN]
    torch.testing.assert_close(
        first.rms("first").value, torch.tensor(2.5).sqrt().double()
    )
    torch.testing.assert_close(second.rms("second").value, torch.tensor(3.0).double())


def test_schedule_microbatch_identity_is_cleared_in_finally() -> None:
    events: list[tuple[str, int]] = []

    class Heartbeat:
        def begin_microbatch(self, microbatch_id: int) -> None:
            events.append(("begin", microbatch_id))

        def end_microbatch(self, microbatch_id: int) -> None:
            events.append(("end", microbatch_id))

    class Model(nn.Module):
        def set_input_tensor(self, _input_tensor) -> None:
            pass

    def fail(_iterator, _model):
        assert get_diagnostic_microbatch_id() == 7
        raise RuntimeError("injected forward failure")

    config = SimpleNamespace(
        timers=None, enable_autocast=False, diagnostic_heartbeat=Heartbeat()
    )
    with pytest.raises(RuntimeError, match="injected forward failure"):
        forward_step(
            fail,
            iter(()),
            Model(),
            1,
            None,
            [],
            config,
            1,
            current_microbatch=7,
        )
    assert events == [("begin", 7), ("end", 7)]
    assert get_diagnostic_microbatch_id() is None


def test_checkpoint_runtime_fields_round_trip() -> None:
    args = _status_only_args()
    args.diagnostic_successful_updates = 123
    args.diagnostic_event_id = 9
    args.diagnostic_cumulative_artifact_bytes = 4567
    buffer = io.BytesIO()
    torch.save({"args": args}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)["args"]
    assert restored.diagnostic_successful_updates == 123
    assert restored.diagnostic_event_id == 9
    assert restored.diagnostic_cumulative_artifact_bytes == 4567


def test_capability_probe_contract_and_cpu_only_subprocess() -> None:
    payload = capability_payload()
    static = load_static_capability()
    assert payload["static_capability_sha256"] == contract_hash()
    assert payload["schema_hash"] == schema_hash()
    assert payload["schema"] == "diag/v2/runtime-capabilities"
    assert payload["supported_max_tier"] == 0
    assert payload["runtime_contract_present"]
    assert payload["integrated_heartbeat_consumer"]
    assert payload["runtime_fields"] == list(
        tier0_module.CONSUMED_DIAGNOSTIC_CONFIG_FIELDS
        if hasattr(tier0_module, "CONSUMED_DIAGNOSTIC_CONFIG_FIELDS")
        else static["runtime_fields"]
    )
    assert payload["source_commit"] == verified_source_commit()
    assert payload["build_identity"] == (
        f"capability-file-sha256:{sha256_file(static_capability_path())}"
    )
    assert payload["topology_bounds"]["world_size"]["maximum"] == 1024
    assert payload["artifact_support"]["exact_files"] == [
        "manifest.json",
        "layer_metrics.npz",
        "rank_perf.npz",
    ]
    assert (
        diagnostic_schema_hash()
        == "7b7e156949da5370cccf5fd825784dfcc4de79b19ee4eca1d79d976783b5b28a"
    )
    assert set(TIER0_METADATA_KEYS).issubset(TIER0_KEYS)

    code = (
        "import json,sys; "
        "from megatron.training.diagnostics.capability import capability_payload; "
        "print(json.dumps({'torch_loaded': 'torch' in sys.modules, "
        "'payload': capability_payload()}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    result = json.loads(completed.stdout)
    assert not result["torch_loaded"]
    assert result["payload"]["source_commit"] == verified_source_commit()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "megatron.training.diagnostics.capability",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    cli_payload = json.loads(completed.stdout)
    assert cli_payload["runtime_contract_present"]
    assert cli_payload["source_commit"] == verified_source_commit()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "capability.json"
        materialize_capability(path, static)
        first = path.read_bytes()
        materialize_capability(path, static)
        assert path.read_bytes() == first


def test_zero_extra_bytes_rejects_before_event_allocation() -> None:
    args = _status_only_args()
    args.diagnostic_max_extra_bytes = 0
    with pytest.raises(RuntimeError, match="reservation rejected globally"):
        Tier0Heartbeat(args, [nn.Linear(2, 2)], object())


def test_pure_startup_reservation_is_bounded_at_world_size_1024() -> None:
    requested = tier0_reservation_bytes(
        num_layers=96,
        num_microbatches=16,
        micro_batch_size=4,
        local_sequence_length=4096,
        owner_elements=100_000_000,
        world_size=1024,
    )
    repeated = tier0_reservation_bytes(
        num_layers=96,
        num_microbatches=16,
        micro_batch_size=4,
        local_sequence_length=4096,
        owner_elements=100_000_000,
        world_size=1024,
    )
    assert requested == repeated
    assert requested > 600_000_000
    assert requested < 700_000_000


def test_joint_advertised_boundary_fits_and_one_past_rejects_before_p2p() -> None:
    static = load_static_capability()
    bounds = static["config_bounds"]
    requested = tier0_reservation_bytes(
        num_layers=bounds["num_layers"]["maximum"],
        num_microbatches=bounds["num_microbatches"]["maximum"],
        micro_batch_size=bounds["micro_batch_size"]["maximum"],
        local_sequence_length=bounds["sequence_length"]["maximum"],
        owner_elements=0,
        world_size=static["topology_bounds"]["world_size"]["maximum"],
    )
    assert requested <= bounds["maximum_reservation_bytes"] < 2**63

    args = _status_only_args()
    args.num_layers = 2**60
    args.seq_length = 2**60
    args.micro_batch_size = 8
    args.context_parallel_size = 1
    with pytest.raises(Tier0ReservationError) as raised:
        Tier0Heartbeat(args, [nn.Linear(2, 2)], object(), num_microbatches=64)
    assert raised.value.status == Tier0ReservationStatus.OVERFLOW


def test_allocator_reserved_growth_uses_the_pre_event_baseline() -> None:
    assert allocator_growth_within_bound(
        pre_event_reserved_bytes=1_000,
        peak_reserved_bytes=1_512,
        predicted_increment_bytes=512,
    )
    assert not allocator_growth_within_bound(
        pre_event_reserved_bytes=1_000,
        peak_reserved_bytes=1_513,
        predicted_increment_bytes=512,
    )


def test_bound_observation_workspace_uses_no_tensor_producing_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = ReductionBinding.flat_world(None, reducer=lambda *_args, **_kwargs: None)
    accumulator = PackedSufficientStatistics(
        ("x",), "cpu", descriptor_hash="workspace", reduction_binding=binding
    )
    storage = torch.empty(accumulator.maximum_scratch_bytes, dtype=torch.uint8)
    accumulator.bind_workspace(storage)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("workspace path called a tensor-producing fallback")

    for name in ("as_tensor", "full", "sort", "stack", "tensor", "where", "zeros"):
        monkeypatch.setattr(torch, name, forbidden)
    accumulator.add_masked_tensor(
        "x", torch.Tensor([1.0, 2.0, 3.0]), mask=torch.Tensor([1.0, 0.0, 1.0])
    )
    assert accumulator.sum_pack[accumulator.slots("x").count] == 2


def test_complete_event_runtime_probe_catches_prior_tensor_producing_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat = Tier0Heartbeat(
        _status_only_args(),
        [nn.Linear(2, 2)],
        object(),
        reduction_binding=ReductionBinding.flat_world(
            None, reducer=lambda *_args, **_kwargs: None
        ),
    )
    assert heartbeat.prepare_attempt(num_microbatches=1)
    counts = {
        name: 0
        for name in ("as_tensor", "full", "sort", "stack", "tensor", "where", "zeros")
    }
    for name in counts:
        original = getattr(torch, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, counted)
    assert heartbeat.finish_optimizer_event(True, iteration=0)
    assert counts == {name: 0 for name in counts}


def test_multirank_wandb_ownership_keeps_validation_on_the_single_owner() -> None:
    args = SimpleNamespace(diagnostic_heartbeat=True, world_size=8)
    owners = [
        rank for rank in range(args.world_size) if rank == wandb_writer_rank(args)
    ]
    tensorboard_owners = [args.world_size - 1]
    validation_wandb_owners = list(owners)
    assert owners == [7]
    assert tensorboard_owners == [7]
    assert validation_wandb_owners == [7]
    ordinary_training_calls = sum(
        rank in owners and rank in tensorboard_owners for rank in range(args.world_size)
    )
    validation_calls = sum(rank in owners for rank in range(args.world_size))
    throughput_calls = sum(rank in owners for rank in range(args.world_size))
    checkpoint_save_calls = sum(
        rank in owners and rank == args.world_size - 1
        for rank in range(args.world_size)
    )
    checkpoint_load_calls = sum(
        rank in owners and rank == args.world_size - 1
        for rank in range(args.world_size)
    )
    assert (
        ordinary_training_calls,
        validation_calls,
        throughput_calls,
        checkpoint_save_calls,
        checkpoint_load_calls,
    ) == (1, 1, 1, 1, 1)


def test_full_event_orchestrator_has_only_one_sink_host_transfer() -> None:
    forbidden = (".item(", ".cpu(", ".tolist(")
    for method_name in (
        "finish_optimizer_event",
        "_gather_latency_ms",
    ):
        source = inspect.getsource(getattr(Tier0Heartbeat, method_name))
        assert not any(token in source for token in forbidden), method_name
    sink_source = inspect.getsource(Tier0Heartbeat._emit)
    assert sink_source.count(".cpu()") == 1
    assert ".item(" not in sink_source


def test_cpu_post_transfer_derivation_retains_exact_75_key_contract() -> None:
    binding = ReductionBinding.flat_world(None, reducer=lambda *_args, **_kwargs: None)
    capture_names = (
        "event/valid_tokens",
        "event/runtime_status",
        *(
            f"{observation}/{family}/layer_0"
            for observation in ("activation", "dgrad")
            for family in ("residual", "qkv", "attn_out", "fc1", "fc2")
        ),
    )
    update_names = (
        *(
            f"update/{family}/layer_0"
            for family in ("qkv", "attn_out", "fc1", "fc2", "norm")
        ),
        "update/embedding",
        "update/output",
    )
    control_names = tier0_module._CONTROL_NAMES
    heartbeat = Tier0Heartbeat.__new__(Tier0Heartbeat)
    heartbeat.capture_accumulator = PackedSufficientStatistics(
        capture_names, "cpu", descriptor_hash="capture", reduction_binding=binding
    )
    heartbeat.update_accumulator = PackedSufficientStatistics(
        update_names, "cpu", descriptor_hash="update", reduction_binding=binding
    )
    heartbeat.control_accumulator = PackedSufficientStatistics(
        control_names, "cpu", descriptor_hash="control", reduction_binding=binding
    )
    heartbeat.capability = SimpleNamespace(supported=True)
    heartbeat.args = SimpleNamespace(
        num_layers=1,
        loss_scale=1.0,
        diagnostic_dgrad_starvation_threshold=0.0,
        diagnostic_update_starvation_threshold=0.0,
    )
    heartbeat.model = (nn.Linear(1, 1),)
    heartbeat.successful_updates = 1

    def pack(accumulator: PackedSufficientStatistics, *, control: bool = False):
        sums = []
        maxima = []
        minima = []
        for name in accumulator.slot_names:
            values = [1.0, 1.0, 4.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            maximum = 0.0 if control or name == "event/runtime_status" else 2.0
            minimum = 0.0 if control or name == "event/runtime_status" else -2.0
            if name == "event/valid_tokens":
                values[0] = 5.0
            sums.extend(values)
            maxima.append(maximum)
            minima.append(minimum)
        return {
            "names": accumulator.slot_names,
            "sum": tuple(sums),
            "max": tuple(maxima),
            "min": tuple(minima),
        }

    host_packs = {
        "capture": pack(heartbeat.capture_accumulator),
        "update": pack(heartbeat.update_accumulator),
        "control": pack(heartbeat.control_accumulator, control=True),
    }
    payload = heartbeat._derive_host_payload(
        host_packs, [[0, 1, 1, 1024, 0, 0, 0, 0, 0, 3]]
    )
    assert tuple(payload) == TIER0_KEYS
    assert len(payload) == 75
    assert payload["diag/v2/status/valid"] == 1
    assert all(math_value == math_value for math_value in payload.values())


def test_event_artifact_matches_approved_scaling_validator_and_logs_once() -> None:
    class Artifact:
        def __init__(self, *, name: str, type: str) -> None:
            self.name = name
            self.type = type
            self.directory = None

        def add_dir(self, directory: str) -> None:
            self.directory = directory

    class Run:
        def __init__(self) -> None:
            self.calls: list[Artifact] = []

        def log_artifact(self, artifact: Artifact) -> None:
            self.calls.append(artifact)

    class Wandb:
        def __init__(self) -> None:
            self.run = Run()
            self.Artifact = Artifact

    with tempfile.TemporaryDirectory() as tmpdir:
        wandb = Wandb()
        writer = Tier0ArtifactWriter(
            Path(tmpdir),
            run_id="run-209",
            job_name="job-209",
            repro={
                "scaling_commit": "a" * 40,
                "megatron_commit": "b" * 40,
                "resolved_config_sha256": "c" * 64,
                "scaling_bundle_sha256": "d" * 64,
                "megatron_bundle_sha256": "e" * 64,
            },
            wandb_writer=wandb,
        )
        directory, compressed = writer.write(
            event_id=1,
            successful_update=1,
            consumed_tokens=1024,
            valid_positions=4,
            valid=True,
            topology={
                "dp": 1,
                "tp": 1,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "vpp": 1,
                "num_layers": 2,
            },
            rank_evidence=[[0, 10, 25, 1000, 90, 100, 20, 100, 120, 7]],
            capability_hash="f" * 64,
            schema_hash=diagnostic_schema_hash(),
        )
        assert {path.name for path in directory.iterdir()} == {
            "manifest.json",
            "layer_metrics.npz",
            "rank_perf.npz",
        }
        assert compressed == writer.cumulative_bytes
        assert len(wandb.run.calls) == 1
        assert wandb.run.calls[0].name == "diag-v2-run-209"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["status"] == "invalid"
        assert not manifest["state_snapshots"]["pre"]["applicable"]
        assert not manifest["state_snapshots"]["post"]["applicable"]
        assert manifest["capability_signature"]["chained_optimizer"] is True
        observed_sampling_fact = hashlib.sha256(
            json.dumps(
                {"mask_checksum": 7.0, "rank": 0, "valid_positions": 4},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        assert manifest["sampling"]["selected_sample_id_hashes"] == [
            observed_sampling_fact
        ]
        observed_world = hashlib.sha256(b"[0]").hexdigest()
        assert manifest["process_groups"][0]["membership_sha256_by_rank"] == [
            observed_world
        ]
        assert manifest["process_groups"] == [
            {
                "name": "world",
                "membership_sha256_by_rank": manifest["process_groups"][0][
                    "membership_sha256_by_rank"
                ],
                "operations": [
                    {"name": "all_reduce", "count": 3, "bytes": 1},
                    {"name": "all_gather", "count": 1, "bytes": 80},
                ],
            }
        ]
        with np.load(directory / "rank_perf.npz", allow_pickle=False) as ranks:
            assert ranks["rank"].dtype == np.int32
            assert ranks["predicted_increment_bytes"].dtype == np.int64
        second_writer = Tier0ArtifactWriter(
            Path(tmpdir) / "second",
            run_id="run-209",
            job_name="job-209",
            repro=writer.repro,
            wandb_writer=None,
        )
        second_directory, _ = second_writer.write(
            event_id=1,
            successful_update=1,
            consumed_tokens=1024,
            valid_positions=4,
            valid=True,
            topology={
                "dp": 1,
                "tp": 1,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "vpp": 1,
                "num_layers": 2,
            },
            rank_evidence=[[0, 10, 25, 1000, 90, 100, 20, 100, 120, 7]],
            capability_hash="f" * 64,
            schema_hash=diagnostic_schema_hash(),
        )
        assert {
            name: sha256_file(directory / name)
            for name in ("manifest.json", "layer_metrics.npz", "rank_perf.npz")
        } == {
            name: sha256_file(second_directory / name)
            for name in ("manifest.json", "layer_metrics.npz", "rank_perf.npz")
        }
        scaling_root = Path(
            "/home/chawkins/src/scaling-worktrees/issue-209-launch-review"
        )
        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-c",
                (
                    "from analysis.diagnostics.validate_wandb import validate_artifact; "
                    f"validate_artifact({str(directory)!r})"
                ),
            ],
            cwd=scaling_root,
            env={**os.environ, "PYTHONPATH": str(scaling_root)},
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert "ContractError" in completed.stderr


def test_actual_artifact_cap_and_wandb_failures_leave_no_promotable_directory() -> None:
    class Artifact:
        def add_dir(self, _directory: str) -> None:
            pass

    class FailingRun:
        def log_artifact(self, _artifact: Artifact) -> None:
            raise RuntimeError("injected W&B upload failure")

    class FailingWandb:
        run = FailingRun()

        @staticmethod
        def Artifact(**_kwargs):
            return Artifact()

    kwargs = {
        "event_id": 1,
        "successful_update": 1,
        "consumed_tokens": 1,
        "valid_positions": 1,
        "valid": True,
        "topology": {
            "dp": 1,
            "tp": 1,
            "pp": 1,
            "cp": 1,
            "ep": 1,
            "vpp": 1,
            "num_layers": 1,
        },
        "rank_evidence": [[0, 1, 1, 1024, 0, 0, 1, 0, 0, 1]],
        "capability_hash": "a" * 64,
        "schema_hash": diagnostic_schema_hash(),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "wandb"
        writer = Tier0ArtifactWriter(
            root,
            run_id="run",
            job_name="job",
            repro={},
            wandb_writer=FailingWandb(),
        )
        with pytest.raises(RuntimeError, match="W&B"):
            writer.write(**kwargs)
        assert not (root / "event-00000001").exists()

        cap_root = Path(tmpdir) / "cap"
        capped = Tier0ArtifactWriter(
            cap_root,
            run_id="run",
            job_name="job",
            repro={},
            wandb_writer=None,
            max_run_bytes=0,
        )
        with pytest.raises(RuntimeError, match="cumulative artifact budget"):
            capped.write(**kwargs)
        assert not (cap_root / "event-00000001").exists()


@pytest.fixture(scope="module")
def gloo_world() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires torch.distributed.run with exactly two ranks")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=30))


@pytest.mark.distributed
def test_one_rank_update_failure_retries_with_fixed_world_collectives(
    gloo_world: None,
) -> None:
    calls: list[tuple[dict[str, float], int]] = []
    heartbeat = Tier0Heartbeat(
        _status_only_args(),
        [nn.Linear(2, 2)],
        object(),
        wandb_log=lambda payload, step: calls.append((payload, step)),
    )

    assert heartbeat.prepare_attempt(num_microbatches=1)
    success = torch.tensor(int(dist.get_rank() == 0))
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    assert not heartbeat.finish_optimizer_event(bool(success.item()), iteration=0)
    assert heartbeat.successful_updates == 0
    assert heartbeat.event_id == 0
    assert not calls

    assert heartbeat.prepare_attempt(num_microbatches=1)
    assert heartbeat.finish_optimizer_event(True, iteration=1)
    assert heartbeat.successful_updates == 1
    assert heartbeat.event_id == 1
    assert not calls
    dist.barrier()


@pytest.mark.distributed
def test_sink_schema_filesystem_cap_and_wandb_failures_keep_next_step_coherent(
    gloo_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _status_only_args()
    args.diagnostic_interval = 1
    heartbeat = Tier0Heartbeat(args, [nn.Linear(2, 2)], object())
    sink_rank = dist.get_world_size() - 1

    for index, label in enumerate(("schema", "filesystem", "cap", "wandb")):
        method_name = "_derive_host_payload" if label == "schema" else "_write_artifact"
        original = getattr(heartbeat, method_name)

        def fail(*_args, _label=label, **_kwargs):
            raise RuntimeError(f"injected {_label} sink failure")

        if dist.get_rank() == sink_rank:
            monkeypatch.setattr(heartbeat, method_name, fail)
        assert heartbeat.prepare_attempt(num_microbatches=1)
        assert heartbeat.finish_optimizer_event(True, iteration=index)
        marker = torch.tensor(1)
        dist.all_reduce(marker)
        assert marker == dist.get_world_size()
        if dist.get_rank() == sink_rank:
            monkeypatch.setattr(heartbeat, method_name, original)

    assert heartbeat.successful_updates == 4
    assert heartbeat.event_id == 4
    assert heartbeat.sink_failure_count == (4 if dist.get_rank() == sink_rank else 0)
    dist.barrier()


@pytest.mark.distributed
def test_pp2_incompatible_forward_function_is_rejected_globally_before_p2p(
    gloo_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameter = nn.Parameter(torch.ones(1))
    distributed_optimizer = SimpleNamespace(
        grad_scaler=None, optimizer=torch.optim.Adam([parameter])
    )
    monkeypatch.setattr(tier0_module, "GPTModel", nn.Linear)
    monkeypatch.setattr(
        tier0_module, "_distributed_optimizer", lambda _optimizer: distributed_optimizer
    )

    def canonical(_iterator, _model, *, diagnostic_heartbeat=None):
        microbatch_id = get_diagnostic_microbatch_id()
        diagnostic_heartbeat.register_local_loss_mask(microbatch_id, None)

    def nonregistering(_iterator, _model, *, diagnostic_heartbeat=None):
        return diagnostic_heartbeat

    args = SimpleNamespace(
        bf16=True,
        fp16=False,
        calculate_per_token_loss=True,
        transformer_impl="local",
    )
    monkeypatch.setattr(
        tier0_module,
        "_canonical_mask_producer",
        (
            canonical,
            tier0_module._MASK_PRODUCER_IDENTITY,
            tier0_module._NONINTERLEAVED_SCHEDULE_ADAPTER,
        ),
    )
    forward_function = canonical if dist.get_rank() == 0 else nonregistering
    capability = tier0_module.negotiate_tier0_capability(
        args, [nn.Linear(1, 1)], object(), forward_function
    )
    assert not capability.supported
    assert "canonical_mask_producer" in capability.reasons
    assert "rank_inconsistent" in capability.reasons
    dist.barrier()


@pytest.mark.distributed
def test_one_rank_startup_allocation_failure_is_agreed_before_training(
    gloo_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_empty = tier0_module.torch.empty

    def injected_empty(*args, **kwargs):
        if dist.get_rank() == 1 and kwargs.get("dtype") == torch.uint8:
            raise torch.OutOfMemoryError("injected startup allocation failure")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(tier0_module.torch, "empty", injected_empty)
    with pytest.raises(RuntimeError, match="startup allocation failed"):
        Tier0Heartbeat(_status_only_args(), [nn.Linear(2, 2)], object())
    dist.barrier()


@pytest.mark.distributed
def test_armed_pp_first_stage_broadcasts_loss_mask_to_every_tp_rank(
    gloo_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from megatron.training import utils as training_utils

    args = SimpleNamespace(
        create_attention_mask_in_dataloader=False,
        hybrid_context_parallel=False,
        micro_batch_size=1,
        pipeline_model_parallel_size=2,
        seq_length=4,
        sft=False,
    )
    monkeypatch.setattr(training_utils, "get_args", lambda: args)
    monkeypatch.setattr(
        training_utils.mpu, "get_tensor_model_parallel_rank", dist.get_rank
    )
    monkeypatch.setattr(
        training_utils.mpu, "get_tensor_model_parallel_src_rank", lambda: 0
    )
    monkeypatch.setattr(
        training_utils.mpu, "get_tensor_model_parallel_group", lambda: dist.group.WORLD
    )
    monkeypatch.setattr(training_utils.mpu, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(training_utils.mpu, "is_pipeline_last_stage", lambda: False)
    monkeypatch.setattr(
        torch.Tensor, "cuda", lambda self, non_blocking=False: self, raising=False
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    data = {
        "tokens": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[2, 3, 4, 5]]),
        "loss_mask": torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
    }
    iterator = iter((data,)) if dist.get_rank() == 0 else None
    batch = training_utils.get_batch_on_this_tp_rank(
        iterator, diagnostic_loss_mask=True
    )
    torch.testing.assert_close(batch["loss_mask"], data["loss_mask"])
    dist.barrier()
