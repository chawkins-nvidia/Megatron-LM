# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tier-0 heartbeat orchestration and runtime-contract tests."""

import io
import json
import os
import subprocess
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from megatron.core.diagnostics import get_diagnostic_microbatch_id
from megatron.core.pipeline_parallel.schedules import forward_step
from megatron.training.argument_utils import _default_config_from_args
from megatron.training.config.training_config import LoggerConfig
from megatron.training.diagnostics import tier0 as tier0_module
from megatron.training.diagnostics.tier0 import Tier0Cadence, Tier0Heartbeat

from megatron.training.diagnostics.accumulator import (  # isort: skip
    PackedSufficientStatistics,
    ReductionBinding,
)
from megatron.training.diagnostics.capability import (  # isort: skip
    capability_payload,
    contract_hash,
    schema_hash,
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
    assert len(calls) == (1 if rank == 0 else 0)
    if rank == 0:
        payload, step = calls[0]
        assert step == 6
        assert tuple(payload) == TIER0_KEYS
        assert len(payload) == 75
        assert payload["diag/v2/status/valid"] == 0
        assert payload["diag/v2/event/successful_update"] == 1
        assert all(
            math_value != math_value
            for key, math_value in payload.items()
            if key in TIER0_METRIC_KEYS
        )
        assert "diag/v2/status/status_code" not in payload
        assert "diag/v2/event/event_id" not in payload
    assert set(tensorboard.values) == set(TIER0_KEYS)


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
    buffer = io.BytesIO()
    torch.save({"args": args}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)["args"]
    assert restored.diagnostic_successful_updates == 123
    assert restored.diagnostic_event_id == 9


def test_capability_probe_contract_and_cpu_only_subprocess() -> None:
    payload = capability_payload(
        source_commit="abc123", build_identity="image@sha256:def"
    )
    assert payload["contract_hash"] == contract_hash()
    assert payload["schema_hash"] == schema_hash()
    assert payload["schema_key_count"] == 75
    assert payload["supported_max_tier"] == 0
    assert payload["runtime_contract_present"]
    assert payload["heartbeat_consumer"] == (
        "megatron.training.diagnostics.tier0.Tier0Heartbeat"
    )
    assert payload["config_consumer"] == (
        "megatron.training.config.training_config.LoggerConfig"
    )
    assert payload["source_commit"] == "abc123"
    assert payload["build_identity"] == "image@sha256:def"
    assert payload["writer_policy"] == {
        "wandb_rank": 0,
        "wandb_calls_per_event": 1,
        "wandb_step": "iteration_plus_one",
        "tensorboard_ownership": "existing",
    }
    assert set(TIER0_METADATA_KEYS).issubset(TIER0_KEYS)

    code = (
        "import json,sys; "
        "from megatron.training.diagnostics.capability import capability_payload; "
        "print(json.dumps({'torch_loaded': 'torch' in sys.modules, "
        "'payload': capability_payload(source_commit='commit', build_identity='build')}))"
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
    assert result["payload"]["source_commit"] == "commit"
    assert result["payload"]["build_identity"] == "build"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "megatron.training.diagnostics.capability",
            "--json",
            "--source-commit",
            "commit",
            "--build-identity",
            "build",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    cli_payload = json.loads(completed.stdout)
    assert cli_payload["runtime_contract_present"]
    assert cli_payload["source_commit"] == "commit"
    assert cli_payload["build_identity"] == "build"


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
    assert not heartbeat.finish_optimizer_event(dist.get_rank() == 0, iteration=0)
    assert heartbeat.successful_updates == 0
    assert heartbeat.event_id == 0
    assert not calls

    assert heartbeat.prepare_attempt(num_microbatches=1)
    assert heartbeat.finish_optimizer_event(True, iteration=1)
    assert heartbeat.successful_updates == 1
    assert heartbeat.event_id == 1
    assert len(calls) == (1 if dist.get_rank() == 0 else 0)
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
