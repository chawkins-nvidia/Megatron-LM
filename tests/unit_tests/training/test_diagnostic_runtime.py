# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused tests for the first Tier-1/2 trainer vertical slice."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from megatron.training.config.training_config import LoggerConfig
from megatron.training.datasets.data_samplers import SamplerIssuedIndex
from megatron.training.diagnostics.function_response import TIER1_KEYS
from megatron.training.diagnostics.runtime import (
    TieredDiagnosticRuntime,
    diagnostics_requested_tier,
    wrap_stable_training_dataset,
)
from megatron.training.diagnostics.schema import (
    TIER0_KEYS,
    assert_tiered_payload_schema,
)
from megatron.training.diagnostics.secant import TIER2_OUTPUT_KEYS


class _Dataset:
    split = "train"

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.tensor([index], dtype=torch.int64),
            "labels": torch.tensor([index + 1], dtype=torch.int64),
            "loss_mask": torch.ones(1, dtype=torch.float32),
            "position_ids": torch.zeros(1, dtype=torch.int64),
        }


@pytest.mark.parametrize(
    ("heartbeat", "enabled", "tier", "expected"),
    (
        (False, True, 2, -1),
        (True, False, 2, 0),
        (True, True, 0, 0),
        (True, True, 1, 1),
        (True, True, 2, 2),
    ),
)
def test_requested_tier_preserves_disabled_and_tier0_paths(
    heartbeat: bool, enabled: bool, tier: int, expected: int
) -> None:
    args = SimpleNamespace(
        diagnostic_heartbeat=heartbeat,
        diag_enabled=enabled,
        diag_max_tier=tier,
    )

    assert diagnostics_requested_tier(args) == expected


def test_train_resolves_pipeline_schedule_before_diagnostic_construction() -> None:
    from megatron.training.training import train

    source = inspect.getsource(train)
    schedule = "forward_backward_func = get_forward_backward_func()"
    heartbeat = "diagnostic_heartbeat = Tier0Heartbeat("

    assert source.count(schedule) == 1
    assert source.index(schedule) < source.index(heartbeat)


def test_stable_dataset_wrapper_is_opt_in_and_delegates_metadata() -> None:
    dataset = _Dataset()
    disabled = SimpleNamespace(diagnostic_heartbeat=True, diag_enabled=False)
    enabled = SimpleNamespace(
        diagnostic_heartbeat=True, diag_enabled=True, diag_max_tier=2
    )

    assert wrap_stable_training_dataset(dataset, disabled) is dataset
    wrapped = wrap_stable_training_dataset(dataset, enabled)
    assert wrapped.split == "train"
    sample = wrapped[SamplerIssuedIndex(epoch=7, sampler_index=2)]
    assert sample["__diag_sample_epoch"] == 7
    assert sample["__diag_sample_index"] == 2


def test_logger_config_consumes_scaling_tiered_fields() -> None:
    config = LoggerConfig()
    required = {
        "diag_schema",
        "diag_enabled",
        "diag_max_tier",
        "diag_require_tier",
        "diag_sample_selector",
        "diag_sample_seed",
        "diag_max_valid_positions_global",
        "diag_replay_input_bytes_per_rank",
        "diag_max_extra_allocated_bytes_per_rank",
        "diag_tier1_replay_mode",
        "diag_tier2_midpoint_fraction",
        "diag_tier2_midpoint_tolerance",
        "diag_tier2_min_response_over_replay_floor",
        "diag_capability_policy",
    }

    assert required <= vars(config).keys()
    assert config.diag_max_tier == 0
    assert config.diag_tier2_midpoint_fraction == 0.5


def test_cumulative_tier_payload_contract_is_exact_and_ordered() -> None:
    tier0 = dict.fromkeys(TIER0_KEYS, 0.0)
    tier1 = {**tier0, **dict.fromkeys(TIER1_KEYS, 1.0)}
    tier2 = {**tier1, **dict.fromkeys(TIER2_OUTPUT_KEYS, 2.0)}

    assert_tiered_payload_schema(tier0, effective_tier=0)
    assert_tiered_payload_schema(tier1, effective_tier=1)
    assert_tiered_payload_schema(tier2, effective_tier=2)
    assert len(tier1) == 105
    assert len(tier2) == 122

    reordered = dict(reversed(tuple(tier2.items())))
    with pytest.raises(ValueError, match="order_matches=False"):
        assert_tiered_payload_schema(reordered, effective_tier=2)


def test_secant_runtime_orders_commit_endpoints_midpoint_and_restore() -> None:
    source = inspect.getsource(TieredDiagnosticRuntime._complete_secant)

    operations = (
        "commit_secant_delta",
        'run_endpoint("post")',
        'run_endpoint("post_repeat")',
        "install_secant_midpoint",
        'run_endpoint("midpoint")',
        "restore_secant_post",
    )
    offsets = tuple(source.index(operation) for operation in operations)
    assert offsets == tuple(sorted(offsets))
    assert source.count("_materialize_parameters()") == 2


def test_first_backend_accepts_tensor_parallelism_and_still_rejects_cp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TieredDiagnosticRuntime.__new__(TieredDiagnosticRuntime)
    runtime.required_tier = 1
    runtime.tier = 2
    runtime.models = (object(),)
    runtime.args = SimpleNamespace(
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        transformer_impl="local",
        diagnostic_unsupported_policy="error",
    )
    monkeypatch.setattr(runtime, "_tp_size", lambda: 2)

    runtime._validate_first_backend()
    assert runtime.tier == 2

    runtime.args.context_parallel_size = 2
    with pytest.raises(
        RuntimeError,
        match="context_parallel_size_must_be_1",
    ):
        runtime._validate_first_backend()


def test_complete_optimizer_event_reports_typed_begin_status_before_unarmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TieredDiagnosticRuntime.__new__(TieredDiagnosticRuntime)
    runtime._attempt_due = True
    runtime.transaction = object()
    errors: list[BaseException] = []
    runtime.fatal_abort = errors.append
    adapter = SimpleNamespace(
        armed=False,
        status_for_event_consensus=lambda: torch.tensor([2], dtype=torch.int64),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="fatal-abort protocol returned"):
        runtime.complete_optimizer_event(adapter, None, update_successful=True)

    assert len(errors) == 1
    assert str(errors[0]) == "pre-update snapshot failed with adapter status 2"
