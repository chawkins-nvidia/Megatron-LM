# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging

import pytest
import torch
import torch.nn as nn

from megatron.training import activation_logging
from megatron.training.activation_logging import ActivationLogger


class TinyDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(16)
        self.proj = nn.Linear(16, 16)

    def forward(self, x):
        return self.proj(self.norm(x))


class TinyDecoderModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(
            [TinyDecoderBlock() for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(16, 8)

    def forward(self, x):
        for layer in self.decoder.layers:
            x = layer(x)
        return self.output_layer(x)


@pytest.mark.parametrize(
    ("statistic", "expected"),
    (
        ("rms", torch.sqrt(torch.tensor(169.0 / 3.0))),
        ("abs_mean", torch.tensor(19.0 / 3.0)),
        ("std", torch.tensor([3.0, 4.0, -12.0]).std(unbiased=False)),
        ("abs_max", torch.tensor(12.0)),
        ("abs_min", torch.tensor(3.0)),
    ),
)
def test_streaming_finite_summary_matches_reference(statistic, expected):
    tensor = torch.tensor([3.0, float("nan"), 4.0, float("inf"), -12.0])

    actual = activation_logging._streaming_finite_summary(
        tensor, statistic, chunk_numel=2
    )

    assert torch.isclose(actual, expected)


def test_streaming_finite_summary_all_nonfinite_is_nan():
    actual = activation_logging._streaming_finite_summary(
        torch.tensor([float("nan"), float("inf")]), "rms", chunk_numel=1
    )

    assert torch.isnan(actual)


@pytest.fixture()
def logger(tmp_path):
    return ActivationLogger(save_dir=str(tmp_path))


@pytest.fixture()
def simple_model():
    """A decoder-shaped model wrapped as a single-element model chunk list."""
    return [TinyDecoderModel(num_layers=6)]


class TestMakeTpeHook:
    """Tests for _make_tpe_hook regex layer extraction."""

    def test_extracts_decoder_layer_number(self, logger):
        hook = logger._make_tpe_hook(
            "chunk0", "decoder.layers.3.mlp.experts.linear_fc1"
        )
        assert hook is not None
        fake_tpe = [128, 64, 96, 80]
        hook(None, (torch.zeros(1), fake_tpe), {}, torch.zeros(1))
        assert logger._decoder_tpe_records[3] == [fake_tpe]

    def test_extracts_mtp_layer_number(self, logger):
        hook = logger._make_tpe_hook(
            "chunk0", "mtp.layers.0.mtp_model_layer.layers.1.mlp.experts.linear_fc1"
        )
        assert hook is not None
        fake_tpe = [50, 50]
        hook(None, (torch.zeros(1), fake_tpe), {}, torch.zeros(1))
        assert logger._mtp_tpe_records[(0, 1)] == [fake_tpe]

    def test_returns_none_for_non_matching_name(self, logger, caplog):
        with caplog.at_level(logging.WARNING):
            hook = logger._make_tpe_hook("chunk0", "some.module.without.layer.number")
        assert hook is None
        assert "Cannot extract layer number" in caplog.text


class TestSaveTpe:
    """Tests for save_tpe JSONL output."""

    def test_creates_jsonl(self, tmp_path, logger):
        logger._decoder_tpe_records[3].append([10, 20])
        logger._decoder_tpe_records[3].append([30, 40])
        logger._decoder_tpe_records[7].append([50, 60])
        logger._mtp_tpe_records[(0, 1)].append([70, 80])

        logger.save_tpe(iteration=100)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        filepath = tmp_path / "tokens_per_expert" / f"rank{rank}.jsonl"
        assert filepath.exists()

        records = [
            json.loads(line) for line in filepath.read_text().strip().split("\n")
        ]
        assert records == [
            {"iter": 100, "block": "decoder", "layer": 3, "tpe": [[10, 20], [30, 40]]},
            {"iter": 100, "block": "decoder", "layer": 7, "tpe": [[50, 60]]},
            {"iter": 100, "block": "mtp", "mtp_idx": 0, "layer": 1, "tpe": [[70, 80]]},
        ]

    def test_appends_across_calls(self, tmp_path, logger):
        logger._decoder_tpe_records[0].append([10, 20])
        logger.save_tpe(iteration=100)

        logger._decoder_tpe_records[0].append([30, 40])
        logger.save_tpe(iteration=200)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        filepath = tmp_path / "tokens_per_expert" / f"rank{rank}.jsonl"
        records = [
            json.loads(line) for line in filepath.read_text().strip().split("\n")
        ]
        assert len(records) == 2
        assert records[0]["iter"] == 100
        assert records[1]["iter"] == 200


class TestActivationHookLifecycle:
    """Tests for activation hook registration and removal."""

    def test_register_and_remove(self, logger, simple_model):
        logger.register_activation_hooks(simple_model)
        # log4pluslast on six 0-indexed layers selects 0/1, 3/4, and last=5.
        # Each selected block has norm+proj hooks; output_layer is a special anchor.
        assert len(logger._activation_hooks) == 11

        logger.remove_activation_hooks()
        assert len(logger._activation_hooks) == 0

    def test_hooks_capture_activations(self, logger, simple_model):
        logger.register_activation_hooks(simple_model)

        simple_model[0](torch.randn(2, 16))

        assert len(logger._activations_state_dict) > 0
        keys = logger._activations_state_dict["model_chunk0"]
        assert "decoder.layers.0.norm/input0" in keys
        assert "decoder.layers.1.proj/output0" in keys
        assert "decoder.layers.2.norm/input0" not in keys
        assert "decoder.layers.3.norm/input0" in keys
        assert "decoder.layers.4.proj/output0" in keys
        assert "decoder.layers.5.proj/output0" in keys
        assert "output_layer/output0" in keys
        logger.remove_activation_hooks()

    def test_hooks_average_across_microbatches(self, logger, simple_model):
        logger.register_activation_hooks(simple_model)

        simple_model[0](torch.ones(2, 16))
        simple_model[0](torch.full((2, 16), 3.0))

        keys = logger._activations_state_dict["model_chunk0"]
        counts = logger._activation_counts["model_chunk0"]
        assert counts["decoder.layers.0.norm/input0"] == 2
        assert torch.isclose(keys["decoder.layers.0.norm/input0"], torch.tensor(2.0))
        logger.remove_activation_hooks()

    def test_removed_hooks_dont_capture(self, logger, simple_model):
        logger.register_activation_hooks(simple_model)
        logger.remove_activation_hooks()

        simple_model[0](torch.randn(2, 16))
        assert len(logger._activations_state_dict) == 0
