# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn

from megatron.training.dgrad_logging import DataGradLogger


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
        self.decoder.layers = nn.ModuleList([TinyDecoderBlock() for _ in range(num_layers)])
        self.output_layer = nn.Linear(16, 8)

    def forward(self, x):
        for layer in self.decoder.layers:
            x = layer(x)
        return self.output_layer(x)


def test_dgrad_logger_captures_norm_and_linear_inputs(tmp_path):
    model = [TinyDecoderModel(num_layers=6)]
    inputs = torch.randn(2, 16, requires_grad=True)

    logger = DataGradLogger(save_dir=str(tmp_path))
    logger.register_hooks(model)
    model[0](inputs).sum().backward()

    keys = logger._dgrads_state_dict["model_chunk0"]
    assert "decoder.layers.0.norm/input0" in keys
    assert "decoder.layers.1.proj/output" in keys
    assert "decoder.layers.2.norm/input0" not in keys
    assert "decoder.layers.3.norm/input0" in keys
    assert "decoder.layers.4.proj/output" in keys
    assert "decoder.layers.5.proj/output" in keys
    assert "output_layer/output" in keys
    logger.remove_hooks()


def test_dgrad_logger_averages_across_microbatches(tmp_path):
    model = [TinyDecoderModel(num_layers=6)]
    logger = DataGradLogger(save_dir=str(tmp_path))
    logger.register_hooks(model)

    model[0](torch.ones(2, 16, requires_grad=True)).sum().backward()
    model[0](torch.full((2, 16), 2.0, requires_grad=True)).sum().backward()

    counts = logger._dgrads_counts["model_chunk0"]
    assert counts["output_layer/output"] == 2
    logger.remove_hooks()
