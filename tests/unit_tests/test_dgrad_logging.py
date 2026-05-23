# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn

from megatron.training.dgrad_logging import DataGradLogger


def test_dgrad_logger_captures_norm_and_linear_inputs(tmp_path):
    model = [nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 8))]
    inputs = torch.randn(2, 16, requires_grad=True)

    logger = DataGradLogger(save_dir=str(tmp_path))
    logger.register_hooks(model)
    model[0](inputs).sum().backward()

    keys = logger._dgrads_state_dict["model_chunk0"]
    assert "0/input0" in keys
    assert "0/output" in keys
    assert "1/input0" in keys
    logger.remove_hooks()
