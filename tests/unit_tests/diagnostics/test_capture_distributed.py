# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real Gloo coverage for CP-sharded Tier-0 capture.

Run with::

    /tmp/megatron-r6-validation-venv/bin/python -m torch.distributed.run \
        --nproc-per-node 2 -m pytest -q \
        tests/unit_tests/diagnostics/test_capture_distributed.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.training.diagnostics.accumulator import ReductionBinding
from megatron.training.diagnostics.capture import CaptureTopology, Tier0CaptureSession
from megatron.training.diagnostics.normalization import CanonicalDgradNormalizer
from megatron.training.diagnostics.schema import Tier0Status
from tests.unit_tests.diagnostics.test_capture import _FakeModel, _FakeTransformerLayer


@pytest.fixture(scope="module")
def gloo_world() -> None:
    """Initialize the two-rank CPU world used by this module."""

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires torch.distributed.run with exactly two ranks")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")


@pytest.mark.distributed
def test_cp_token_shards_pool_masks_and_canonical_dgrad_once(gloo_world: None) -> None:
    rank = dist.get_rank()
    masks = (torch.tensor([[1.0, 0.0, 1.0]]), torch.tensor([[0.0, 1.0, 1.0]]))
    values = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]))
    mask = masks[rank]
    inputs = values[rank].view(3, 1, 1).requires_grad_(True)
    layer = _FakeTransformerLayer()
    session = Tier0CaptureSession(
        _FakeModel(layer),
        num_layers=1,
        topology=CaptureTopology(context_parallel_rank=rank, context_parallel_size=2),
        device="cpu",
        micro_batch_size=1,
        local_sequence_length=3,
        calculate_per_token_loss=True,
        dgrad_normalizer=CanonicalDgradNormalizer(),
        reduction_binding=ReductionBinding.flat_world(None),
    )
    session.arm()
    session.begin_microbatch(0)
    session.register_valid_token_mask(0, mask)
    output = layer(inputs)[0]
    session.end_microbatch(0)
    (output * mask.transpose(0, 1).unsqueeze(-1)).sum().backward()
    result = session.finalize()

    selected_residuals = torch.tensor([0.75, 2.25, 3.75, 4.5])
    expected_activation_rms = selected_residuals.square().mean().sqrt().double()
    torch.testing.assert_close(result.global_valid_tokens, torch.tensor(4.0).double())
    torch.testing.assert_close(
        result.accumulator.rms("activation/residual/layer_0").value, expected_activation_rms
    )
    torch.testing.assert_close(
        result.accumulator.rms("dgrad/residual/layer_0").value, torch.tensor(0.25).double()
    )
    assert result.status == Tier0Status.OK
    session.close()
    dist.barrier()
