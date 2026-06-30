# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real two-rank CUDA/NCCL coverage for Tier-0 transformer-layer capture.

Run with::

    python -m torch.distributed.run --standalone --nproc-per-node=2 -m pytest \
        --confcutdir=tests/unit_tests/diagnostics -q \
        tests/unit_tests/diagnostics/test_capture_cuda.py
"""

import os
from collections.abc import Iterator
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training.diagnostics.accumulator import ReductionBinding
from megatron.training.diagnostics.capture import (
    CaptureTopology,
    Tier0CaptureSession,
    TokenLayout,
)
from megatron.training.diagnostics.normalization import CanonicalDgradNormalizer
from megatron.training.diagnostics.schema import Tier0Status

_HIDDEN_SIZE = 8
_SEQUENCE_LENGTH = 8


@pytest.fixture(scope="module")
def cuda_nccl_world() -> Iterator[torch.device]:
    """Initialize the exact two-rank NCCL topology used by this module."""

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires torch.distributed.run with exactly two ranks")
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        pytest.skip("requires CUDA and NCCL")
    if "LOCAL_RANK" not in os.environ:
        pytest.skip("requires LOCAL_RANK from torch.distributed.run")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            distributed_timeout_minutes=1,
            create_gloo_process_groups=False,
        )
        model_parallel_cuda_manual_seed(1234)

        assert dist.get_world_size() == 2
        assert dist.get_backend(dist.group.WORLD) == "nccl"
        assert (
            dist.get_backend(parallel_state.get_tensor_model_parallel_group()) == "nccl"
        )
        assert parallel_state.get_tensor_model_parallel_world_size() == 2
        assert parallel_state.get_pipeline_model_parallel_world_size() == 1
        yield device
    finally:
        if dist.is_initialized():
            try:
                torch.cuda.synchronize(device)
                dist.barrier()
            finally:
                try:
                    parallel_state.destroy_model_parallel()
                finally:
                    dist.destroy_process_group()


def _build_local_layer(
    sequence_parallel: bool, device: torch.device
) -> TransformerLayer:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=4 * _HIDDEN_SIZE,
        num_attention_heads=2,
        tensor_model_parallel_size=2,
        sequence_parallel=sequence_parallel,
        transformer_impl="local",
        use_cpu_initialization=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
        apply_rope_fusion=False,
        persist_layer_norm=False,
    )
    layer = TransformerLayer(config, get_gpt_layer_local_submodules()).to(
        device=device, dtype=torch.bfloat16
    )
    assert all(
        "transformer_engine" not in type(module).__module__
        for module in layer.modules()
    )
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    layer.train()
    return layer


@pytest.mark.distributed
@pytest.mark.parametrize("sequence_parallel", (False, True))
@pytest.mark.parametrize("micro_batch_size", (1, 4))
def test_real_cuda_nccl_tp_capture(
    cuda_nccl_world: torch.device,
    sequence_parallel: bool,
    micro_batch_size: int,
) -> None:
    """Capture a real local BF16 layer over TP/SP NCCL collectives."""

    device = cuda_nccl_world
    layer = _build_local_layer(sequence_parallel, device)
    topology = CaptureTopology.from_parallel_state(sequence_parallel=sequence_parallel)
    session = Tier0CaptureSession(
        layer,
        num_layers=1,
        topology=topology,
        device=device,
        micro_batch_size=micro_batch_size,
        local_sequence_length=_SEQUENCE_LENGTH,
        calculate_per_token_loss=True,
        dgrad_normalizer=CanonicalDgradNormalizer(),
        reduction_binding=ReductionBinding.flat_world(dist.group.WORLD),
    )
    try:
        layouts = {
            target.family.value: target.token_layout for target in session.targets
        }
        assert layouts["qkv"] == TokenLayout.CP_LOCAL_SEQUENCE
        assert layouts["fc1"] == TokenLayout.CP_LOCAL_SEQUENCE
        expected_row_layout = (
            TokenLayout.TP_SEQUENCE_SHARD
            if sequence_parallel
            else TokenLayout.CP_LOCAL_SEQUENCE
        )
        for family in ("attn_out", "fc2", "residual"):
            assert layouts[family] == expected_row_layout

        for observation in ("activation", "dgrad"):
            for family in ("qkv", "fc1", "attn_out", "fc2", "residual"):
                expected_owner = (
                    family in ("qkv", "fc1")
                    or sequence_parallel
                    or topology.tensor_parallel_rank == 0
                )
                assert (
                    session.registry.owns(f"{observation}/{family}/layer_0")
                    == expected_owner
                )

        valid_token_mask = torch.ones(
            (micro_batch_size, _SEQUENCE_LENGTH), dtype=torch.float32, device=device
        )
        valid_token_mask[:, :4] = 0
        valid_token_mask[0, 0] = 1
        full_hidden_states = (
            torch.arange(
                _SEQUENCE_LENGTH * micro_batch_size * _HIDDEN_SIZE,
                dtype=torch.float32,
                device=device,
            )
            .reshape(_SEQUENCE_LENGTH, micro_batch_size, _HIDDEN_SIZE)
            .mul_(1.0 / 128.0)
            .to(dtype=torch.bfloat16)
        )
        if sequence_parallel:
            hidden_states = full_hidden_states.chunk(
                topology.tensor_parallel_size, dim=0
            )[topology.tensor_parallel_rank]
        else:
            hidden_states = full_hidden_states
        hidden_states = hidden_states.detach().clone().requires_grad_(True)
        assert hidden_states.is_leaf and hidden_states.requires_grad

        attention_mask = torch.triu(
            torch.ones(
                (1, 1, _SEQUENCE_LENGTH, _SEQUENCE_LENGTH),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )
        session.arm(expected_microbatch_ids=(0,))
        session.begin_microbatch(0)
        session.register_valid_token_mask(0, valid_token_mask)
        output = layer(hidden_states, attention_mask=attention_mask)[0]
        session.end_microbatch(0)

        full_loss_mask = valid_token_mask.transpose(0, 1).unsqueeze(-1)
        loss_mask = (
            full_loss_mask.chunk(topology.tensor_parallel_size, dim=0)[
                topology.tensor_parallel_rank
            ]
            if sequence_parallel
            else full_loss_mask
        )
        expected_local_sequence = _SEQUENCE_LENGTH // (2 if sequence_parallel else 1)
        assert tuple(output.shape) == (
            expected_local_sequence,
            micro_batch_size,
            _HIDDEN_SIZE,
        )
        (output * loss_mask).sum().backward()
        result = session.finalize()

        expected_valid_tokens = {1: 5, 4: 17}[micro_batch_size]
        assert result.status == Tier0Status.OK
        torch.testing.assert_close(
            result.global_valid_tokens,
            torch.tensor(
                float(expected_valid_tokens), dtype=torch.float64, device=device
            ),
        )
        expected_widths = {
            "qkv": 3 * _HIDDEN_SIZE,
            "fc1": layer.config.ffn_hidden_size,
            "attn_out": _HIDDEN_SIZE,
            "fc2": _HIDDEN_SIZE,
            "residual": _HIDDEN_SIZE,
        }
        for family, width in expected_widths.items():
            for observation in ("activation", "dgrad"):
                name = f"{observation}/{family}/layer_0"
                slots = result.accumulator.slots(name)
                torch.testing.assert_close(
                    result.accumulator.sum_pack[slots.count],
                    torch.tensor(
                        expected_valid_tokens * width,
                        dtype=torch.float64,
                        device=device,
                    ),
                )
                for error_slot in (
                    slots.observation_error,
                    slots.mask_error,
                    slots.nonfinite,
                    slots.nonfinite_arithmetic,
                ):
                    torch.testing.assert_close(
                        result.accumulator.sum_pack[error_slot],
                        torch.zeros((), dtype=torch.float64, device=device),
                    )
                rms = result.accumulator.rms(name)
                assert rms.valid
                assert torch.isfinite(rms.value)
    finally:
        session.close()
