# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real Gloo coverage for CP-sharded Tier-0 capture.

Run with::

    /tmp/megatron-r6-validation-venv/bin/python -m torch.distributed.run \
        --nproc-per-node 2 -m pytest -q \
        tests/unit_tests/diagnostics/test_capture_distributed.py
"""

import os
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training.diagnostics.accumulator import ReductionBinding
from megatron.training.diagnostics.capture import (
    CaptureTopology,
    Tier0CaptureSession,
    TokenLayout,
    pack_valid_token_mask_sideband,
    unpack_valid_token_mask_sideband,
)
from megatron.training.diagnostics.normalization import CanonicalDgradNormalizer
from megatron.training.diagnostics.schema import Tier0Status
from tests.unit_tests.diagnostics.test_capture import _FakeModel, _FakeTransformerLayer


@pytest.fixture(scope="module")
def gloo_world() -> None:
    """Initialize the two-rank CPU world used by this module."""

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires torch.distributed.run with exactly two ranks")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=30))


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


@pytest.mark.distributed
@pytest.mark.parametrize("micro_batch_size", (1, 4))
def test_fixed_shape_pp_mask_sideband_preserves_order_without_broadcast(
    gloo_world: None, micro_batch_size: int
) -> None:
    """Exercise timeout-protected point-to-point mask transport in microbatch order."""

    rank = dist.get_rank()
    sequence_length = 8
    for microbatch_id in range(8):
        if rank == 0:
            mask = torch.full(
                (micro_batch_size, sequence_length), float(microbatch_id + 1)
            )
            payload = pack_valid_token_mask_sideband(
                mask,
                micro_batch_size=micro_batch_size,
                sequence_length=sequence_length,
                device="cpu",
            )
            dist.send(payload, dst=1)
        else:
            payload = torch.empty(micro_batch_size * sequence_length + 1)
            dist.recv(payload, src=0)
            staged = unpack_valid_token_mask_sideband(
                payload,
                micro_batch_size=micro_batch_size,
                sequence_length=sequence_length,
                device="cpu",
            )
            assert staged.valid
            torch.testing.assert_close(
                staged.values,
                torch.full(
                    (sequence_length, micro_batch_size, 1), float(microbatch_id + 1)
                ),
            )
    dist.barrier()


@pytest.mark.distributed
@pytest.mark.parametrize("micro_batch_size", (1, 4))
@pytest.mark.parametrize("sequence_parallel", (False, True))
def test_real_mcore_tp_sp_module_shapes_use_typed_token_layout(
    gloo_world: None,
    monkeypatch: pytest.MonkeyPatch,
    micro_batch_size: int,
    sequence_parallel: bool,
) -> None:
    """Exercise a real local layer forward/backward on a two-rank Gloo TP group."""

    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    class _CpuRngTracker:
        def fork(self, *_args: object, **_kwargs: object):
            return nullcontext()

    monkeypatch.setattr(tensor_parallel, "get_cuda_rng_tracker", _CpuRngTracker)
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1
    )
    try:
        hidden_size = 8
        sequence_length = 8
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=2,
            tensor_model_parallel_size=2,
            sequence_parallel=False,
            transformer_impl="local",
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        layer = TransformerLayer(config, get_gpt_layer_local_submodules())
        if sequence_parallel:
            # The validation venv lacks Apex, so the local spec falls back to a
            # torch LayerNorm that rejects SP construction. The parallel linear
            # modules themselves are real MCore modules; enable their normal SP
            # paths after the unused norm modules have been constructed.
            config.sequence_parallel = True
            for column in (layer.self_attention.linear_qkv, layer.mlp.linear_fc1):
                column.sequence_parallel = True
                column.allreduce_dgrad = False
            for row in (layer.self_attention.linear_proj, layer.mlp.linear_fc2):
                row.sequence_parallel = True
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
            parameter.fill_(0.03125)
        layer.train()

        topology = CaptureTopology.from_parallel_state(sequence_parallel=sequence_parallel)
        session = Tier0CaptureSession(
            layer,
            num_layers=1,
            topology=topology,
            device="cpu",
            micro_batch_size=micro_batch_size,
            local_sequence_length=sequence_length,
            calculate_per_token_loss=True,
            dgrad_normalizer=CanonicalDgradNormalizer(),
            reduction_binding=ReductionBinding.flat_world(None),
        )
        layouts = {target.family.value: target.token_layout for target in session.targets}
        assert layouts["qkv"] == TokenLayout.CP_LOCAL_SEQUENCE
        assert layouts["fc1"] == TokenLayout.CP_LOCAL_SEQUENCE
        expected_row_layout = (
            TokenLayout.TP_SEQUENCE_SHARD if sequence_parallel else TokenLayout.CP_LOCAL_SEQUENCE
        )
        assert layouts["attn_out"] == expected_row_layout
        assert layouts["fc2"] == expected_row_layout
        assert layouts["residual"] == expected_row_layout

        mask = torch.ones(micro_batch_size, sequence_length)
        mask[:, : sequence_length // 2] = 0
        mask[0, 0] = 1
        local_input_sequence = (
            sequence_length // topology.tensor_parallel_size
            if sequence_parallel
            else sequence_length
        )
        rank = dist.get_rank()
        session.arm(expected_microbatch_ids=(0,))
        session.begin_microbatch(0)
        session.register_valid_token_mask(0, mask)
        hidden_states = torch.full(
            (local_input_sequence, micro_batch_size, hidden_size),
            rank + 1.0,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        attention_mask = torch.triu(
            torch.ones(1, 1, sequence_length, sequence_length, dtype=torch.bool), diagonal=1
        )
        output = layer(hidden_states, attention_mask=attention_mask)[0]
        session.end_microbatch(0)

        full_mask = mask.transpose(0, 1).unsqueeze(-1)
        row_mask = (
            full_mask.chunk(topology.tensor_parallel_size, dim=0)[rank]
            if sequence_parallel
            else full_mask
        )
        assert tuple(output.shape) == (local_input_sequence, micro_batch_size, hidden_size)
        loss = (output * row_mask).sum()
        loss.backward()
        result = session.finalize()

        valid_tokens = mask.sum(dtype=torch.float64)
        expected_widths = {
            "qkv": hidden_size * 3,
            "fc1": config.ffn_hidden_size,
            "attn_out": hidden_size,
            "fc2": hidden_size,
            "residual": hidden_size,
        }
        for family, width in expected_widths.items():
            for observation in ("activation", "dgrad"):
                slots = result.accumulator.slots(f"{observation}/{family}/layer_0")
                torch.testing.assert_close(
                    result.accumulator.sum_pack[slots.count], valid_tokens * width
                )
                assert result.accumulator.sum_pack[slots.observation_error] == 0
        assert result.status == Tier0Status.OK
        session.close()
    finally:
        parallel_state.destroy_model_parallel()
        dist.barrier()
