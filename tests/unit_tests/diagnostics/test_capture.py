# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import megatron.core.tensor_parallel.random as random_module
from megatron.core.diagnostics import (
    diagnostic_recompute,
    get_diagnostic_microbatch_id,
    is_diagnostic_recompute,
    set_diagnostic_microbatch_id,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import CheckpointFunction, CheckpointWithoutOutput
from megatron.core.transformer.cuda_graphs import set_current_microbatch
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training.diagnostics.accumulator import ReductionBinding
from megatron.training.diagnostics.capture import (
    CaptureTopology,
    Tier0CaptureSession,
    slice_sequence_parallel_mask,
    stage_valid_token_mask,
    transport_valid_token_mask,
)
from megatron.training.diagnostics.normalization import CanonicalDgradNormalizer
from megatron.training.diagnostics.registry import MetricFamily
from megatron.training.diagnostics.schema import Tier0Status


class _FakeColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, scale: float) -> None:
        nn.Module.__init__(self)
        self.scale = scale

    def forward(self, values: torch.Tensor):
        return values * self.scale, None


class _FakeRowParallelLinear(RowParallelLinear):
    def __init__(self, scale: float) -> None:
        nn.Module.__init__(self)
        self.scale = scale

    def forward(self, values: torch.Tensor):
        return values * self.scale, None


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_qkv = _FakeColumnParallelLinear(2.0)
        self.linear_proj = _FakeRowParallelLinear(0.5)


class _FakeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_fc1 = _FakeColumnParallelLinear(3.0)
        self.linear_fc2 = _FakeRowParallelLinear(0.25)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values, _ = self.linear_fc1(values)
        values, _ = self.linear_fc2(values)
        return values


class _FakeTransformerLayer(TransformerLayer):
    def __init__(self, layer_number: int = 1, *, selective_recompute: bool = False) -> None:
        nn.Module.__init__(self)
        self.layer_number = layer_number
        self.is_moe_layer = False
        self.config = SimpleNamespace(
            transformer_impl="local",
            fp8=None,
            fp4=None,
            cuda_graph_impl="none",
            mlp_chunks_for_training=1,
        )
        self.self_attention = _FakeAttention()
        self.mlp = _FakeMLP()
        self.selective_recompute = selective_recompute

    def forward(self, values: torch.Tensor):
        values, _ = self.self_attention.linear_qkv(values)
        values, _ = self.self_attention.linear_proj(values)
        if self.selective_recompute:
            values = CheckpointFunction.apply(self.mlp, False, values)
        else:
            values = self.mlp(values)
        return values, None


class _FakeModel(nn.Module):
    def __init__(self, layer: _FakeTransformerLayer) -> None:
        super().__init__()
        self.layer = layer


def _identity_reducer(tensor: torch.Tensor, *, op: object, group: object | None) -> None:
    del tensor, op, group


def _binding() -> ReductionBinding:
    return ReductionBinding.flat_world(None, reducer=_identity_reducer)


@pytest.fixture
def checkpoint_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random_module, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(random_module, "_set_all_rng_states", lambda *args: None)


def _run_capture(
    *,
    masks: tuple[torch.Tensor, ...],
    loss_scale: float = 1.0,
    recompute: str = "off",
    values: tuple[torch.Tensor, ...] | None = None,
) -> tuple[Tier0CaptureSession, object]:
    layer = _FakeTransformerLayer(selective_recompute=recompute == "selective")
    model = _FakeModel(layer)
    session = Tier0CaptureSession(
        model,
        num_layers=1,
        topology=CaptureTopology(),
        device="cpu",
        micro_batch_size=masks[0].shape[0],
        local_sequence_length=masks[0].shape[1],
        calculate_per_token_loss=True,
        dgrad_normalizer=CanonicalDgradNormalizer(loss_scale),
        reduction_binding=_binding(),
    )
    session.arm()
    for microbatch_id, mask in enumerate(masks):
        session.begin_microbatch(microbatch_id)
        session.register_valid_token_mask(microbatch_id, mask)
        tensor = (
            values[microbatch_id].detach().clone()
            if values is not None
            else torch.arange(mask.numel() * 2, dtype=torch.float32).view(
                mask.shape[1], mask.shape[0], 2
            )
        ).requires_grad_(True)
        if recompute == "full":
            output = CheckpointFunction.apply(lambda data: layer(data)[0], False, tensor)
        else:
            output = layer(tensor)[0]
        staged = mask.transpose(0, 1).unsqueeze(-1)
        loss = (output * staged).sum() * loss_scale
        session.end_microbatch(microbatch_id)
        loss.backward()
    return session, session.finalize()


def test_execution_identity_context_is_nested_and_restored() -> None:
    set_diagnostic_microbatch_id(7)
    assert get_diagnostic_microbatch_id() == 7
    assert not is_diagnostic_recompute()
    with diagnostic_recompute(3):
        assert get_diagnostic_microbatch_id() == 3
        assert is_diagnostic_recompute()
    assert get_diagnostic_microbatch_id() == 7
    assert not is_diagnostic_recompute()
    set_diagnostic_microbatch_id(None)


def test_schedule_microbatch_setter_updates_diagnostic_identity() -> None:
    set_current_microbatch(nn.Module(), 5)
    assert get_diagnostic_microbatch_id() == 5
    set_diagnostic_microbatch_id(None)


@pytest.mark.parametrize("recompute", ("off", "full", "selective"))
def test_recompute_modes_capture_one_activation_and_dgrad(
    recompute: str, checkpoint_rng: None
) -> None:
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    session, result = _run_capture(masks=(mask,), recompute=recompute)
    expected_count = torch.tensor(4.0, dtype=torch.float64)

    for family in ("residual", "qkv", "attn_out", "fc1", "fc2"):
        for observation in ("activation", "dgrad"):
            slots = result.accumulator.slots(f"{observation}/{family}/layer_0")
            torch.testing.assert_close(result.accumulator.sum_pack[slots.count], expected_count)
    assert result.status == Tier0Status.OK
    session.close()


def test_checkpoint_without_output_recompute_restores_saved_identity(
    checkpoint_rng: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    executions: list[tuple[int | None, bool]] = []

    def run(values: torch.Tensor) -> torch.Tensor:
        executions.append((get_diagnostic_microbatch_id(), is_diagnostic_recompute()))
        return values.square()

    monkeypatch.setattr(random_module, "_get_share_storage", lambda: lambda _dst, _src: None)
    checkpoint = CheckpointWithoutOutput(fp8=None)
    set_diagnostic_microbatch_id(11)
    checkpoint.checkpoint(run, torch.ones(2, requires_grad=True))
    set_diagnostic_microbatch_id(19)
    checkpoint._recompute(None)

    assert executions == [(11, False), (11, True)]
    assert get_diagnostic_microbatch_id() == 19
    set_diagnostic_microbatch_id(None)


def test_canonical_dgrad_is_invariant_to_loss_scale_and_microbatch_partition() -> None:
    values = torch.arange(16, dtype=torch.float32).view(4, 2, 2)
    combined_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    _, combined = _run_capture(masks=(combined_mask,), loss_scale=1.0, values=(values,))
    split_masks = (combined_mask[:1], combined_mask[1:])
    split_values = (values[:, :1], values[:, 1:])
    _, split = _run_capture(masks=split_masks, loss_scale=8.0, values=split_values)

    assert combined.global_valid_tokens == split.global_valid_tokens == 5
    for family in ("residual", "qkv", "attn_out", "fc1", "fc2"):
        name = f"dgrad/{family}/layer_0"
        torch.testing.assert_close(
            combined.accumulator.rms(name).value, split.accumulator.rms(name).value
        )


def test_topology_ownership_and_full_pipeline_slots() -> None:
    model = _FakeModel(_FakeTransformerLayer(layer_number=3))
    rank_one = Tier0CaptureSession(
        model,
        num_layers=4,
        topology=CaptureTopology(tensor_parallel_rank=1, tensor_parallel_size=2),
        device="cpu",
        micro_batch_size=1,
        local_sequence_length=2,
        calculate_per_token_loss=True,
        dgrad_normalizer=CanonicalDgradNormalizer(),
        reduction_binding=_binding(),
    )

    assert len(rank_one.registry.descriptors) == 42
    assert rank_one.registry.owns("activation/qkv/layer_2")
    assert rank_one.registry.owns("activation/fc1/layer_2")
    assert not rank_one.registry.owns("activation/attn_out/layer_2")
    assert not rank_one.registry.owns("activation/residual/layer_2")
    assert not rank_one.registry.owns("activation/qkv/layer_0")
    assert (
        rank_one.registry.descriptors[
            rank_one.registry.slot_names.index("activation/residual/layer_0")
        ].family
        == MetricFamily.RESIDUAL
    )
    rank_one.close()

    sequence_parallel = Tier0CaptureSession(
        model,
        num_layers=4,
        topology=CaptureTopology(
            tensor_parallel_rank=1, tensor_parallel_size=2, sequence_parallel=True
        ),
        device="cpu",
        micro_batch_size=1,
        local_sequence_length=2,
        calculate_per_token_loss=True,
        dgrad_normalizer=CanonicalDgradNormalizer(),
        reduction_binding=_binding(),
    )
    assert sequence_parallel.registry.owns("activation/attn_out/layer_2")
    assert sequence_parallel.registry.owns("activation/residual/layer_2")
    sequence_parallel.close()


def test_mask_staging_sequence_parallel_slice_and_single_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    staged = stage_valid_token_mask(mask, micro_batch_size=1, sequence_length=4, device="cpu")
    sliced = slice_sequence_parallel_mask(staged, tensor_parallel_rank=1, tensor_parallel_size=2)
    torch.testing.assert_close(sliced.values[:, 0, 0], torch.tensor([3.0, 4.0]))
    assert sliced.valid

    calls = []

    def broadcast(payload, *, src, group):
        calls.append((payload.shape, src, group))

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    group = object()
    assert (
        transport_valid_token_mask(
            mask,
            armed=False,
            micro_batch_size=1,
            sequence_length=4,
            device="cpu",
            pipeline_group=group,
            pipeline_last_global_rank=7,
            is_pipeline_last_stage=True,
        )
        is None
    )
    transported = transport_valid_token_mask(
        mask,
        armed=True,
        micro_batch_size=1,
        sequence_length=4,
        device="cpu",
        pipeline_group=group,
        pipeline_last_global_rank=7,
        is_pipeline_last_stage=True,
    )
    assert calls == [(torch.Size([5]), 7, group)]
    torch.testing.assert_close(transported.values, staged.values)
    assert transported.valid


@pytest.mark.parametrize("failure", ("all_masked", "nonfinite", "mismatch"))
def test_collective_safe_capture_failures_are_invalid(failure: str) -> None:
    mask = torch.ones(1, 2)
    values = torch.ones(2, 1, 2)
    if failure == "all_masked":
        mask.zero_()
    elif failure == "nonfinite":
        values[0, 0, 0] = torch.nan
    else:
        mask = torch.ones(2, 2)
    _, result = _run_capture(masks=(mask,), values=(values,))

    assert result.status == Tier0Status.INVALID_STATISTICS
    assert not result.valid


def test_session_installs_no_duplicate_hooks_across_events() -> None:
    layer = _FakeTransformerLayer()
    model = _FakeModel(layer)
    session = Tier0CaptureSession(
        model,
        num_layers=1,
        topology=CaptureTopology(),
        device="cpu",
        micro_batch_size=1,
        local_sequence_length=2,
        calculate_per_token_loss=True,
        dgrad_normalizer=CanonicalDgradNormalizer(),
        reduction_binding=_binding(),
    )
    assert len(session._hook_handles) == 5
    target_hook_counts = [len(target.module._forward_hooks) for target in session.targets]
    assert target_hook_counts == [1, 1, 1, 1, 1]

    for _ in range(2):
        session.arm()
        session.begin_microbatch(0)
        session.register_valid_token_mask(0, torch.ones(1, 2))
        output = layer(torch.ones(2, 1, 2, requires_grad=True))[0]
        session.end_microbatch(0)
        output.sum().backward()
        result = session.finalize()
        slots = result.accumulator.slots("activation/residual/layer_0")
        assert result.accumulator.sum_pack[slots.count] == 4
    session.close()
    assert all(not target.module._forward_hooks for target in session.targets)


def test_capture_hooks_have_no_host_synchronization_calls() -> None:
    methods = (
        Tier0CaptureSession._make_forward_hook,
        Tier0CaptureSession._register_dgrad_hook,
        Tier0CaptureSession._record_hook_error,
    )
    violations = []
    for method in methods:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"cpu", "item", "numpy", "tolist"}:
                    violations.append((method.__name__, node.func.attr, node.lineno))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "float":
                    violations.append((method.__name__, node.func.id, node.lineno))
    assert violations == []


def test_legacy_mean_loss_and_dynamic_scaling_fail_closed() -> None:
    with pytest.raises(ValueError, match="per-token summed loss"):
        Tier0CaptureSession(
            _FakeModel(_FakeTransformerLayer()),
            num_layers=1,
            topology=CaptureTopology(),
            device="cpu",
            micro_batch_size=1,
            local_sequence_length=2,
            calculate_per_token_loss=False,
            dgrad_normalizer=CanonicalDgradNormalizer(),
            reduction_binding=_binding(),
        )

    from megatron.core.optimizer.grad_scaler import DynamicGradScaler

    dynamic = object.__new__(DynamicGradScaler)
    with pytest.raises(ValueError, match="dynamic loss scaling"):
        CanonicalDgradNormalizer.from_grad_scaler(dynamic)
