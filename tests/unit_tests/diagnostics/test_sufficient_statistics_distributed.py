# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real collective tests for packed Tier-0 sufficient statistics.

Run the CPU path with::

    uv run python -m torch.distributed.run --nproc-per-node 2 -m pytest -q \
        tests/unit_tests/diagnostics/test_sufficient_statistics_distributed.py -k gloo

Run both paths on a CUDA host with::

    uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
        tests/unit_tests/diagnostics/test_sufficient_statistics_distributed.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.training.diagnostics.accumulator import (
    PackedSufficientStatistics,
    ReductionBinding,
)


@pytest.fixture(scope="module")
def distributed_world() -> None:
    """Initialize the torchrun world without replacing an existing default group."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2 or "RANK" not in os.environ:
        pytest.skip("requires torch.distributed.run with at least two ranks")

    if not dist.is_initialized():
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)


@pytest.mark.distributed
def test_default_all_reduce_path_pools_real_multi_rank_gloo_statistics(
    distributed_world: None,
) -> None:
    group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    accumulator = PackedSufficientStatistics(
        ("gloo",),
        "cpu",
        descriptor_hash="distributed-gloo-v1",
        reduction_binding=ReductionBinding.flat_world(group),
    )
    if rank == 0:
        accumulator.add_masked_tensor("gloo", torch.tensor([1.0, 2.0]))
    elif rank == 1:
        accumulator.add_masked_tensor("gloo", torch.tensor([3.0, 4.0]))

    accumulator.reduce_()

    assert accumulator.sum_pack.dtype == torch.float64
    assert accumulator.max_pack.dtype == torch.float32
    assert accumulator.min_pack.dtype == torch.float32
    torch.testing.assert_close(
        accumulator.rms("gloo").value,
        torch.sqrt(torch.tensor(7.5, dtype=torch.float64)),
    )
    torch.testing.assert_close(accumulator.maximum("gloo").value, torch.tensor(4.0))
    torch.testing.assert_close(accumulator.minimum("gloo").value, torch.tensor(1.0))
    dist.barrier(group=group)
    dist.destroy_process_group(group)


@pytest.mark.distributed
@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL is required",
)
def test_default_all_reduce_path_supports_nccl_dtypes_and_zero_contributor(
    distributed_world: None,
) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    group = dist.new_group(backend="nccl")
    accumulator = PackedSufficientStatistics(
        ("nccl",),
        device,
        descriptor_hash="distributed-nccl-v1",
        reduction_binding=ReductionBinding.flat_world(group),
    )
    if dist.get_rank() == 0:
        accumulator.add_masked_tensor(
            "nccl", torch.tensor([2.0, 4.0], dtype=torch.bfloat16, device=device)
        )

    accumulator.reduce_()

    assert accumulator.sum_pack.dtype == torch.float64
    assert accumulator.sum_pack.device == device
    assert accumulator.max_pack.dtype == torch.float32
    assert accumulator.min_pack.dtype == torch.float32
    torch.testing.assert_close(
        accumulator.rms("nccl").value,
        torch.sqrt(torch.tensor(10.0, dtype=torch.float64, device=device)),
    )
    torch.testing.assert_close(
        accumulator.maximum("nccl").value,
        torch.tensor(4.0, dtype=torch.float32, device=device),
    )
    torch.testing.assert_close(
        accumulator.minimum("nccl").value,
        torch.tensor(2.0, dtype=torch.float32, device=device),
    )
    dist.barrier(group=group)
    dist.destroy_process_group(group)
