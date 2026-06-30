# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for authoritative BF16 distributed-optimizer diagnostics."""

import gc
import inspect
import os
import weakref
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

import megatron.training.diagnostics.distributed_optimizer as adapter_module
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.training.diagnostics.accumulator import (
    PackedSlots,
    PackedSufficientStatistics,
    ReductionBinding,
)
from megatron.training.diagnostics.distributed_optimizer import (
    Bf16DistributedOptimizerDiagnosticAdapter,
    DeviceMemoryState,
    DistributedOptimizerDiagnosticReason,
    DistributedOptimizerDiagnosticUnsupportedError,
    DistributedOptimizerEventStatus,
    SnapshotMemoryReason,
)
from megatron.training.diagnostics.normalization import CanonicalDgradNormalizer
from megatron.training.diagnostics.registry import (
    DenominatorKind,
    MaskKind,
    MetricDescriptor,
    MetricFamily,
    MetricRegistry,
    NormalizationKind,
    Ownership,
    PartitionAxis,
    ProcessGroupIdentity,
    ReductionKind,
    StatisticKind,
)


class _TensorParallelGroup:
    def __init__(self, rank: int) -> None:
        self._rank = rank

    def rank(self) -> int:
        return self._rank


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        bf16=True,
        fp16=False,
        fp8_recipe=None,
        use_precision_aware_optimizer=False,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        use_layer_wise_distributed_optimizer=False,
        optimizer_cpu_offload=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
    )


def _ddp_config() -> SimpleNamespace:
    return SimpleNamespace(
        use_megatron_fsdp=False,
        fp8_param_gather=False,
        fp4_param_gather=False,
        overlap_param_gather=False,
    )


def _fake_optimizer(
    *, parameter_sizes: tuple[int, ...] = (5, 6), tp_rank: int = 0
) -> tuple[DistributedOptimizer, tuple[torch.nn.Parameter, ...]]:
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.config = _config()
    optimizer.ddp_config = _ddp_config()
    optimizer.is_stub_optimizer = False
    optimizer.tp_group = _TensorParallelGroup(tp_rank)
    optimizer.model_chunks = [SimpleNamespace(pre_process=True, post_process=False)]
    optimizer.gbuf_idx_to_model_idx_map = {0: 0}

    parameters = tuple(
        torch.nn.Parameter(torch.arange(1, size + 1, dtype=torch.float32).to(torch.bfloat16))
        for size in parameter_sizes
    )
    bucket_data = torch.full((32,), -99.0, dtype=torch.bfloat16)
    dtype_key = (torch.bfloat16, torch.float32)
    param_map = {}
    model_shards = []
    main_shards = []
    bucket_ranges = ((4, 7), (13, 17))
    param_ranges = ((2, 5), (0, 4))
    for index, parameter in enumerate(parameters):
        param_start, param_end = param_ranges[index]
        bucket_start, bucket_end = bucket_ranges[index]
        model_shard = parameter.detach().view(-1)[param_start:param_end]
        bucket_data[bucket_start:bucket_end].copy_(model_shard)
        main_shard = torch.nn.Parameter(model_shard.detach().float().clone())
        model_shards.append(model_shard)
        main_shards.append(main_shard)
        param_map[parameter] = {
            "gbuf_world": Range(bucket_start, bucket_end),
            "gbuf_world_in_bucket": Range(bucket_start, bucket_end),
            "gbuf_local": Range(index * 8, index * 8 + model_shard.numel()),
            "param": Range(param_start, param_end),
        }

    optimizer.model_float16_groups = [list(parameters)]
    optimizer.model_fp32_groups = []
    optimizer.shard_float16_groups = [model_shards]
    optimizer.shard_fp32_groups = []
    optimizer.shard_fp32_from_float16_groups = [main_shards]
    optimizer.gbuf_ranges = [{dtype_key: [{"param_map": param_map}]}]
    optimizer.model_param_gbuf_map = {parameter: (0, dtype_key, 0) for parameter in parameters}
    optimizer.buffers = [SimpleNamespace(buckets=[SimpleNamespace(param_data=bucket_data)])]
    return optimizer, parameters


def _real_buffer_optimizer() -> tuple[DistributedOptimizer, tuple[torch.nn.Parameter, ...]]:
    """Build the closest CPU-practical real param-buffer owner fixture.

    The actual ``_ParamAndGradBuffer`` layout, bucket padding, parameter remapping,
    and ``DistributedOptimizer`` range builders are used. Full optimizer
    construction still requires CUDA because it identifies CUDA tensor types and
    initializes distributed optimizer state.
    """

    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=False,
        bucket_size=18,
        average_in_collective=False,
    )
    dp_group = mock.MagicMock()
    dp_group.size.return_value = 2
    dp_group.rank.return_value = 0
    pg_collection = SimpleNamespace(dp_cp=dp_group, tp=_TensorParallelGroup(0))
    parameter_groups = (
        tuple(
            torch.nn.Parameter(torch.arange(size, dtype=torch.float32).to(torch.bfloat16))
            for size in (17, 13, 9)
        ),
        tuple(
            torch.nn.Parameter(torch.arange(size, dtype=torch.float32).to(torch.bfloat16))
            for size in (15, 7)
        ),
    )
    buffers = []
    with (
        mock.patch("torch.cuda.current_device", return_value="cpu"),
        mock.patch("megatron.core.distributed.param_and_grad_buffer.log_on_each_pipeline_stage"),
    ):
        for buffer_index, parameters in enumerate(parameter_groups):
            layout = DistributedOptimizer._compute_per_buffer_param_layout(
                list(parameters),
                ddp_config.bucket_size,
                2,
                ddp_config,
                list(range(len(parameters))),
            )
            buffers.append(
                _ParamAndGradBuffer(
                    ddp_config=ddp_config,
                    param_dtype=torch.bfloat16,
                    grad_dtype=torch.float32,
                    params_with_names=[
                        (parameter, f"buffer{buffer_index}.parameter{index}")
                        for index, parameter in enumerate(parameters)
                    ],
                    data_parallel_group=dp_group,
                    bucket_size=ddp_config.bucket_size,
                    param_to_name={
                        parameter: f"buffer{buffer_index}.parameter{index}"
                        for index, parameter in enumerate(parameters)
                    },
                    gradient_scaling_factor=1.0,
                    param_indices=list(range(len(parameters))),
                    nccl_ub=False,
                    pg_collection=pg_collection,
                    param_layout=layout,
                )
            )

    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.config = _config()
    optimizer.ddp_config = _ddp_config()
    optimizer.is_stub_optimizer = False
    optimizer.tp_group = _TensorParallelGroup(0)
    optimizer.model_chunks = [
        SimpleNamespace(pre_process=True, post_process=False),
        SimpleNamespace(pre_process=False, post_process=True),
    ]
    optimizer.buffers = buffers
    optimizer.gbuf_idx_to_model_idx_map = {0: 0, 1: 1}
    optimizer.gbuf_ranges = [
        DistributedOptimizer._build_gbuf_range_map(buffer) for buffer in buffers
    ]
    optimizer.model_param_gbuf_map = DistributedOptimizer._build_model_param_gbuf_map(
        optimizer.gbuf_ranges
    )
    owned_groups = []
    model_shard_groups = []
    main_groups = []
    for parameters in parameter_groups:
        owned = [
            parameter for parameter in parameters if parameter in optimizer.model_param_gbuf_map
        ]
        owned_groups.append(owned)
        model_shards = []
        main_shards = []
        for parameter in owned:
            range_map = optimizer._get_model_param_range_map(parameter)
            model_shard = parameter.detach().view(-1)[
                range_map["param"].start : range_map["param"].end
            ]
            model_shards.append(model_shard)
            main_shards.append(torch.nn.Parameter(model_shard.float().clone()))
        model_shard_groups.append(model_shards)
        main_groups.append(main_shards)
    optimizer.model_float16_groups = owned_groups
    optimizer.model_fp32_groups = []
    optimizer.shard_float16_groups = model_shard_groups
    optimizer.shard_fp32_groups = []
    optimizer.shard_fp32_from_float16_groups = main_groups
    return optimizer, tuple(parameter for group in owned_groups for parameter in group)


def _registry(
    bindings: dict[torch.nn.Parameter, tuple[str, MetricFamily]],
    *,
    local_owners: tuple[bool, ...] | None = None,
) -> tuple[MetricRegistry, dict[torch.nn.Parameter, str]]:
    descriptors = []
    names = {}
    for index, (parameter, (logical_name, family)) in enumerate(bindings.items()):
        names[parameter] = logical_name
        descriptors.append(
            MetricDescriptor(
                logical_name=logical_name,
                family=family,
                global_layer=(
                    index if family not in (MetricFamily.EMBEDDING, MetricFamily.OUTPUT) else None
                ),
                partition_axes=(PartitionAxis.OPTIMIZER_SHARD,),
                replication_axes=(),
                replication_multiplicity=1,
                ownership=Ownership.AUTHORITATIVE_SHARD,
                mask_kind=MaskKind.NONE,
                statistic_kind=StatisticKind.UPDATE,
                denominator_kind=DenominatorKind.PRE_UPDATE_SUMSQ,
                normalization_kind=NormalizationKind.NONE,
                process_group_identity=ProcessGroupIdentity.WORLD,
                reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
                tied_owner_identity=(
                    "embedding.first_pipeline_stage" if family == MetricFamily.EMBEDDING else None
                ),
                packed_slots=PackedSlots.for_index(index),
            )
        )
    return (
        MetricRegistry(
            descriptors,
            reduction_binding=ReductionBinding.flat_world(None),
            local_owners=local_owners,
        ),
        names,
    )


def _adapter(
    optimizer: DistributedOptimizer | ChainedOptimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    families: tuple[MetricFamily, ...] | None = None,
    max_extra_bytes: int = 1_000_000,
    memory_state_provider=None,
    finish_chunk_elements: int = 65_536,
) -> tuple[Bf16DistributedOptimizerDiagnosticAdapter, MetricRegistry]:
    selected_families = families or tuple(MetricFamily.FC1 for _ in parameters)
    bindings = {
        parameter: (f"update/{family.value}/{index}", family)
        for index, (parameter, family) in enumerate(zip(parameters, selected_families))
    }
    registry, names = _registry(bindings)
    return (
        Bf16DistributedOptimizerDiagnosticAdapter(
            optimizer,
            registry,
            names,
            diagnostic_max_extra_bytes=max_extra_bytes,
            finish_chunk_elements=finish_chunk_elements,
            memory_state_provider=memory_state_provider,
        ),
        registry,
    )


def _cross_lane_registry(
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    local_owners: tuple[bool, ...] | None = None,
    reducer=None,
) -> tuple[MetricRegistry, dict[torch.nn.Parameter, str]]:
    descriptors = [
        MetricDescriptor(
            logical_name="event/runtime_status",
            family=MetricFamily.EVENT,
            global_layer=None,
            partition_axes=(),
            replication_axes=(),
            replication_multiplicity=1,
            ownership=Ownership.EVERY_RANK,
            mask_kind=MaskKind.NONE,
            statistic_kind=StatisticKind.TENSOR_MOMENTS,
            denominator_kind=DenominatorKind.SELECTED_ELEMENTS,
            normalization_kind=NormalizationKind.NONE,
            process_group_identity=ProcessGroupIdentity.WORLD,
            reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
            tied_owner_identity=None,
            packed_slots=PackedSlots.for_index(0),
        ),
        MetricDescriptor(
            logical_name="dgrad/residual/layer_0",
            family=MetricFamily.RESIDUAL,
            global_layer=0,
            partition_axes=(PartitionAxis.DATA_SAMPLE, PartitionAxis.PIPELINE_LAYER),
            replication_axes=(),
            replication_multiplicity=1,
            ownership=Ownership.PIPELINE_STAGE,
            mask_kind=MaskKind.TOKEN,
            statistic_kind=StatisticKind.TENSOR_MOMENTS,
            denominator_kind=DenominatorKind.SELECTED_ELEMENTS,
            normalization_kind=NormalizationKind.LOSS_SCALE_AND_GLOBAL_VALID_TOKENS,
            process_group_identity=ProcessGroupIdentity.WORLD,
            reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
            tied_owner_identity=None,
            packed_slots=PackedSlots.for_index(1),
        ),
    ]
    names = {}
    for parameter_index, parameter in enumerate(parameters):
        descriptor_index = len(descriptors)
        logical_name = f"update/fc1/{parameter_index}"
        names[parameter] = logical_name
        descriptors.append(
            MetricDescriptor(
                logical_name=logical_name,
                family=MetricFamily.FC1,
                global_layer=parameter_index,
                partition_axes=(
                    PartitionAxis.OPTIMIZER_SHARD,
                    PartitionAxis.TENSOR_FEATURE,
                    PartitionAxis.PIPELINE_LAYER,
                ),
                replication_axes=(),
                replication_multiplicity=1,
                ownership=Ownership.AUTHORITATIVE_SHARD,
                mask_kind=MaskKind.NONE,
                statistic_kind=StatisticKind.UPDATE,
                denominator_kind=DenominatorKind.PRE_UPDATE_SUMSQ,
                normalization_kind=NormalizationKind.NONE,
                process_group_identity=ProcessGroupIdentity.WORLD,
                reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
                tied_owner_identity=None,
                packed_slots=PackedSlots.for_index(descriptor_index),
            )
        )
    return (
        MetricRegistry(
            descriptors,
            reduction_binding=ReductionBinding.flat_world(None, reducer=reducer),
            local_owners=local_owners,
            normalization_adapters=(CanonicalDgradNormalizer(loss_scale=2.0),),
        ),
        names,
    )


def _applied_update_moments() -> dict[str, torch.Tensor]:
    return {
        "count": torch.tensor(2.0, dtype=torch.float64),
        "master_sumsq": torch.tensor(4.0, dtype=torch.float64),
        "applied_sumsq": torch.tensor(1.0, dtype=torch.float64),
        "applied_pre_sumsq": torch.tensor(16.0, dtype=torch.float64),
        "master_nonzero": torch.tensor(2.0, dtype=torch.float64),
        "applied_nonzero": torch.tensor(1.0, dtype=torch.float64),
        "cast_zero": torch.tensor(1.0, dtype=torch.float64),
        "nonfinite": torch.tensor(0.0, dtype=torch.float64),
        "maximum": torch.tensor(0.5, dtype=torch.float32),
        "minimum": torch.tensor(0.0, dtype=torch.float32),
        "arithmetic_error": torch.tensor(0.0, dtype=torch.float64),
        "materialization_error": torch.tensor(0.0, dtype=torch.float64),
        "materialization_valid": torch.tensor(True),
    }


def test_cross_lane_normalization_changes_only_the_capture_slot() -> None:
    _, parameters = _fake_optimizer()
    registry, _ = _cross_lane_registry(parameters)
    accumulator = registry.new_accumulator("cpu")
    registry.add_masked_tensor(
        accumulator, "dgrad/residual/layer_0", torch.tensor([4.0, 8.0]), mask=torch.ones(2)
    )
    registry.add_applied_update_moments(accumulator, "update/fc1/0", **_applied_update_moments())
    accumulator.finalize_local_()

    update_slots = accumulator.slots("update/fc1/0")
    update_sum_before = accumulator.sum_pack[
        update_slots.sum : update_slots.observation_error + 1
    ].clone()
    update_max_before = accumulator.max_pack[update_slots.maximum].clone()
    update_min_before = accumulator.min_pack[update_slots.minimum].clone()
    registry.apply_normalizations_(accumulator, global_valid_tokens=torch.tensor(2.0))

    torch.testing.assert_close(
        accumulator.rms("dgrad/residual/layer_0").value,
        torch.sqrt(torch.tensor(40.0, dtype=torch.float64)) / 4.0,
    )
    torch.testing.assert_close(
        accumulator.norm_retention("update/fc1/0").value, torch.tensor(0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(
        accumulator.sum_pack[update_slots.sum : update_slots.observation_error + 1],
        update_sum_before,
    )
    torch.testing.assert_close(accumulator.max_pack[update_slots.maximum], update_max_before)
    torch.testing.assert_close(accumulator.min_pack[update_slots.minimum], update_min_before)


def test_cross_lane_capture_missing_and_optimizer_invalid_status_share_one_reduction() -> None:
    collective_calls = []

    def identity_reducer(tensor, *, op, group):
        collective_calls.append((tensor, op, group))

    optimizer, parameters = _fake_optimizer()
    registry, names = _cross_lane_registry(parameters, reducer=identity_reducer)
    adapter = Bf16DistributedOptimizerDiagnosticAdapter(
        optimizer, registry, names, diagnostic_max_extra_bytes=1_000_000, finish_chunk_elements=2
    )
    accumulator = registry.new_accumulator("cpu")
    registry.mark_observation_error(accumulator, "dgrad/residual/layer_0")
    assert adapter.begin_event() is not None
    with torch.no_grad():
        for shard in adapter.iter_owner_shards():
            shard.main_shard.add_(0.25)
    assert adapter.finish_event(accumulator, update_successful=True) is not None
    registry.add_masked_tensor(
        accumulator, "event/runtime_status", adapter.status_for_event_consensus()
    )
    accumulator.reduce_()

    capture_slots = accumulator.slots("dgrad/residual/layer_0")
    event_slots = accumulator.slots("event/runtime_status")
    assert len(collective_calls) == 3
    assert accumulator.sum_pack[capture_slots.observation_error] == 1
    assert accumulator.max_pack[event_slots.maximum] == int(
        DistributedOptimizerEventStatus.MATERIALIZATION_MISMATCH
    )
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.MATERIALIZATION_MISMATCH
    for parameter in parameters:
        update_slots = accumulator.slots(names[parameter])
        assert accumulator.sum_pack[update_slots.mask_error] > 0
        assert accumulator.sum_pack[update_slots.observation_error] == 0


def test_cross_lane_nonowners_keep_every_slot_neutral_and_the_same_layout() -> None:
    _, parameters = _fake_optimizer()
    owner_registry, _ = _cross_lane_registry(parameters)
    nonowner_registry, _ = _cross_lane_registry(
        parameters, local_owners=tuple(False for _ in owner_registry.descriptors)
    )
    accumulator = nonowner_registry.new_accumulator("cpu")
    nonowner_registry.add_masked_tensor(
        accumulator, "dgrad/residual/layer_0", torch.tensor([4.0]), mask=torch.ones(1)
    )
    nonowner_registry.add_applied_update_moments(
        accumulator, "update/fc1/0", **_applied_update_moments()
    )
    nonowner_registry.add_masked_tensor(
        accumulator,
        "event/runtime_status",
        torch.tensor([DistributedOptimizerEventStatus.MATERIALIZATION_MISMATCH]),
    )
    accumulator.finalize_local_()
    nonowner_registry.apply_normalizations_(accumulator, global_valid_tokens=torch.tensor(2.0))

    assert nonowner_registry.slot_names == owner_registry.slot_names
    assert nonowner_registry.descriptor_hash == owner_registry.descriptor_hash
    assert torch.count_nonzero(accumulator.sum_pack) == 0
    assert torch.all(accumulator.max_pack == -torch.inf)
    assert torch.all(accumulator.min_pack == torch.inf)


def test_cross_lane_exact_hbm_adds_capture_scratch_and_optimizer_event_only() -> None:
    optimizer, parameters = _fake_optimizer()
    registry, names = _cross_lane_registry(parameters)
    adapter = Bf16DistributedOptimizerDiagnosticAdapter(
        optimizer, registry, names, diagnostic_max_extra_bytes=2_000_000
    )
    accumulator = registry.new_accumulator("cpu")
    estimate = adapter.estimate_snapshot_memory()
    persistent_pack_baseline = (
        accumulator.sum_pack.nbytes + accumulator.max_pack.nbytes + accumulator.min_pack.nbytes
    )
    combined_incremental_hbm = accumulator.maximum_scratch_bytes + estimate.total_bytes

    assert accumulator.maximum_scratch_bytes == 1_572_864
    assert estimate.snapshot_bytes == 42
    assert estimate.finish_scratch_bytes == 304
    assert estimate.total_bytes == estimate.snapshot_bytes + estimate.finish_scratch_bytes == 346
    assert combined_incremental_hbm == 1_573_210
    assert persistent_pack_baseline == 384
    assert combined_incremental_hbm + persistent_pack_baseline == 1_573_594


def test_iterator_exposes_existing_cross_parameter_and_padding_ranges() -> None:
    optimizer, parameters = _fake_optimizer()
    shards = tuple(optimizer.iter_model_main_param_shards())

    assert [shard.model_param for shard in shards] == list(parameters)
    assert [(shard.param_range.start, shard.param_range.end) for shard in shards] == [
        (2, 5),
        (0, 4),
    ]
    assert [(shard.bucket_range.start, shard.bucket_range.end) for shard in shards] == [
        (4, 7),
        (13, 17),
    ]
    assert [shard.model_chunk_index for shard in shards] == [0, 0]
    assert [shard.bucket_index for shard in shards] == [0, 0]
    torch.testing.assert_close(shards[0].model_shard, parameters[0][2:5])
    torch.testing.assert_close(shards[1].model_shard, parameters[1][0:4])
    assert torch.all(optimizer.buffers[0].buckets[0].param_data[7:13] == -99)


def test_iterator_applies_tp_duplicate_and_tied_embedding_owner_rules() -> None:
    optimizer, parameters = _fake_optimizer(tp_rank=1)
    parameters[0].tensor_model_parallel = True
    parameters[1].shared_embedding = True
    parameters[1].shared = True

    sharded, tied_copy = tuple(optimizer.iter_model_main_param_shards())

    assert sharded.tensor_parallel_sharded
    assert not sharded.tensor_parallel_duplicate
    assert sharded.logical_owner
    assert tied_copy.tied
    assert tied_copy.shared
    assert not tied_copy.tied_owner
    assert tied_copy.tensor_parallel_duplicate
    assert not tied_copy.logical_owner


def test_iterator_distinguishes_tp_replicated_owner_from_nonowner() -> None:
    owner_optimizer, _ = _fake_optimizer(tp_rank=0)
    nonowner_optimizer, _ = _fake_optimizer(tp_rank=1)

    owner_shards = tuple(owner_optimizer.iter_model_main_param_shards())
    nonowner_shards = tuple(nonowner_optimizer.iter_model_main_param_shards())

    assert all(not shard.tensor_parallel_sharded for shard in owner_shards)
    assert all(shard.logical_owner for shard in owner_shards)
    assert all(shard.tensor_parallel_duplicate for shard in nonowner_shards)
    assert all(not shard.logical_owner for shard in nonowner_shards)


def test_real_param_buffers_cover_multiple_buffers_buckets_boundaries_and_padding() -> None:
    optimizer, parameters = _real_buffer_optimizer()
    shards = tuple(optimizer.iter_model_main_param_shards())

    assert len(optimizer.buffers) == 2
    assert all(isinstance(buffer, _ParamAndGradBuffer) for buffer in optimizer.buffers)
    assert sum(len(buffer.buckets) for buffer in optimizer.buffers) >= 3
    assert {shard.buffer_index for shard in shards} == {0, 1}
    assert len({(shard.buffer_index, shard.bucket_index) for shard in shards}) >= 3
    assert any(shard.local_buffer_range.start == 0 for shard in shards)
    assert any(shard.bucket_range.start == 0 for shard in shards)
    assert any(buffer.numel > buffer.numel_unpadded for buffer in optimizer.buffers)
    assert {shard.model_param for shard in shards} == set(parameters)


def test_one_child_chain_is_accepted_and_multiple_children_fail_closed() -> None:
    optimizer, parameters = _fake_optimizer()
    chain = ChainedOptimizer([optimizer])
    adapter, _ = _adapter(chain, parameters)
    assert adapter.optimizer is optimizer

    multi = ChainedOptimizer([optimizer, optimizer])
    report = Bf16DistributedOptimizerDiagnosticAdapter.negotiate_capabilities(multi)
    assert not report.supported
    assert report.reasons == (DistributedOptimizerDiagnosticReason.CHAIN_ARITY,)


@pytest.mark.parametrize(
    ("mutator", "reason"),
    (
        (
            lambda optimizer, parameters: setattr(optimizer, "is_stub_optimizer", True),
            DistributedOptimizerDiagnosticReason.STUB_OPTIMIZER,
        ),
        (
            lambda optimizer, parameters: setattr(optimizer.config, "bf16", False),
            DistributedOptimizerDiagnosticReason.NOT_BF16,
        ),
        (
            lambda optimizer, parameters: setattr(
                optimizer,
                "model_fp32_groups",
                [[torch.nn.Parameter(torch.ones(1, dtype=torch.float32))]],
            ),
            DistributedOptimizerDiagnosticReason.NOT_BF16,
        ),
        (
            lambda optimizer, parameters: setattr(optimizer.ddp_config, "use_megatron_fsdp", True),
            DistributedOptimizerDiagnosticReason.FSDP,
        ),
        (
            lambda optimizer, parameters: setattr(
                optimizer.config, "use_precision_aware_optimizer", True
            ),
            DistributedOptimizerDiagnosticReason.PRECISION_AWARE,
        ),
        (
            lambda optimizer, parameters: setattr(optimizer.config, "fp8_recipe", "delayed"),
            DistributedOptimizerDiagnosticReason.FP8,
        ),
        (
            lambda optimizer, parameters: setattr(optimizer.ddp_config, "fp4_param_gather", True),
            DistributedOptimizerDiagnosticReason.FP4,
        ),
        (
            lambda optimizer, parameters: setattr(parameters[0], "allreduce", False),
            DistributedOptimizerDiagnosticReason.MOE,
        ),
        (
            lambda optimizer, parameters: setattr(
                optimizer.config, "use_layer_wise_distributed_optimizer", True
            ),
            DistributedOptimizerDiagnosticReason.LAYERWISE,
        ),
        (
            lambda optimizer, parameters: setattr(optimizer.config, "optimizer_cpu_offload", True),
            DistributedOptimizerDiagnosticReason.CPU_OFFLOAD,
        ),
        (
            lambda optimizer, parameters: setattr(
                optimizer.ddp_config, "overlap_param_gather", True
            ),
            DistributedOptimizerDiagnosticReason.OVERLAP_PARAM_GATHER,
        ),
        (
            lambda optimizer, parameters: optimizer.shard_fp32_from_float16_groups[0].__setitem__(
                0, parameters[0].detach().to(torch.bfloat16)
            ),
            DistributedOptimizerDiagnosticReason.NON_FP32_MAIN,
        ),
    ),
)
def test_every_unsupported_first_backend_capability_has_a_stable_reason(
    mutator, reason: DistributedOptimizerDiagnosticReason
) -> None:
    optimizer, parameters = _fake_optimizer()
    mutator(optimizer, parameters)

    report = Bf16DistributedOptimizerDiagnosticAdapter.negotiate_capabilities(optimizer)

    assert not report.supported
    assert reason in report.reasons
    registry, names = _registry({parameters[0]: ("update/fc1/0", MetricFamily.FC1)})
    with pytest.raises(DistributedOptimizerDiagnosticUnsupportedError, match=reason.name):
        Bf16DistributedOptimizerDiagnosticAdapter(
            optimizer, registry, names, diagnostic_max_extra_bytes=1_000_000
        )


def test_wrong_optimizer_type_fails_closed() -> None:
    report = Bf16DistributedOptimizerDiagnosticAdapter.negotiate_capabilities(object())
    assert report.reasons == (DistributedOptimizerDiagnosticReason.OPTIMIZER_TYPE,)
    assert report.startup_host_sync


def test_only_startup_negotiation_contains_an_accepted_host_synchronization() -> None:
    event_sources = (
        inspect.getsource(Bf16DistributedOptimizerDiagnosticAdapter.begin_event),
        inspect.getsource(Bf16DistributedOptimizerDiagnosticAdapter.finish_event),
        inspect.getsource(Bf16DistributedOptimizerDiagnosticAdapter._begin_event),
        inspect.getsource(Bf16DistributedOptimizerDiagnosticAdapter._finish_event),
        inspect.getsource(Bf16DistributedOptimizerDiagnosticAdapter._verify_materialization),
        inspect.getsource(Bf16DistributedOptimizerDiagnosticAdapter._accumulate_update_moments),
        inspect.getsource(Bf16DistributedOptimizerDiagnosticAdapter._chunk_moments),
        inspect.getsource(PackedSufficientStatistics.add_applied_update_moments),
    )
    prohibited = (".item(", ".cpu(", ".tolist(", ".numpy(")

    assert not any(operation in source for source in event_sources for operation in prohibited)


def test_overlapping_local_owner_shards_become_typed_constructor_status() -> None:
    optimizer, parameters = _fake_optimizer()
    backing = torch.nn.Parameter(torch.arange(7, dtype=torch.float32))
    optimizer.shard_fp32_from_float16_groups = [[backing[:3], backing[2:6]]]
    registry, names = _registry(
        {
            parameters[0]: ("update/fc1/0", MetricFamily.FC1),
            parameters[1]: ("update/fc1/1", MetricFamily.FC1),
        }
    )

    adapter = Bf16DistributedOptimizerDiagnosticAdapter(
        optimizer, registry, names, diagnostic_max_extra_bytes=1_000_000
    )

    assert (
        adapter.local_status.item() == DistributedOptimizerEventStatus.CONSTRUCTOR_OWNERSHIP_FAILED
    )
    assert adapter.begin_event() is None
    assert not adapter.armed


def test_constructor_iterator_and_binding_failures_are_typed_local_status() -> None:
    iterator_optimizer, iterator_parameters = _fake_optimizer()

    def fail_iterator():
        raise RuntimeError("injected iterator failure")

    iterator_optimizer.iter_model_main_param_shards = fail_iterator
    registry, names = _registry(
        {
            iterator_parameters[0]: ("update/fc1/0", MetricFamily.FC1),
            iterator_parameters[1]: ("update/fc1/1", MetricFamily.FC1),
        }
    )
    iterator_adapter = Bf16DistributedOptimizerDiagnosticAdapter(
        iterator_optimizer, registry, names, diagnostic_max_extra_bytes=1_000_000
    )
    assert (
        iterator_adapter.local_status.item()
        == DistributedOptimizerEventStatus.CONSTRUCTOR_ITERATOR_FAILED
    )

    binding_optimizer, binding_parameters = _fake_optimizer()
    binding_registry, binding_names = _registry(
        {binding_parameters[0]: ("update/fc1/0", MetricFamily.FC1)}
    )
    binding_adapter = Bf16DistributedOptimizerDiagnosticAdapter(
        binding_optimizer, binding_registry, binding_names, diagnostic_max_extra_bytes=1_000_000
    )
    assert (
        binding_adapter.local_status.item()
        == DistributedOptimizerEventStatus.CONSTRUCTOR_BINDING_FAILED
    )


def test_begin_and_finish_identity_failures_release_without_raising() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters)
    original_main = optimizer.shard_fp32_from_float16_groups[0][0]
    optimizer.shard_fp32_from_float16_groups[0][0] = torch.nn.Parameter(
        original_main.detach().clone()
    )
    assert adapter.begin_event() is None
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.BEGIN_IDENTITY_CHANGED
    assert not adapter.armed

    finish_optimizer, finish_parameters = _fake_optimizer()
    finish_adapter, finish_registry = _adapter(finish_optimizer, finish_parameters)
    assert finish_adapter.begin_event() is not None
    finish_optimizer.shard_fp32_from_float16_groups[0][0] = torch.nn.Parameter(
        finish_optimizer.shard_fp32_from_float16_groups[0][0].detach().clone()
    )
    assert (
        finish_adapter.finish_event(finish_registry.new_accumulator("cpu"), update_successful=True)
        is None
    )
    assert (
        finish_adapter.local_status.item()
        == DistributedOptimizerEventStatus.FINISH_SHARD_IDENTITY_CHANGED
    )
    assert not finish_adapter.armed


def test_accumulation_failure_is_typed_and_releases_snapshot() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters)
    assert adapter.begin_event() is not None
    optimizer._copy_main_params_to_model_params()
    unrelated_parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    unrelated_registry, _ = _registry(
        {unrelated_parameter: ("update/fc2/unrelated", MetricFamily.FC2)}
    )

    assert (
        adapter.finish_event(unrelated_registry.new_accumulator("cpu"), update_successful=True)
        is None
    )
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.ACCUMULATION_FAILED
    assert not adapter.armed


def test_adamw_decoupled_weight_decay_accumulates_actual_master_and_applied_updates() -> None:
    optimizer, parameters = _fake_optimizer(parameter_sizes=(5, 6))
    adapter, registry = _adapter(optimizer, parameters)
    accumulator = registry.new_accumulator("cpu")
    shards = adapter.iter_owner_shards()
    master_before = torch.cat([shard.main_shard.detach().clone() for shard in shards])
    applied_before = torch.cat([shard.model_shard.detach().float().clone() for shard in shards])
    adapter.begin_event()

    adamw = torch.optim.AdamW(
        [shard.main_shard for shard in shards], lr=0.1, betas=(0.0, 0.0), eps=1.0, weight_decay=0.2
    )
    for shard in shards:
        shard.main_shard.grad = torch.full_like(shard.main_shard, 0.5)
    adamw.step()
    with torch.no_grad():
        for shard in shards:
            shard.model_shard.copy_(shard.main_shard)

    adapter.finish_event(accumulator, update_successful=True)
    accumulator.finalize_local_()
    master_after = torch.cat([shard.main_shard.detach() for shard in shards])
    applied_after = torch.cat([shard.model_shard.detach().float() for shard in shards])
    master_delta = master_after - master_before
    applied_delta = applied_after - applied_before

    for index in range(len(parameters)):
        logical_name = f"update/fc1/{index}"
        shard = shards[index]
        torch.testing.assert_close(
            accumulator.master_delta_rms(logical_name).value,
            shard.main_shard.detach()
            .sub(
                master_before[
                    sum(s.main_shard.numel() for s in shards[:index]) : sum(
                        s.main_shard.numel() for s in shards[: index + 1]
                    )
                ]
            )
            .square()
            .mean()
            .sqrt()
            .double(),
        )
    expected_retention = torch.sqrt(
        applied_delta.square().sum() / master_delta.square().sum()
    ).double()
    pooled_registry, pooled_names = _registry({parameters[0]: ("update/fc1/all", MetricFamily.FC1)})
    pooled = pooled_registry.new_accumulator("cpu")
    pooled_registry.add_applied_update(
        pooled,
        pooled_names[parameters[0]],
        master_before,
        master_after,
        applied_before,
        applied_after,
    )
    pooled.finalize_local_()
    torch.testing.assert_close(pooled.norm_retention("update/fc1/all").value, expected_retention)


def test_bf16_cast_to_zero_records_nonzero_master_and_zero_applied_delta() -> None:
    optimizer, parameters = _fake_optimizer(parameter_sizes=(5, 6))
    adapter, registry = _adapter(optimizer, parameters)
    accumulator = registry.new_accumulator("cpu")
    adapter.begin_event()
    first = adapter.iter_owner_shards()[0]
    first.main_shard.data.add_(1.0e-4)
    first.model_shard.copy_(first.main_shard)

    adapter.finish_event(accumulator, update_successful=True)
    accumulator.finalize_local_()

    assert accumulator.master_nonzero_fraction("update/fc1/0").value == 1
    assert accumulator.applied_nonzero_fraction("update/fc1/0").value == 0
    assert accumulator.cast_zero_fraction("update/fc1/0").value == 1
    assert accumulator.norm_retention("update/fc1/0").value == 0


def test_tied_copy_and_tp_nonowner_emit_neutral_statistics_while_untied_weights_contribute() -> (
    None
):
    optimizer, parameters = _fake_optimizer(tp_rank=1)
    parameters[0].tensor_model_parallel = True
    parameters[1].shared_embedding = True
    parameters[1].shared = True
    adapter, registry = _adapter(
        optimizer, parameters, families=(MetricFamily.EMBEDDING, MetricFamily.OUTPUT)
    )
    accumulator = registry.new_accumulator("cpu")
    adapter.begin_event()
    with torch.no_grad():
        for shard in adapter.iter_owner_shards():
            shard.main_shard.add_(0.25)
            shard.model_shard.copy_(shard.main_shard)
    adapter.finish_event(accumulator, update_successful=True)
    accumulator.finalize_local_()

    assert accumulator.relative_rms("update/embedding/0").valid
    assert not accumulator.relative_rms("update/output/1").valid

    untied_optimizer, untied_parameters = _fake_optimizer()
    untied_adapter, untied_registry = _adapter(
        untied_optimizer, untied_parameters, families=(MetricFamily.EMBEDDING, MetricFamily.OUTPUT)
    )
    untied_accumulator = untied_registry.new_accumulator("cpu")
    untied_adapter.begin_event()
    with torch.no_grad():
        for shard in untied_adapter.iter_owner_shards():
            shard.main_shard.add_(0.25)
            shard.model_shard.copy_(shard.main_shard)
    untied_adapter.finish_event(untied_accumulator, update_successful=True)
    untied_accumulator.finalize_local_()
    assert untied_accumulator.relative_rms("update/embedding/0").valid
    assert untied_accumulator.relative_rms("update/output/1").valid


def test_skipped_update_releases_snapshots_and_accumulates_nothing() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters)
    accumulator = registry.new_accumulator("cpu")
    adapter.begin_event()

    assert adapter.finish_event(accumulator, update_successful=False) is None
    assert not adapter.armed
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.UPDATE_SKIPPED
    assert torch.count_nonzero(accumulator.sum_pack) == 0


def test_finish_requires_explicit_success_and_early_finish_is_typed() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters)
    accumulator = registry.new_accumulator("cpu")

    with pytest.raises(TypeError, match="update_successful"):
        adapter.finish_event(accumulator)
    assert adapter.finish_event(accumulator, update_successful=True) is None
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.FINISH_NOT_ARMED


@pytest.mark.parametrize("corruption", ("stale", "corrupt"))
def test_stale_or_corrupt_materialization_invalidates_all_update_metrics(corruption: str) -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters, finish_chunk_elements=2)
    accumulator = registry.new_accumulator("cpu")
    assert adapter.begin_event() is not None
    with torch.no_grad():
        for shard in adapter.iter_owner_shards():
            shard.main_shard.add_(0.25)
        if corruption == "corrupt":
            optimizer._copy_main_params_to_model_params()
            adapter.iter_owner_shards()[-1].model_shard[-1].add_(1)

    measurement = adapter.finish_event(accumulator, update_successful=True)
    accumulator.finalize_local_()

    assert measurement is not None
    assert not adapter.armed
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.MATERIALIZATION_MISMATCH
    for logical_name in accumulator.slot_names:
        assert not accumulator.norm_retention(logical_name).valid
        assert torch.isnan(accumulator.norm_retention(logical_name).value)


def test_normal_main_to_param_copy_materializes_and_accumulates_in_bounded_chunks() -> None:
    optimizer, parameters = _real_buffer_optimizer()
    adapter, registry = _adapter(
        optimizer, parameters, finish_chunk_elements=3, max_extra_bytes=1_000_000
    )
    accumulator = registry.new_accumulator("cpu")
    assert adapter.begin_event() is not None
    with torch.no_grad():
        for shard in adapter.iter_owner_shards():
            shard.main_shard.add_(0.5)
        optimizer._copy_main_params_to_model_params()

    measurement = adapter.finish_event(accumulator, update_successful=True)
    accumulator.finalize_local_()

    assert measurement is not None
    assert measurement.payload_bytes == measurement.estimated_bytes
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.OK
    for logical_name in accumulator.slot_names:
        assert accumulator.relative_rms(logical_name).valid
        assert accumulator.norm_retention(logical_name).value == 1


def test_exact_memory_estimation_measurement_and_preflight_rejections() -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters)
    estimate = adapter.estimate_snapshot_memory()
    assert estimate.owner_elements == 7
    assert estimate.fp32_master_bytes == 28
    assert estimate.bf16_applied_bytes == 14
    assert estimate.snapshot_bytes == 42
    assert estimate.finish_elementwise_bytes == 198
    assert estimate.finish_scalar_bytes == 106
    assert estimate.finish_scratch_bytes == 304
    assert estimate.total_bytes == 346
    measurement = adapter.begin_event()
    assert measurement is not None
    assert measurement.estimated_bytes == 346
    assert measurement.payload_bytes == 42
    adapter.abort_event()

    rereview_cap, _ = _adapter(optimizer, parameters, max_extra_bytes=240)
    assert not rereview_cap.preflight_snapshot_memory().accepted
    assert rereview_cap.preflight_snapshot_memory().reason == SnapshotMemoryReason.MAX_EXTRA_BYTES
    exact, _ = _adapter(optimizer, parameters, max_extra_bytes=346)
    assert exact.preflight_snapshot_memory().accepted
    capped, _ = _adapter(optimizer, parameters, max_extra_bytes=345)
    assert capped.preflight_snapshot_memory().reason == SnapshotMemoryReason.MAX_EXTRA_BYTES
    assert capped.begin_event() is None
    assert capped.last_memory_reason == SnapshotMemoryReason.MAX_EXTRA_BYTES
    assert capped.local_status.item() == DistributedOptimizerEventStatus.BEGIN_PREFLIGHT_REJECTED
    assert not capped.armed

    fraction, _ = _adapter(
        optimizer,
        parameters,
        memory_state_provider=lambda device: DeviceMemoryState(700, 700, 500, 1000),
    )
    assert (
        fraction.preflight_snapshot_memory().reason == SnapshotMemoryReason.DEVICE_MEMORY_FRACTION
    )

    headroom, _ = _adapter(
        optimizer,
        parameters,
        memory_state_provider=lambda device: DeviceMemoryState(100, 100, 345, 1000),
    )
    assert headroom.preflight_snapshot_memory().reason == SnapshotMemoryReason.DEVICE_HEADROOM

    def fail_memory_query(device):
        raise RuntimeError("injected memory query failure")

    query_failure, _ = _adapter(optimizer, parameters, memory_state_provider=fail_memory_query)
    assert (
        query_failure.preflight_snapshot_memory().reason == SnapshotMemoryReason.ALLOCATION_FAILED
    )
    assert query_failure.begin_event() is None
    assert (
        query_failure.local_status.item()
        == DistributedOptimizerEventStatus.BEGIN_PREFLIGHT_REJECTED
    )


@pytest.mark.parametrize(
    ("failure_dtype", "expected_status"),
    (
        (torch.float32, DistributedOptimizerEventStatus.BEGIN_FIRST_ALLOCATION_FAILED),
        (torch.bfloat16, DistributedOptimizerEventStatus.BEGIN_SECOND_ALLOCATION_FAILED),
    ),
)
def test_partial_snapshot_allocation_failures_release_every_tensor(
    monkeypatch: pytest.MonkeyPatch,
    failure_dtype: torch.dtype,
    expected_status: DistributedOptimizerEventStatus,
) -> None:
    optimizer, parameters = _fake_optimizer()
    original_empty = adapter_module.torch.empty

    def raise_oom(*args, **kwargs):
        if kwargs.get("dtype") == failure_dtype:
            raise torch.OutOfMemoryError("injected")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(adapter_module.torch, "empty", raise_oom)
    adapter, _ = _adapter(optimizer, parameters)

    assert adapter.begin_event() is None
    assert adapter.local_status.item() == expected_status
    assert adapter.last_memory_reason == SnapshotMemoryReason.ALLOCATION_FAILED
    assert not adapter.armed


def test_finish_scratch_allocation_failure_releases_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters)
    accumulator = registry.new_accumulator("cpu")
    assert adapter.begin_event() is not None
    original_empty = adapter_module.torch.empty

    def raise_oom(*args, **kwargs):
        if kwargs.get("dtype") == torch.uint8:
            raise torch.OutOfMemoryError("injected")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(adapter_module.torch, "empty", raise_oom)

    assert adapter.finish_event(accumulator, update_successful=True) is None
    assert (
        adapter.local_status.item()
        == DistributedOptimizerEventStatus.FINISH_SCRATCH_ALLOCATION_FAILED
    )
    assert adapter.last_memory_reason == SnapshotMemoryReason.ALLOCATION_FAILED
    assert not adapter.armed


def test_allocator_peak_is_tracked_through_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters)
    allocated = iter((100, 142, 340, 330))
    monkeypatch.setattr(adapter, "_allocator_bytes", lambda device: next(allocated))
    accumulator = registry.new_accumulator("cpu")
    begin_measurement = adapter.begin_event()
    assert begin_measurement is not None
    assert begin_measurement.allocator_peak_delta_bytes == 42
    optimizer._copy_main_params_to_model_params()

    finish_measurement = adapter.finish_event(accumulator, update_successful=True)

    assert finish_measurement is not None
    assert finish_measurement.allocator_peak_delta_bytes == 240
    assert finish_measurement.payload_bytes == finish_measurement.estimated_bytes == 346


def test_finish_allocator_query_failure_is_typed_and_releases_all_event_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters)
    accumulator = registry.new_accumulator("cpu")
    allocator_calls = 0

    def fail_after_finish_scratch(device):
        nonlocal allocator_calls
        allocator_calls += 1
        if allocator_calls == 1:
            return 100
        if allocator_calls == 2:
            return 142
        raise RuntimeError("injected finish allocator query failure")

    scratch_storage_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original_allocate = adapter._allocate_finish_scratch

    def capture_scratch(device):
        scratch = original_allocate(device)
        scratch_storage_refs.append(weakref.ref(scratch.storage))
        return scratch

    monkeypatch.setattr(adapter, "_allocator_bytes", fail_after_finish_scratch)
    monkeypatch.setattr(adapter, "_allocate_finish_scratch", capture_scratch)
    assert adapter.begin_event() is not None
    assert adapter._snapshot is not None
    snapshot_refs = (
        weakref.ref(adapter._snapshot.master_before),
        weakref.ref(adapter._snapshot.applied_before),
    )
    optimizer._copy_main_params_to_model_params()

    assert adapter.finish_event(accumulator, update_successful=True) is None
    gc.collect()

    assert allocator_calls == 3
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.ALLOCATOR_QUERY_FAILED
    assert not adapter.armed
    assert scratch_storage_refs and scratch_storage_refs[0]() is None
    assert all(reference() is None for reference in snapshot_refs)


@pytest.mark.parametrize("failure_call", (1, 2))
def test_begin_allocator_query_failures_are_typed_and_release_partial_snapshots(
    monkeypatch: pytest.MonkeyPatch, failure_call: int
) -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, _ = _adapter(optimizer, parameters)
    allocator_calls = 0
    snapshot_refs = []
    original_empty = adapter_module.torch.empty

    def fail_allocator_query(device):
        nonlocal allocator_calls
        allocator_calls += 1
        if allocator_calls == failure_call:
            raise RuntimeError("injected begin allocator query failure")
        return 100

    def capture_snapshot(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if kwargs.get("dtype") in (torch.float32, torch.bfloat16):
            snapshot_refs.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(adapter, "_allocator_bytes", fail_allocator_query)
    monkeypatch.setattr(adapter_module.torch, "empty", capture_snapshot)

    assert adapter.begin_event() is None
    gc.collect()

    assert allocator_calls == failure_call
    assert adapter.local_status.item() == DistributedOptimizerEventStatus.ALLOCATOR_QUERY_FAILED
    assert not adapter.armed
    assert all(reference() is None for reference in snapshot_refs)


@pytest.mark.parametrize("finish_chunk_elements", (1, 2, 3, 7, 11))
def test_preflight_estimate_covers_independently_enumerated_peak_live_storages(
    monkeypatch: pytest.MonkeyPatch, finish_chunk_elements: int
) -> None:
    optimizer, parameters = _fake_optimizer()
    adapter, registry = _adapter(optimizer, parameters, finish_chunk_elements=finish_chunk_elements)
    accumulator = registry.new_accumulator("cpu")
    estimate = adapter.estimate_snapshot_memory()
    scratch_box = []
    observed_peak_bytes = []
    original_allocate = adapter._allocate_finish_scratch
    original_add = registry.add_applied_update_moments

    def storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
        storage = tensor.untyped_storage()
        return str(tensor.device), storage.data_ptr(), storage.nbytes()

    def capture_scratch(device):
        scratch = original_allocate(device)
        scratch_box.append(scratch)
        return scratch

    def enumerate_live_storages(accumulator_arg, logical_name, **moments):
        assert adapter._snapshot is not None
        scratch = scratch_box[-1]
        tensors = (
            adapter._snapshot.master_before,
            adapter._snapshot.applied_before,
            scratch.storage,
            *moments.values(),
        )
        storages = {storage_key(tensor) for tensor in tensors}
        observed_peak_bytes.append(sum(nbytes for _, _, nbytes in storages))
        assert {storage_key(moment) for moment in moments.values()} == {
            storage_key(scratch.storage)
        }
        return original_add(accumulator_arg, logical_name, **moments)

    monkeypatch.setattr(adapter, "_allocate_finish_scratch", capture_scratch)
    monkeypatch.setattr(registry, "add_applied_update_moments", enumerate_live_storages)
    assert adapter.begin_event() is not None
    optimizer._copy_main_params_to_model_params()

    assert adapter.finish_event(accumulator, update_successful=True) is not None

    assert observed_peak_bytes
    assert max(observed_peak_bytes) == estimate.total_bytes
    assert max(observed_peak_bytes) <= estimate.total_bytes


@pytest.fixture(scope="module")
def distributed_world() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2 or "RANK" not in os.environ:
        pytest.skip("requires torch.distributed.run with at least two ranks")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")


@pytest.mark.distributed
def test_real_gloo_capability_negotiation_fails_every_rank_on_one_rank_mismatch(
    distributed_world: None,
) -> None:
    group = dist.new_group(backend="gloo")
    optimizer, _ = _fake_optimizer()
    if dist.get_rank() == 1:
        optimizer.ddp_config.overlap_param_gather = True

    report = Bf16DistributedOptimizerDiagnosticAdapter.negotiate_capabilities(
        optimizer, process_group=group
    )

    assert not report.supported
    assert DistributedOptimizerDiagnosticReason.OVERLAP_PARAM_GATHER in report.reasons
    assert DistributedOptimizerDiagnosticReason.RANK_INCONSISTENT in report.reasons
    dist.barrier(group=group)
    dist.destroy_process_group(group)


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    (
        ("iterator", DistributedOptimizerEventStatus.CONSTRUCTOR_ITERATOR_FAILED),
        ("binding", DistributedOptimizerEventStatus.CONSTRUCTOR_BINDING_FAILED),
        ("oom", DistributedOptimizerEventStatus.BEGIN_FIRST_ALLOCATION_FAILED),
        ("identity", DistributedOptimizerEventStatus.FINISH_SHARD_IDENTITY_CHANGED),
        ("materialization", DistributedOptimizerEventStatus.MATERIALIZATION_MISMATCH),
        ("telemetry", DistributedOptimizerEventStatus.ALLOCATOR_QUERY_FAILED),
    ),
)
def test_real_two_rank_one_rank_runtime_failure_reaches_one_fixed_consensus_and_releases(
    distributed_world: None, failure: str, expected_status: DistributedOptimizerEventStatus
) -> None:
    assert dist.get_world_size() == 2
    group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    optimizer, parameters = _fake_optimizer()

    if failure == "iterator" and rank == 1:

        def fail_iterator():
            raise RuntimeError("injected iterator failure")

        optimizer.iter_model_main_param_shards = fail_iterator

    if failure == "binding":
        bindings = {
            parameters[0]: ("update/fc1/0", MetricFamily.FC1),
            **({parameters[1]: ("update/fc1/1", MetricFamily.FC1)} if rank == 0 else {}),
        }
        registry, names = _registry(bindings)
        adapter = Bf16DistributedOptimizerDiagnosticAdapter(
            optimizer, registry, names, diagnostic_max_extra_bytes=1_000_000, process_group=group
        )
    else:
        selected = {
            parameter: (f"update/fc1/{index}", MetricFamily.FC1)
            for index, parameter in enumerate(parameters)
        }
        registry, names = _registry(selected)
        adapter = Bf16DistributedOptimizerDiagnosticAdapter(
            optimizer, registry, names, diagnostic_max_extra_bytes=1_000_000, process_group=group
        )

    accumulator = registry.new_accumulator("cpu")
    if failure == "telemetry" and rank == 1:
        allocator_calls = 0

        def rank_local_allocator_failure(device):
            nonlocal allocator_calls
            allocator_calls += 1
            if allocator_calls <= 2:
                return None
            raise RuntimeError("injected finish allocator query failure")

        adapter._allocator_bytes = rank_local_allocator_failure
    if failure == "oom":
        original_empty = adapter_module.torch.empty

        def rank_local_oom(*args, **kwargs):
            if rank == 1 and kwargs.get("dtype") == torch.float32:
                raise torch.OutOfMemoryError("injected")
            return original_empty(*args, **kwargs)

        adapter_module.torch.empty = rank_local_oom
        try:
            adapter.begin_event()
        finally:
            adapter_module.torch.empty = original_empty
    elif failure in ("identity", "materialization", "telemetry"):
        assert adapter.begin_event() is not None
        with torch.no_grad():
            if failure == "identity" and rank == 1:
                optimizer.shard_fp32_from_float16_groups[0][0] = torch.nn.Parameter(
                    optimizer.shard_fp32_from_float16_groups[0][0].detach().clone()
                )
            elif failure == "materialization":
                for shard in adapter.iter_owner_shards():
                    shard.main_shard.add_(0.25)
                if rank == 0:
                    optimizer._copy_main_params_to_model_params()
            else:
                optimizer._copy_main_params_to_model_params()
        adapter.finish_event(accumulator, update_successful=True)

    if failure == "telemetry":
        assert not adapter.armed

    # This test-only standalone consensus is exactly one MAX reduction in the
    # same order on every rank; production consensus remains heartbeat-owned.
    global_status = adapter.status_for_event_consensus().clone()
    dist.all_reduce(global_status, op=dist.ReduceOp.MAX, group=group)
    assert global_status.item() == expected_status

    adapter.abort_event()
    assert not adapter.armed
    dist.barrier(group=group)
    dist.destroy_process_group(group)


@pytest.mark.distributed
def test_real_gloo_owner_updates_pool_sufficient_statistics_before_ratios(
    distributed_world: None,
) -> None:
    group = dist.new_group(backend="gloo")
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    base_registry, names = _registry({parameter: ("update/fc1/pooled", MetricFamily.FC1)})
    registry = MetricRegistry(
        base_registry.descriptors, reduction_binding=ReductionBinding.flat_world(group)
    )
    accumulator = registry.new_accumulator("cpu")
    if dist.get_rank() == 0:
        master_before = torch.tensor([1.0])
        master_after = torch.tensor([2.0])
        applied_before = torch.tensor([1.0], dtype=torch.bfloat16)
        applied_after = torch.tensor([2.0], dtype=torch.bfloat16)
    else:
        master_before = torch.ones(3)
        master_after = torch.tensor([4.0, 4.0, 4.0])
        applied_before = torch.ones(3, dtype=torch.bfloat16)
        applied_after = torch.tensor([1.0, 4.0, 4.0], dtype=torch.bfloat16)
    registry.add_applied_update(
        accumulator, names[parameter], master_before, master_after, applied_before, applied_after
    )

    accumulator.reduce_()

    torch.testing.assert_close(
        accumulator.norm_retention("update/fc1/pooled").value,
        torch.sqrt(torch.tensor(19.0 / 28.0, dtype=torch.float64)),
    )
    torch.testing.assert_close(
        accumulator.relative_rms("update/fc1/pooled").value,
        torch.sqrt(torch.tensor(19.0 / 4.0, dtype=torch.float64)),
    )
    torch.testing.assert_close(
        accumulator.cast_zero_fraction("update/fc1/pooled").value,
        torch.tensor(0.25, dtype=torch.float64),
    )
    dist.barrier(group=group)
    dist.destroy_process_group(group)
