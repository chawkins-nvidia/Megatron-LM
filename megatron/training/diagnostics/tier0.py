# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Successful-update orchestration for the Tier-0 diagnostic heartbeat."""

from __future__ import annotations

import functools
import inspect
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import nn

from megatron.core import parallel_state
from megatron.core.diagnostics import (
    get_diagnostic_global_valid_tokens,
    set_diagnostic_global_valid_tokens,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.transformer.transformer_layer import TransformerLayer

from .accumulator import PackedSlots, PackedSufficientStatistics, ReductionBinding
from .artifact import Tier0ArtifactWriter, writer_from_runtime
from .capability import SUPPORT_SIGNATURE, diagnostic_schema_hash
from .capability import sha256_file as capability_sha256_file
from .capability import static_capability_path
from .capture import (
    CaptureTopology,
    StagedTokenMask,
    Tier0CaptureResult,
    Tier0CaptureSession,
    pack_valid_token_mask_sideband,
    unpack_valid_token_mask_sideband,
)
from .distributed_optimizer import (
    Bf16DistributedOptimizerDiagnosticAdapter,
    SnapshotMemoryPreflight,
    snapshot_memory_estimate,
)
from .normalization import CanonicalDgradNormalizer
from .registry import (
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
    ReplicationAxis,
    StatisticKind,
)

from .schema import (  # isort: skip
    TIER0_KEYS,
    TIER0_METADATA_KEYS,
    TIER0_METRIC_KEYS,
    assert_payload_schema,
)

_UPDATE_FAMILIES = (
    MetricFamily.QKV,
    MetricFamily.ATTN_OUT,
    MetricFamily.FC1,
    MetricFamily.FC2,
    MetricFamily.NORM,
)
_MATRIX_FAMILIES = (
    MetricFamily.EMBEDDING,
    MetricFamily.QKV,
    MetricFamily.ATTN_OUT,
    MetricFamily.FC1,
    MetricFamily.FC2,
    MetricFamily.OUTPUT,
)
_CONTROL_NAMES = (
    "control/update_failure",
    "control/unsupported",
    "control/preflight_failure",
    "control/runtime_failure",
    "control/adapter_status",
    "control/peak_hbm_bytes",
)
_MASK_PRODUCER_MARKER = "megatron.tier0.canonical-gpt-mask-producer.v1"


@dataclass(frozen=True)
class Tier0Cadence:
    """Successful-update cadence for diagnostic events."""

    interval: int = 1000
    early_updates: tuple[int, ...] = (1, 10, 100)

    def __post_init__(self) -> None:
        """Validate the fixed cadence."""

        if self.interval <= 0:
            raise ValueError("diagnostic interval must be positive")
        if any(update <= 0 for update in self.early_updates):
            raise ValueError("early diagnostic updates must be positive")

    @classmethod
    def parse(cls, interval: int, early_updates: str) -> "Tier0Cadence":
        """Parse a comma-separated early-update declaration."""

        try:
            parsed = tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in early_updates.split(",")
                        if value.strip()
                    }
                )
            )
        except ValueError as error:
            raise ValueError(
                "diagnostic early updates must be comma-separated integers"
            ) from error
        return cls(interval=interval, early_updates=parsed)

    def is_due(self, successful_update: int) -> bool:
        """Return whether a prospective successful update is diagnostic."""

        return (
            successful_update in self.early_updates
            or successful_update % self.interval == 0
        )


@dataclass(frozen=True)
class Tier0Capability:
    """Collectively negotiated narrow-backend support result."""

    supported: bool
    reasons: tuple[str, ...]
    support_signature: str


@dataclass(frozen=True)
class Tier0Reservation:
    """Pure startup reservation and measured allocation evidence."""

    requested_bytes: int
    max_extra_bytes: int
    accepted: bool
    measured_allocated_bytes: int = 0


def tier0_reservation_bytes(
    *,
    num_layers: int,
    num_microbatches: int,
    micro_batch_size: int,
    local_sequence_length: int,
    owner_elements: int,
    world_size: int,
) -> int:
    """Calculate the exact worst-case event reservation without allocating."""

    dimensions = (
        num_layers,
        num_microbatches,
        micro_batch_size,
        local_sequence_length,
        world_size,
    )
    if any(value <= 0 for value in dimensions) or owner_elements < 0:
        raise ValueError("Tier-0 reservation dimensions are invalid")
    capture_slots = 2 + 10 * num_layers
    update_slots = 5 * num_layers + 2
    control_slots = len(_CONTROL_NAMES)
    pack_bytes = (capture_slots + update_slots + control_slots) * (11 * 8 + 4 + 4)
    sideband_bytes = (
        2 * num_microbatches * (micro_batch_size * local_sequence_length + 1) * 4
    )
    optimizer_bytes = snapshot_memory_estimate(owner_elements).total_bytes
    capture_scratch = PackedSufficientStatistics.scratch_bytes_for_capacity()
    rank_evidence_bytes = (world_size + 1) * 9 * 8
    sink_staging_bytes = (75 + 9 * world_size + 13 * (capture_slots + update_slots)) * 8
    return (
        2 * pack_bytes
        + sideband_bytes
        + optimizer_bytes
        + capture_scratch
        + rank_evidence_bytes
        + sink_staging_bytes
        + 24
    )


def _unwrap_module(module: nn.Module) -> nn.Module:
    while hasattr(module, "module") and isinstance(module.module, nn.Module):
        module = module.module
    return module


def _distributed_optimizer(optimizer: object) -> DistributedOptimizer | None:
    if (
        not isinstance(optimizer, ChainedOptimizer)
        or len(optimizer.chained_optimizers) != 1
    ):
        return None
    child = optimizer.chained_optimizers[0]
    return child if isinstance(child, DistributedOptimizer) else None


def mark_tier0_mask_producer(
    forward_step_func: Callable[..., Any],
) -> Callable[..., Any]:
    """Mark a mechanically verifiable forward function as a Tier-0 mask producer."""

    setattr(
        forward_step_func, "__megatron_tier0_mask_producer__", _MASK_PRODUCER_MARKER
    )
    return forward_step_func


def _verified_mask_producer(forward_step_func: Callable[..., Any] | None) -> bool:
    """Require both the positive marker and the canonical registration operations."""

    if forward_step_func is None:
        return False
    candidate = (
        forward_step_func.func
        if isinstance(forward_step_func, functools.partial)
        else forward_step_func
    )
    candidate = inspect.unwrap(candidate)
    if (
        getattr(candidate, "__megatron_tier0_mask_producer__", None)
        != _MASK_PRODUCER_MARKER
    ):
        return False
    try:
        parameters = inspect.signature(candidate).parameters
        names = set(candidate.__code__.co_names)
    except (TypeError, ValueError, AttributeError):
        return False
    return (
        "diagnostic_heartbeat" in parameters
        and "register_local_loss_mask" in names
        and "get_diagnostic_microbatch_id" in names
    )


def _local_capability_reasons(
    args: Any,
    model: Sequence[nn.Module],
    optimizer: object,
    forward_step_func: Callable[..., Any] | None = None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    chunks = tuple(_unwrap_module(chunk) for chunk in model)
    if len(chunks) != 1 or not isinstance(chunks[0], GPTModel):
        reasons.append("dense_local_mcore_gpt")
    if getattr(args, "transformer_impl", "local") != "local":
        reasons.append("transformer_engine")
    if not getattr(args, "bf16", False) or getattr(args, "fp16", False):
        reasons.append("bf16")
    if not getattr(args, "calculate_per_token_loss", False):
        reasons.append("per_token_loss")
    distributed_optimizer = _distributed_optimizer(optimizer)
    if distributed_optimizer is None:
        reasons.append("single_distributed_optimizer_chain")
    else:
        grad_scaler = distributed_optimizer.grad_scaler
        if isinstance(grad_scaler, DynamicGradScaler) or (
            grad_scaler is not None and not isinstance(grad_scaler, ConstantGradScaler)
        ):
            reasons.append("dynamic_loss_scaling")
        inner_optimizer = distributed_optimizer.optimizer
        if not isinstance(inner_optimizer, (torch.optim.Adam, torch.optim.AdamW)) and (
            type(inner_optimizer).__name__ != "FusedAdam"
        ):
            reasons.append("adam_optimizer")
    if getattr(args, "overlap_param_gather", False):
        reasons.append("param_gather_overlap")
    if getattr(args, "use_megatron_fsdp", False) or getattr(
        args, "use_torch_fsdp2", False
    ):
        reasons.append("fsdp")
    if getattr(args, "fp8", None) or getattr(args, "fp4", None):
        reasons.append("fp8_fp4")
    if getattr(args, "num_experts", None) not in (None, 0):
        reasons.append("moe")
    if getattr(args, "qk_layernorm", False):
        reasons.append("qk_layernorm")
    if getattr(args, "expert_model_parallel_size", 1) != 1:
        reasons.append("expert_parallel")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) is not None:
        reasons.append("virtual_pipeline")
    if getattr(args, "pipeline_model_parallel_layout", None) is not None:
        reasons.append("custom_pipeline_layout")
    if getattr(args, "mtp_num_layers", None) not in (None, 0):
        reasons.append("mtp")
    if getattr(args, "hybrid_context_parallel", False):
        reasons.append("hybrid_context_parallel")
    if getattr(args, "cuda_graph_impl", "none") not in (None, "none") or getattr(
        args, "optimizer_cuda_graph", False
    ):
        reasons.append("cuda_graph")
    if getattr(args, "sft", False) and getattr(args, "micro_batch_size", 1) != 1:
        reasons.append("packed_micro_batch_size")
    if getattr(args, "variable_seq_lengths", False):
        reasons.append("variable_sequence_lengths")
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(getattr(args, "world_size", 1))
    )
    bounded_values = (
        (world_size, 1, 1024),
        (int(getattr(args, "data_parallel_size", 1)), 1, 1024),
        (int(getattr(args, "tensor_model_parallel_size", 1)), 1, 1024),
        (int(getattr(args, "pipeline_model_parallel_size", 1)), 1, 1024),
        (int(getattr(args, "context_parallel_size", 1)), 1, 1024),
        (int(getattr(args, "num_layers", 1)), 1, 10_000),
        (int(getattr(args, "micro_batch_size", 1)), 1, 1_048_576),
        (int(getattr(args, "seq_length", 1)), 1, 16_777_216),
    )
    if any(
        value < minimum or value > maximum for value, minimum, maximum in bounded_values
    ):
        reasons.append("capability_bounds")
    if not _verified_mask_producer(forward_step_func):
        reasons.append("canonical_mask_producer")
    return tuple(sorted(set(reasons)))


def negotiate_tier0_capability(
    args: Any,
    model: Sequence[nn.Module],
    optimizer: object,
    forward_step_func: Callable[..., Any] | None = None,
) -> Tier0Capability:
    """Collectively fail closed for the first supported runtime signature."""

    local_reasons = _local_capability_reasons(args, model, optimizer, forward_step_func)
    known = (
        "adam_optimizer",
        "bf16",
        "capability_bounds",
        "canonical_mask_producer",
        "cuda_graph",
        "custom_pipeline_layout",
        "dense_local_mcore_gpt",
        "dynamic_loss_scaling",
        "expert_parallel",
        "fp8_fp4",
        "fsdp",
        "hybrid_context_parallel",
        "moe",
        "mtp",
        "packed_micro_batch_size",
        "param_gather_overlap",
        "per_token_loss",
        "qk_layernorm",
        "single_distributed_optimizer_chain",
        "transformer_engine",
        "virtual_pipeline",
        "variable_sequence_lengths",
    )
    backend = dist.get_backend() if dist.is_initialized() else None
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    flags = torch.zeros(len(known), dtype=torch.int64, device=device)
    for reason in local_reasons:
        flags[known.index(reason)] = 1
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    counts = flags.cpu().tolist()
    reasons = tuple(reason for reason, count in zip(known, counts) if count)
    if any(
        count not in (0, dist.get_world_size() if dist.is_initialized() else 1)
        for count in counts
    ):
        reasons = (*reasons, "rank_inconsistent")
    return Tier0Capability(not reasons, tuple(sorted(set(reasons))), SUPPORT_SIGNATURE)


def _update_descriptor(
    name: str, family: MetricFamily, layer: int | None, index: int
) -> MetricDescriptor:
    return MetricDescriptor(
        logical_name=name,
        family=family,
        global_layer=layer,
        partition_axes=(PartitionAxis.OPTIMIZER_SHARD, PartitionAxis.PIPELINE_LAYER),
        replication_axes=(ReplicationAxis.TENSOR, ReplicationAxis.PIPELINE_TIED),
        replication_multiplicity=1,
        ownership=Ownership.AUTHORITATIVE_SHARD,
        mask_kind=MaskKind.NONE,
        statistic_kind=StatisticKind.UPDATE,
        denominator_kind=DenominatorKind.PRE_UPDATE_SUMSQ,
        normalization_kind=NormalizationKind.NONE,
        process_group_identity=ProcessGroupIdentity.WORLD,
        reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
        tied_owner_identity=None,
        packed_slots=PackedSlots.for_index(index),
    )


def _bind_module_parameters(
    bindings: dict[nn.Parameter, str], module: nn.Module | None, logical_name: str
) -> None:
    if module is None:
        return
    for parameter in module.parameters(recurse=True):
        existing = bindings.setdefault(parameter, logical_name)
        if existing != logical_name:
            raise ValueError(
                "one optimizer parameter cannot belong to multiple diagnostic families"
            )


def build_update_registry(
    model: Sequence[nn.Module],
    *,
    num_layers: int,
    reduction_binding: ReductionBinding,
) -> tuple[MetricRegistry, Mapping[nn.Parameter, str]]:
    """Build rank-independent update slots and typed local parameter bindings."""

    descriptors: list[MetricDescriptor] = []
    names: list[tuple[str, MetricFamily, int | None]] = []
    for layer in range(num_layers):
        for family in _UPDATE_FAMILIES:
            names.append((f"update/{family.value}/layer_{layer}", family, layer))
    names.extend(
        (
            ("update/embedding", MetricFamily.EMBEDDING, None),
            ("update/output", MetricFamily.OUTPUT, None),
        )
    )
    for index, (name, family, layer) in enumerate(names):
        descriptors.append(_update_descriptor(name, family, layer, index))

    bindings: dict[nn.Parameter, str] = {}
    local_names: set[str] = set()
    for wrapped_chunk in model:
        chunk = _unwrap_module(wrapped_chunk)
        if not isinstance(chunk, GPTModel):
            continue
        if getattr(chunk, "pre_process", False):
            name = "update/embedding"
            _bind_module_parameters(bindings, getattr(chunk, "embedding", None), name)
            local_names.add(name)
        if getattr(chunk, "post_process", False):
            name = "update/output"
            output_layer = getattr(chunk, "output_layer", None)
            output_parameters = (
                tuple(output_layer.parameters()) if output_layer is not None else ()
            )
            if not (
                getattr(chunk, "share_embeddings_and_output_weights", False)
                and any(parameter in bindings for parameter in output_parameters)
            ):
                _bind_module_parameters(bindings, output_layer, name)
                local_names.add(name)
        for module in chunk.modules():
            if not isinstance(module, TransformerLayer):
                continue
            layer = module.layer_number - 1
            typed_modules = (
                (MetricFamily.QKV, module.self_attention.linear_qkv),
                (MetricFamily.ATTN_OUT, module.self_attention.linear_proj),
                (MetricFamily.FC1, module.mlp.linear_fc1),
                (MetricFamily.FC2, module.mlp.linear_fc2),
                (MetricFamily.NORM, getattr(module, "input_layernorm", None)),
                (MetricFamily.NORM, getattr(module, "pre_mlp_layernorm", None)),
            )
            for family, typed_module in typed_modules:
                name = f"update/{family.value}/layer_{layer}"
                _bind_module_parameters(bindings, typed_module, name)
                local_names.add(name)
        final_layernorm = getattr(
            getattr(chunk, "decoder", None), "final_layernorm", None
        )
        if final_layernorm is not None and num_layers:
            name = f"update/{MetricFamily.NORM.value}/layer_{num_layers - 1}"
            _bind_module_parameters(bindings, final_layernorm, name)
            local_names.add(name)

    registry = MetricRegistry(
        descriptors,
        reduction_binding=reduction_binding,
        local_owners=tuple(
            descriptor.logical_name in local_names for descriptor in descriptors
        ),
    )
    return registry, bindings


class Tier0Heartbeat:
    """Own one training run's Tier-0 cadence and event state machine."""

    def __init__(
        self,
        args: Any,
        model: Sequence[nn.Module],
        optimizer: object,
        forward_step_func: Callable[..., Any] | None = None,
        *,
        wandb_log: Callable[..., None] | None = None,
        wandb_writer: object | None = None,
        artifact_writer: Tier0ArtifactWriter | None = None,
        tensorboard_writer: object | None = None,
        reduction_binding: ReductionBinding | None = None,
        num_microbatches: int = 1,
    ) -> None:
        """Collectively negotiate support and construct dormant run state."""

        self.args = args
        self.model = tuple(model)
        self.optimizer = optimizer
        self.device = self._model_device()
        self.reduction_binding = reduction_binding or ReductionBinding.flat_world(None)
        self.cadence = Tier0Cadence.parse(
            args.diagnostic_interval, args.diagnostic_early_updates
        )
        self.unsupported_policy: Literal["error", "status-only"] = (
            args.diagnostic_unsupported_policy
        )
        if self.unsupported_policy not in ("error", "status-only"):
            raise ValueError(
                "diagnostic unsupported policy must be error or status-only"
            )
        self.capability = negotiate_tier0_capability(
            args, self.model, optimizer, forward_step_func
        )
        self.forward_step_func = forward_step_func
        if not self.capability.supported and self.unsupported_policy == "error":
            raise RuntimeError(
                "Tier-0 heartbeat unsupported on every rank: "
                + ", ".join(self.capability.reasons)
            )

        self.successful_updates = int(getattr(args, "diagnostic_successful_updates", 0))
        self.event_id = int(getattr(args, "diagnostic_event_id", 0))
        self.cumulative_artifact_bytes = int(
            getattr(args, "diagnostic_cumulative_artifact_bytes", 0)
        )
        self.wandb_log = wandb_log
        self.wandb_writer = wandb_writer
        self.artifact_writer = artifact_writer
        self.tensorboard_writer = tensorboard_writer
        self.capture: Tier0CaptureSession | None = None
        self.capture_accumulator: PackedSufficientStatistics | None = None
        self.capture_result: Tier0CaptureResult | None = None
        self.update_registry: MetricRegistry | None = None
        self.update_accumulator: PackedSufficientStatistics | None = None
        self.adapter: Bf16DistributedOptimizerDiagnosticAdapter | None = None
        self.control_accumulator: PackedSufficientStatistics | None = None
        self.preflight: SnapshotMemoryPreflight | None = None
        self._attempt_due = False
        self._attempt_started = 0.0
        self._step_started = 0.0
        self._ordinary_step_latency_ms = 0.0
        self._expected_num_microbatches = 0
        self._sideband_payloads: dict[int, torch.Tensor] = {}
        self._local_sideband_payloads: dict[int, torch.Tensor] = {}
        self._received_sidebands: set[int] = set()
        self._reduction_arenas: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._capture_scratch: torch.Tensor | None = None
        self._rank_evidence: torch.Tensor | None = None
        self._local_rank_evidence: torch.Tensor | None = None
        self._sink_staging: torch.Tensor | None = None
        self._startup_num_microbatches = int(num_microbatches)
        if not 1 <= self._startup_num_microbatches <= 1_048_576:
            raise ValueError("Tier-0 startup microbatch bound is out of range")

        self.reservation = self._startup_reservation()
        self._allocate_startup_state()
        self._initialize_artifact_writer()

    def _initialize_artifact_writer(self) -> None:
        """Construct the sole rank-0 artifact owner and agree setup before training."""

        rank = dist.get_rank() if dist.is_initialized() else 0
        failed = False
        if self.capability.supported and rank == 0 and self.artifact_writer is None:
            try:
                if self.wandb_writer is None:
                    raise RuntimeError("Tier-0 requires a rank-0 W&B writer")
                self.artifact_writer = writer_from_runtime(
                    self.args,
                    self.wandb_writer,
                    cumulative_bytes=self.cumulative_artifact_bytes,
                )
            except Exception:
                failed = True
        status = torch.tensor(int(failed), dtype=torch.int64, device=self.device)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
        if bool(status.cpu().item()):
            raise RuntimeError(
                "Tier-0 artifact ownership/provenance setup failed globally"
            )

    def _startup_reservation(self) -> Tier0Reservation:
        """Calculate and globally agree the complete peak before event allocation."""

        max_extra = getattr(self.args, "diagnostic_max_extra_bytes", None)
        max_extra_bytes = 2**63 - 1 if max_extra is None else int(max_extra)
        num_layers = int(getattr(self.args, "num_layers", 1))
        local_sequence = int(getattr(self.args, "seq_length", 1)) // max(
            1,
            int(getattr(self.args, "context_parallel_size", 1)),
        )
        owner_elements = 0
        distributed_optimizer = _distributed_optimizer(self.optimizer)
        if self.capability.supported and distributed_optimizer is not None:
            owner_elements = sum(
                shard.main_shard.numel()
                for shard in distributed_optimizer.iter_model_main_param_shards()
            )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        requested = tier0_reservation_bytes(
            num_layers=num_layers,
            num_microbatches=self._startup_num_microbatches,
            micro_batch_size=int(getattr(self.args, "micro_batch_size", 1)),
            local_sequence_length=local_sequence,
            owner_elements=owner_elements,
            world_size=world_size,
        )

        locally_accepted = requested <= max_extra_bytes
        if self.device.type == "cuda":
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                allocated = torch.cuda.memory_allocated(self.device)
                reusable = max(0, reserved - allocated)
                driver_need = max(0, requested - reusable)
                locally_accepted = locally_accepted and driver_need <= free_bytes
                locally_accepted = locally_accepted and (
                    reserved + driver_need <= int(total_bytes * 0.90)
                )
            except Exception:
                locally_accepted = False

        control = torch.tensor(
            [requested, max_extra_bytes, int(locally_accepted)],
            dtype=torch.int64,
            device=self.device,
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            maximum = control.clone()
            minimum = control.clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            maximum_host = maximum.cpu().tolist()
            minimum_host = minimum.cpu().tolist()
            globally_accepted = minimum_host[2] == 1
            requested = int(maximum_host[0])
            max_extra_bytes = int(minimum_host[1])
        else:
            globally_accepted = locally_accepted
        if not globally_accepted:
            raise RuntimeError(
                "Tier-0 startup reservation rejected globally before event allocation "
                f"(requested={requested}, max_extra={max_extra_bytes})"
            )
        return Tier0Reservation(requested, max_extra_bytes, True)

    def _allocate_startup_state(self) -> None:
        """Allocate all persistent/event buffers and globally agree measured success."""

        before = (
            torch.cuda.memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0
        )
        allocation_failed = False
        try:
            self.control_accumulator = PackedSufficientStatistics(
                _CONTROL_NAMES,
                self.device,
                descriptor_hash="tier0_control_v1",
                reduction_binding=self.reduction_binding,
            )
            if self.capability.supported:
                self._construct_update_registry()
                self.update_accumulator = self.update_registry.new_accumulator(
                    self.device
                )
                self._construct_optimizer_adapter()
                assert self.adapter is not None
                self.adapter.allocate_event_buffers()
                self.capture = self._construct_capture_session()
                self.capture_accumulator = self.capture.registry.new_accumulator(
                    self.device
                )
            accumulators = tuple(
                accumulator
                for accumulator in (
                    self.capture_accumulator,
                    self.update_accumulator,
                    self.control_accumulator,
                )
                if accumulator is not None
            )
            self._reduction_arenas = (
                PackedSufficientStatistics.allocate_reduction_arenas(accumulators)
            )
            self._capture_scratch = torch.empty(
                PackedSufficientStatistics.scratch_bytes_for_capacity(),
                dtype=torch.uint8,
                device=self.device,
            )
            local_length = int(getattr(self.args, "seq_length", 1)) // max(
                1, int(getattr(self.args, "context_parallel_size", 1))
            )
            payload_elements = (
                int(getattr(self.args, "micro_batch_size", 1)) * local_length + 1
            )
            self._sideband_payloads = {
                index: torch.empty(
                    payload_elements, dtype=torch.float32, device=self.device
                )
                for index in range(self._startup_num_microbatches)
            }
            self._local_sideband_payloads = {
                index: torch.empty(
                    payload_elements, dtype=torch.float32, device=self.device
                )
                for index in range(self._startup_num_microbatches)
            }
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            self._rank_evidence = torch.empty(
                (world_size, 9), dtype=torch.float64, device=self.device
            )
            self._local_rank_evidence = torch.empty(
                9, dtype=torch.float64, device=self.device
            )
            sink_pack_elements = sum(
                accumulator.sum_pack.numel()
                + accumulator.max_pack.numel()
                + accumulator.min_pack.numel()
                for accumulator in (
                    self.capture_accumulator,
                    self.update_accumulator,
                )
                if accumulator is not None
            )
            self._sink_staging = torch.empty(
                75 + world_size * 9 + sink_pack_elements,
                dtype=torch.float64,
                device=self.device,
            )
        except Exception:
            allocation_failed = True

        measured = (
            max(0, torch.cuda.memory_allocated(self.device) - before)
            if self.device.type == "cuda"
            else 0
        )
        failed = torch.tensor(
            int(allocation_failed or measured > self.reservation.requested_bytes),
            dtype=torch.int64,
            device=self.device,
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if bool(failed.cpu().item()):
            raise RuntimeError("Tier-0 startup allocation failed on at least one rank")
        self.reservation = Tier0Reservation(
            self.reservation.requested_bytes,
            self.reservation.max_extra_bytes,
            True,
            measured,
        )

    @property
    def armed(self) -> bool:
        """Return whether the current rerun attempt is diagnostic."""

        return self._attempt_due

    @property
    def next_successful_update(self) -> int:
        """Return the successful-update index that the next step may commit."""

        return self.successful_updates + 1

    def _model_device(self) -> torch.device:
        for chunk in self.model:
            parameter = next(chunk.parameters(), None)
            if parameter is not None:
                return parameter.device
        return (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def _construct_update_registry(self) -> None:
        num_layers = int(self.args.num_layers)
        self.update_registry, bindings = build_update_registry(
            self.model, num_layers=num_layers, reduction_binding=self.reduction_binding
        )
        self._update_bindings = bindings

    def _construct_optimizer_adapter(self) -> None:
        assert self.update_registry is not None
        max_extra = self.args.diagnostic_max_extra_bytes
        self.adapter = Bf16DistributedOptimizerDiagnosticAdapter(
            self.optimizer,
            self.update_registry,
            self._update_bindings,
            diagnostic_max_extra_bytes=(2**63 - 1 if max_extra is None else max_extra),
            max_memory_fraction=0.90,
        )

    def _construct_capture_session(self) -> Tier0CaptureSession:
        """Construct dormant hooks against startup-preallocated event state."""

        normalizer = CanonicalDgradNormalizer.from_grad_scaler(
            getattr(_distributed_optimizer(self.optimizer), "grad_scaler", None)
        )
        local_sequence_length = int(self.args.seq_length) // max(
            1, parallel_state.get_context_parallel_world_size()
        )
        return Tier0CaptureSession(
            self.model,
            num_layers=int(self.args.num_layers),
            topology=CaptureTopology.from_parallel_state(
                sequence_parallel=bool(self.args.sequence_parallel)
            ),
            device=self.device,
            micro_batch_size=int(self.args.micro_batch_size),
            local_sequence_length=local_sequence_length,
            calculate_per_token_loss=True,
            dgrad_normalizer=normalizer,
            reduction_binding=self.reduction_binding,
        )

    def prepare_attempt(self, *, num_microbatches: int) -> bool:
        """Replace prior rerun state and arm capture when the next success is due."""

        self.abort_attempt()
        set_diagnostic_global_valid_tokens(None)
        self._step_started = time.perf_counter()
        self._attempt_due = self.cadence.is_due(self.next_successful_update)
        if not self._attempt_due:
            return False
        self._attempt_started = time.perf_counter()
        self._expected_num_microbatches = num_microbatches
        if (
            self.capability.supported
            and num_microbatches > self._startup_num_microbatches
        ):
            raise RuntimeError("Tier-0 microbatch count exceeds the startup bound")
        assert self.control_accumulator is not None
        self.control_accumulator.reset_()
        self._received_sidebands.clear()
        assert self._local_rank_evidence is not None
        self._local_rank_evidence.zero_()
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._local_rank_evidence[0].fill_(rank)
        if self.device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            del free_bytes
            self._local_rank_evidence[3].fill_(total_bytes)
            self._local_rank_evidence[4].fill_(torch.cuda.memory_allocated(self.device))
            self._local_rank_evidence[5].fill_(torch.cuda.memory_reserved(self.device))
        else:
            self._local_rank_evidence[3].fill_(1)
        self._local_rank_evidence[6].fill_(self.reservation.requested_bytes)
        if not self.capability.supported:
            return True
        assert self.update_registry is not None
        assert self.update_accumulator is not None
        self.update_accumulator.reset_()
        return True

    def install_capture(self) -> None:
        """Install capture hooks immediately before the forward/backward schedule."""

        if not self._attempt_due or not self.capability.supported:
            return
        if self.capture is None or self.capture_accumulator is None:
            raise RuntimeError("Tier-0 startup capture allocation is absent")
        self.capture.arm(
            expected_microbatch_ids=tuple(range(self._expected_num_microbatches)),
            accumulator=self.capture_accumulator,
        )

    def seal_capture(self) -> None:
        """Seal the attempt's local capture and remove hooks immediately."""

        if self.capture is None:
            return
        self.capture_accumulator = self.capture.seal()

    def begin_microbatch(self, microbatch_id: int) -> None:
        """Begin one centrally identified schedule microbatch."""

        if self.capture is None or not self.capture.armed:
            return
        self.capture.begin_microbatch(microbatch_id)
        payload = self._sideband_payloads.get(microbatch_id)
        if (
            microbatch_id in self._received_sidebands
            and not parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            staged = unpack_valid_token_mask_sideband(
                payload,
                micro_batch_size=int(self.args.micro_batch_size),
                sequence_length=int(self.args.seq_length)
                // max(1, parallel_state.get_context_parallel_world_size()),
                device=self.device,
            )
            self.capture.register_valid_token_mask(microbatch_id, staged)

    def end_microbatch(self, microbatch_id: int) -> None:
        """End one centrally identified schedule microbatch."""

        if self.capture is not None and self.capture.armed:
            self.capture.end_microbatch(microbatch_id)

    def register_local_loss_mask(
        self, microbatch_id: int, loss_mask: torch.Tensor | None
    ) -> None:
        """Register/compare a CP-local first/last-stage mask before model compute."""

        if self.capture is None or not self.capture.armed:
            return
        local_length = int(self.args.seq_length) // max(
            1, parallel_state.get_context_parallel_world_size()
        )
        local_payload = pack_valid_token_mask_sideband(
            loss_mask,
            micro_batch_size=int(self.args.micro_batch_size),
            sequence_length=local_length,
            device=self.device,
            out=self._local_sideband_payloads[microbatch_id],
        )
        staged = unpack_valid_token_mask_sideband(
            local_payload,
            micro_batch_size=int(self.args.micro_batch_size),
            sequence_length=local_length,
            device=self.device,
        )
        if (
            parallel_state.is_pipeline_last_stage(ignore_virtual=True)
            and microbatch_id in self._received_sidebands
        ):
            received = self._sideband_payloads[microbatch_id]
            staged = StagedTokenMask(
                staged.values, staged.valid & torch.eq(local_payload, received).all()
            )
        self.capture.register_valid_token_mask(microbatch_id, staged)
        if parallel_state.is_pipeline_first_stage(
            ignore_virtual=True
        ) and not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            dist.send(
                local_payload,
                dst=parallel_state.get_pipeline_model_parallel_next_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
            )

    def receive_mask_sideband(self, microbatch_id: int) -> None:
        """Receive the ordered fixed-shape mask immediately before activation receive."""

        if (
            not self._attempt_due
            or not self.capability.supported
            or parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        ):
            return
        local_length = int(self.args.seq_length) // max(
            1, parallel_state.get_context_parallel_world_size()
        )
        payload = self._sideband_payloads[microbatch_id]
        dist.recv(
            payload,
            src=parallel_state.get_pipeline_model_parallel_prev_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        self._received_sidebands.add(microbatch_id)

    def forward_mask_sideband(self, microbatch_id: int) -> None:
        """Forward an ordered mask before intermediate-stage model compute."""

        if (
            not self._attempt_due
            or not self.capability.supported
            or parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            return
        dist.send(
            self._sideband_payloads[microbatch_id],
            dst=parallel_state.get_pipeline_model_parallel_next_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )

    def begin_optimizer_event(self) -> SnapshotMemoryPreflight | None:
        """Snapshot into buffers admitted and allocated during startup."""

        if not self._attempt_due or self.adapter is None:
            return None
        if self.capture_accumulator is None or self.update_accumulator is None:
            raise RuntimeError(
                "diagnostic capture must be sealed before the optimizer snapshot"
            )
        self.adapter.begin_event()
        return None

    def finish_optimizer_event(
        self, update_successful: bool, *, iteration: int
    ) -> bool:
        """Complete fixed collectives, retry failures, and emit successful events."""

        if not self._attempt_due:
            if update_successful:
                self._commit_successful_update()
                self._ordinary_step_latency_ms = (
                    time.perf_counter() - self._step_started
                ) * 1000.0
            return update_successful
        assert self.control_accumulator is not None
        if self.adapter is not None and self.update_accumulator is not None:
            if update_successful and self.adapter.armed:
                self.adapter.finish_event(
                    self.update_accumulator, update_successful=True
                )
            else:
                self.adapter.abort_event()

        self._add_control("control/update_failure", not update_successful)
        self._add_control("control/unsupported", not self.capability.supported)
        self._add_control(
            "control/preflight_failure",
            self.preflight is not None and not self.preflight.accepted,
        )
        self._add_control(
            "control/runtime_failure",
            self.capability.supported and get_diagnostic_global_valid_tokens() is None,
        )
        adapter_status = (
            self.adapter.local_status.squeeze(0)
            if self.adapter is not None
            else torch.zeros((), dtype=torch.int64, device=self.device)
        )
        self._add_control("control/adapter_status", adapter_status)
        peak_hbm = (
            max(
                torch.cuda.max_memory_allocated(self.device),
                torch.cuda.max_memory_reserved(self.device),
            )
            if self.device.type == "cuda"
            else 0
        )
        self._add_control("control/peak_hbm_bytes", peak_hbm)

        accumulators = tuple(
            accumulator
            for accumulator in (
                self.capture_accumulator,
                self.update_accumulator,
                self.control_accumulator,
            )
            if accumulator is not None
        )
        PackedSufficientStatistics.reduce_many_(accumulators, self._reduction_arenas)
        latencies = self._gather_latency_ms()
        if not update_successful:
            self.abort_attempt()
            return False

        self._commit_successful_update()
        rank = dist.get_rank() if dist.is_initialized() else 0
        diagnostic_valid = True
        if rank == 0:
            payload = self._derive_payload(latencies)
            diagnostic_valid = self._emit(payload, iteration=iteration)
        self.event_id += 1
        self.args.diagnostic_event_id = self.event_id
        self.abort_attempt()
        if rank == 0 and not diagnostic_valid and self.unsupported_policy == "error":
            raise RuntimeError("Tier-0 heartbeat event failed after sink validation")
        return True

    def _add_control(self, name: str, value: int | bool | torch.Tensor) -> None:
        assert self.control_accumulator is not None
        tensor = torch.as_tensor(value, dtype=torch.float64, device=self.device)
        self.control_accumulator.add_masked_tensor(name, tensor)

    def _gather_latency_ms(self) -> torch.Tensor:
        assert self._local_rank_evidence is not None
        assert self._rank_evidence is not None
        elapsed = (time.perf_counter() - self._attempt_started) * 1000.0
        self._local_rank_evidence[1].fill_(elapsed)
        self._local_rank_evidence[2].fill_(self._ordinary_step_latency_ms or elapsed)
        if self.device.type == "cuda":
            self._local_rank_evidence[7].fill_(
                torch.cuda.max_memory_allocated(self.device)
            )
            self._local_rank_evidence[8].fill_(
                torch.cuda.max_memory_reserved(self.device)
            )
        else:
            self._local_rank_evidence[7].copy_(self._local_rank_evidence[4])
            self._local_rank_evidence[8].copy_(self._local_rank_evidence[5])
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            self._rank_evidence[0].copy_(self._local_rank_evidence)
        else:
            dist.all_gather_into_tensor(
                self._rank_evidence.reshape(-1), self._local_rank_evidence
            )
        return self._rank_evidence[:, 1]

    def _commit_successful_update(self) -> None:
        self.successful_updates += 1
        self.args.diagnostic_successful_updates = self.successful_updates

    def _derive_payload(self, latencies: torch.Tensor) -> dict[str, torch.Tensor]:
        nan = torch.full((), torch.nan, dtype=torch.float64, device=self.device)
        payload = {key: nan.clone() for key in TIER0_KEYS}
        if (
            not self.capability.supported
            or self.capture is None
            or self.capture_accumulator is None
            or self.update_accumulator is None
        ):
            valid = False
            capture_result = None
        else:
            global_valid_tokens = get_diagnostic_global_valid_tokens()
            capture_result = self.capture.derive_result(
                self.capture_accumulator, global_valid_tokens=global_valid_tokens
            )
            valid = capture_result.valid & self._updates_valid()
            self.capture_result = capture_result
            self._derive_capture_metrics(payload, capture_result)
            self._derive_update_metrics(payload)
            self._derive_nonfinite_health(payload)

        assert self.control_accumulator is not None
        control_failure = torch.stack(
            tuple(
                self.control_accumulator.maximum(name).value != 0
                for name in (
                    "control/unsupported",
                    "control/preflight_failure",
                    "control/runtime_failure",
                    "control/adapter_status",
                )
            )
        ).any()
        valid = (
            torch.as_tensor(valid, dtype=torch.bool, device=self.device)
            & ~control_failure
        )
        for key in TIER0_METRIC_KEYS:
            payload[key] = torch.where(valid, payload[key], nan)
        payload["diag/v2/event/successful_update"] = torch.tensor(
            self.successful_updates, dtype=torch.float64, device=self.device
        )
        if capture_result is not None:
            payload["diag/v2/event/valid_positions"] = (
                capture_result.global_valid_tokens
            )
        else:
            payload["diag/v2/event/valid_positions"] = torch.zeros(
                (), dtype=torch.float64, device=self.device
            )
        payload["diag/v2/status/valid"] = torch.tensor(
            valid, dtype=torch.float64, device=self.device
        )
        payload["diag/v2/perf/peak_hbm_bytes_max_rank"] = (
            self.control_accumulator.maximum("control/peak_hbm_bytes").value
        )
        sorted_latency = torch.sort(latencies).values
        count = sorted_latency.numel()
        if count % 2:
            median = sorted_latency[count // 2]
        else:
            median = (sorted_latency[count // 2 - 1] + sorted_latency[count // 2]) / 2
        payload["diag/v2/perf/latency_ms_median_rank"] = median
        payload["diag/v2/perf/latency_ms_max_rank"] = sorted_latency[-1]
        assert_payload_schema(payload)
        return payload

    def _updates_valid(self) -> torch.Tensor:
        """Return reduced update-pack validity, allowing tied output absence."""

        assert self.update_registry is not None
        assert self.update_accumulator is not None
        tied_output = any(
            isinstance(_unwrap_module(chunk), GPTModel)
            and getattr(
                _unwrap_module(chunk), "share_embeddings_and_output_weights", False
            )
            for chunk in self.model
        )
        valid = torch.ones((), dtype=torch.bool, device=self.device)
        for descriptor in self.update_registry.descriptors:
            slots = descriptor.packed_slots
            errors = sum(
                self.update_accumulator.sum_pack[offset]
                for offset in (
                    slots.nonfinite,
                    slots.mask_error,
                    slots.nonfinite_arithmetic,
                    slots.observation_error,
                )
            )
            has_contributors = self.update_accumulator.sum_pack[slots.count] > 0
            if tied_output and descriptor.family == MetricFamily.OUTPUT:
                has_contributors = torch.ones_like(has_contributors)
            valid &= (errors == 0) & has_contributors
        return valid

    def _capture_layer_values(
        self,
        result: Tier0CaptureResult,
        observation: str,
        family: MetricFamily,
        statistic: str,
    ) -> torch.Tensor:
        values = []
        for layer in range(int(self.args.num_layers)):
            name = f"{observation}/{family.value}/layer_{layer}"
            derived = getattr(result.accumulator, statistic)(name)
            values.append(derived.value)
        return torch.stack(values)

    @staticmethod
    def _summary(values: torch.Tensor, quantile: float) -> torch.Tensor:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return torch.full((), torch.nan, dtype=values.dtype, device=values.device)
        return torch.quantile(finite, quantile)

    def _derive_capture_metrics(
        self, payload: dict[str, torch.Tensor], result: Tier0CaptureResult
    ) -> None:
        residual_activation = self._capture_layer_values(
            result, "activation", MetricFamily.RESIDUAL, "rms"
        )
        residual_dgrad = self._capture_layer_values(
            result, "dgrad", MetricFamily.RESIDUAL, "rms"
        )
        anchors = {
            "first": 0,
            "q1": round((int(self.args.num_layers) - 1) * 0.25),
            "middle": round((int(self.args.num_layers) - 1) * 0.50),
            "q3": round((int(self.args.num_layers) - 1) * 0.75),
            "last": int(self.args.num_layers) - 1,
        }
        for observation, values in (
            ("activation", residual_activation),
            ("dgrad", residual_dgrad),
        ):
            for summary, index in anchors.items():
                payload[f"diag/v2/t0/{observation}/residual/rms/{summary}"] = values[
                    index
                ]
            for summary, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
                payload[f"diag/v2/t0/{observation}/residual/rms/{summary}"] = (
                    self._summary(values, quantile)
                )
        for family in (
            MetricFamily.QKV,
            MetricFamily.ATTN_OUT,
            MetricFamily.FC1,
            MetricFamily.FC2,
        ):
            values = self._capture_layer_values(result, "dgrad", family, "rms")
            payload[f"diag/v2/t0/dgrad/{family.value}/p10"] = self._summary(values, 0.1)
            payload[f"diag/v2/t0/dgrad/{family.value}/p50"] = self._summary(values, 0.5)
            finite = torch.isfinite(values)
            denominator = finite.sum()
            payload[f"diag/v2/t0/dgrad/{family.value}/zero_fraction"] = torch.where(
                denominator > 0,
                (
                    (values <= self.args.diagnostic_dgrad_starvation_threshold) & finite
                ).sum()
                / denominator,
                torch.full((), torch.nan, dtype=values.dtype, device=values.device),
            )
        for family in (
            MetricFamily.QKV,
            MetricFamily.ATTN_OUT,
            MetricFamily.FC1,
            MetricFamily.FC2,
            MetricFamily.RESIDUAL,
        ):
            maxima = self._capture_layer_values(result, "activation", family, "maximum")
            minima = self._capture_layer_values(result, "activation", family, "minimum")
            payload[f"diag/v2/t0/activation/{family.value}/max_abs"] = torch.maximum(
                torch.amax(maxima.abs()), torch.amax(minima.abs())
            )

    def _update_layer_values(
        self, family: MetricFamily, statistic: str
    ) -> torch.Tensor:
        assert self.update_accumulator is not None
        return torch.stack(
            [
                getattr(self.update_accumulator, statistic)(
                    f"update/{family.value}/layer_{layer}"
                ).value
                for layer in range(int(self.args.num_layers))
            ]
        )

    def _derive_update_metrics(self, payload: dict[str, torch.Tensor]) -> None:
        assert self.update_accumulator is not None
        for family in _UPDATE_FAMILIES:
            values = self._update_layer_values(family, "relative_rms")
            for summary, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
                payload[f"diag/v2/t0/update/{family.value}/{summary}"] = self._summary(
                    values, quantile
                )
            finite = torch.isfinite(values)
            denominator = finite.sum()
            payload[f"diag/v2/t0/update/{family.value}/starved_fraction"] = torch.where(
                denominator > 0,
                (
                    (values <= self.args.diagnostic_update_starvation_threshold)
                    & finite
                ).sum()
                / denominator,
                torch.full((), torch.nan, dtype=values.dtype, device=values.device),
            )
        for family in (MetricFamily.EMBEDDING, MetricFamily.OUTPUT):
            payload[f"diag/v2/t0/update/{family.value}/relative_rms"] = (
                self.update_accumulator.relative_rms(f"update/{family.value}").value
            )
        cast_zero = torch.zeros((), dtype=torch.float64, device=self.device)
        master_nonzero = torch.zeros_like(cast_zero)
        for descriptor in (
            self.update_registry.descriptors if self.update_registry is not None else ()
        ):
            slots = descriptor.packed_slots
            if descriptor.family in _MATRIX_FAMILIES:
                cast_zero += self.update_accumulator.sum_pack[slots.zero]
                master_nonzero += self.update_accumulator.sum_pack[slots.sum]
        payload["diag/v2/health/underflow_fraction"] = torch.where(
            master_nonzero > 0,
            cast_zero / master_nonzero,
            torch.full_like(cast_zero, torch.nan),
        )
        for family in _MATRIX_FAMILIES:
            if family in (MetricFamily.EMBEDDING, MetricFamily.OUTPUT):
                names = (f"update/{family.value}",)
            else:
                names = tuple(
                    f"update/{family.value}/layer_{layer}"
                    for layer in range(int(self.args.num_layers))
                )
            retention = torch.stack(
                [self.update_accumulator.norm_retention(name).value for name in names]
            )
            cast_loss = torch.stack(
                [
                    self.update_accumulator.cast_zero_fraction(name).value
                    for name in names
                ]
            )
            payload[f"diag/v2/t0/retention/{family.value}/median"] = self._summary(
                retention, 0.5
            )
            payload[f"diag/v2/t0/retention/{family.value}/zero_fraction"] = (
                self._summary(cast_loss, 0.5)
            )

    def _derive_nonfinite_health(self, payload: dict[str, torch.Tensor]) -> None:
        """Pool nonfinite counts across capture and update descriptors."""

        nonfinite = torch.zeros((), dtype=torch.float64, device=self.device)
        finite_count = torch.zeros_like(nonfinite)
        for registry, accumulator in (
            (
                self.capture.registry if self.capture is not None else None,
                self.capture_accumulator,
            ),
            (self.update_registry, self.update_accumulator),
        ):
            if registry is None or accumulator is None:
                continue
            for descriptor in registry.descriptors:
                if descriptor.family == MetricFamily.EVENT:
                    continue
                slots = descriptor.packed_slots
                nonfinite += accumulator.sum_pack[slots.nonfinite]
                finite_count += accumulator.sum_pack[slots.count]
        payload["diag/v2/health/nonfinite_fraction"] = torch.where(
            finite_count + nonfinite > 0,
            nonfinite / (finite_count + nonfinite),
            torch.full_like(nonfinite, torch.nan),
        )

    def _emit(self, payload: Mapping[str, torch.Tensor], *, iteration: int) -> bool:
        """Perform the sink's sole consolidated device-to-host transfer and log once."""

        assert_payload_schema(payload)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            raise RuntimeError("only the global Tier-0 sink may emit an event")
        assert self._rank_evidence is not None
        assert self._sink_staging is not None
        for index, key in enumerate(TIER0_KEYS):
            self._sink_staging[index].copy_(payload[key])
        offset = len(TIER0_KEYS)
        rank_elements = self._rank_evidence.numel()
        self._sink_staging[offset : offset + rank_elements].copy_(
            self._rank_evidence.reshape(-1)
        )
        offset += rank_elements
        pack_ranges: list[tuple[PackedSufficientStatistics, int, int]] = []
        for accumulator in (self.capture_accumulator, self.update_accumulator):
            if accumulator is None:
                continue
            start = offset
            for tensor in (
                accumulator.sum_pack,
                accumulator.max_pack,
                accumulator.min_pack,
            ):
                end = offset + tensor.numel()
                self._sink_staging[offset:end].copy_(tensor)
                offset = end
            pack_ranges.append((accumulator, start, offset))
        host_combined = self._sink_staging.cpu().tolist()
        host_values = host_combined[: len(TIER0_KEYS)]
        rank_values = host_combined[len(TIER0_KEYS) : len(TIER0_KEYS) + rank_elements]
        rank_evidence = [
            rank_values[offset : offset + 9] for offset in range(0, len(rank_values), 9)
        ]
        host_payload: dict[str, float | int] = dict(zip(TIER0_KEYS, host_values))
        for key in (
            "diag/v2/event/successful_update",
            "diag/v2/event/valid_positions",
            "diag/v2/status/valid",
            "diag/v2/perf/peak_hbm_bytes_max_rank",
        ):
            host_payload[key] = int(host_payload[key])
        assert_payload_schema(host_payload)
        host_packs: dict[str, dict[str, Sequence[Any]]] = {}
        for accumulator, start, end in pack_ranges:
            packed = host_combined[start:end]
            sum_count = accumulator.sum_pack.numel()
            max_count = accumulator.max_pack.numel()
            host_packs[accumulator.descriptor_hash] = {
                "names": accumulator.slot_names,
                "sum": tuple(packed[:sum_count]),
                "max": tuple(packed[sum_count : sum_count + max_count]),
                "min": tuple(packed[sum_count + max_count :]),
            }
        if self.wandb_log is not None:
            self.wandb_log(host_payload, step=iteration + 1)
        if self.artifact_writer is not None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            tp = int(getattr(self.args, "tensor_model_parallel_size", 1))
            pp = int(getattr(self.args, "pipeline_model_parallel_size", 1))
            cp = int(getattr(self.args, "context_parallel_size", 1))
            denominator = tp * pp * cp
            if world_size % denominator:
                raise RuntimeError("Tier-0 topology does not divide world size")
            topology = {
                "dp": world_size // denominator,
                "tp": tp,
                "pp": pp,
                "cp": cp,
                "ep": 1,
                "vpp": 1,
                "num_layers": int(self.args.num_layers),
            }
            _, artifact_bytes = self.artifact_writer.write(
                event_id=self.event_id + 1,
                successful_update=self.successful_updates,
                consumed_tokens=int(getattr(self.args, "consumed_train_samples", 0))
                * int(getattr(self.args, "seq_length", 1)),
                valid_positions=int(host_payload["diag/v2/event/valid_positions"]),
                valid=host_payload["diag/v2/status/valid"] == 1,
                topology=topology,
                rank_evidence=rank_evidence,
                capability_hash=capability_sha256_file(static_capability_path()),
                schema_hash=diagnostic_schema_hash(),
                layer_evidence=self._host_layer_evidence(host_packs),
            )
            self.cumulative_artifact_bytes += artifact_bytes
            self.args.diagnostic_cumulative_artifact_bytes = (
                self.cumulative_artifact_bytes
            )
        if self.tensorboard_writer is not None:
            for key in (*TIER0_METRIC_KEYS, *TIER0_METADATA_KEYS):
                self.tensorboard_writer.add_scalar(
                    key, host_payload[key], iteration + 1
                )
        return host_payload["diag/v2/status/valid"] == 1

    def _host_layer_evidence(
        self, host_packs: Mapping[str, Mapping[str, Sequence[Any]]]
    ) -> dict[tuple[int, str, str], dict[str, float | int]]:
        """Derive O(layers) artifact rows from the consolidated sink transfer."""

        if self.capture_accumulator is None or self.update_accumulator is None:
            raise RuntimeError(
                "valid Tier-0 artifact requires capture and update packs"
            )
        capture = host_packs[self.capture_accumulator.descriptor_hash]
        update = host_packs[self.update_accumulator.descriptor_hash]

        def fields(
            pack: Mapping[str, Sequence[Any]], name: str
        ) -> tuple[list[float], float, float]:
            index = pack["names"].index(name)
            values = [
                float(value) for value in pack["sum"][index * 11 : (index + 1) * 11]
            ]
            return values, float(pack["max"][index]), float(pack["min"][index])

        def row(
            pack: Mapping[str, Sequence[Any]],
            name: str,
            metric: str,
        ) -> dict[str, float | int]:
            values, maximum, minimum = fields(pack, name)
            count = values[1]
            if metric in ("activation_rms", "dgrad_rms"):
                numerator, denominator = values[2], count
                value = (
                    math.sqrt(numerator / denominator) if denominator > 0 else math.nan
                )
            elif metric == "activation_max_abs":
                numerator, denominator = values[2], count
                value = max(abs(maximum), abs(minimum))
            elif metric == "update_relative_rms":
                numerator, denominator = values[4], values[5]
                value = (
                    math.sqrt(numerator / denominator) if denominator > 0 else math.nan
                )
            elif metric == "retention":
                numerator, denominator = values[4], values[2]
                value = (
                    math.sqrt(numerator / denominator) if denominator > 0 else math.nan
                )
            else:
                raise AssertionError(f"unknown Tier-0 layer metric {metric}")
            valid = (
                count > 0
                and denominator > 0
                and values[7] == 0
                and values[8] == 0
                and values[9] == 0
                and values[10] == 0
                and math.isfinite(value)
            )
            return {
                "value": value,
                "valid": int(valid),
                "count": int(count),
                "sum": values[0],
                "sum_sq": numerator,
                "denominator_sum_sq": denominator,
                "zero_count": int(values[6]),
                "nonfinite_count": int(values[7]),
            }

        evidence: dict[tuple[int, str, str], dict[str, float | int]] = {}
        for layer in range(int(self.args.num_layers)):
            for family, metric, observation in (
                ("residual", "activation_rms", "activation"),
                ("residual", "activation_max_abs", "activation"),
                ("residual", "dgrad_rms", "dgrad"),
            ):
                evidence[(layer, family, metric)] = row(
                    capture, f"{observation}/{family}/layer_{layer}", metric
                )
            evidence[(layer, "norm", "update_relative_rms")] = row(
                update, f"update/norm/layer_{layer}", "update_relative_rms"
            )
            for family in ("qkv", "attn_out", "fc1", "fc2"):
                for metric, observation in (
                    ("activation_max_abs", "activation"),
                    ("dgrad_rms", "dgrad"),
                ):
                    evidence[(layer, family, metric)] = row(
                        capture, f"{observation}/{family}/layer_{layer}", metric
                    )
                for metric in ("update_relative_rms", "retention"):
                    evidence[(layer, family, metric)] = row(
                        update, f"update/{family}/layer_{layer}", metric
                    )
        return evidence

    def abort_attempt(self) -> None:
        """Release all attempt-local state without changing cadence counters."""

        if self.adapter is not None:
            self.adapter.abort_event()
        if self.capture is not None:
            self.capture.abort()
        self.capture_result = None
        self.preflight = None
        self._received_sidebands.clear()
        self._attempt_due = False
        self._expected_num_microbatches = 0
