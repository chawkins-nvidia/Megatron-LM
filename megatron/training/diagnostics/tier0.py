# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Successful-update orchestration for the Tier-0 diagnostic heartbeat."""

from __future__ import annotations

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
from .capability import SUPPORT_SIGNATURE
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
    DistributedOptimizerDiagnosticUnsupportedError,
    SnapshotMemoryPreflight,
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


def _local_capability_reasons(
    args: Any, model: Sequence[nn.Module], optimizer: object
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
    return tuple(sorted(set(reasons)))


def negotiate_tier0_capability(
    args: Any, model: Sequence[nn.Module], optimizer: object
) -> Tier0Capability:
    """Collectively fail closed for the first supported runtime signature."""

    local_reasons = _local_capability_reasons(args, model, optimizer)
    known = (
        "adam_optimizer",
        "bf16",
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
        *,
        wandb_log: Callable[..., None] | None = None,
        tensorboard_writer: object | None = None,
        reduction_binding: ReductionBinding | None = None,
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
        self.capability = negotiate_tier0_capability(args, self.model, optimizer)
        if not self.capability.supported and self.unsupported_policy == "error":
            raise RuntimeError(
                "Tier-0 heartbeat unsupported on every rank: "
                + ", ".join(self.capability.reasons)
            )

        self.successful_updates = int(getattr(args, "diagnostic_successful_updates", 0))
        self.event_id = int(getattr(args, "diagnostic_event_id", 0))
        self.wandb_log = wandb_log
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
        self._expected_num_microbatches = 0
        self._sideband_payloads: dict[int, torch.Tensor] = {}

        if self.capability.supported:
            try:
                self._construct_update_registry()
            except Exception:
                registry_failed = torch.ones((), dtype=torch.int64, device=self.device)
            else:
                registry_failed = torch.zeros((), dtype=torch.int64, device=self.device)
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(registry_failed, op=dist.ReduceOp.MAX)
            if bool(registry_failed.item()):
                self._set_startup_unsupported(("optimizer_registry",))

        if self.capability.supported:
            try:
                self._construct_optimizer_adapter()
            except DistributedOptimizerDiagnosticUnsupportedError as error:
                self._set_startup_unsupported((f"optimizer_adapter:{error}",))
            else:
                assert self.adapter is not None
                construction_failed = (self.adapter.local_status != 0).to(
                    dtype=torch.int64
                )
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(construction_failed, op=dist.ReduceOp.MAX)
                if bool(construction_failed.item()):
                    self._set_startup_unsupported(("optimizer_binding",))

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

    def _set_startup_unsupported(self, reasons: tuple[str, ...]) -> None:
        self.capability = Tier0Capability(
            supported=False,
            reasons=tuple(sorted(set((*self.capability.reasons, *reasons)))),
            support_signature=self.capability.support_signature,
        )
        self.adapter = None
        self.update_registry = None
        self._update_bindings = {}
        if self.unsupported_policy == "error":
            raise RuntimeError(
                "Tier-0 heartbeat unsupported on every rank: "
                + ", ".join(self.capability.reasons)
            )

    def prepare_attempt(self, *, num_microbatches: int) -> bool:
        """Replace prior rerun state and arm capture when the next success is due."""

        self.abort_attempt()
        set_diagnostic_global_valid_tokens(None)
        self._attempt_due = self.cadence.is_due(self.next_successful_update)
        if not self._attempt_due:
            return False
        self._attempt_started = time.perf_counter()
        self._expected_num_microbatches = num_microbatches
        self.control_accumulator = PackedSufficientStatistics(
            _CONTROL_NAMES,
            self.device,
            descriptor_hash="tier0_control_v1",
            reduction_binding=self.reduction_binding,
        )
        if not self.capability.supported:
            return True
        assert self.update_registry is not None
        self.update_accumulator = self.update_registry.new_accumulator(self.device)
        return True

    def install_capture(self) -> None:
        """Install capture hooks immediately before the forward/backward schedule."""

        if not self._attempt_due or not self.capability.supported:
            return
        if self.capture is not None:
            raise RuntimeError("Tier-0 capture hooks are already installed")
        normalizer = CanonicalDgradNormalizer.from_grad_scaler(
            getattr(_distributed_optimizer(self.optimizer), "grad_scaler", None)
        )
        local_sequence_length = int(self.args.seq_length) // max(
            1, parallel_state.get_context_parallel_world_size()
        )
        self.capture = Tier0CaptureSession(
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
        self.capture.arm(
            expected_microbatch_ids=tuple(range(self._expected_num_microbatches))
        )

    def seal_capture(self) -> None:
        """Seal the attempt's local capture and remove hooks immediately."""

        if self.capture is None:
            return
        try:
            self.capture_accumulator = self.capture.seal()
        finally:
            self.capture.close()

    def begin_microbatch(self, microbatch_id: int) -> None:
        """Begin one centrally identified schedule microbatch."""

        if self.capture is None or not self.capture.armed:
            return
        self.capture.begin_microbatch(microbatch_id)
        payload = self._sideband_payloads.get(microbatch_id)
        if payload is not None and not parallel_state.is_pipeline_last_stage(
            ignore_virtual=True
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
        )
        staged = unpack_valid_token_mask_sideband(
            local_payload,
            micro_batch_size=int(self.args.micro_batch_size),
            sequence_length=local_length,
            device=self.device,
        )
        if (
            parallel_state.is_pipeline_last_stage(ignore_virtual=True)
            and microbatch_id in self._sideband_payloads
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
        payload = torch.empty(
            int(self.args.micro_batch_size) * local_length + 1,
            dtype=torch.float32,
            device=self.device,
        )
        dist.recv(
            payload,
            src=parallel_state.get_pipeline_model_parallel_prev_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        self._sideband_payloads[microbatch_id] = payload

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
        """Run exact HBM preflight, then snapshot immediately before the step."""

        if not self._attempt_due or self.adapter is None:
            return None
        if (
            self.capture_accumulator is None
            or self.update_accumulator is None
            or self.control_accumulator is None
        ):
            raise RuntimeError(
                "diagnostic capture must be sealed before optimizer preflight"
            )
        accumulators = (
            self.capture_accumulator,
            self.update_accumulator,
            self.control_accumulator,
        )
        persistent = sum(
            accumulator.sum_pack.nbytes
            + accumulator.max_pack.nbytes
            + accumulator.min_pack.nbytes
            for accumulator in accumulators
        )
        mask_arena = self._retained_storage_bytes(
            (*self.capture.retained_event_tensors, *self._sideband_payloads.values())
        )
        capture_scratch = self.capture_accumulator.maximum_scratch_bytes
        reduction_arena = PackedSufficientStatistics.reduction_arena_bytes(accumulators)
        additional = persistent + mask_arena + capture_scratch + reduction_arena
        self.preflight = self.adapter.preflight_snapshot_memory(
            additional_bytes=additional
        )
        self.adapter.begin_event(additional_bytes=additional)
        return self.preflight

    def finish_optimizer_event(
        self, update_successful: bool, *, iteration: int
    ) -> bool:
        """Complete fixed collectives, retry failures, and emit successful events."""

        if not self._attempt_due:
            if update_successful:
                self._commit_successful_update()
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
        peak_hbm = self.preflight.requested_bytes if self.preflight is not None else 0
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
        PackedSufficientStatistics.reduce_many_(accumulators)
        global_update_success = not bool(
            self.control_accumulator.maximum("control/update_failure").value.item()
        )
        latencies = self._gather_latency_ms()
        if not global_update_success:
            self.abort_attempt()
            return False

        self._commit_successful_update()
        payload, diagnostic_valid = self._derive_payload(latencies)
        if not diagnostic_valid and self.unsupported_policy == "error":
            self.abort_attempt()
            raise RuntimeError(
                "Tier-0 heartbeat event failed collectively after optimizer step"
            )
        self._emit(payload, iteration=iteration)
        self.event_id += 1
        self.args.diagnostic_event_id = self.event_id
        self.abort_attempt()
        return True

    def _add_control(self, name: str, value: int | bool | torch.Tensor) -> None:
        assert self.control_accumulator is not None
        tensor = torch.as_tensor(value, dtype=torch.float64, device=self.device)
        self.control_accumulator.add_masked_tensor(name, tensor)

    def _gather_latency_ms(self) -> torch.Tensor:
        local = torch.tensor(
            (time.perf_counter() - self._attempt_started) * 1000.0,
            dtype=torch.float64,
            device=self.device,
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        gathered = torch.empty(world_size, dtype=torch.float64, device=self.device)
        if world_size == 1:
            gathered.copy_(local.reshape(1))
        else:
            dist.all_gather_into_tensor(gathered, local.reshape(1))
        return gathered

    def _commit_successful_update(self) -> None:
        self.successful_updates += 1
        self.args.diagnostic_successful_updates = self.successful_updates

    def _derive_payload(
        self, latencies: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], bool]:
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
            valid = bool(capture_result.valid.item()) and self._updates_valid()
            self.capture_result = capture_result
            self._derive_capture_metrics(payload, capture_result)
            self._derive_update_metrics(payload)
            self._derive_nonfinite_health(payload)

        assert self.control_accumulator is not None
        control_failure = any(
            self.control_accumulator.maximum(name).value.item() != 0
            for name in (
                "control/unsupported",
                "control/preflight_failure",
                "control/runtime_failure",
                "control/adapter_status",
            )
        )
        valid = valid and not control_failure
        if not valid:
            for key in TIER0_METRIC_KEYS:
                payload[key] = nan.clone()
        payload["diag/v2/event/successful_update"] = torch.tensor(
            self.successful_updates, dtype=torch.float64, device=self.device
        )
        if capture_result is not None:
            payload["diag/v2/event/valid_positions"] = (
                capture_result.global_valid_tokens
            )
        payload["diag/v2/status/valid"] = torch.tensor(
            float(valid), dtype=torch.float64, device=self.device
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
        return payload, valid

    def _updates_valid(self) -> bool:
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
        return bool(valid.item())

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

    def _emit(self, payload: Mapping[str, torch.Tensor], *, iteration: int) -> None:
        assert_payload_schema(payload)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and self.wandb_log is not None:
            values = torch.stack(
                tuple(payload[key].to(dtype=torch.float64) for key in TIER0_KEYS)
            )
            host_values = values.cpu().tolist()
            host_payload = dict(zip(TIER0_KEYS, host_values))
            assert_payload_schema(host_payload)
            self.wandb_log(host_payload, step=iteration + 1)
        if self.tensorboard_writer is not None:
            assert_payload_schema(payload)
            for key in (*TIER0_METRIC_KEYS, *TIER0_METADATA_KEYS):
                self.tensorboard_writer.add_scalar(key, payload[key], iteration + 1)

    @staticmethod
    def _retained_storage_bytes(tensors: Sequence[torch.Tensor]) -> int:
        """Count unique live tensor storages exactly once."""

        storages: dict[tuple[torch.device, int], int] = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    def abort_attempt(self) -> None:
        """Release all attempt-local state without changing cadence counters."""

        if self.adapter is not None:
            self.adapter.abort_event()
        if self.capture is not None:
            self.capture.close()
        self.capture = None
        self.capture_accumulator = None
        self.capture_result = None
        self.update_accumulator = None
        self.control_accumulator = None
        self.preflight = None
        self._sideband_payloads.clear()
        self._attempt_due = False
        self._expected_num_microbatches = 0
