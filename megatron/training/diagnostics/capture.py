# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recompute-safe, topology-aware Tier-0 activation and dgrad capture."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import torch
from torch import nn

from megatron.core import parallel_state
from megatron.core.diagnostics import (
    get_diagnostic_microbatch_id,
    is_diagnostic_recompute,
    set_diagnostic_microbatch_id,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.transformer_layer import TransformerLayer

from .accumulator import (
    PackedSlots,
    PackedSufficientStatistics,
    ProcessGroupIdentity,
    ReductionBinding,
    ReductionKind,
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
    ReplicationAxis,
    StatisticKind,
)
from .schema import Tier0Status

_CAPTURE_FAMILIES = (
    MetricFamily.RESIDUAL,
    MetricFamily.QKV,
    MetricFamily.ATTN_OUT,
    MetricFamily.FC1,
    MetricFamily.FC2,
)
_FEATURE_SHARDED_FAMILIES = (MetricFamily.QKV, MetricFamily.FC1)
_EVENT_VALID_TOKENS = "event/valid_tokens"
_EVENT_RUNTIME_STATUS = "event/runtime_status"


@dataclass(frozen=True)
class CaptureTopology:
    """Topology coordinates needed to select unique activation contributors."""

    tensor_parallel_rank: int = 0
    tensor_parallel_size: int = 1
    pipeline_parallel_rank: int = 0
    pipeline_parallel_size: int = 1
    context_parallel_rank: int = 0
    context_parallel_size: int = 1
    sequence_parallel: bool = False

    def __post_init__(self) -> None:
        """Validate all topology coordinates."""

        for name, rank, size in (
            ("tensor", self.tensor_parallel_rank, self.tensor_parallel_size),
            ("pipeline", self.pipeline_parallel_rank, self.pipeline_parallel_size),
            ("context", self.context_parallel_rank, self.context_parallel_size),
        ):
            if size <= 0 or rank < 0 or rank >= size:
                raise ValueError(f"invalid {name}-parallel coordinate {rank}/{size}")

    @property
    def is_pipeline_last_stage(self) -> bool:
        """Return whether this rank owns the final physical pipeline stage."""

        return self.pipeline_parallel_rank == self.pipeline_parallel_size - 1

    @classmethod
    def from_parallel_state(cls, *, sequence_parallel: bool) -> "CaptureTopology":
        """Read the supported physical topology from Megatron parallel state."""

        return cls(
            tensor_parallel_rank=parallel_state.get_tensor_model_parallel_rank(),
            tensor_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
            pipeline_parallel_rank=parallel_state.get_pipeline_model_parallel_rank(),
            pipeline_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
            context_parallel_rank=parallel_state.get_context_parallel_rank(),
            context_parallel_size=parallel_state.get_context_parallel_world_size(),
            sequence_parallel=sequence_parallel,
        )


class TokenLayout(StrEnum):
    """Describe the first dimension of one typed MCore module output."""

    CP_LOCAL_SEQUENCE = "cp_local_sequence"
    TP_SEQUENCE_SHARD = "tp_sequence_shard"


@dataclass(frozen=True)
class LayerCaptureTarget:
    """Bind one typed local MCore module to a global layer/family identity."""

    global_layer: int
    family: MetricFamily
    module: nn.Module
    token_layout: TokenLayout


@dataclass(frozen=True)
class StagedTokenMask:
    """Carry a canonical ``[sequence, batch, 1]`` mask and device validity."""

    values: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class Tier0CaptureResult:
    """Return reduced capture packs and device-resident event validity."""

    accumulator: PackedSufficientStatistics
    global_valid_tokens: torch.Tensor
    status: torch.Tensor
    valid: torch.Tensor


def stage_valid_token_mask(
    loss_mask: torch.Tensor | None,
    *,
    micro_batch_size: int,
    sequence_length: int,
    device: torch.device | str,
) -> StagedTokenMask:
    """Stage a GPT loss mask into the capture layout without host synchronization.

    Args:
        loss_mask: CP-local GPT mask in ``[batch, sequence]`` order.
        micro_batch_size: Configured fixed local microbatch size.
        sequence_length: Configured fixed CP-local sequence length.
        device: Capture device.

    Returns:
        Canonical values plus a scalar device validity flag. Shape/device
        failures produce a neutral invalid mask rather than raising.
    """

    target_device = torch.device(device)
    values = torch.zeros(
        (sequence_length, micro_batch_size, 1), dtype=torch.float32, device=target_device
    )
    valid = torch.zeros((), dtype=torch.bool, device=target_device)
    if (
        loss_mask is not None
        and loss_mask.device == target_device
        and not loss_mask.is_complex()
        and tuple(loss_mask.shape) == (micro_batch_size, sequence_length)
    ):
        values.copy_(loss_mask.detach().transpose(0, 1).unsqueeze(-1).to(dtype=torch.float32))
        valid.fill_(True)
    return StagedTokenMask(values=values, valid=valid)


def pack_valid_token_mask_sideband(
    loss_mask: torch.Tensor | None,
    *,
    micro_batch_size: int,
    sequence_length: int,
    device: torch.device | str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack one fixed-shape mask/status payload without communication.

    The supported non-interleaved schedule owns ordered pipeline transport. This
    helper only prepares the caller-provided CP-local payload that the schedule
    can send alongside its normal forward traffic.

    Args:
        loss_mask: Caller-provided CP-local mask in ``[batch, sequence]`` order.
        micro_batch_size: Configured fixed local microbatch size.
        sequence_length: Configured fixed CP-local sequence length.
        device: Sideband payload device.

    Returns:
        Flat FP32 mask values followed by one device-validity value.
    """

    target_device = torch.device(device)
    num_mask_values = micro_batch_size * sequence_length
    payload = out
    if (
        payload is None
        or payload.device != target_device
        or payload.dtype != torch.float32
        or payload.numel() != num_mask_values + 1
    ):
        if out is not None:
            raise ValueError("preallocated mask payload has the wrong layout")
        payload = torch.empty(num_mask_values + 1, dtype=torch.float32, device=target_device)
    payload.zero_()
    if (
        loss_mask is not None
        and loss_mask.device == target_device
        and not loss_mask.is_complex()
        and tuple(loss_mask.shape) == (micro_batch_size, sequence_length)
    ):
        payload[:-1].view(sequence_length, micro_batch_size).copy_(
            loss_mask.detach().transpose(0, 1)
        )
        payload[-1].fill_(1)
    return payload


def unpack_valid_token_mask_sideband(
    payload: torch.Tensor | None,
    *,
    micro_batch_size: int,
    sequence_length: int,
    device: torch.device | str,
) -> StagedTokenMask:
    """Validate and unpack a schedule-delivered mask payload without collectives.

    Args:
        payload: Flat schedule-delivered FP32 payload, or ``None`` on failure.
        micro_batch_size: Configured fixed local microbatch size.
        sequence_length: Configured fixed CP-local sequence length.
        device: Capture device.

    Returns:
        Canonical values and a device-resident validity scalar. Malformed
        payloads return a neutral invalid mask.
    """

    target_device = torch.device(device)
    expected_numel = micro_batch_size * sequence_length + 1
    if (
        payload is None
        or payload.device != target_device
        or payload.is_complex()
        or payload.dtype != torch.float32
        or payload.dim() != 1
        or payload.numel() != expected_numel
    ):
        return stage_valid_token_mask(
            None,
            micro_batch_size=micro_batch_size,
            sequence_length=sequence_length,
            device=target_device,
        )
    return StagedTokenMask(
        values=payload[:-1].view(sequence_length, micro_batch_size, 1), valid=payload[-1] > 0
    )


def slice_sequence_parallel_mask(
    staged: StagedTokenMask, *, tensor_parallel_rank: int, tensor_parallel_size: int
) -> StagedTokenMask:
    """Take the contiguous first-dimension TP slice used by MCore SP mappings.

    Args:
        staged: Full CP-local staged mask.
        tensor_parallel_rank: Local TP coordinate.
        tensor_parallel_size: TP group size.

    Returns:
        The rank's contiguous first-dimension slice, or an invalid mask when
        the sequence dimension is not evenly partitionable.
    """

    sequence_length = staged.values.shape[0]
    if tensor_parallel_size <= 0 or sequence_length % tensor_parallel_size != 0:
        return StagedTokenMask(staged.values, staged.valid & torch.zeros_like(staged.valid))
    local_length = sequence_length // tensor_parallel_size
    offset = tensor_parallel_rank * local_length
    return StagedTokenMask(staged.values[offset : offset + local_length].contiguous(), staged.valid)


def discover_layer_capture_targets(
    model: nn.Module | Sequence[nn.Module], *, num_layers: int
) -> tuple[LayerCaptureTarget, ...]:
    """Discover dense local MCore capture points without parsing parameter names.

    Args:
        model: One physical, non-VPP model chunk.
        num_layers: Global transformer depth used to validate layer numbers.

    Returns:
        Targets sorted by global layer and canonical family order.

    Raises:
        ValueError: If the model is outside the narrow dense local backend.
    """

    models = (model,) if isinstance(model, nn.Module) else tuple(model)
    if len(models) != 1 or not isinstance(models[0], nn.Module):
        raise ValueError("Tier-0 capture does not support virtual pipeline model chunks")
    if num_layers <= 0:
        raise ValueError("Tier-0 capture requires a positive global layer count")

    targets: list[LayerCaptureTarget] = []
    seen_layers: set[int] = set()
    for module in models[0].modules():
        if not isinstance(module, TransformerLayer):
            continue
        global_layer = module.layer_number - 1
        if global_layer < 0 or global_layer >= num_layers or global_layer in seen_layers:
            raise ValueError("TransformerLayer.layer_number is not a unique global layer")
        if getattr(module, "is_moe_layer", False):
            raise ValueError("Tier-0 capture does not support MoE layers")
        config = getattr(module, "config", None)
        if config is not None and (
            getattr(config, "transformer_impl", "local") != "local"
            or getattr(config, "params_dtype", torch.bfloat16) != torch.bfloat16
            or getattr(config, "fp8", None)
            or getattr(config, "fp4", None)
            or getattr(config, "cuda_graph_impl", "none") not in ("none", None)
            or getattr(config, "mlp_chunks_for_training", 1) != 1
        ):
            raise ValueError("TransformerLayer is outside the supported local eager backend")

        qkv = module.self_attention.linear_qkv
        attention_output = module.self_attention.linear_proj
        fc1 = module.mlp.linear_fc1
        fc2 = module.mlp.linear_fc2
        if not isinstance(qkv, ColumnParallelLinear) or not isinstance(fc1, ColumnParallelLinear):
            raise ValueError("qkv/fc1 must be local ColumnParallelLinear modules")
        if not isinstance(attention_output, RowParallelLinear) or not isinstance(
            fc2, RowParallelLinear
        ):
            raise ValueError("attention output/fc2 must be local RowParallelLinear modules")

        seen_layers.add(global_layer)
        sequence_parallel = bool(getattr(config, "sequence_parallel", False))
        targets.extend(
            (
                LayerCaptureTarget(
                    global_layer,
                    MetricFamily.RESIDUAL,
                    module,
                    (
                        TokenLayout.TP_SEQUENCE_SHARD
                        if sequence_parallel
                        else TokenLayout.CP_LOCAL_SEQUENCE
                    ),
                ),
                LayerCaptureTarget(
                    global_layer, MetricFamily.QKV, qkv, TokenLayout.CP_LOCAL_SEQUENCE
                ),
                LayerCaptureTarget(
                    global_layer,
                    MetricFamily.ATTN_OUT,
                    attention_output,
                    (
                        TokenLayout.TP_SEQUENCE_SHARD
                        if getattr(attention_output, "sequence_parallel", sequence_parallel)
                        else TokenLayout.CP_LOCAL_SEQUENCE
                    ),
                ),
                LayerCaptureTarget(
                    global_layer, MetricFamily.FC1, fc1, TokenLayout.CP_LOCAL_SEQUENCE
                ),
                LayerCaptureTarget(
                    global_layer,
                    MetricFamily.FC2,
                    fc2,
                    (
                        TokenLayout.TP_SEQUENCE_SHARD
                        if getattr(fc2, "sequence_parallel", sequence_parallel)
                        else TokenLayout.CP_LOCAL_SEQUENCE
                    ),
                ),
            )
        )
    if not targets:
        raise ValueError("Tier-0 capture found no local TransformerLayer objects")
    family_order = {family: index for index, family in enumerate(_CAPTURE_FAMILIES)}
    return tuple(
        sorted(targets, key=lambda target: (target.global_layer, family_order[target.family]))
    )


class Tier0CaptureSession:
    """Capture armed activation/dgrad moments into one fixed packed registry."""

    def __init__(
        self,
        model: nn.Module | Sequence[nn.Module],
        *,
        num_layers: int,
        topology: CaptureTopology,
        device: torch.device | str,
        micro_batch_size: int,
        local_sequence_length: int,
        calculate_per_token_loss: bool,
        dgrad_normalizer: CanonicalDgradNormalizer,
        reduction_binding: ReductionBinding,
    ) -> None:
        """Build fixed global slots and install dormant typed forward hooks."""

        if not calculate_per_token_loss:
            raise ValueError("Tier-0 canonical dgrad requires per-token summed loss")
        if micro_batch_size <= 0 or local_sequence_length <= 0:
            raise ValueError("Tier-0 capture requires fixed positive mask dimensions")
        self.topology = topology
        self.device = torch.device(device)
        self.micro_batch_size = micro_batch_size
        self.local_sequence_length = local_sequence_length
        self.dgrad_normalizer = dgrad_normalizer
        self.targets = discover_layer_capture_targets(model, num_layers=num_layers)
        for target in self.targets:
            if target.family == MetricFamily.RESIDUAL and (
                getattr(target.module.config, "sequence_parallel", topology.sequence_parallel)
                != topology.sequence_parallel
            ):
                raise ValueError("capture topology does not match layer sequence parallelism")
        local_layers = {target.global_layer for target in self.targets}
        descriptors, local_owners = self._build_descriptors(num_layers, local_layers)
        self.registry = MetricRegistry(
            descriptors,
            reduction_binding=reduction_binding,
            local_owners=local_owners,
            normalization_adapters=(dgrad_normalizer,),
        )
        self._hook_handles = tuple(
            target.module.register_forward_hook(self._make_forward_hook(target))
            for target in self.targets
        )
        self._accumulator: PackedSufficientStatistics | None = None
        self._armed = False
        self._begun: set[int] = set()
        self._ended: set[int] = set()
        self._masks: dict[int, StagedTokenMask] = {}
        self._activation_seen: set[tuple[int, str]] = set()
        self._dgrad_registered: set[tuple[int, str]] = set()
        self._dgrad_seen: set[tuple[int, str]] = set()
        self._expected_microbatch_ids: set[int] | None = None
        self._runtime_status = torch.zeros((), dtype=torch.int64, device=self.device)

    @property
    def armed(self) -> bool:
        """Return whether hooks currently collect diagnostic observations."""

        return self._armed

    @property
    def retained_event_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return tensors retained until event reduction for exact preflight accounting."""

        tensors = [self._runtime_status]
        for staged in self._masks.values():
            tensors.extend((staged.values, staged.valid))
        return tuple(tensors)

    def arm(
        self,
        *,
        expected_microbatch_ids: Sequence[int] | None = None,
        accumulator: PackedSufficientStatistics | None = None,
    ) -> None:
        """Allocate neutral event packs and arm dormant hooks.

        Args:
            expected_microbatch_ids: Optional complete schedule-local microbatch
                identity sequence. When provided, a wholly skipped microbatch is
                reported through packed observation-completeness status.
        """

        if self._armed:
            raise RuntimeError("Tier-0 capture session is already armed")
        self._accumulator = (
            self.registry.new_accumulator(self.device)
            if accumulator is None
            else accumulator.reset_()
        )
        self._begun.clear()
        self._ended.clear()
        self._masks.clear()
        self._activation_seen.clear()
        self._dgrad_registered.clear()
        self._dgrad_seen.clear()
        if expected_microbatch_ids is None:
            self._expected_microbatch_ids = None
        else:
            expected = tuple(expected_microbatch_ids)
            if any(microbatch_id < 0 for microbatch_id in expected) or len(expected) != len(
                set(expected)
            ):
                raise ValueError("expected diagnostic microbatch identities must be unique")
            self._expected_microbatch_ids = set(expected)
        self._runtime_status.zero_()
        self._armed = True

    def begin_microbatch(self, microbatch_id: int) -> None:
        """Begin one armed schedule-local microbatch."""

        if not self._armed:
            raise RuntimeError("Tier-0 capture session is not armed")
        if microbatch_id < 0 or microbatch_id in self._begun:
            raise ValueError("diagnostic microbatch must be unique and nonnegative")
        if (
            self._expected_microbatch_ids is not None
            and microbatch_id not in self._expected_microbatch_ids
        ):
            self._set_runtime_error()
        self._begun.add(microbatch_id)
        set_diagnostic_microbatch_id(microbatch_id)

    def register_valid_token_mask(
        self, microbatch_id: int, loss_mask: torch.Tensor | StagedTokenMask | None
    ) -> None:
        """Register the CP-local valid-token mask before the model forward."""

        if not self._armed or microbatch_id not in self._begun:
            raise RuntimeError("begin_microbatch must precede mask registration")
        if microbatch_id in self._masks:
            raise ValueError("a diagnostic microbatch mask may only be registered once")
        staged = (
            loss_mask
            if isinstance(loss_mask, StagedTokenMask)
            else stage_valid_token_mask(
                loss_mask,
                micro_batch_size=self.micro_batch_size,
                sequence_length=self.local_sequence_length,
                device=self.device,
            )
        )
        expected_shape = (self.local_sequence_length, self.micro_batch_size, 1)
        if (
            not isinstance(staged.values, torch.Tensor)
            or not isinstance(staged.valid, torch.Tensor)
            or staged.values.device != self.device
            or staged.valid.device != self.device
            or staged.values.is_complex()
            or tuple(staged.values.shape) != expected_shape
            or staged.valid.numel() != 1
        ):
            staged = stage_valid_token_mask(
                None,
                micro_batch_size=self.micro_batch_size,
                sequence_length=self.local_sequence_length,
                device=self.device,
            )
        numeric_valid = torch.isfinite(staged.values).all() & (staged.values >= 0).all()
        staged = StagedTokenMask(staged.values, staged.valid & numeric_valid)
        self._masks[microbatch_id] = staged

        if self.registry.owns(_EVENT_VALID_TOKENS):
            accumulator = self._require_accumulator()
            self.registry.mark_mask_error(accumulator, _EVENT_VALID_TOKENS, ~staged.valid)
            valid_token_count = torch.where(
                staged.valid,
                staged.values.sum(dtype=torch.float64),
                torch.zeros((), dtype=torch.float64, device=self.device),
            )
            self.registry.add_masked_tensor(accumulator, _EVENT_VALID_TOKENS, valid_token_count)

    def end_microbatch(self, microbatch_id: int) -> None:
        """Mark completion of one microbatch forward; backward may follow later."""

        if microbatch_id not in self._begun or microbatch_id in self._ended:
            raise ValueError("diagnostic microbatch was not active")
        self._ended.add(microbatch_id)
        if get_diagnostic_microbatch_id() == microbatch_id:
            set_diagnostic_microbatch_id(None)

    def finalize(self) -> Tier0CaptureResult:
        """Reduce fixed packs, canonicalize dgrad, and derive event validity."""

        accumulator = self.seal()
        accumulator.reduce_()
        return self.derive_result(accumulator)

    def seal(self) -> PackedSufficientStatistics:
        """Finish local capture without launching diagnostic collectives."""

        if not self._armed:
            raise RuntimeError("Tier-0 capture session is not armed")
        accumulator = self._require_accumulator()
        if self._begun != self._ended or self._begun != self._masks.keys():
            self._set_runtime_error()
        self._mark_incomplete_observations(accumulator)
        self.registry.add_masked_tensor(accumulator, _EVENT_RUNTIME_STATUS, self._runtime_status)
        self._armed = False
        set_diagnostic_microbatch_id(None)
        return accumulator

    def derive_result(
        self,
        accumulator: PackedSufficientStatistics,
        *,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> Tier0CaptureResult:
        """Canonicalize and validate capture packs after combined reduction."""

        if not accumulator.reduced:
            raise RuntimeError("Tier-0 capture derivation requires reduced packs")
        token_slots = accumulator.slots(_EVENT_VALID_TOKENS)
        packed_valid_tokens = accumulator.sum_pack[token_slots.sum]
        global_valid_tokens = (
            packed_valid_tokens
            if global_valid_tokens is None
            else global_valid_tokens.to(dtype=torch.float64, device=packed_valid_tokens.device)
        )
        self.registry.apply_normalizations_(accumulator, global_valid_tokens=global_valid_tokens)

        runtime_slots = accumulator.slots(_EVENT_RUNTIME_STATUS)
        runtime_status = accumulator.max_pack[runtime_slots.maximum].to(dtype=torch.int64)
        packed_errors = torch.stack(
            tuple(
                accumulator.sum_pack[offset]
                for descriptor in self.registry.descriptors
                for offset in (
                    descriptor.packed_slots.nonfinite,
                    descriptor.packed_slots.mask_error,
                    descriptor.packed_slots.nonfinite_arithmetic,
                    descriptor.packed_slots.observation_error,
                )
            )
        )
        observation_counts = torch.stack(
            tuple(
                accumulator.sum_pack[descriptor.packed_slots.count]
                for descriptor in self.registry.descriptors
                if descriptor.family != MetricFamily.EVENT
            )
        )
        invalid_statistics = (
            (packed_errors > 0).any()
            | (observation_counts <= 0).any()
            | ~torch.isfinite(global_valid_tokens)
            | (global_valid_tokens <= 0)
        )
        status = torch.where(
            runtime_status >= Tier0Status.RUNTIME_ERROR,
            runtime_status,
            torch.where(
                invalid_statistics,
                torch.full_like(runtime_status, Tier0Status.INVALID_STATISTICS),
                torch.full_like(runtime_status, Tier0Status.OK),
            ),
        )
        return Tier0CaptureResult(
            accumulator=accumulator,
            global_valid_tokens=global_valid_tokens,
            status=status,
            valid=status == Tier0Status.OK,
        )

    def close(self) -> None:
        """Remove all dormant forward hooks owned by this session."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = ()
        self._armed = False
        set_diagnostic_microbatch_id(None)

    def abort(self) -> None:
        """Disarm one attempt while retaining startup-allocated hooks and buffers."""

        self._armed = False
        self._begun.clear()
        self._ended.clear()
        self._masks.clear()
        self._activation_seen.clear()
        self._dgrad_registered.clear()
        self._dgrad_seen.clear()
        set_diagnostic_microbatch_id(None)

    def _build_descriptors(
        self, num_layers: int, local_layers: set[int]
    ) -> tuple[tuple[MetricDescriptor, ...], tuple[bool, ...]]:
        descriptors: list[MetricDescriptor] = []
        owners: list[bool] = []

        def append_descriptor(
            logical_name: str,
            family: MetricFamily,
            global_layer: int | None,
            *,
            owner: bool,
            ownership: Ownership,
            mask_kind: MaskKind,
            normalization_kind: NormalizationKind,
            partition_axes: tuple[PartitionAxis, ...],
            replication_axes: tuple[ReplicationAxis, ...] = (),
        ) -> None:
            index = len(descriptors)
            descriptors.append(
                MetricDescriptor(
                    logical_name=logical_name,
                    family=family,
                    global_layer=global_layer,
                    partition_axes=partition_axes,
                    replication_axes=replication_axes,
                    replication_multiplicity=1,
                    ownership=ownership,
                    mask_kind=mask_kind,
                    statistic_kind=StatisticKind.TENSOR_MOMENTS,
                    denominator_kind=DenominatorKind.SELECTED_ELEMENTS,
                    normalization_kind=normalization_kind,
                    process_group_identity=ProcessGroupIdentity.WORLD,
                    reduction_kind=ReductionKind.PACKED_SUM_MAX_MIN,
                    tied_owner_identity=None,
                    packed_slots=PackedSlots.for_index(index),
                )
            )
            owners.append(owner)

        append_descriptor(
            _EVENT_VALID_TOKENS,
            MetricFamily.EVENT,
            None,
            owner=(
                self.topology.is_pipeline_last_stage and self.topology.tensor_parallel_rank == 0
            ),
            ownership=Ownership.TENSOR_PARALLEL_RANK_ZERO,
            mask_kind=MaskKind.NONE,
            normalization_kind=NormalizationKind.NONE,
            partition_axes=(PartitionAxis.DATA_SAMPLE, PartitionAxis.CONTEXT_SEQUENCE),
        )
        append_descriptor(
            _EVENT_RUNTIME_STATUS,
            MetricFamily.EVENT,
            None,
            owner=True,
            ownership=Ownership.EVERY_RANK,
            mask_kind=MaskKind.NONE,
            normalization_kind=NormalizationKind.NONE,
            partition_axes=(),
        )

        for observation in ("activation", "dgrad"):
            for global_layer in range(num_layers):
                for family in _CAPTURE_FAMILIES:
                    feature_sharded = family in _FEATURE_SHARDED_FAMILIES
                    mask_kind = (
                        MaskKind.SEQUENCE_PARALLEL_TOKEN
                        if self.topology.sequence_parallel and not feature_sharded
                        else MaskKind.TOKEN
                    )
                    tp_owner = (
                        feature_sharded
                        or self.topology.sequence_parallel
                        or self.topology.tensor_parallel_rank == 0
                    )
                    partition_axes = [
                        PartitionAxis.DATA_SAMPLE,
                        PartitionAxis.CONTEXT_SEQUENCE,
                        PartitionAxis.PIPELINE_LAYER,
                    ]
                    if feature_sharded:
                        partition_axes.append(PartitionAxis.TENSOR_FEATURE)
                    replication_axes = (
                        (ReplicationAxis.TENSOR,)
                        if not feature_sharded and not self.topology.sequence_parallel
                        else ()
                    )
                    append_descriptor(
                        f"{observation}/{family.value}/layer_{global_layer}",
                        family,
                        global_layer,
                        owner=global_layer in local_layers and tp_owner,
                        ownership=(
                            Ownership.PIPELINE_STAGE
                            if feature_sharded or self.topology.sequence_parallel
                            else Ownership.TENSOR_PARALLEL_RANK_ZERO
                        ),
                        mask_kind=mask_kind,
                        normalization_kind=(
                            NormalizationKind.LOSS_SCALE_AND_GLOBAL_VALID_TOKENS
                            if observation == "dgrad"
                            else NormalizationKind.NONE
                        ),
                        partition_axes=tuple(partition_axes),
                        replication_axes=replication_axes,
                    )
        return tuple(descriptors), tuple(owners)

    def _make_forward_hook(self, target: LayerCaptureTarget):
        activation_name = f"activation/{target.family.value}/layer_{target.global_layer}"
        dgrad_name = f"dgrad/{target.family.value}/layer_{target.global_layer}"

        def capture_forward(_module, _inputs, output):
            if not self._armed:
                return
            try:
                microbatch_id = get_diagnostic_microbatch_id()
                if microbatch_id is None or microbatch_id not in self._masks:
                    self._set_runtime_error()
                    return
                tensor = self._output_tensor(output)
                mask = self._mask_for_target(self._masks[microbatch_id], target)
                if is_diagnostic_recompute():
                    self._register_dgrad_hook(tensor, microbatch_id, dgrad_name, mask)
                    return

                activation_key = (microbatch_id, activation_name)
                if activation_key in self._activation_seen:
                    self.registry.mark_observation_error(
                        self._require_accumulator(), activation_name
                    )
                else:
                    self._activation_seen.add(activation_key)
                    accumulator = self._require_accumulator()
                    self.registry.mark_mask_error(accumulator, activation_name, ~mask.valid)
                    self.registry.add_masked_tensor(
                        accumulator, activation_name, tensor, mask=mask.values
                    )
                self._register_dgrad_hook(tensor, microbatch_id, dgrad_name, mask)
            except Exception:  # Hooks must convert rank-local runtime failures to packed status.
                self._record_hook_error(activation_name)

        return capture_forward

    def _register_dgrad_hook(
        self, tensor: torch.Tensor, microbatch_id: int, logical_name: str, mask: StagedTokenMask
    ) -> None:
        key = (microbatch_id, logical_name)
        if key in self._dgrad_registered:
            self.registry.mark_observation_error(self._require_accumulator(), logical_name)
            return
        if not tensor.requires_grad:
            return
        self._dgrad_registered.add(key)

        def capture_dgrad(gradient: torch.Tensor) -> torch.Tensor:
            try:
                if self._armed and key not in self._dgrad_seen:
                    self._dgrad_seen.add(key)
                    accumulator = self._require_accumulator()
                    self.registry.mark_mask_error(accumulator, logical_name, ~mask.valid)
                    self.registry.add_masked_tensor(
                        accumulator, logical_name, gradient, mask=mask.values
                    )
                elif self._armed:
                    self.registry.mark_observation_error(self._require_accumulator(), logical_name)
            except Exception:  # Hooks must convert rank-local runtime failures to packed status.
                self._record_hook_error(logical_name)
            return gradient

        tensor.register_hook(capture_dgrad)

    def _mask_for_target(
        self, staged: StagedTokenMask, target: LayerCaptureTarget
    ) -> StagedTokenMask:
        if target.token_layout != TokenLayout.TP_SEQUENCE_SHARD:
            return staged
        if not self.topology.sequence_parallel:
            return StagedTokenMask(staged.values, staged.valid & torch.zeros_like(staged.valid))
        return slice_sequence_parallel_mask(
            staged,
            tensor_parallel_rank=self.topology.tensor_parallel_rank,
            tensor_parallel_size=self.topology.tensor_parallel_size,
        )

    def _mark_incomplete_observations(self, accumulator: PackedSufficientStatistics) -> None:
        expected_microbatches = (
            self._begun if self._expected_microbatch_ids is None else self._expected_microbatch_ids
        )
        if expected_microbatches != self._begun:
            self._set_runtime_error()
        for microbatch_id in expected_microbatches:
            for target in self.targets:
                for observation, seen in (
                    ("activation", self._activation_seen),
                    ("dgrad", self._dgrad_seen),
                ):
                    logical_name = (
                        f"{observation}/{target.family.value}/layer_{target.global_layer}"
                    )
                    if (
                        self.registry.owns(logical_name)
                        and (microbatch_id, logical_name) not in seen
                    ):
                        self.registry.mark_observation_error(accumulator, logical_name)

    @staticmethod
    def _output_tensor(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError("capture target did not return a leading tensor")

    def _record_hook_error(self, logical_name: str) -> None:
        self._set_runtime_error()
        if self._accumulator is not None and not self._accumulator.reduced:
            try:
                self._accumulator.mark_arithmetic_error(logical_name)
            except Exception:
                pass

    def _set_runtime_error(self) -> None:
        self._runtime_status.copy_(
            torch.maximum(
                self._runtime_status,
                torch.full_like(self._runtime_status, Tier0Status.RUNTIME_ERROR),
            )
        )

    def _require_accumulator(self) -> PackedSufficientStatistics:
        if self._accumulator is None:
            raise RuntimeError("Tier-0 capture packs are not allocated")
        return self._accumulator
