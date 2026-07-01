# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Selected-token function response on the canonical diagnostic accumulator."""

from __future__ import annotations

import contextlib
import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.diagnostics import diagnostic_attention_observer
from megatron.core.transformer.transformer_layer import TransformerLayer

from .accumulator import PackedSlots, PackedSufficientStatistics, ReductionBinding
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


class ResponseFamily(StrEnum):
    """Fixed Tier-1 response families."""

    RESIDUAL = "residual"
    QKV = "qkv"
    ATTN_OUT = "attn_out"
    FC1 = "fc1"
    FC2 = "fc2"


RESPONSE_FAMILIES: tuple[ResponseFamily, ...] = tuple(ResponseFamily)
_FAMILY_INDEX = {family: index for index, family in enumerate(RESPONSE_FAMILIES)}
_RESIDUAL_DEPTH_NAMES = ("first", "q1", "middle", "q3", "last")
_QUANTILE_NAMES = ("p10", "p50", "p90")
_ATTENTION_METRICS = ("logit_abs", "entropy", "collapse")
_STARVATION_FAMILIES = (
    ResponseFamily.QKV,
    ResponseFamily.ATTN_OUT,
    ResponseFamily.FC1,
    ResponseFamily.FC2,
    ResponseFamily.RESIDUAL,
)
TIER1_PREFIX = "diag/v2/t1/response/"
TIER1_ATTENTION_PREFIX = "diag/v2/t1/attention/"
TIER1_KEYS: tuple[str, ...] = (
    *(
        f"{TIER1_PREFIX}residual/dy_rel/{name}"
        for name in (*_RESIDUAL_DEPTH_NAMES, *_QUANTILE_NAMES)
    ),
    *(
        f"{TIER1_PREFIX}{family}/dy_rel/{name}"
        for family in ("qkv", "attn_out", "fc1", "fc2")
        for name in _QUANTILE_NAMES
    ),
    *(f"{TIER1_PREFIX}{family.value}/starved_fraction" for family in _STARVATION_FAMILIES),
    f"{TIER1_ATTENTION_PREFIX}logit_abs_p50",
    f"{TIER1_ATTENTION_PREFIX}logit_abs_p90",
    f"{TIER1_ATTENTION_PREFIX}entropy_p10",
    f"{TIER1_ATTENTION_PREFIX}entropy_p50",
    f"{TIER1_ATTENTION_PREFIX}collapse_fraction",
)


@dataclass(frozen=True)
class ResponseHookDescriptor:
    """Bind one typed dense-MCore response hook to a global slot."""

    global_layer: int
    family: ResponseFamily
    module: torch.nn.Module = field(compare=False, repr=False)
    owner: bool
    sequence_sharded: bool
    affine_bias_output: bool

    @property
    def key(self) -> tuple[int, ResponseFamily]:
        """Return the global canonical hook key."""

        return self.global_layer, self.family


@dataclass(frozen=True)
class ResponseHookRegistration:
    """Describe one exact probe-owned hook registration."""

    descriptor: ResponseHookDescriptor
    module: torch.nn.Module
    registry_name: str
    handle_id: int
    hook: Any = field(repr=False)


def discover_response_hooks(
    models: Sequence[torch.nn.Module],
    *,
    global_layers: int,
    expected_local_layers: Sequence[int],
    tensor_parallel_rank: int,
    sequence_parallel: bool,
    layer_type: type[torch.nn.Module] = TransformerLayer,
) -> tuple[ResponseHookDescriptor, ...]:
    """Discover exact dense layer hooks and reject missing or duplicate local slots."""

    if global_layers <= 0:
        raise ValueError("Tier-1 response requires a positive global layer count")
    expected = tuple(sorted(expected_local_layers))
    if len(expected) != len(set(expected)) or any(
        not 0 <= layer < global_layers for layer in expected
    ):
        raise ValueError("expected local response layers must be unique global indices")
    descriptors: list[ResponseHookDescriptor] = []
    discovered: set[int] = set()
    for model in models:
        for layer in model.modules():
            if not isinstance(layer, layer_type):
                continue
            number = getattr(layer, "layer_number", None)
            if not isinstance(number, int):
                raise ValueError("MCore TransformerLayer lacks an integer layer_number")
            global_layer = number - 1
            if global_layer in discovered or global_layer not in expected:
                raise ValueError("response layer ownership is duplicate or unexpected")
            discovered.add(global_layer)
            try:
                modules = {
                    ResponseFamily.RESIDUAL: layer,
                    ResponseFamily.QKV: layer.self_attention.linear_qkv,
                    ResponseFamily.ATTN_OUT: layer.self_attention.linear_proj,
                    ResponseFamily.FC1: layer.mlp.linear_fc1,
                    ResponseFamily.FC2: layer.mlp.linear_fc2,
                }
            except AttributeError as error:
                raise ValueError("unsupported dense MCore response layer") from error
            for family in RESPONSE_FAMILIES:
                feature_sharded = family in (ResponseFamily.QKV, ResponseFamily.FC1)
                sequence_sharded = sequence_parallel and not feature_sharded
                descriptors.append(
                    ResponseHookDescriptor(
                        global_layer=global_layer,
                        family=family,
                        module=modules[family],
                        owner=feature_sharded or sequence_parallel or tensor_parallel_rank == 0,
                        sequence_sharded=sequence_sharded,
                        affine_bias_output=family != ResponseFamily.RESIDUAL,
                    )
                )
    if discovered != set(expected):
        raise ValueError(f"missing local response layers: {sorted(set(expected) - discovered)}")
    descriptors.sort(
        key=lambda descriptor: (descriptor.global_layer, _FAMILY_INDEX[descriptor.family])
    )
    return tuple(descriptors)


def _validate_descriptor_order(descriptors: Sequence[ResponseHookDescriptor]) -> None:
    keys = tuple(descriptor.key for descriptor in descriptors)
    if len(keys) != len(set(keys)):
        raise ValueError("response hook descriptors contain duplicate global slots")
    if keys != tuple(sorted(keys, key=lambda key: (key[0], _FAMILY_INDEX[key[1]]))):
        raise ValueError("response hook descriptors are not in canonical global slot order")


def _response_registry(
    descriptors: Sequence[ResponseHookDescriptor],
    *,
    global_layers: int,
    sequence_parallel: bool,
    attention_owner: bool,
    reduction_binding: ReductionBinding,
) -> MetricRegistry:
    _validate_descriptor_order(descriptors)
    local = {descriptor.key: descriptor for descriptor in descriptors}
    metrics: list[MetricDescriptor] = []
    owners: list[bool] = []
    family_map = {
        ResponseFamily.RESIDUAL: MetricFamily.RESIDUAL,
        ResponseFamily.QKV: MetricFamily.QKV,
        ResponseFamily.ATTN_OUT: MetricFamily.ATTN_OUT,
        ResponseFamily.FC1: MetricFamily.FC1,
        ResponseFamily.FC2: MetricFamily.FC2,
    }
    for layer in range(global_layers):
        for family in RESPONSE_FAMILIES:
            hook = local.get((layer, family))
            feature_sharded = family in (ResponseFamily.QKV, ResponseFamily.FC1)
            sequence_sharded = sequence_parallel and not feature_sharded
            partition_axes = [
                PartitionAxis.DATA_SAMPLE,
                PartitionAxis.PIPELINE_LAYER,
                PartitionAxis.CONTEXT_SEQUENCE,
            ]
            if feature_sharded:
                partition_axes.append(PartitionAxis.TENSOR_FEATURE)
            replicated = not feature_sharded and not sequence_sharded
            metrics.append(
                MetricDescriptor(
                    logical_name=f"tier1/response/layer/{layer}/{family.value}",
                    family=family_map[family],
                    global_layer=layer,
                    partition_axes=tuple(partition_axes),
                    replication_axes=(ReplicationAxis.TENSOR,) if replicated else (),
                    replication_multiplicity=1,
                    ownership=(
                        Ownership.TENSOR_PARALLEL_RANK_ZERO
                        if replicated
                        else Ownership.PIPELINE_STAGE
                    ),
                    mask_kind=MaskKind.NONE,
                    statistic_kind=StatisticKind.UPDATE,
                    denominator_kind=DenominatorKind.PRE_UPDATE_SUMSQ,
                    normalization_kind=NormalizationKind.NONE,
                    process_group_identity=reduction_binding.process_group_identity,
                    reduction_kind=reduction_binding.reduction_kind,
                    tied_owner_identity=None,
                    packed_slots=PackedSlots.for_index(len(metrics)),
                )
            )
            owners.append(bool(hook is not None and hook.owner))
    for layer in range(global_layers):
        for metric in _ATTENTION_METRICS:
            metrics.append(
                MetricDescriptor(
                    logical_name=f"tier1/attention/layer/{layer}/{metric}",
                    family=MetricFamily.ATTN_OUT,
                    global_layer=layer,
                    partition_axes=(
                        PartitionAxis.DATA_SAMPLE,
                        PartitionAxis.PIPELINE_LAYER,
                        PartitionAxis.CONTEXT_SEQUENCE,
                        PartitionAxis.TENSOR_FEATURE,
                    ),
                    replication_axes=(),
                    replication_multiplicity=1,
                    ownership=Ownership.PIPELINE_STAGE,
                    mask_kind=MaskKind.NONE,
                    statistic_kind=StatisticKind.TENSOR_MOMENTS,
                    denominator_kind=DenominatorKind.SELECTED_ELEMENTS,
                    normalization_kind=NormalizationKind.NONE,
                    process_group_identity=reduction_binding.process_group_identity,
                    reduction_kind=reduction_binding.reduction_kind,
                    tied_owner_identity=None,
                    packed_slots=PackedSlots.for_index(len(metrics)),
                )
            )
            owners.append(attention_owner and layer in {key[0] for key in local})
    return MetricRegistry(metrics, reduction_binding=reduction_binding, local_owners=owners)


@dataclass
class ResponseAccumulator:
    """Typed Tier-1 view over the canonical registry and packed accumulator."""

    registry: MetricRegistry
    statistics: PackedSufficientStatistics
    global_layers: int

    @property
    def descriptor_hash(self) -> str:
        """Return the canonical global descriptor hash."""

        return self.registry.descriptor_hash

    def finalize_local_(self) -> "ResponseAccumulator":
        """Finalize an intentional single-process accumulation."""

        self.statistics.finalize_local_()
        return self


def verify_response_descriptor_consensus(
    accumulator: ResponseAccumulator, group: object | None = None
) -> None:
    """Reject slot-count or descriptor-hash disagreement using typed collectives."""

    if not dist.is_available() or not dist.is_initialized():
        return
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend(group) == "nccl"
        else torch.device("cpu")
    )
    digest = bytes.fromhex(accumulator.descriptor_hash)
    wire = torch.tensor(
        [len(accumulator.registry.slot_names), *digest], dtype=torch.int64, device=device
    )
    minimum = wire.clone()
    maximum = wire.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(minimum, maximum):
        raise RuntimeError("Tier-1 response descriptor/slot hash mismatch")


def _canonical_output(
    output: Any, *, affine_bias_output: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if affine_bias_output:
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError("affine response expected (output, output_bias)")
            tensor, bias = output
        else:
            tensor, bias = output, None
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("affine response output must contain a tensor")
        if bias is not None and not isinstance(bias, torch.Tensor):
            raise TypeError("affine response bias must be a tensor or None")
        return tensor, bias
    tensor = output[0] if isinstance(output, tuple) else output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("residual response output must contain a tensor")
    return tensor, None


def _selected_rows(
    activation: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Select rows before bias materialization, bounding work by selected tokens."""

    boolean = mask if mask.dtype == torch.bool else mask != 0
    if activation.ndim == 2:
        if activation.shape[0] != boolean.numel():
            raise ValueError("flattened response activation does not match its mask")
        rows = activation.reshape(-1, activation.shape[-1])[boolean.reshape(-1)]
    elif activation.ndim == 3 and boolean.ndim == 2:
        if activation.shape[:2] == boolean.shape:
            aligned = activation
        elif activation.shape[:2] == tuple(reversed(boolean.shape)):
            aligned = activation.transpose(0, 1)
        else:
            raise ValueError("response activation shape does not match its mask")
        rows = aligned.reshape(-1, aligned.shape[-1])[boolean.reshape(-1)]
    else:
        raise ValueError("Tier-1 response expects 2D or [sequence,batch,hidden] output")
    return rows if bias is None else rows + bias


class FunctionResponseProbe:
    """Capture bounded selected pre rows and accumulate post deltas canonically."""

    def __init__(
        self,
        descriptors: Sequence[ResponseHookDescriptor],
        *,
        global_layers: int,
        device: torch.device | str,
        expected_hook_calls: int,
        sequence_parallel: bool = False,
        attention_owner: bool = True,
        attention_required: bool = True,
        retain_secant_endpoints: bool = False,
        reduction_binding: ReductionBinding | None = None,
        scratch_element_capacity: int = 16 * 1024,
    ) -> None:
        if expected_hook_calls < 0 or scratch_element_capacity <= 0:
            raise ValueError("response hook counts and scratch capacity must be valid")
        self.descriptors = tuple(descriptors)
        _validate_descriptor_order(self.descriptors)
        self.global_layers = global_layers
        self.device = torch.device(device)
        self.expected_hook_calls = expected_hook_calls
        self.sequence_parallel = sequence_parallel
        self.attention_required = attention_required
        self.retain_secant_endpoints = retain_secant_endpoints
        binding = reduction_binding or ReductionBinding.flat_world(None)
        self.registry = _response_registry(
            self.descriptors,
            global_layers=global_layers,
            sequence_parallel=sequence_parallel,
            attention_owner=attention_owner,
            reduction_binding=binding,
        )
        statistics = PackedSufficientStatistics(
            self.registry.slot_names,
            self.device,
            descriptor_hash=self.registry.descriptor_hash,
            reduction_binding=binding,
            scratch_element_capacity=scratch_element_capacity,
        )
        self.accumulator = ResponseAccumulator(self.registry, statistics, global_layers)
        self._pre_rows: dict[tuple[int, ResponseFamily], list[torch.Tensor]] = {}
        self._pre_calls: dict[tuple[int, ResponseFamily], int] = {}
        self._post_calls: dict[tuple[int, ResponseFamily], int] = {}
        self._endpoint_calls: dict[str, dict[tuple[int, ResponseFamily], int]] = {
            "post_repeat": {},
            "midpoint": {},
        }
        self._secant_rows: dict[
            str, dict[tuple[int, ResponseFamily], list[torch.Tensor]]
        ] = {
            phase: {} for phase in ("pre", "post", "post_repeat", "midpoint")
        }
        self._attention_calls: dict[int, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._phase: str | None = None
        self._full_mask: torch.Tensor | None = None
        self._sequence_mask: torch.Tensor | None = None
        self._finalized = False
        self._selected_row_capacity: int | None = None
        self._response_widths: tuple[tuple[ResponseFamily, int], ...] | None = None
        self._response_dtype: torch.dtype | None = None
        self._attention_heads: int | None = None
        self._attention_key_length: int | None = None
        self._preflight_binding: tuple[Any, ...] | None = None

    def bind_preflight(
        self,
        *,
        selected_row_capacity: int,
        response_widths: Mapping[ResponseFamily, int],
        response_dtype: torch.dtype,
        attention_heads: int,
        attention_key_length: int,
    ) -> None:
        """Bind immutable live-response limits before either replay schedule."""

        if (
            selected_row_capacity <= 0
            or set(response_widths) != set(RESPONSE_FAMILIES)
            or any(width <= 0 for width in response_widths.values())
            or not isinstance(response_dtype, torch.dtype)
            or attention_heads <= 0
            or attention_key_length <= 0
        ):
            raise ValueError("response preflight limits are invalid")
        binding = (
            selected_row_capacity,
            tuple((family, response_widths[family]) for family in RESPONSE_FAMILIES),
            response_dtype,
            attention_heads,
            attention_key_length,
        )
        if self._preflight_binding is not None and self._preflight_binding != binding:
            raise RuntimeError("response probe preflight binding changed")
        (
            self._selected_row_capacity,
            self._response_widths,
            self._response_dtype,
            self._attention_heads,
            self._attention_key_length,
        ) = binding
        self._preflight_binding = binding

    def validate_preflight_binding(self) -> None:
        """Require all response allocation dimensions to be engine-bound."""

        if (
            self._selected_row_capacity is None
            or self._response_widths is None
            or self._response_dtype is None
            or self._attention_heads is None
            or self._attention_key_length is None
        ):
            raise RuntimeError("response probe is not bound to memory preflight")
        current = (
            self._selected_row_capacity,
            self._response_widths,
            self._response_dtype,
            self._attention_heads,
            self._attention_key_length,
        )
        if current != self._preflight_binding:
            raise RuntimeError("response probe changed after memory preflight")

    @property
    def descriptor_hash(self) -> str:
        """Return the response registry descriptor hash."""

        return self.registry.descriptor_hash

    def set_masks(
        self, full_mask: torch.Tensor, *, sequence_parallel_mask: torch.Tensor | None = None
    ) -> None:
        """Set CP-local and optional SP-local masks for one schedule microbatch."""

        if full_mask.device != self.device:
            raise ValueError("response mask is on the wrong device")
        if sequence_parallel_mask is not None and sequence_parallel_mask.device != self.device:
            raise ValueError("sequence-parallel response mask is on the wrong device")
        self._full_mask = full_mask
        self._sequence_mask = sequence_parallel_mask

    @contextlib.contextmanager
    def capture_pre(self) -> Iterator[tuple[ResponseHookRegistration, ...]]:
        """Install hooks for the pre-update forward-only schedule."""

        with self._capture("pre") as registrations:
            yield registrations

    @contextlib.contextmanager
    def capture_post(self) -> Iterator[tuple[ResponseHookRegistration, ...]]:
        """Install hooks for the post-update forward-only schedule."""

        with self._capture("post") as registrations:
            yield registrations

    @contextlib.contextmanager
    def capture_endpoint(
        self, phase: str
    ) -> Iterator[tuple[ResponseHookRegistration, ...]]:
        """Install the same bounded hooks for one named replay endpoint.

        ``pre`` and ``post`` retain their Tier-1 semantics. ``post_repeat`` and
        ``midpoint`` are admitted only when the probe was constructed for a
        Tier-2 secant event.
        """

        if phase not in ("pre", "post", "post_repeat", "midpoint"):
            raise ValueError(f"unknown response endpoint phase: {phase}")
        if phase in ("post_repeat", "midpoint") and not self.retain_secant_endpoints:
            raise RuntimeError("additional response endpoints require Tier-2 retention")
        with self._capture(phase) as registrations:
            yield registrations

    @contextlib.contextmanager
    def _capture(self, phase: str) -> Iterator[tuple[ResponseHookRegistration, ...]]:
        if self._phase is not None or self._finalized:
            raise RuntimeError("response capture phases cannot overlap or follow finalize")
        self._phase = phase
        try:
            registrations: list[ResponseHookRegistration] = []
            for descriptor in self.descriptors:
                hook = self._make_hook(descriptor)
                handle = descriptor.module.register_forward_hook(hook)
                self._handles.append(handle)
                registrations.append(
                    ResponseHookRegistration(
                        descriptor=descriptor,
                        module=descriptor.module,
                        registry_name="_forward_hooks",
                        handle_id=handle.id,
                        hook=hook,
                    )
                )
            observer_context = (
                diagnostic_attention_observer(self.observe_attention)
                if phase == "post"
                else contextlib.nullcontext()
            )
            with observer_context:
                yield tuple(registrations)
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._phase = None
            self._full_mask = None
            self._sequence_mask = None

    def _make_hook(self, descriptor: ResponseHookDescriptor):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            try:
                self._observe(descriptor, output)
            except (RuntimeError, TypeError, ValueError):
                self.registry.mark_observation_error(
                    self.accumulator.statistics, self.registry.slot_names[self._slot(descriptor)]
                )
                raise

        return hook

    def _slot(self, descriptor: ResponseHookDescriptor) -> int:
        return descriptor.global_layer * len(RESPONSE_FAMILIES) + _FAMILY_INDEX[descriptor.family]

    def _attention_slot(self, layer: int, metric: str) -> int:
        return (
            self.global_layers * len(RESPONSE_FAMILIES)
            + layer * len(_ATTENTION_METRICS)
            + _ATTENTION_METRICS.index(metric)
        )

    def observe_attention(
        self,
        layer: int,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        *,
        collapse_threshold: float = 0.9,
    ) -> None:
        """Accumulate selected-query attention statistics on the canonical pack.

        The integration hook calls this with pre-softmax logits and normalized
        probabilities shaped ``[batch, heads, query, key]`` during the post
        replay.  Quantiles are later taken across pooled per-layer means; no
        response value is relabeled as an attention statistic.
        """

        if self._phase != "post":
            raise RuntimeError("attention observations belong to post replay")
        if not 0 <= layer < self.global_layers:
            raise ValueError("attention observation has an invalid global layer")
        if (
            logits.shape != probabilities.shape
            or logits.ndim != 4
            or logits.device != self.device
            or probabilities.device != self.device
            or logits.dtype != self._response_dtype
            or probabilities.dtype != self._response_dtype
        ):
            raise ValueError("attention logits/probabilities have an invalid layout")
        if (
            logits.shape[1] != self._attention_heads
            or logits.shape[3] != self._attention_key_length
        ):
            raise ValueError("attention response shape disagrees with memory preflight")
        if not 0 < collapse_threshold <= 1:
            raise ValueError("attention collapse threshold must be in (0, 1]")
        mask = self._full_mask
        if mask is None or mask.shape != (logits.shape[0], logits.shape[2]):
            raise ValueError("attention observation does not match its selected-token mask")
        selected = mask if mask.dtype == torch.bool else mask != 0
        selected_logits = logits.permute(0, 2, 1, 3)[selected]
        selected_probabilities = probabilities.permute(0, 2, 1, 3)[selected]
        if selected_logits.shape[0] > self._selected_row_capacity:
            raise ValueError("attention selected rows exceed memory preflight")
        logit_abs = selected_logits.abs().mean(dim=-1)
        entropy = -(
            selected_probabilities
            * selected_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1)
        collapse = (selected_probabilities.amax(dim=-1) >= collapse_threshold).to(
            dtype=torch.float32
        )
        for metric, values in (
            ("logit_abs", logit_abs),
            ("entropy", entropy),
            ("collapse", collapse),
        ):
            self.registry.add_masked_tensor(
                self.accumulator.statistics,
                self.registry.slot_names[self._attention_slot(layer, metric)],
                values,
            )
        self._attention_calls[layer] = self._attention_calls.get(layer, 0) + 1

    def _observe(self, descriptor: ResponseHookDescriptor, output: Any) -> None:
        if self._phase not in ("pre", "post", "post_repeat", "midpoint"):
            raise RuntimeError("response hook fired outside replay capture")
        self.validate_preflight_binding()
        if self._phase == "pre":
            calls = self._pre_calls
        elif self._phase == "post":
            calls = self._post_calls
        else:
            calls = self._endpoint_calls[self._phase]
        if calls.get(descriptor.key, 0) >= self.expected_hook_calls:
            raise RuntimeError("response hook exceeded its preflight cardinality")
        mask = self._sequence_mask if descriptor.sequence_sharded else self._full_mask
        if mask is None:
            raise ValueError("response schedule did not supply the required mask")
        activation, bias = _canonical_output(
            output, affine_bias_output=descriptor.affine_bias_output
        )
        widths = dict(self._response_widths)
        if (
            activation.device != self.device
            or activation.dtype != self._response_dtype
            or activation.shape[-1] != widths[descriptor.family]
        ):
            raise ValueError("response activation disagrees with memory preflight")
        if bias is not None and (
            bias.device != self.device
            or bias.dtype != self._response_dtype
            or bias.ndim != 1
            or bias.shape[0] != widths[descriptor.family]
        ):
            raise ValueError("response bias disagrees with memory preflight")
        rows = _selected_rows(activation, mask, bias)
        calls[descriptor.key] = calls.get(descriptor.key, 0) + 1
        if not descriptor.owner:
            return
        if self._phase == "pre":
            retained_rows = sum(value.shape[0] for value in self._pre_rows.get(descriptor.key, ()))
            if retained_rows + rows.shape[0] > self._selected_row_capacity:
                raise RuntimeError("response rows exceed their preallocated retention cap")
            retained = rows.detach().clone()
            self._pre_rows.setdefault(descriptor.key, []).append(retained)
            if self.retain_secant_endpoints:
                self._secant_rows["pre"].setdefault(descriptor.key, []).append(retained)
            return
        if self._phase in ("post_repeat", "midpoint"):
            self._secant_rows[self._phase].setdefault(descriptor.key, []).append(
                rows.detach().clone()
            )
            return
        before_values = self._pre_rows.get(descriptor.key, [])
        before = before_values.pop(0) if before_values else None
        if not before_values:
            self._pre_rows.pop(descriptor.key, None)
        logical_name = self.registry.slot_names[self._slot(descriptor)]
        if before is None or before.shape != rows.shape:
            self.registry.mark_observation_error(self.accumulator.statistics, logical_name)
            return
        post = rows.detach().clone() if self.retain_secant_endpoints else rows.detach()
        self.registry.add_update(self.accumulator.statistics, logical_name, before, post)
        if self.retain_secant_endpoints:
            self._secant_rows["post"].setdefault(descriptor.key, []).append(post)

    def finalize(self) -> ResponseAccumulator:
        """Mark exact hook-cardinality errors and return neutral global slots."""

        if self._phase is not None or self._finalized:
            raise RuntimeError("response probe finalizes exactly once after both schedules")
        for descriptor in self.descriptors:
            if (
                self._pre_calls.get(descriptor.key, 0) != self.expected_hook_calls
                or self._post_calls.get(descriptor.key, 0) != self.expected_hook_calls
                or descriptor.key in self._pre_rows
            ):
                self.registry.mark_observation_error(
                    self.accumulator.statistics, self.registry.slot_names[self._slot(descriptor)]
                )
            if self.retain_secant_endpoints and any(
                self._endpoint_calls[phase].get(descriptor.key, 0)
                != self.expected_hook_calls
                for phase in ("post_repeat", "midpoint")
            ):
                self.registry.mark_observation_error(
                    self.accumulator.statistics,
                    self.registry.slot_names[self._slot(descriptor)],
                )
        if self.attention_required:
            local_layers = {descriptor.global_layer for descriptor in self.descriptors}
            for layer in local_layers:
                if self._attention_calls.get(layer, 0) != self.expected_hook_calls:
                    for metric in _ATTENTION_METRICS:
                        self.registry.mark_observation_error(
                            self.accumulator.statistics,
                            self.registry.slot_names[self._attention_slot(layer, metric)],
                        )
        self._pre_rows.clear()
        self._finalized = True
        return self.accumulator

    def secant_endpoint_rows(
        self, phase: str, key: tuple[int, ResponseFamily]
    ) -> tuple[torch.Tensor, ...]:
        """Return one immutable selected-row endpoint sequence for Tier 2."""

        if not self.retain_secant_endpoints or phase not in self._secant_rows:
            raise RuntimeError("secant endpoint rows are unavailable")
        return tuple(self._secant_rows[phase].get(key, ()))

    def reset_event(self, *, expected_hook_calls: int) -> None:
        """Reset preallocated packs and all event-local endpoint references."""

        if self._phase is not None or self._handles:
            raise RuntimeError("cannot reset an active response capture")
        if expected_hook_calls <= 0:
            raise ValueError("response events require a positive replay count")
        self.expected_hook_calls = expected_hook_calls
        self.accumulator.statistics.reset_()
        self._pre_rows.clear()
        self._pre_calls.clear()
        self._post_calls.clear()
        for calls in self._endpoint_calls.values():
            calls.clear()
        for rows in self._secant_rows.values():
            rows.clear()
        self._attention_calls.clear()
        self._full_mask = None
        self._sequence_mask = None
        self._finalized = False

    @property
    def retained_pre_bytes(self) -> int:
        """Return exact currently retained selected-row storage."""

        return sum(
            value.numel() * value.element_size()
            for values in self._pre_rows.values()
            for value in values
        )

    @property
    def maximum_accumulator_scratch_bytes(self) -> int:
        """Return canonical bounded FP32/FP64 update scratch."""

        return self.accumulator.statistics.maximum_scratch_bytes

    @property
    def packed_statistics_bytes(self) -> int:
        """Return bytes in the actual canonical persistent packs."""

        statistics = self.accumulator.statistics
        return statistics.sum_pack.nbytes + statistics.max_pack.nbytes + statistics.min_pack.nbytes

    @property
    def reduction_arena_bytes(self) -> int:
        """Return bytes in the actual canonical temporary reduction arena."""

        return PackedSufficientStatistics.reduction_arena_bytes((self.accumulator.statistics,))

    def release(self) -> None:
        """Remove temporary hooks and release all retained response rows."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pre_rows.clear()
        for rows in self._secant_rows.values():
            rows.clear()
        self._full_mask = None
        self._sequence_mask = None
        self._phase = None


@dataclass(frozen=True)
class FunctionResponseResult:
    """All-layer pooled response values and canonical validity."""

    dy_rel: torch.Tensor
    valid: torch.Tensor
    count: torch.Tensor
    delta_square_sum: torch.Tensor

    @classmethod
    def from_reduced(cls, accumulator: ResponseAccumulator) -> "FunctionResponseResult":
        """Derive ratios only after canonical packed reduction/finalization."""

        if not accumulator.statistics.reduced:
            raise RuntimeError("Tier-1 response statistics must be reduced before derivation")
        values = torch.empty(
            (accumulator.global_layers, len(RESPONSE_FAMILIES)),
            dtype=torch.float64,
            device=accumulator.statistics.sum_pack.device,
        )
        valid = torch.empty_like(values, dtype=torch.bool)
        count = torch.empty_like(values)
        delta = torch.empty_like(values)
        for layer in range(accumulator.global_layers):
            for family in range(len(RESPONSE_FAMILIES)):
                slot = layer * len(RESPONSE_FAMILIES) + family
                statistic = accumulator.statistics.relative_rms(slot)
                packed = accumulator.statistics.slots(slot)
                values[layer, family] = statistic.value
                slot_valid = statistic.valid & (
                    accumulator.statistics.sum_pack[packed.observation_error] == 0
                )
                valid[layer, family] = slot_valid
                values[layer, family] = torch.where(
                    slot_valid, statistic.value, torch.full_like(statistic.value, torch.nan)
                )
                count[layer, family] = accumulator.statistics.sum_pack[packed.count]
                delta[layer, family] = accumulator.statistics.sum_pack[packed.lhs_sumsq]
        return cls(values, valid, count, delta)


def derive_tier1_summaries(accumulator: ResponseAccumulator) -> dict[str, torch.Tensor]:
    """Derive the exact fixed 30-key Tier-1 payload from pooled sufficient sums."""

    result = FunctionResponseResult.from_reduced(accumulator)
    values_by_key: dict[str, torch.Tensor] = {}
    quantiles = torch.tensor((0.1, 0.5, 0.9), dtype=torch.float64, device=result.dy_rel.device)
    for family in RESPONSE_FAMILIES:
        family_index = _FAMILY_INDEX[family]
        response = result.dy_rel[:, family_index]
        validity = result.valid[:, family_index]
        prefix = f"{TIER1_PREFIX}{family.value}"
        if family == ResponseFamily.RESIDUAL:
            last = response.numel() - 1
            anchors = (
                0,
                math.floor(last * 0.25 + 0.5),
                math.floor(last * 0.5 + 0.5),
                math.floor(last * 0.75 + 0.5),
                last,
            )
            for name, index in zip(_RESIDUAL_DEPTH_NAMES, anchors, strict=True):
                values_by_key[f"{prefix}/dy_rel/{name}"] = response[index]
        values = torch.nanquantile(response, quantiles)
        for name, value in zip(_QUANTILE_NAMES, values, strict=True):
            values_by_key[f"{prefix}/dy_rel/{name}"] = value
        contributors = validity
        contributor_count = contributors.sum(dtype=torch.float64)
        values_by_key[f"{prefix}/starved_fraction"] = torch.where(
            contributor_count > 0,
            ((result.delta_square_sum[:, family_index] == 0) & contributors).sum(
                dtype=torch.float64
            )
            / contributor_count,
            torch.full((), torch.nan, dtype=torch.float64, device=response.device),
        )
    attention_means = torch.full(
        (accumulator.global_layers, len(_ATTENTION_METRICS)),
        torch.nan,
        dtype=torch.float64,
        device=result.dy_rel.device,
    )
    attention_valid = torch.zeros_like(attention_means, dtype=torch.bool)
    attention_sums = torch.zeros_like(attention_means)
    attention_counts = torch.zeros_like(attention_means)
    base = accumulator.global_layers * len(RESPONSE_FAMILIES)
    for layer in range(accumulator.global_layers):
        for metric_index in range(len(_ATTENTION_METRICS)):
            slot = base + layer * len(_ATTENTION_METRICS) + metric_index
            statistic = accumulator.statistics.mean(slot)
            packed = accumulator.statistics.slots(slot)
            valid = statistic.valid & (
                accumulator.statistics.sum_pack[packed.observation_error] == 0
            )
            attention_valid[layer, metric_index] = valid
            attention_means[layer, metric_index] = torch.where(
                valid, statistic.value, torch.full_like(statistic.value, torch.nan)
            )
            attention_sums[layer, metric_index] = accumulator.statistics.sum_pack[packed.sum]
            attention_counts[layer, metric_index] = accumulator.statistics.sum_pack[packed.count]
    for metric_index, names, quantiles_requested in (
        (0, ("logit_abs_p50", "logit_abs_p90"), (0.5, 0.9)),
        (1, ("entropy_p10", "entropy_p50"), (0.1, 0.5)),
    ):
        values = torch.nanquantile(
            attention_means[:, metric_index],
            torch.tensor(quantiles_requested, dtype=torch.float64, device=attention_means.device),
        )
        for name, value in zip(names, values, strict=True):
            values_by_key[f"{TIER1_ATTENTION_PREFIX}{name}"] = value
    collapse_valid = attention_valid[:, 2]
    collapse_count = torch.where(
        collapse_valid, attention_counts[:, 2], torch.zeros_like(attention_counts[:, 2])
    ).sum()
    collapse_sum = torch.where(
        collapse_valid, attention_sums[:, 2], torch.zeros_like(attention_sums[:, 2])
    ).sum()
    values_by_key[f"{TIER1_ATTENTION_PREFIX}collapse_fraction"] = torch.where(
        collapse_count > 0,
        collapse_sum / collapse_count,
        torch.full((), torch.nan, dtype=torch.float64, device=result.dy_rel.device),
    )
    payload = {key: values_by_key[key] for key in TIER1_KEYS}
    if len(set(payload)) != 30:
        raise RuntimeError("derived Tier-1 payload does not match the canonical 30-key list")
    return payload


def descriptor_fingerprint(descriptors: Sequence[ResponseHookDescriptor]) -> str:
    """Hash local canonical hook identities for preflight diagnostics."""

    _validate_descriptor_order(descriptors)
    digest = hashlib.sha256()
    for descriptor in descriptors:
        digest.update(f"{descriptor.global_layer}:{descriptor.family.value}".encode())
    return digest.hexdigest()
