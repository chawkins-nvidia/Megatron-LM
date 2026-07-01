# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded Tier-1 replay primitives with no training-loop integration.

The core in this module is deliberately additive.  A later heartbeat change can
bind it to the application batch function and optimizer boundary without
changing Tier-0 registry, accumulator, cadence, or artifact ownership.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum, StrEnum
from typing import Any, Protocol

import numpy as np
import torch
import torch.distributed as dist

from megatron.training.datasets.data_samplers import SamplerIssuedIndex

from .accumulator import PackedSlots, PackedSufficientStatistics

SAMPLE_INDEX_FIELD = "__diag_sample_index"
SAMPLE_EPOCH_FIELD = "__diag_sample_epoch"
DIAGNOSTIC_MASK_FIELD = "__diag_token_mask"
_RESERVED_FIELDS = frozenset((SAMPLE_INDEX_FIELD, SAMPLE_EPOCH_FIELD))
_MODEL_FIELDS: tuple[tuple[str, torch.dtype], ...] = (
    ("tokens", torch.int64),
    ("labels", torch.int64),
    ("loss_mask", torch.float32),
    ("position_ids", torch.int64),
)
_CAPACITY_LIMIT = (1 << 63) - 1
_PLAN_MAGIC = 0x5449455231504C4E
_PLAN_VERSION = 2


def _hash64(*values: int) -> int:
    payload = b"".join(int(value).to_bytes(8, "little", signed=True) for value in values)
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _systematic_positions(total: int, selected: int, offset: int) -> frozenset[int]:
    """Return one cyclic fixed grid; exhaustive offsets give exact inclusion."""

    if total <= 0 or not 0 < selected <= total or not 0 <= offset < total:
        raise ValueError("systematic selection dimensions are invalid")
    return frozenset((offset + (index * total) // selected) % total for index in range(selected))


@dataclass(frozen=True, order=True)
class SampleId:
    """Stable sampler-issued sample identity."""

    epoch: int
    sampler_index: int


@dataclass(frozen=True, order=True)
class TokenId:
    """Stable logical token identity before CP or SP slicing."""

    sample: SampleId
    sequence_column: int


class StableSampleDataset:
    """Add immutable sampler identity fields to mapping dataset samples."""

    accepts_sampler_issued_identity = True

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: SamplerIssuedIndex | int) -> dict[str, Any]:
        if not isinstance(index, SamplerIssuedIndex):
            raise TypeError("StableSampleDataset requires a sampler-issued identity")
        source_index = (
            index
            if getattr(self.dataset, "accepts_sampler_issued_identity", False)
            else index.sampler_index
        )
        sample = self.dataset[source_index]
        if not isinstance(sample, Mapping):
            raise TypeError("Tier-1 stable identity requires mapping samples")
        result = dict(sample)
        result[SAMPLE_INDEX_FIELD] = torch.tensor(index.sampler_index, dtype=torch.int64)
        result[SAMPLE_EPOCH_FIELD] = torch.tensor(index.epoch, dtype=torch.int64)
        return result


@dataclass(frozen=True)
class RecordedSample:
    """Reference one lane in an unchanged raw CPU training batch."""

    sample_id: SampleId
    batch: Mapping[str, Any]
    lane: int
    batch_size: int

    def row(self, key: str) -> Any:
        """Return one sample lane without cloning its storage."""

        value = self.batch[key]
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.batch_size:
            return value[self.lane]
        return value

    @property
    def loss_mask(self) -> torch.Tensor:
        """Return this sample's full-sequence loss mask."""

        value = self.row("loss_mask")
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError("Tier-1 replay requires one-dimensional sample loss masks")
        return value

    @property
    def valid_columns(self) -> tuple[int, ...]:
        """Return strictly positive loss-mask columns."""

        return tuple(
            int(column) for column in torch.nonzero(self.loss_mask > 0, as_tuple=False).flatten()
        )


@dataclass(frozen=True)
class RecordedBatch:
    """Retain references to one raw CPU batch and its stable identities."""

    raw: Mapping[str, Any]
    samples: tuple[RecordedSample, ...]

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "RecordedBatch":
        """Validate a raw pre-broadcast batch and retain only references."""

        expected_fields = {name for name, _dtype in _MODEL_FIELDS} | _RESERVED_FIELDS
        if set(raw) != expected_fields:
            raise ValueError(
                "Tier-1 replay requires exactly the fixed tokens, labels, loss_mask, "
                "position_ids, and identity fields"
            )
        for name in _RESERVED_FIELDS:
            value = raw[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.int64
            ):
                raise ValueError("replay identity fields must be CPU int64 tensors")
        indices = raw[SAMPLE_INDEX_FIELD].view(-1)
        epochs = raw[SAMPLE_EPOCH_FIELD].view(-1)
        if indices.numel() == 0 or indices.numel() != epochs.numel():
            raise ValueError("sample identity fields must have equal nonzero length")
        batch_size = indices.numel()
        sequence_length: int | None = None
        for name, dtype in _MODEL_FIELDS:
            value = raw[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"replay field {name!r} must be a tensor")
            if value.device.type != "cpu" or value.dtype != dtype or value.ndim != 2:
                raise ValueError(
                    f"replay field {name!r} must be a two-dimensional CPU {dtype} tensor"
                )
            if value.shape[0] != batch_size:
                raise ValueError(f"replay field {name!r} has the wrong batch dimension")
            if sequence_length is None:
                sequence_length = value.shape[1]
            elif value.shape[1] != sequence_length:
                raise ValueError("replay model fields must have one sequence length")
        samples = tuple(
            RecordedSample(SampleId(int(epochs[lane]), int(indices[lane])), raw, lane, batch_size)
            for lane in range(batch_size)
        )
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("stable sample identities must be unique within a raw batch")
        return cls(raw, samples)

    @property
    def tensor_bytes(self) -> int:
        """Return referenced raw tensor bytes for host-memory accounting."""

        return sum(
            value.numel() * value.element_size()
            for value in self.raw.values()
            if isinstance(value, torch.Tensor)
        )


class ReplayBatchRecorder(Iterator[Mapping[str, Any]]):
    """Read through a training iterator while recording raw batch references."""

    def __init__(
        self, source: Iterator[Mapping[str, Any]], *, maximum_batches: int, maximum_host_bytes: int
    ) -> None:
        if maximum_batches <= 0 or not 0 <= maximum_host_bytes <= _CAPACITY_LIMIT:
            raise ValueError("replay recorder caps are invalid")
        self.source = source
        self.maximum_batches = maximum_batches
        self.maximum_host_bytes = maximum_host_bytes
        self.recorded: list[RecordedBatch] = []
        self._host_capture_bytes = 0

    def __iter__(self) -> "ReplayBatchRecorder":
        return self

    def __next__(self) -> Mapping[str, Any]:
        raw = next(self.source)
        batch = RecordedBatch.from_raw(raw)
        retained_bytes = _checked_sum(self._host_capture_bytes, batch.tensor_bytes)
        if len(self.recorded) >= self.maximum_batches:
            raise ReplayPreflightError("replay recorder batch cap exceeded")
        if retained_bytes > self.maximum_host_bytes:
            raise ReplayPreflightError("replay recorder byte cap exceeded")
        self.recorded.append(batch)
        self._host_capture_bytes = retained_bytes
        return raw

    @property
    def host_capture_bytes(self) -> int:
        """Return event-local referenced host tensor bytes."""

        return self._host_capture_bytes

    def clear(self) -> None:
        """Release references without advancing or replacing the source iterator."""

        self.recorded.clear()
        self._host_capture_bytes = 0


@dataclass(frozen=True)
class SamplePopulation:
    """Stable identity and number of selectable token positions."""

    sample_id: SampleId
    valid_count: int


def local_sample_populations(recorded: Sequence[RecordedBatch]) -> tuple[SamplePopulation, ...]:
    """Return validated local sample populations."""

    samples = tuple(sample for batch in recorded for sample in batch.samples)
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("recorded sample identities must be unique on a data owner")
    return tuple(
        SamplePopulation(sample.sample_id, len(sample.valid_columns)) for sample in samples
    )


@dataclass(frozen=True)
class CollectiveBinding:
    """Bind a pre-existing process group to its declared fixed size."""

    identity: str
    group: object | None
    size: int

    def validate(self) -> None:
        """Reject a topology that is not an existing CPU/Gloo control group."""

        if not self.identity or self.size <= 0:
            raise ValueError("collective bindings require an identity and positive size")
        if dist.is_available() and dist.is_initialized():
            if dist.get_world_size(self.group) != self.size:
                raise ValueError(f"{self.identity} binding size disagrees with its process group")
            if str(dist.get_backend(self.group)).lower() != "gloo":
                raise RuntimeError(
                    f"{self.identity} control binding must use a pre-existing Gloo group"
                )
        elif self.size != 1 or self.group is not None:
            raise RuntimeError("multi-rank bindings require initialized torch.distributed")

    def require_same(self, other: "CollectiveBinding") -> None:
        """Require control status and payload wires to use one verified group."""

        if (
            self.identity != other.identity
            or self.group is not other.group
            or self.size != other.size
        ):
            raise ValueError("control payload and readiness bindings disagree")


class ReplayPreflightError(RuntimeError):
    """Raised by a globally settled failure before schedule/P2P entry."""


class ReadinessConsensus:
    """Own a status scalar allocated before any replay-risky phase."""

    def __init__(self, binding: CollectiveBinding, device: torch.device | str = "cpu") -> None:
        binding.validate()
        if torch.device(device).type != "cpu":
            raise RuntimeError("readiness consensus requires a CPU/Gloo control wire")
        self.binding = binding
        self.status = torch.ones(1, dtype=torch.int32, device=device)

    def settle(self, error: BaseException | None, phase: str) -> None:
        """Make one local preflight result identical across the bound group."""

        self.status.fill_(0 if error is not None else 1)
        if self.binding.size > 1:
            dist.all_reduce(self.status, op=dist.ReduceOp.MIN, group=self.binding.group)
        # This is a CPU scalar read.  Control wires never reside on CUDA, so it
        # cannot introduce event-path device synchronization.
        if not bool(int(self.status[0])):
            raise ReplayPreflightError(f"Tier-1 {phase} failed collectively") from error


class PopulationCollectiveWorkspace:
    """Preallocate fixed typed DP population and schedule-count buffers."""

    def __init__(
        self,
        binding: CollectiveBinding,
        *,
        maximum_local_samples: int,
        device: torch.device | str = "cpu",
    ) -> None:
        binding.validate()
        if torch.device(device).type != "cpu":
            raise RuntimeError("population exchange requires CPU/Gloo control wires")
        if maximum_local_samples <= 0:
            raise ValueError("population workspace needs a positive local sample cap")
        self.binding = binding
        self.maximum_local_samples = maximum_local_samples
        self.local_count = torch.zeros(1, dtype=torch.int64, device=device)
        self.counts = torch.zeros(binding.size, dtype=torch.int64, device=device)
        self.local = torch.full((maximum_local_samples, 3), -1, dtype=torch.int64, device=device)
        self.gathered = torch.full(
            (binding.size * maximum_local_samples, 3), -1, dtype=torch.int64, device=device
        )
        self.schedule_count = torch.zeros(1, dtype=torch.int64, device=device)

    def gather(
        self, local: Sequence[SamplePopulation], readiness: ReadinessConsensus
    ) -> tuple[SamplePopulation, ...]:
        """Exchange fixed sample metadata and reject duplicate global identities."""

        self.binding.require_same(readiness.binding)
        error: BaseException | None = None
        try:
            if len(local) > self.maximum_local_samples:
                raise ValueError("local sample population exceeds its fixed cap")
            self.local_count.fill_(len(local))
            self.local.fill_(-1)
            for index, population in enumerate(local):
                self.local[index] = torch.tensor(
                    (
                        population.sample_id.epoch,
                        population.sample_id.sampler_index,
                        population.valid_count,
                    ),
                    dtype=torch.int64,
                    device=self.local.device,
                )
        except BaseException as caught:
            error = caught
        readiness.settle(error, "DP population packing")
        if self.binding.size > 1:
            dist.all_gather_into_tensor(self.counts, self.local_count, group=self.binding.group)
            dist.all_gather_into_tensor(self.gathered, self.local, group=self.binding.group)
        else:
            self.counts.copy_(self.local_count)
            self.gathered.copy_(self.local)
        populations = []
        for rank, count_value in enumerate(self.counts):
            count = int(count_value)
            start = rank * self.maximum_local_samples
            for row in self.gathered[start : start + count]:
                epoch, sample, valid = (int(value) for value in row)
                populations.append(SamplePopulation(SampleId(epoch, sample), valid))
        if len({population.sample_id for population in populations}) != len(populations):
            raise ValueError("DP population contains duplicate stable sample identities")
        return tuple(populations)

    def maximum_schedule_count(self, local_count: int) -> int:
        """Return the fixed DP maximum schedule length."""

        if local_count < 0:
            raise ValueError("local replay count cannot be negative")
        self.schedule_count.fill_(local_count)
        if self.binding.size > 1:
            dist.all_reduce(self.schedule_count, op=dist.ReduceOp.MAX, group=self.binding.group)
        return int(self.schedule_count[0])


@dataclass
class ReplayMicrobatch:
    """One fixed-shape replay microbatch on a TP source rank."""

    data: dict[str, Any]
    sample_ids: tuple[SampleId, ...]

    def model_batch(self) -> dict[str, Any]:
        """Return model-path fields, including the diagnostic mask."""

        return {key: value for key, value in self.data.items() if key not in _RESERVED_FIELDS}

    @property
    def tensor_bytes(self) -> int:
        """Return exact tensor bytes retained by this batch."""

        return sum(
            value.numel() * value.element_size()
            for value in self.data.values()
            if isinstance(value, torch.Tensor)
        )


@dataclass(frozen=True)
class ReplayPlanMetadata:
    """Rank-independent, fixed-capacity replay schedule metadata."""

    micro_batch_size: int
    sequence_length: int
    num_microbatches: int
    global_selected_tokens: int
    selected_tokens: tuple[TokenId, ...]
    ordered_sample_ids: tuple[SampleId, ...]
    diagnostic_masks: tuple[tuple[bool, ...], ...]

    @property
    def descriptor_hash(self) -> str:
        """Return the canonical metadata/slot-order hash."""

        digest = hashlib.sha256()
        for value in (
            self.micro_batch_size,
            self.sequence_length,
            self.num_microbatches,
            self.global_selected_tokens,
        ):
            digest.update(value.to_bytes(8, "little", signed=True))
        for token in self.selected_tokens:
            for value in (token.sample.epoch, token.sample.sampler_index, token.sequence_column):
                digest.update(value.to_bytes(8, "little", signed=True))
        for sample in self.ordered_sample_ids:
            digest.update(sample.epoch.to_bytes(8, "little", signed=True))
            digest.update(sample.sampler_index.to_bytes(8, "little", signed=True))
        for mask in self.diagnostic_masks:
            digest.update(bytes(mask))
        return digest.hexdigest()


@dataclass
class ReplayPlan:
    """Bounded replay batches plus metadata available on every TP rank."""

    metadata: ReplayPlanMetadata
    microbatches: list[ReplayMicrobatch]
    source_rank: bool
    _filler: RecordedSample | None = field(default=None, repr=False)

    @property
    def num_microbatches(self) -> int:
        """Return the globally fixed schedule count."""

        return self.metadata.num_microbatches

    @property
    def tensor_bytes(self) -> int:
        """Return reconstructed source-plan tensor bytes."""

        return sum(batch.tensor_bytes for batch in self.microbatches)

    def diagnostic_mask(self, microbatch: int, device: torch.device | str) -> torch.Tensor:
        """Construct one fixed mask on any TP or PP rank."""

        if not 0 <= microbatch < self.num_microbatches:
            raise IndexError("replay microbatch is out of range")
        flat = self.metadata.diagnostic_masks[microbatch]
        return torch.tensor(flat, dtype=torch.bool, device=device).view(
            self.metadata.micro_batch_size, self.metadata.sequence_length
        )

    def release(self) -> None:
        """Release all plan-owned source tensors and sample references."""

        self.microbatches.clear()
        self._filler = None
        self.metadata = ReplayPlanMetadata(
            self.metadata.micro_batch_size, self.metadata.sequence_length, 0, 0, (), (), ()
        )


@dataclass(frozen=True)
class _ReplayPlanFacts:
    descriptor_hash: str
    selected_tokens: int
    microbatches: int
    host_tensor_bytes: int
    tensor_bindings: tuple[tuple[int, str, int, tuple[int, ...], torch.dtype, int], ...]

    @classmethod
    def observe(cls, plan: ReplayPlan) -> "_ReplayPlanFacts":
        metadata = plan.metadata
        if (
            metadata.micro_batch_size <= 0
            or metadata.sequence_length <= 0
            or metadata.num_microbatches <= 0
            or metadata.global_selected_tokens <= 0
        ):
            raise ValueError("replay plan dimensions and selection must be positive")
        if len(metadata.diagnostic_masks) != metadata.num_microbatches:
            raise ValueError("replay plan mask cardinality disagrees with its schedule")
        mask_size = _checked_product(metadata.micro_batch_size, metadata.sequence_length)
        if any(len(mask) != mask_size for mask in metadata.diagnostic_masks):
            raise ValueError("replay plan contains a malformed diagnostic mask")
        actual_selected = sum(
            sum(bool(value) for value in mask) for mask in metadata.diagnostic_masks
        )
        if actual_selected != metadata.global_selected_tokens:
            raise ValueError("replay plan selected-token metadata disagrees with fixed masks")
        if (
            len(metadata.selected_tokens) != actual_selected
            or len(set(metadata.selected_tokens)) != actual_selected
        ):
            raise ValueError("replay plan selected-token identities are inconsistent")
        expected_fields = {
            **dict(_MODEL_FIELDS),
            DIAGNOSTIC_MASK_FIELD: torch.bool,
            SAMPLE_INDEX_FIELD: torch.int64,
            SAMPLE_EPOCH_FIELD: torch.int64,
        }
        if plan.source_rank and len(plan.microbatches) != metadata.num_microbatches:
            raise ValueError("TP source replay plan has the wrong microbatch count")
        if not plan.source_rank and plan.microbatches:
            raise ValueError("non-source replay plans cannot retain source microbatches")
        bindings = []
        total_bytes = 0
        for batch_index, batch in enumerate(plan.microbatches):
            if set(batch.data) != set(expected_fields):
                raise ValueError("replay microbatch fields changed after fixed-field admission")
            if len(batch.sample_ids) != metadata.micro_batch_size:
                raise ValueError("replay microbatch sample identities have the wrong size")
            for name, dtype in expected_fields.items():
                value = batch.data[name]
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                    raise TypeError(f"replay field {name!r} must remain a CPU tensor")
                expected_shape = (
                    (metadata.micro_batch_size,)
                    if name in _RESERVED_FIELDS
                    else (metadata.micro_batch_size, metadata.sequence_length)
                )
                if value.dtype != dtype or tuple(value.shape) != expected_shape:
                    raise ValueError(f"replay field {name!r} changed shape or dtype")
                field_bytes = _checked_product(value.numel(), value.element_size())
                total_bytes = _checked_sum(total_bytes, field_bytes)
                bindings.append(
                    (batch_index, name, id(value), tuple(value.shape), value.dtype, field_bytes)
                )
        return cls(
            descriptor_hash=metadata.descriptor_hash,
            selected_tokens=actual_selected,
            microbatches=metadata.num_microbatches,
            host_tensor_bytes=total_bytes,
            tensor_bindings=tuple(bindings),
        )

    def revalidate(self, plan: ReplayPlan) -> None:
        if _ReplayPlanFacts.observe(plan) != self:
            raise RuntimeError("replay plan changed after memory preflight")


def _assemble_microbatch(
    samples: Sequence[RecordedSample], selected: set[TokenId], real_lanes: int
) -> ReplayMicrobatch:
    if not samples:
        raise ValueError("a replay microbatch requires at least one lane")
    data: dict[str, Any] = {}
    for key in (key for key in samples[0].batch if key not in _RESERVED_FIELDS):
        rows = [sample.row(key) for sample in samples]
        first = rows[0]
        if isinstance(first, torch.Tensor):
            if not all(isinstance(row, torch.Tensor) and row.shape == first.shape for row in rows):
                raise ValueError(f"replay tensor field {key!r} has inconsistent shapes")
            data[key] = torch.stack(rows)
        elif all(row is None for row in rows):
            data[key] = None
        elif all(row == first for row in rows):
            data[key] = first
        else:
            raise ValueError(f"replay field {key!r} cannot be reconstructed exactly")
    loss_mask = data.get("loss_mask")
    if not isinstance(loss_mask, torch.Tensor) or loss_mask.ndim != 2:
        raise ValueError("replay reconstruction requires [batch, sequence] loss_mask")
    diagnostic_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
    for lane, sample in enumerate(samples[:real_lanes]):
        for column in range(loss_mask.shape[1]):
            diagnostic_mask[lane, column] = TokenId(sample.sample_id, column) in selected
    if real_lanes < len(samples):
        loss_mask = loss_mask.clone()
        loss_mask[real_lanes:] = 0
        data["loss_mask"] = loss_mask
    data[DIAGNOSTIC_MASK_FIELD] = diagnostic_mask
    data[SAMPLE_INDEX_FIELD] = torch.tensor(
        [sample.sample_id.sampler_index for sample in samples], dtype=torch.int64
    )
    data[SAMPLE_EPOCH_FIELD] = torch.tensor(
        [sample.sample_id.epoch for sample in samples], dtype=torch.int64
    )
    return ReplayMicrobatch(data, tuple(sample.sample_id for sample in samples))


def select_local_token_ids(
    recorded: Sequence[RecordedBatch],
    global_population: Sequence[SamplePopulation],
    *,
    probe_tokens: int,
    run_seed: int,
    event_id: int,
) -> tuple[TokenId, ...]:
    """Select a seed-shifted systematic grid over the full token population.

    A uniform cyclic offset makes every one of the ``total`` logical positions
    appear in exactly ``selected_count`` of the ``total`` possible grids.  The
    resulting per-token inclusion probability is therefore exactly ``k / N``;
    sample block sizes and their order cannot bias selection.
    """

    if probe_tokens < 0:
        raise ValueError("probe token count must be nonnegative")
    ordered = sorted(global_population, key=lambda item: item.sample_id)
    total = sum(item.valid_count for item in ordered)
    selected_count = min(probe_tokens, total)
    if not selected_count:
        return ()
    if selected_count == total:
        positions = set(range(total))
    else:
        random_offset = random.Random(_hash64(run_seed, event_id, total, selected_count)).randrange(
            total
        )
        positions = _systematic_positions(total, selected_count, random_offset)
    local = {sample.sample_id: sample for batch in recorded for sample in batch.samples}
    result: list[TokenId] = []
    cursor = 0
    for population in ordered:
        offsets = sorted(
            position - cursor
            for position in positions
            if cursor <= position < cursor + population.valid_count
        )
        sample = local.get(population.sample_id)
        if sample is not None:
            columns = sample.valid_columns
            if len(columns) != population.valid_count:
                raise ValueError("local loss mask disagrees with gathered population")
            result.extend(TokenId(sample.sample_id, columns[offset]) for offset in offsets)
        cursor += population.valid_count
    return tuple(sorted(result))


def build_local_replay_plan(
    recorded: Sequence[RecordedBatch],
    selected_tokens: Sequence[TokenId],
    *,
    micro_batch_size: int,
    target_microbatches: int | None = None,
    global_selected_tokens: int | None = None,
) -> ReplayPlan:
    """Reconstruct deterministic fixed batches without advancing a sampler."""

    if micro_batch_size <= 0:
        raise ValueError("replay micro batch size must be positive")
    by_id = {sample.sample_id: sample for batch in recorded for sample in batch.samples}
    if len(by_id) != sum(len(batch.samples) for batch in recorded):
        raise ValueError("recorded sample identities must be unique")
    selected_set = set(selected_tokens)
    global_selected = (
        len(selected_set) if global_selected_tokens is None else global_selected_tokens
    )
    if global_selected < len(selected_set):
        raise ValueError("global selected-token count is smaller than local selection")
    selected_samples = [by_id[sample_id] for sample_id in sorted({t.sample for t in selected_set})]
    filler = min(by_id.values(), key=lambda sample: sample.sample_id) if by_id else None
    batches: list[ReplayMicrobatch] = []
    for start in range(0, len(selected_samples), micro_batch_size):
        real = selected_samples[start : start + micro_batch_size]
        if filler is None:
            raise ValueError("a selected replay plan requires a filler sample")
        lanes = [*real, *([filler] * (micro_batch_size - len(real)))]
        batches.append(_assemble_microbatch(lanes, selected_set, len(real)))
    target = len(batches) if target_microbatches is None else target_microbatches
    if target < len(batches):
        raise ValueError("target replay count cannot discard selected samples")
    while len(batches) < target:
        if filler is None:
            raise ValueError("padding requires one shape-compatible local sample")
        batches.append(_assemble_microbatch([filler] * micro_batch_size, set(), 0))
    sequence_length = 0
    if batches:
        sequence_length = int(batches[0].data["loss_mask"].shape[1])
    elif by_id:
        sequence_length = int(next(iter(by_id.values())).loss_mask.numel())
    masks = tuple(
        tuple(bool(value) for value in batch.data[DIAGNOSTIC_MASK_FIELD].reshape(-1))
        for batch in batches
    )
    metadata = ReplayPlanMetadata(
        micro_batch_size=micro_batch_size,
        sequence_length=sequence_length,
        num_microbatches=len(batches),
        global_selected_tokens=global_selected,
        selected_tokens=tuple(selected_tokens),
        ordered_sample_ids=tuple(sample for batch in batches for sample in batch.sample_ids),
        diagnostic_masks=masks,
    )
    return ReplayPlan(metadata, batches, True, filler)


def build_distributed_source_plan(
    recorded: Sequence[RecordedBatch],
    *,
    workspace: PopulationCollectiveWorkspace,
    readiness: ReadinessConsensus,
    probe_tokens: int,
    run_seed: int,
    event_id: int,
    micro_batch_size: int,
    maximum_microbatches: int,
) -> ReplayPlan:
    """Build and equalize a bounded source plan across a pre-existing DP group."""

    local: tuple[SamplePopulation, ...] = ()
    error: BaseException | None = None
    try:
        local = local_sample_populations(recorded)
    except BaseException as caught:
        error = caught
    readiness.settle(error, "local population validation")
    global_population = workspace.gather(local, readiness)
    global_selected_tokens = min(
        probe_tokens, sum(population.valid_count for population in global_population)
    )
    if probe_tokens <= 0 or not any(population.valid_count for population in global_population):
        readiness.settle(ReplayPreflightError("Tier-1 replay has no valid tokens"), "selection")
    selected: tuple[TokenId, ...] = ()
    plan: ReplayPlan | None = None
    error = None
    try:
        selected = select_local_token_ids(
            recorded,
            global_population,
            probe_tokens=probe_tokens,
            run_seed=run_seed,
            event_id=event_id,
        )
        plan = build_local_replay_plan(
            recorded,
            selected,
            micro_batch_size=micro_batch_size,
            global_selected_tokens=global_selected_tokens,
        )
    except BaseException as caught:
        error = caught
    readiness.settle(error, "local plan reconstruction")
    assert plan is not None
    target = workspace.maximum_schedule_count(plan.num_microbatches)
    error = (
        ReplayPreflightError("Tier-1 replay exceeds its fixed microbatch cap")
        if target > maximum_microbatches
        else None
    )
    readiness.settle(error, "schedule count cap")
    if target == plan.num_microbatches:
        return plan
    plan.release()
    return build_local_replay_plan(
        recorded,
        selected,
        micro_batch_size=micro_batch_size,
        target_microbatches=target,
        global_selected_tokens=global_selected_tokens,
    )


@dataclass(frozen=True)
class FixedPlanCodec:
    """Encode replay metadata and masks into two fixed-size typed tensors."""

    maximum_tokens: int
    maximum_microbatches: int
    micro_batch_size: int
    sequence_length: int

    def __post_init__(self) -> None:
        if (
            min(
                self.maximum_tokens,
                self.maximum_microbatches,
                self.micro_batch_size,
                self.sequence_length,
            )
            <= 0
        ):
            raise ValueError("fixed replay codec dimensions must be positive")

    @property
    def integer_count(self) -> int:
        """Return the fixed int64 metadata tensor length."""

        return 8 + 3 * self.maximum_tokens + 2 * self.maximum_microbatches * self.micro_batch_size

    @property
    def mask_count(self) -> int:
        """Return the fixed uint8 mask tensor length."""

        return self.maximum_microbatches * self.micro_batch_size * self.sequence_length

    def allocate(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate one reusable fixed wire pair."""

        if torch.device(device).type != "cpu":
            raise RuntimeError("replay plan control wires require CPU/Gloo")
        return (
            torch.full((self.integer_count,), -1, dtype=torch.int64, device=device),
            torch.zeros((self.mask_count,), dtype=torch.uint8, device=device),
        )

    def encode(self, plan: ReplayPlan, integer: torch.Tensor, mask: torch.Tensor) -> None:
        """Encode one source plan into preallocated tensors."""

        if integer.device.type != "cpu" or mask.device.type != "cpu":
            raise RuntimeError("replay plan control wires must remain CPU-resident")
        metadata = plan.metadata
        if (
            metadata.micro_batch_size != self.micro_batch_size
            or metadata.sequence_length != self.sequence_length
            or metadata.num_microbatches > self.maximum_microbatches
            or len(metadata.selected_tokens) > self.maximum_tokens
            or not len(metadata.selected_tokens)
            <= metadata.global_selected_tokens
            <= self.maximum_tokens
        ):
            raise ValueError("replay plan exceeds its fixed codec")
        if integer.numel() != self.integer_count or mask.numel() != self.mask_count:
            raise ValueError("replay codec received incorrectly sized wire tensors")
        integer.fill_(-1)
        mask.zero_()
        integer[:8] = torch.tensor(
            (
                _PLAN_MAGIC,
                _PLAN_VERSION,
                metadata.num_microbatches,
                metadata.micro_batch_size,
                metadata.sequence_length,
                metadata.global_selected_tokens,
                len(metadata.selected_tokens),
                len(metadata.ordered_sample_ids),
            ),
            dtype=torch.int64,
            device=integer.device,
        )
        offset = 8
        for token in metadata.selected_tokens:
            integer[offset : offset + 3] = torch.tensor(
                (token.sample.epoch, token.sample.sampler_index, token.sequence_column),
                dtype=torch.int64,
                device=integer.device,
            )
            offset += 3
        offset = 8 + 3 * self.maximum_tokens
        for sample in metadata.ordered_sample_ids:
            integer[offset : offset + 2] = torch.tensor(
                (sample.epoch, sample.sampler_index), dtype=torch.int64, device=integer.device
            )
            offset += 2
        for index, values in enumerate(metadata.diagnostic_masks):
            start = index * self.micro_batch_size * self.sequence_length
            mask[start : start + len(values)] = torch.tensor(
                values, dtype=torch.uint8, device=mask.device
            )

    def decode(self, integer: torch.Tensor, mask: torch.Tensor) -> ReplayPlanMetadata:
        """Validate and decode one fixed wire pair."""

        if integer.device.type != "cpu" or mask.device.type != "cpu":
            raise RuntimeError("replay plan control wires must remain CPU-resident")
        header = tuple(int(value) for value in integer[:8])
        (magic, version, count, mbs, sequence, global_selected, selected_count, sample_count) = (
            header
        )
        if magic != _PLAN_MAGIC or version != _PLAN_VERSION:
            raise ValueError("invalid replay plan wire identity")
        if not 0 <= count <= self.maximum_microbatches:
            raise ValueError("invalid replay microbatch count")
        if mbs != self.micro_batch_size or sequence != self.sequence_length:
            raise ValueError("replay plan wire shape disagrees with its codec")
        if not 0 <= selected_count <= self.maximum_tokens:
            raise ValueError("invalid replay selected-token count")
        if not selected_count <= global_selected <= self.maximum_tokens:
            raise ValueError("invalid global selected-token count")
        expected_samples = count * mbs
        if sample_count != expected_samples:
            raise ValueError("replay sample metadata cardinality is inconsistent")
        offset = 8
        tokens = []
        for _ in range(selected_count):
            epoch, sample, column = (int(value) for value in integer[offset : offset + 3])
            tokens.append(TokenId(SampleId(epoch, sample), column))
            offset += 3
        offset = 8 + 3 * self.maximum_tokens
        samples = []
        for _ in range(sample_count):
            epoch, sample = (int(value) for value in integer[offset : offset + 2])
            samples.append(SampleId(epoch, sample))
            offset += 2
        width = mbs * sequence
        masks = tuple(
            tuple(bool(value) for value in mask[index * width : (index + 1) * width])
            for index in range(count)
        )
        return ReplayPlanMetadata(
            mbs, sequence, count, global_selected, tuple(tokens), tuple(samples), masks
        )


def broadcast_replay_plan(
    source_plan: ReplayPlan | None,
    *,
    codec: FixedPlanCodec,
    binding: CollectiveBinding,
    source_group_rank: int,
    readiness: ReadinessConsensus,
    device: torch.device | str = "cpu",
) -> ReplayPlan:
    """Broadcast immutable plan data and construct a neutral TP non-source plan."""

    binding.validate()
    binding.require_same(readiness.binding)
    if torch.device(device).type != "cpu":
        raise RuntimeError("replay plan broadcast requires a CPU/Gloo control wire")
    if not 0 <= source_group_rank < binding.size:
        raise ValueError("TP source group rank is outside the verified binding")
    group_rank = dist.get_rank(binding.group) if binding.size > 1 else 0
    integer: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    error: BaseException | None = None
    try:
        integer, mask = codec.allocate(device)
    except BaseException as caught:
        error = caught
    readiness.settle(error, "plan wire allocation")
    assert integer is not None and mask is not None
    error = None
    if group_rank == source_group_rank:
        if source_plan is None:
            error = ValueError("the TP source rank did not provide a replay plan")
        else:
            try:
                codec.encode(source_plan, integer, mask)
            except Exception as caught:
                error = caught
    elif source_plan is not None:
        error = ValueError("a TP non-source rank supplied source replay batches")
    readiness.settle(error, "plan encoding")
    if binding.size > 1:
        global_source = (
            source_group_rank
            if binding.group is None
            else dist.get_global_rank(binding.group, source_group_rank)
        )
        dist.broadcast(integer, src=global_source, group=binding.group)
        dist.broadcast(mask, src=global_source, group=binding.group)
    metadata = codec.decode(integer, mask)
    if source_plan is not None:
        if metadata.descriptor_hash != source_plan.metadata.descriptor_hash:
            raise ReplayPreflightError("source replay plan changed during typed broadcast")
        return source_plan
    return ReplayPlan(metadata, [], False)


def verify_replay_plan_consensus(
    plan: ReplayPlan,
    *,
    binding: CollectiveBinding,
    readiness: ReadinessConsensus,
    device: torch.device | str = "cpu",
) -> None:
    """Verify fixed plan shape and hash across TP/PP participants before P2P."""

    binding.validate()
    binding.require_same(readiness.binding)
    error: BaseException | None = None
    minimum: torch.Tensor | None = None
    maximum: torch.Tensor | None = None
    try:
        digest = bytes.fromhex(plan.metadata.descriptor_hash)
        wire = torch.tensor(
            (
                plan.num_microbatches,
                plan.metadata.micro_batch_size,
                plan.metadata.sequence_length,
                *digest,
            ),
            dtype=torch.int64,
            device=device,
        )
        minimum = wire.clone()
        maximum = wire.clone()
    except BaseException as caught:
        error = caught
    readiness.settle(error, "plan descriptor construction")
    assert minimum is not None and maximum is not None
    if binding.size > 1:
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=binding.group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=binding.group)
        if not torch.equal(minimum, maximum):
            raise ReplayPreflightError("Tier-1 replay plan descriptor mismatch")


def cp_sequence_columns(sequence_length: int, cp_size: int, cp_rank: int) -> torch.Tensor:
    """Return Megatron's two-chunk zigzag CP column indices."""

    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError("invalid context-parallel topology")
    if sequence_length % (2 * cp_size):
        raise ValueError("sequence length must be divisible by twice the CP size")
    width = sequence_length // (2 * cp_size)
    chunks = (cp_rank, 2 * cp_size - cp_rank - 1)
    return torch.tensor(
        [column for chunk in chunks for column in range(chunk * width, (chunk + 1) * width)],
        dtype=torch.int64,
    )


def slice_replay_mask(
    mask: torch.Tensor,
    *,
    context_parallel_size: int,
    context_parallel_rank: int,
    sequence_parallel: bool = False,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply CP zigzag and optional contiguous SP slicing to a replay mask."""

    if mask.ndim != 2:
        raise ValueError("replay masks must have [batch, sequence] layout")
    columns = cp_sequence_columns(mask.shape[1], context_parallel_size, context_parallel_rank)
    cp_mask = mask.index_select(1, columns.to(mask.device))
    if not sequence_parallel:
        return cp_mask, None
    if tensor_parallel_size <= 0 or not 0 <= tensor_parallel_rank < tensor_parallel_size:
        raise ValueError("invalid sequence-parallel TP coordinate")
    if cp_mask.shape[1] % tensor_parallel_size:
        raise ValueError("CP-local sequence length is not divisible by TP size")
    width = cp_mask.shape[1] // tensor_parallel_size
    start = tensor_parallel_rank * width
    return cp_mask, cp_mask[:, start : start + width].contiguous()


class ReplayIterator(Iterator[dict[str, Any]]):
    """Resettable iterator over source-rank replay batches."""

    def __init__(self, plan: ReplayPlan) -> None:
        if not plan.source_rank:
            raise ValueError("only a TP source plan can create a replay iterator")
        self.plan = plan
        self.index = 0

    def __iter__(self) -> "ReplayIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self.index >= self.plan.num_microbatches:
            raise StopIteration
        batch = self.plan.microbatches[self.index].model_batch()
        self.index += 1
        return batch

    def reset(self) -> None:
        """Rewind replay only; the training iterator and sampler are untouched."""

        self.index = 0


class NeutralReplayIterator(Iterator[dict[str, Any]]):
    """Iterator that fails if a non-data rank tries to advance raw replay data."""

    def __iter__(self) -> "NeutralReplayIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        raise RuntimeError("a replay nonowner attempted to advance a data iterator")


def _clone_tracker_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    clone_state = getattr(value, "clone_state", None)
    if callable(clone_state):
        return clone_state()
    raise TypeError("Megatron RNG tracker values need tensor or clone_state semantics")


def _validated_tracker(getter: Callable[[], Any] | None) -> Any | None:
    if getter is None:
        return None
    tracker = getter()
    get_states = getattr(tracker, "get_states", None)
    set_states = getattr(tracker, "set_states", None)
    if not callable(get_states) or not callable(set_states):
        raise TypeError("Megatron RNG tracker lacks get_states/set_states")
    states = get_states()
    if not isinstance(states, Mapping) or not all(isinstance(name, str) for name in states):
        raise TypeError("Megatron RNG tracker returned an invalid state mapping")
    if not states:
        raise TypeError("Megatron RNG tracker returned no inspected RNG states")
    for value in states.values():
        _clone_tracker_value(value)
    return tracker


@dataclass
class ReplayRngState:
    """Independent snapshots of every supported replay RNG stream."""

    python_state: object
    numpy_state: tuple[Any, ...]
    torch_state: torch.Tensor
    cuda_state: torch.Tensor | None
    tracker_states: dict[str, Any]
    tracker_name: str | None

    @classmethod
    def capture(
        cls, *, tracker_getter: Callable[[], Any] | None, cuda_device: torch.device | int | None
    ) -> "ReplayRngState":
        """Capture Python, NumPy, Torch, CUDA, and validated Megatron tracker state."""

        tracker = _validated_tracker(tracker_getter)
        states = (
            {name: _clone_tracker_value(value) for name, value in tracker.get_states().items()}
            if tracker is not None
            else {}
        )
        numpy_state = np.random.get_state()
        numpy_copy = (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
        cuda_state = (
            torch.cuda.get_rng_state(cuda_device).clone() if torch.cuda.is_available() else None
        )
        return cls(
            copy.deepcopy(random.getstate()),
            numpy_copy,
            torch.get_rng_state().clone(),
            cuda_state,
            states,
            getattr(tracker, "_current_state_name", None),
        )

    @property
    def tensor_bytes(self) -> int:
        """Return exact tensor bytes retained by the RNG snapshot."""

        values = [
            self.torch_state,
            *(value for value in self.tracker_states.values() if isinstance(value, torch.Tensor)),
        ]
        if self.cuda_state is not None:
            values.append(self.cuda_state)
        return sum(value.numel() * value.element_size() for value in values)

    def restore_stages(
        self, *, tracker_getter: Callable[[], Any] | None, cuda_device: torch.device | int | None
    ) -> tuple[tuple[str, Callable[[], None]], ...]:
        """Return independently invokable restoration stages."""

        def restore_python() -> None:
            random.setstate(copy.deepcopy(self.python_state))
            if random.getstate() != self.python_state:
                raise RuntimeError("Python RNG restoration verification failed")

        def restore_numpy() -> None:
            np.random.set_state(self.numpy_state)
            current = np.random.get_state()
            if (
                current[0] != self.numpy_state[0]
                or not np.array_equal(current[1], self.numpy_state[1])
                or current[2:] != self.numpy_state[2:]
            ):
                raise RuntimeError("NumPy RNG restoration verification failed")

        def restore_torch() -> None:
            torch.set_rng_state(self.torch_state)
            if not torch.equal(torch.get_rng_state(), self.torch_state):
                raise RuntimeError("Torch RNG restoration verification failed")

        def restore_cuda() -> None:
            if self.cuda_state is not None:
                torch.cuda.set_rng_state(self.cuda_state, cuda_device)
                if not torch.equal(torch.cuda.get_rng_state(cuda_device), self.cuda_state):
                    raise RuntimeError("CUDA RNG restoration verification failed")

        def restore_tracker() -> None:
            if tracker_getter is None:
                if self.tracker_states:
                    raise RuntimeError("captured tracker state has no restoration getter")
                return
            tracker = _validated_tracker(tracker_getter)
            assert tracker is not None
            tracker.set_states(
                {name: _clone_tracker_value(value) for name, value in self.tracker_states.items()}
            )
            if self.tracker_name is not None:
                tracker._current_state_name = self.tracker_name
            restored = tracker.get_states()
            if set(restored) != set(self.tracker_states) or any(
                not isinstance(restored[name], torch.Tensor)
                or not isinstance(expected, torch.Tensor)
                or not torch.equal(restored[name], expected)
                for name, expected in self.tracker_states.items()
            ):
                raise RuntimeError("Megatron RNG tracker restoration verification failed")

        return (
            ("python_rng", restore_python),
            ("numpy_rng", restore_numpy),
            ("torch_rng", restore_torch),
            ("cuda_rng", restore_cuda),
            ("tracker_rng", restore_tracker),
        )


@dataclass
class _ValueSnapshot:
    original: Any
    contents: Any

    @classmethod
    def capture(cls, value: Any) -> "_ValueSnapshot":
        if isinstance(value, torch.Tensor):
            contents = value.detach().clone()
        elif type(value) is list:
            contents = tuple(cls.capture(item) for item in value)
        elif type(value) is tuple:
            contents = tuple(cls.capture(item) for item in value)
        elif type(value) in (dict, OrderedDict):
            if any(not _is_snapshot_key(key) for key in value):
                raise TypeError("mutable model-state mappings require immutable scalar keys")
            contents = tuple((key, cls.capture(item)) for key, item in value.items())
        elif type(value) is set:
            if any(not _is_snapshot_key(item) for item in value):
                raise TypeError("mutable model-state sets require immutable scalar values")
            contents = frozenset(value)
        elif _is_snapshot_leaf(value):
            contents = value
        else:
            raise TypeError(f"unsupported mutable model-state value: {type(value).__qualname__}")
        return cls(value, contents)

    def restore(self) -> Any:
        if isinstance(self.original, torch.Tensor):
            self.original.copy_(self.contents)
        elif type(self.original) is list:
            self.original.clear()
            self.original.extend(item.restore() for item in self.contents)
        elif type(self.original) is tuple:
            for item in self.contents:
                item.restore()
            return self.original
        elif type(self.original) in (dict, OrderedDict):
            self.original.clear()
            self.original.update((key, item.restore()) for key, item in self.contents)
        elif type(self.original) is set:
            self.original.clear()
            self.original.update(self.contents)
        return self.original

    def verify(self, value: Any) -> bool:
        if value is not self.original:
            return False
        if isinstance(value, torch.Tensor):
            return torch.equal(value, self.contents)
        if type(value) is list:
            return len(value) == len(self.contents) and all(
                snapshot.verify(item) for snapshot, item in zip(self.contents, value, strict=True)
            )
        if type(value) is tuple:
            return len(value) == len(self.contents) and all(
                snapshot.verify(item) for snapshot, item in zip(self.contents, value, strict=True)
            )
        if type(value) in (dict, OrderedDict):
            if type(value) is not type(self.original):
                return False
            return tuple(value) == tuple(key for key, _ in self.contents) and all(
                snapshot.verify(value[key]) for key, snapshot in self.contents
            )
        if type(value) is set:
            return value == self.contents
        return value is self.original

    @property
    def tensor_bytes(self) -> int:
        """Return recursively cloned tensor and mutable-container storage."""

        if isinstance(self.original, torch.Tensor):
            return self.contents.numel() * self.contents.element_size() + sys.getsizeof(
                self.contents
            )
        if type(self.original) in (list, tuple):
            return (sys.getsizeof(self.contents) if self.contents else 0) + sum(
                item.tensor_bytes for item in self.contents
            )
        if type(self.original) in (dict, OrderedDict):
            return (sys.getsizeof(self.contents) if self.contents else 0) + sum(
                sys.getsizeof(entry) + entry[1].tensor_bytes for entry in self.contents
            )
        if type(self.original) is set:
            return sys.getsizeof(self.contents) if self.contents else 0
        return 0


def _is_snapshot_key(value: Any) -> bool:
    return value is None or type(value) in (bool, int, float, complex, str, bytes)


def _is_snapshot_leaf(value: Any) -> bool:
    return (
        _is_snapshot_key(value)
        or isinstance(value, (Enum, torch.dtype, torch.device))
        or callable(value)
    )


def _value_tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Conservatively count recursively cloned state storage before capture."""

    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() + sys.getsizeof(value)
    if type(value) in (list, tuple):
        container_bytes = sys.getsizeof(value) if value else 0
        return _checked_sum(container_bytes, *(_value_tensor_bytes(item, seen) for item in value))
    if type(value) in (dict, OrderedDict):
        container_bytes = sys.getsizeof(value) if value else 0
        return _checked_sum(
            container_bytes,
            *(
                _checked_sum(sys.getsizeof((key, item)), _value_tensor_bytes(item, seen))
                for key, item in value.items()
            ),
        )
    if type(value) is set:
        return sys.getsizeof(value) if value else 0
    return 0


@dataclass
class _AttributeSnapshot:
    owner: object
    name: str
    value: _ValueSnapshot | None

    def restore(self) -> None:
        if self.value is None:
            if hasattr(self.owner, self.name):
                delattr(self.owner, self.name)
        else:
            setattr(self.owner, self.name, self.value.restore())

    def verify(self) -> bool:
        if self.value is None:
            return not hasattr(self.owner, self.name)
        return hasattr(self.owner, self.name) and self.value.verify(getattr(self.owner, self.name))

    @property
    def tensor_bytes(self) -> int:
        return 0 if self.value is None else self.value.tensor_bytes


@dataclass
class _ExtraStateSnapshot:
    owner: torch.nn.Module
    value: _ValueSnapshot

    def restore(self) -> None:
        self.owner.set_extra_state(self.value.restore())

    def verify(self) -> bool:
        return self.value.verify(self.owner.get_extra_state())


class DenseGPTStateSnapshot:
    """Snapshot declared model buffers, modes, references, and hook maps."""

    _REFERENCE_ATTRIBUTES = (
        "input_tensor",
        "embedding_activation_buffer",
        "grad_output_buffer",
        "rotary_pos_emb_cache",
        "_decoder_hidden_states_cache",
    )
    _STATIC_ATTRIBUTES = {"config", "pg_collection", "submodules", "transformer_layer_spec"}

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        mutable_buffers: Sequence[str],
        maximum_tensor_bytes: int = _CAPACITY_LIMIT,
    ) -> None:
        if not 0 <= maximum_tensor_bytes <= _CAPACITY_LIMIT:
            raise ValueError("model state snapshot cap is invalid")
        self.models = tuple(models)
        self.mutable_buffers = frozenset(mutable_buffers)
        self.maximum_tensor_bytes = maximum_tensor_bytes
        self.modes: list[tuple[torch.nn.Module, bool]] = []
        self.module_maps: list[tuple[torch.nn.Module, tuple[tuple[str, torch.nn.Module], ...]]] = []
        self.buffer_maps: list[tuple[torch.nn.Module, tuple[str, ...]]] = []
        self.buffers: list[tuple[str, torch.nn.Module, str, torch.Tensor, torch.Tensor]] = []
        self.none_buffers: list[tuple[str, torch.nn.Module, str]] = []
        self.attributes: list[_AttributeSnapshot] = []
        self.extra_states: list[_ExtraStateSnapshot] = []

    def _admit(self, additional_bytes: int) -> None:
        if additional_bytes < 0 or self.tensor_bytes + additional_bytes > self.maximum_tensor_bytes:
            raise ValueError("model state snapshot exceeds its declared byte cap")

    def capture(self) -> None:
        """Capture all explicitly admitted mutable model state."""

        self.modes = [
            (module, module.training) for model in self.models for module in model.modules()
        ]
        found: set[str] = set()
        registered: set[str] = set()
        seen: set[tuple[int, str]] = set()
        for model_index, model in enumerate(self.models):
            for module_name, module in model.named_modules():
                self.module_maps.append((module, tuple(module._modules.items())))
                self.buffer_maps.append((module, tuple(module._buffers)))
                for local_name, buffer in module._buffers.items():
                    name = f"{module_name}.{local_name}" if module_name else local_name
                    qualified = f"{model_index}:{name}"
                    registered.add(qualified)
                    if name in self.mutable_buffers or qualified in self.mutable_buffers:
                        found.add(name if name in self.mutable_buffers else qualified)
                        if buffer is None:
                            self.none_buffers.append((qualified, module, local_name))
                        elif isinstance(buffer, torch.Tensor):
                            size = buffer.numel() * buffer.element_size()
                            self._admit(size)
                            self.buffers.append(
                                (qualified, module, local_name, buffer, buffer.detach().clone())
                            )
                        else:
                            raise TypeError("registered model buffer is not tensor-or-None")
                attribute_names = set(self._REFERENCE_ATTRIBUTES)
                attribute_names.update(
                    name
                    for name in vars(module)
                    if name not in self._STATIC_ATTRIBUTES
                    and not name.endswith("_group")
                    and "process_group" not in name
                )
                attribute_names.difference_update({"_buffers", "_parameters", "_modules"})
                for name in sorted(attribute_names):
                    identity = (id(module), name)
                    if identity not in seen:
                        seen.add(identity)
                        value: _ValueSnapshot | None = None
                        if hasattr(module, name):
                            current = getattr(module, name)
                            self._admit(_value_tensor_bytes(current))
                            value = _ValueSnapshot.capture(current)
                        self.attributes.append(_AttributeSnapshot(module, name, value))
                if type(module).get_extra_state is not torch.nn.Module.get_extra_state:
                    if type(module).set_extra_state is torch.nn.Module.set_extra_state:
                        raise TypeError("model extra state has no verified restoration method")
                    current = module.get_extra_state()
                    self._admit(_value_tensor_bytes(current))
                    value = _ValueSnapshot.capture(current)
                    self.extra_states.append(_ExtraStateSnapshot(module, value))
        missing = self.mutable_buffers - found
        if missing:
            raise ValueError(f"declared mutable model buffers were not found: {sorted(missing)}")
        admitted = {
            qualified
            for qualified in registered
            if qualified.split(":", 1)[1] in self.mutable_buffers
            or qualified in self.mutable_buffers
        }
        undeclared = registered - admitted
        if undeclared:
            raise ValueError(
                f"registered model buffers were not declared mutable: {sorted(undeclared)}"
            )

    def restore(self) -> None:
        """Restore model state without performing verification."""

        for module, names in self.buffer_maps:
            for local_name in tuple(module._buffers):
                if local_name not in names:
                    del module._buffers[local_name]
        for module, entries in self.module_maps:
            module._modules.clear()
            module._modules.update(entries)
        for _name, module, local_name, buffer, snapshot in self.buffers:
            module._buffers[local_name] = buffer
            buffer.copy_(snapshot)
        for _name, module, local_name in self.none_buffers:
            module._buffers[local_name] = None
        for attribute in self.attributes:
            attribute.restore()
        for extra_state in self.extra_states:
            extra_state.restore()
        for module, training in self.modes:
            module.training = training

    def verify(self) -> None:
        """Verify buffer contents, references, hooks, and module modes."""

        if any(tuple(module._buffers) != names for module, names in self.buffer_maps):
            raise RuntimeError("model buffer registry restoration failed")
        if any(tuple(module._modules.items()) != entries for module, entries in self.module_maps):
            raise RuntimeError("model module graph restoration failed")
        for name, module, local_name, buffer, snapshot in self.buffers:
            if module._buffers.get(local_name) is not buffer or not torch.equal(buffer, snapshot):
                raise RuntimeError(f"model buffer restoration failed: {name}")
        if any(
            module._buffers.get(local_name) is not None
            for _, module, local_name in self.none_buffers
        ):
            raise RuntimeError("None model-buffer restoration failed")
        if any(module.training != training for module, training in self.modes):
            raise RuntimeError("model training-mode restoration failed")
        if any(not attribute.verify() for attribute in self.attributes):
            raise RuntimeError("model reference/hook restoration failed")
        if any(not extra_state.verify() for extra_state in self.extra_states):
            raise RuntimeError("model extra-state restoration failed")

    @property
    def tensor_bytes(self) -> int:
        """Return exact cloned model-buffer storage."""

        return (
            sum(snapshot.numel() * snapshot.element_size() for _, _, _, _, snapshot in self.buffers)
            + sum(attribute.tensor_bytes for attribute in self.attributes)
            + sum(extra_state.value.tensor_bytes for extra_state in self.extra_states)
        )

    def release(self) -> None:
        """Release all captured model references."""

        self.modes.clear()
        self.module_maps.clear()
        self.buffer_maps.clear()
        self.buffers.clear()
        self.none_buffers.clear()
        self.attributes.clear()
        self.extra_states.clear()


class IdentityStateSnapshot:
    """Snapshot replay iterator positions and sampler identity counters."""

    def __init__(self, iterators: Sequence[ReplayIterator], samplers: Sequence[object]) -> None:
        self.iterators = tuple(iterators)
        self.samplers = tuple(samplers)
        self.iterator_positions: tuple[int, ...] = ()
        self.sampler_values: tuple[tuple[object, str, Any], ...] = ()

    def capture(self) -> None:
        """Capture bounded identity state without advancing it."""

        self.iterator_positions = tuple(iterator.index for iterator in self.iterators)
        values = []
        for sampler in self.samplers:
            for name in ("consumed_samples", "epoch"):
                if hasattr(sampler, name):
                    values.append((sampler, name, copy.deepcopy(getattr(sampler, name))))
        self.sampler_values = tuple(values)

    def restore(self) -> None:
        """Restore iterator and sampler counters."""

        for iterator, position in zip(self.iterators, self.iterator_positions, strict=True):
            iterator.index = position
        for sampler, name, value in self.sampler_values:
            setattr(sampler, name, copy.deepcopy(value))

    def verify(self) -> None:
        """Verify every captured identity counter."""

        if tuple(iterator.index for iterator in self.iterators) != self.iterator_positions:
            raise RuntimeError("replay iterator identity restoration failed")
        if any(getattr(owner, name) != value for owner, name, value in self.sampler_values):
            raise RuntimeError("training sampler identity restoration failed")


class OverlapStateSnapshot:
    """Snapshot explicitly named overlap handles and reject active work."""

    _ATTRIBUTES = (
        "param_gather_handle",
        "grad_reduce_handle",
        "param_gather_work",
        "grad_reduce_work",
    )

    def __init__(self, owners: Sequence[object]) -> None:
        self.owners = tuple(owners)
        self.attributes: list[_AttributeSnapshot] = []

    def capture(self) -> None:
        """Capture quiescent overlap state and reject active handles."""

        for owner in self.owners:
            for name in self._ATTRIBUTES:
                snapshot: _ValueSnapshot | None = None
                if hasattr(owner, name):
                    value = getattr(owner, name)
                    if value is not None:
                        raise ValueError(f"Tier-1 replay requires quiescent overlap state: {name}")
                    snapshot = _ValueSnapshot.capture(value)
                self.attributes.append(_AttributeSnapshot(owner, name, snapshot))

    def restore(self) -> None:
        """Restore overlap references."""

        for attribute in self.attributes:
            attribute.restore()

    def verify(self) -> None:
        """Verify overlap references remain quiescent."""

        if any(not attribute.verify() for attribute in self.attributes):
            raise RuntimeError("overlap-state restoration failed")


class ReplayRestorationError(RuntimeError):
    """Aggregate all independent restoration/verification failures."""

    def __init__(self, failures: Sequence[tuple[str, BaseException]]) -> None:
        self.failures = tuple(failures)
        names = ", ".join(name for name, _ in failures)
        super().__init__(f"Tier-1 restoration failed in stages: {names}")


class ReplayStateGuard:
    """Nested transaction guard with independent best-effort restoration stages."""

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        *,
        mutable_buffer_names: Sequence[str] = (),
        replay_iterators: Sequence[ReplayIterator] = (),
        samplers: Sequence[object] = (),
        overlap_objects: Sequence[object] = (),
        tracker_getter: Callable[[], Any] | None = None,
        cuda_device: torch.device | int | None = None,
        fault_injections: Mapping[str, Callable[[], None]] | None = None,
        maximum_model_state_bytes: int = _CAPACITY_LIMIT,
    ) -> None:
        self.model = DenseGPTStateSnapshot(models, mutable_buffer_names, maximum_model_state_bytes)
        self.identity = IdentityStateSnapshot(replay_iterators, samplers)
        self.overlap = OverlapStateSnapshot(overlap_objects)
        self.tracker_getter = tracker_getter
        self.cuda_device = cuda_device
        self.faults = dict(fault_injections or {})
        self.rng: ReplayRngState | None = None
        self.prepared = False
        self.entered = False

    def prepare(self) -> None:
        """Complete every fallible capture before collective readiness."""

        if self.prepared or self.entered:
            raise RuntimeError("replay state guard can be prepared exactly once")
        try:
            self.model.capture()
            self.identity.capture()
            self.overlap.capture()
            self.rng = ReplayRngState.capture(
                tracker_getter=self.tracker_getter, cuda_device=self.cuda_device
            )
            self.prepared = True
        except BaseException:
            self.model.release()
            self.overlap.attributes.clear()
            self.rng = None
            raise

    def __enter__(self) -> "ReplayStateGuard":
        if not self.prepared:
            self.prepare()
        if self.entered:
            raise RuntimeError("replay state guard cannot be entered twice")
        self.entered = True
        return self

    def _attempt(
        self, name: str, operation: Callable[[], None], failures: list[tuple[str, BaseException]]
    ) -> None:
        try:
            fault = self.faults.get(name)
            if fault is not None:
                fault()
            operation()
        except BaseException as error:
            failures.append((name, error))

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self.entered:
            raise RuntimeError("unentered replay state guard cannot restore")
        failures: list[tuple[str, BaseException]] = []
        assert self.rng is not None
        restore = (
            ("model_restore", self.model.restore),
            ("identity_restore", self.identity.restore),
            ("overlap_restore", self.overlap.restore),
            *self.rng.restore_stages(
                tracker_getter=self.tracker_getter, cuda_device=self.cuda_device
            ),
        )
        for name, operation in restore:
            self._attempt(name, operation, failures)
        for name, operation in (
            ("model_verify", self.model.verify),
            ("identity_verify", self.identity.verify),
            ("overlap_verify", self.overlap.verify),
        ):
            self._attempt(name, operation, failures)
        self.model.release()
        self.rng = None
        self.entered = False
        if failures:
            raise ReplayRestorationError(failures)
        return False


@dataclass(frozen=True)
class ReplayMemoryConfig:
    """Pure inputs for the exact core replay memory model."""

    sequence_length: int
    micro_batch_size: int
    global_batch_size: int
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    context_parallel_size: int
    global_layers: int
    local_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    selected_tokens: int
    replay_microbatches: int
    element_size: int
    rng_snapshot_bytes: int
    model_state_bytes: int
    fixed_workspace_bytes: int
    alignment: int = 256
    headroom_fraction: float = 0.1
    tp_source: bool = True
    sequence_parallel: bool = False
    gated_linear_unit: bool = False
    observed_host_replay_bytes: int | None = None
    observed_packed_statistics_bytes: int | None = None
    observed_reduction_arena_bytes: int | None = None
    observed_accumulator_scratch_bytes: int | None = None
    observed_attention_workspace_bytes: int | None = None


@dataclass(frozen=True)
class ReplayMemoryPolicy:
    """Caller-owned cap policy; all allocation dimensions are engine-derived."""

    alignment: int = 256
    headroom_fraction: float = 0.1

    def validate(self) -> None:
        if self.alignment <= 0 or self.alignment > _CAPACITY_LIMIT:
            raise ValueError("replay memory alignment is invalid")
        if not 0 <= self.headroom_fraction <= 1:
            raise ValueError("replay memory headroom fraction must be in [0, 1]")


@dataclass(frozen=True)
class ReplayMemoryEstimate:
    """Exact named allocation terms and aligned/headroom total."""

    terms: tuple[tuple[str, int], ...]
    alignment_padding_bytes: int
    headroom_bytes: int

    @property
    def predicted_allocated_bytes(self) -> int:
        """Return the aligned allocation bound before allocator headroom."""

        return sum(value for _, value in self.terms) + self.alignment_padding_bytes

    @property
    def predicted_reserved_bytes(self) -> int:
        """Return allocated bytes plus explicit configured headroom."""

        return self.predicted_allocated_bytes + self.headroom_bytes

    def term(self, name: str) -> int:
        """Return one named exact term."""

        return dict(self.terms)[name]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _checked_sum(*values: int) -> int:
    if any(value < 0 for value in values):
        raise ValueError("memory terms cannot be negative")
    result = sum(values)
    if result > _CAPACITY_LIMIT:
        raise OverflowError("replay memory calculation exceeds int64 capacity")
    return result


def _checked_product(*values: int) -> int:
    result = math.prod(values)
    if any(value < 0 for value in values) or result > _CAPACITY_LIMIT:
        raise OverflowError("replay memory calculation exceeds int64 capacity")
    return result


def _packed_storage_bytes(slots: int) -> int:
    sum_fields = len(fields(PackedSlots)) - 2
    return slots * (
        sum_fields * (torch.finfo(torch.float64).bits // 8)
        + 2 * (torch.finfo(torch.float32).bits // 8)
    )


def estimate_replay_memory(config: ReplayMemoryConfig) -> ReplayMemoryEstimate:
    """Purely estimate all core replay allocations without DP/CP duplication."""

    integers = (
        config.sequence_length,
        config.micro_batch_size,
        config.global_batch_size,
        config.data_parallel_size,
        config.tensor_parallel_size,
        config.pipeline_parallel_size,
        config.context_parallel_size,
        config.global_layers,
        config.local_layers,
        config.hidden_size,
        config.ffn_hidden_size,
        config.num_attention_heads,
        config.replay_microbatches,
        config.element_size,
        config.alignment,
    )
    if any(value <= 0 for value in integers) or config.selected_tokens < 0:
        raise ValueError("replay memory dimensions must be positive")
    observed = (
        config.observed_host_replay_bytes,
        config.observed_packed_statistics_bytes,
        config.observed_reduction_arena_bytes,
        config.observed_accumulator_scratch_bytes,
        config.observed_attention_workspace_bytes,
    )
    if any(value is not None and not 0 <= value <= _CAPACITY_LIMIT for value in observed):
        raise ValueError("observed replay memory facts are invalid")
    if not 0 <= config.headroom_fraction <= 1:
        raise ValueError("replay memory headroom fraction must be in [0, 1]")
    if config.sequence_length % (2 * config.context_parallel_size):
        raise ValueError("sequence length must support zigzag CP")
    if (
        config.hidden_size % config.tensor_parallel_size
        or (config.ffn_hidden_size % config.tensor_parallel_size)
        or config.num_attention_heads % config.tensor_parallel_size
    ):
        raise ValueError("response widths must be divisible by tensor parallelism")
    if config.global_layers % config.pipeline_parallel_size or (
        config.local_layers != config.global_layers // config.pipeline_parallel_size
    ):
        raise ValueError("standard replay requires uniform physical pipeline layers")
    if (
        config.sequence_parallel
        and (config.sequence_length // config.context_parallel_size) % config.tensor_parallel_size
    ):
        raise ValueError("CP-local sequence length must support sequence parallelism")
    # Every selected row and selected sample may land on one DP/CP rank.  DP
    # and CP therefore do not divide the maximum-rank bound.
    local_samples = min(config.global_batch_size, config.selected_tokens)
    minimum_microbatches = _ceil_div(local_samples, config.micro_batch_size) if local_samples else 0
    if config.replay_microbatches < minimum_microbatches:
        raise ValueError("replay microbatch count cannot discard selected samples")
    # tokens, labels, position_ids, loss_mask, diagnostic mask, and two identity vectors.
    full_batch = _checked_product(
        config.micro_batch_size, config.sequence_length, 8 + 8 + 8 + 4 + 1
    )
    full_batch = _checked_sum(full_batch, _checked_product(config.micro_batch_size, 2, 8))
    modeled_host_inputs = (
        _checked_product(config.replay_microbatches, full_batch) if config.tp_source else 0
    )
    host_inputs = (
        modeled_host_inputs
        if config.observed_host_replay_bytes is None
        else config.observed_host_replay_bytes
    )
    cp_local = _checked_product(
        config.micro_batch_size, config.sequence_length // config.context_parallel_size
    )
    cp_outputs = _checked_product(cp_local, 8 + 8 + 8 + 4 + 1)
    device_inputs = _checked_sum(full_batch, cp_outputs)
    sp_mask = cp_local // config.tensor_parallel_size if config.sequence_parallel else 0
    local_rows = config.selected_tokens
    feature_widths = (
        config.hidden_size,
        3 * config.hidden_size // config.tensor_parallel_size,
        config.hidden_size,
        (1 + int(config.gated_linear_unit)) * config.ffn_hidden_size // config.tensor_parallel_size,
        config.hidden_size,
    )
    response = 0
    max_response = 0
    # TP rank zero owns residual, attention-out, and FC2 in addition to the
    # feature-sharded QKV/FC1 families, so it is the conservative max rank.
    for width in feature_widths:
        allocation = _checked_product(config.local_layers, local_rows, width, config.element_size)
        response = _checked_sum(response, allocation)
        max_response = max(max_response, _checked_product(local_rows, width, config.element_size))
    slots = _checked_product(config.global_layers, 8)
    modeled_packed = _packed_storage_bytes(slots)
    packed = (
        modeled_packed
        if config.observed_packed_statistics_bytes is None
        else config.observed_packed_statistics_bytes
    )
    reduction_arena = (
        packed
        if config.observed_reduction_arena_bytes is None
        else config.observed_reduction_arena_bytes
    )
    accumulator_scratch = (
        PackedSufficientStatistics.scratch_bytes_for_capacity()
        if config.observed_accumulator_scratch_bytes is None
        else config.observed_accumulator_scratch_bytes
    )
    attention_elements = _checked_product(
        config.selected_tokens,
        config.num_attention_heads // config.tensor_parallel_size,
        config.sequence_length,
    )
    modeled_attention_workspace = _checked_product(
        8, attention_elements, max(config.element_size, 4)
    )
    attention_workspace = (
        modeled_attention_workspace
        if config.observed_attention_workspace_bytes is None
        else config.observed_attention_workspace_bytes
    )
    hook_workspace = _checked_sum(_checked_product(2, max_response), attention_workspace)
    terms = (
        ("host_replay_inputs", host_inputs),
        ("device_full_and_cp_inputs", device_inputs),
        ("cp_sp_masks", sp_mask),
        ("retained_response_rows", response),
        ("rng_snapshot", config.rng_snapshot_bytes),
        ("model_state_snapshot", config.model_state_bytes),
        ("packed_statistics", packed),
        ("reduction_arena", reduction_arena),
        ("accumulator_scratch", accumulator_scratch),
        ("hook_workspace", hook_workspace),
        ("fixed_schedule_workspace", config.fixed_workspace_bytes),
    )
    padding = _checked_sum(*((-value) % config.alignment for _, value in terms if value))
    allocated = _checked_sum(*(value for _, value in terms), padding)
    headroom_numerator, headroom_denominator = config.headroom_fraction.as_integer_ratio()
    headroom = (allocated * headroom_numerator + headroom_denominator - 1) // headroom_denominator
    _checked_sum(allocated, headroom)
    return ReplayMemoryEstimate(terms, padding, headroom)


def preflight_replay_memory(
    estimate: ReplayMemoryEstimate,
    *,
    maximum_extra_bytes: int,
    currently_reserved_bytes: int,
    total_device_bytes: int,
) -> None:
    """Reject a replay whose reserved bound exceeds either mandatory limit."""

    values = (
        maximum_extra_bytes,
        currently_reserved_bytes,
        total_device_bytes,
        estimate.predicted_reserved_bytes,
    )
    if min(values) < 0 or max(values) > _CAPACITY_LIMIT:
        raise ValueError("memory preflight values must be nonnegative int64 values")
    if estimate.predicted_reserved_bytes > maximum_extra_bytes:
        raise ReplayPreflightError("Tier-1 replay exceeds its configured memory cap")
    if (
        _checked_sum(currently_reserved_bytes, estimate.predicted_reserved_bytes)
        > total_device_bytes
    ):
        raise ReplayPreflightError("Tier-1 replay exceeds physical device headroom")


class ResponseProbe(Protocol):
    """Narrow response-probe surface consumed by the replay schedule."""

    expected_hook_calls: int
    descriptor_hash: str

    def set_masks(
        self, full_mask: torch.Tensor, *, sequence_parallel_mask: torch.Tensor | None = None
    ) -> None: ...

    def capture_pre(self) -> Any: ...

    def capture_post(self) -> Any: ...

    def finalize(self) -> Any: ...

    def release(self) -> None: ...


class ReplaySchedule(Protocol):
    """Standard forward-only schedule adapter surface."""

    p2p_started: bool
    completed: bool

    def __call__(self, plan: ReplayPlan, probe: ResponseProbe, phase: str) -> None: ...


class NonInterleavedReplaySchedule:
    """Invoke an existing eager noninterleaved schedule for PP1 or PP2+."""

    def __init__(
        self,
        *,
        forward_backward_func: Callable[..., Any],
        forward_step_func: Callable[..., Any],
        model: Sequence[torch.nn.Module],
        sequence_length: int,
        micro_batch_size: int,
        probe_device: torch.device | str,
        tensor_parallel_rank: int,
        tensor_parallel_size: int,
        context_parallel_rank: int = 0,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        pipeline_data_owner: bool = True,
        virtual_pipeline_size: int | None = None,
        decoder_sequence_length: int | None = None,
        adjust_tensor_shapes_fn: Callable[..., Any] | None = None,
        pg_collection: object | None = None,
    ) -> None:
        if virtual_pipeline_size not in (None, 1) or len(model) != 1:
            raise ValueError("Tier-1 replay does not support interleaved/virtual pipeline")
        self.forward_backward_func = forward_backward_func
        self.forward_step_func = forward_step_func
        self.model = tuple(model)
        self.sequence_length = sequence_length
        self.micro_batch_size = micro_batch_size
        self.probe_device = torch.device(probe_device)
        self.tp_rank = tensor_parallel_rank
        self.tp_size = tensor_parallel_size
        self.cp_rank = context_parallel_rank
        self.cp_size = context_parallel_size
        self.sequence_parallel = sequence_parallel
        self.pipeline_data_owner = pipeline_data_owner
        self.decoder_sequence_length = decoder_sequence_length
        self.adjust_tensor_shapes_fn = adjust_tensor_shapes_fn
        self.pg_collection = pg_collection
        self.p2p_started = False
        self.completed = False

    def __call__(self, plan: ReplayPlan, probe: ResponseProbe, phase: str) -> None:
        if phase not in ("pre", "post"):
            raise ValueError("unknown replay schedule phase")
        data_owner = self.pipeline_data_owner and self.tp_rank == 0
        iterator: Iterator[dict[str, Any]] = (
            ReplayIterator(plan) if data_owner else NeutralReplayIterator()
        )
        calls = 0

        def replay_forward(data_iterator: Iterator[Any] | None, model: torch.nn.Module, *args: Any):
            nonlocal calls
            mask = plan.diagnostic_mask(calls, self.probe_device)
            cp_mask, sp_mask = slice_replay_mask(
                mask,
                context_parallel_size=self.cp_size,
                context_parallel_rank=self.cp_rank,
                sequence_parallel=self.sequence_parallel,
                tensor_parallel_size=self.tp_size,
                tensor_parallel_rank=self.tp_rank,
            )
            probe.set_masks(cp_mask, sequence_parallel_mask=sp_mask)
            calls += 1
            output, _reducer = self.forward_step_func(data_iterator, model, *args)

            def zero_reducer(output_tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
                return (torch.zeros((), dtype=torch.float32, device=output_tensor.device), {})

            return output, zero_reducer

        self.p2p_started = True
        self.completed = False
        self.forward_backward_func(
            forward_step_func=replay_forward,
            data_iterator=iterator,
            model=self.model,
            num_microbatches=plan.num_microbatches,
            seq_length=self.sequence_length,
            micro_batch_size=self.micro_batch_size,
            decoder_seq_length=self.decoder_sequence_length,
            forward_only=True,
            collect_non_loss_data=False,
            adjust_tensor_shapes_fn=self.adjust_tensor_shapes_fn,
            force_all_reduce=False,
            pg_collection=self.pg_collection,
        )
        if calls != plan.num_microbatches:
            raise RuntimeError("replay schedule executed an unexpected microbatch count")
        self.completed = True


@dataclass(frozen=True)
class _ReplayScheduleFacts:
    model_ids: tuple[int, ...]
    sequence_length: int
    micro_batch_size: int
    probe_device: torch.device
    tp_rank: int
    tp_size: int
    cp_rank: int
    cp_size: int
    sequence_parallel: bool
    pipeline_data_owner: bool
    forward_backward_id: int
    forward_step_id: int

    @classmethod
    def observe(
        cls,
        schedule: NonInterleavedReplaySchedule,
        models: Sequence[torch.nn.Module],
        plan: ReplayPlan,
    ) -> "_ReplayScheduleFacts":
        if len(schedule.model) != len(models) or any(
            scheduled is not guarded
            for scheduled, guarded in zip(schedule.model, models, strict=True)
        ):
            raise ValueError("replay schedule model tuple is not the guarded engine model tuple")
        if (
            schedule.sequence_length != plan.metadata.sequence_length
            or schedule.micro_batch_size != plan.metadata.micro_batch_size
        ):
            raise ValueError("replay schedule dimensions disagree with the fixed plan")
        if (
            schedule.tp_size <= 0
            or schedule.cp_size <= 0
            or not 0 <= schedule.tp_rank < schedule.tp_size
            or not 0 <= schedule.cp_rank < schedule.cp_size
        ):
            raise ValueError("replay schedule topology is invalid")
        return cls(
            model_ids=tuple(id(model) for model in schedule.model),
            sequence_length=schedule.sequence_length,
            micro_batch_size=schedule.micro_batch_size,
            probe_device=schedule.probe_device,
            tp_rank=schedule.tp_rank,
            tp_size=schedule.tp_size,
            cp_rank=schedule.cp_rank,
            cp_size=schedule.cp_size,
            sequence_parallel=schedule.sequence_parallel,
            pipeline_data_owner=schedule.pipeline_data_owner,
            forward_backward_id=id(schedule.forward_backward_func),
            forward_step_id=id(schedule.forward_step_func),
        )

    def revalidate(
        self,
        schedule: NonInterleavedReplaySchedule,
        models: Sequence[torch.nn.Module],
        plan: ReplayPlan,
    ) -> None:
        if _ReplayScheduleFacts.observe(schedule, models, plan) != self:
            raise RuntimeError("replay schedule changed after memory preflight")


class FatalAbort(Protocol):
    """Job-fatal protocol used after schedule/P2P entry."""

    def __call__(self, error: BaseException) -> None: ...


class ProductionFatalAbort:
    """Close the process group and terminate the local worker immediately."""

    def __init__(self, exit_code: int = 86) -> None:
        self.exit_code = exit_code

    def __call__(self, error: BaseException) -> None:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except RuntimeError:
                pass
        os._exit(self.exit_code)


class TransactionState(StrEnum):
    """Replay transaction lifecycle."""

    READY = "ready"
    PRE_COMPLETE = "pre_complete"
    CLOSED = "closed"


def _validate_dense_gpt_models(models: Sequence[torch.nn.Module]) -> None:
    """Admit only the explicitly inspected dense local-MCore GPT surface."""

    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import ModelType
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer

    allowed_prefixes = (
        "megatron.core.models.common.",
        "megatron.core.models.gpt.",
        "megatron.core.tensor_parallel.",
        "megatron.core.transformer.",
        "megatron.core.fusions.",
        "torch.nn.modules.",
    )

    def validate_spec_value(value: Any) -> None:
        if value is None or type(value) in (bool, int, float, str, bytes):
            return
        if type(value) in (tuple, list):
            for item in value:
                validate_spec_value(item)
            return
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError("dense GPT specs require string mapping keys")
            for item in value.values():
                validate_spec_value(item)
            return
        if type(value) is ModuleSpec:
            if isinstance(value.module, tuple):
                raise TypeError("dense GPT specs cannot use dynamic import tuples")
            validate_spec_value(value.module)
            validate_spec_value(value.params)
            validate_spec_value(value.submodules)
            validate_spec_value(value.metainfo)
            return
        if isinstance(value, type) or callable(value):
            module_name = getattr(value, "__module__", "")
            if not module_name.startswith(allowed_prefixes):
                raise TypeError("dense GPT spec contains an unsupported builder")
            return
        if is_dataclass(value) and type(value).__module__.startswith("megatron.core."):
            for spec_field in fields(value):
                validate_spec_value(getattr(value, spec_field.name))
            return
        raise TypeError(f"dense GPT spec contains unsupported value {type(value).__qualname__}")

    if len(models) != 1:
        raise TypeError("Tier-1 replay requires one noninterleaved dense GPTModel chunk")
    for model in models:
        if type(model) is not GPTModel:
            raise TypeError("Tier-1 replay supports exact MCore GPTModel instances only")
        config = getattr(model, "config", None)
        if type(config) is not TransformerConfig:
            raise TypeError("Tier-1 replay requires an exact TransformerConfig")
        unsupported = {
            "transformer_engine": config.transformer_impl != "local",
            "mixture_of_experts": config.num_moe_experts is not None,
            "expert_parallel": config.expert_model_parallel_size != 1,
            "fp8": config.fp8 is not None or config.fp8_param,
            "recompute": config.recompute_granularity is not None,
            "deferred_embedding_wgrad": config.defer_embedding_wgrad_compute,
            "cpu_offloading": config.cpu_offloading,
            "cuda_graph": config.cuda_graph_impl != "none",
            "virtual_pipeline": config.virtual_pipeline_model_parallel_size not in (None, 1),
            "multi_token_prediction": bool(getattr(model, "mtp_process", False)),
            "multi_latent_attention": config.multi_latent_attention,
        }
        enabled = sorted(name for name, value in unsupported.items() if value)
        if enabled:
            raise ValueError(f"Tier-1 replay does not support GPT capabilities: {enabled}")
        if getattr(model, "model_type", None) != ModelType.encoder_or_decoder:
            raise ValueError("Tier-1 replay requires the dense GPT model type")
        if hasattr(model, "transformer_layer_spec"):
            validate_spec_value(model.transformer_layer_spec)
        for module in model.modules():
            if not type(module).__module__.startswith(allowed_prefixes):
                raise TypeError(
                    f"Tier-1 replay rejects unsupported child module {type(module).__qualname__}"
                )
            if type(module) is TransformerLayer:
                if getattr(module, "is_moe_layer", False):
                    raise ValueError("Tier-1 replay rejects MoE transformer layers")
                if type(module.self_attention) is not SelfAttention or type(module.mlp) is not MLP:
                    raise TypeError("Tier-1 replay requires exact dense local layer children")
                if (
                    type(module.self_attention.linear_qkv) is not ColumnParallelLinear
                    or type(module.mlp.linear_fc1) is not ColumnParallelLinear
                ):
                    raise TypeError("Tier-1 replay requires exact local column-parallel linears")
                if (
                    type(module.self_attention.linear_proj) is not RowParallelLinear
                    or type(module.mlp.linear_fc2) is not RowParallelLinear
                ):
                    raise TypeError("Tier-1 replay requires exact local row-parallel linears")
                if type(module.self_attention.core_attention) is not DotProductAttention:
                    raise TypeError("Tier-1 replay requires exact local dot-product attention")


class Tier1ReplayTransaction:
    """Split pre/post replay around an externally owned optimizer update."""

    def __init__(
        self,
        *,
        models: Sequence[torch.nn.Module],
        plan: ReplayPlan,
        probe: ResponseProbe,
        schedule: ReplaySchedule,
        readiness: ReadinessConsensus,
        mutable_buffer_names: Sequence[str],
        tracker_getter: Callable[[], Any] | None,
        cuda_device: torch.device | int | None,
        samplers: Sequence[object],
        overlap_objects: Sequence[object],
        fatal_abort: FatalAbort,
        maximum_model_state_bytes: int = _CAPACITY_LIMIT,
        memory_estimate: ReplayMemoryEstimate | None = None,
        preflight_validator: Callable[[], None] | None = None,
    ) -> None:
        self.models = tuple(models)
        self.plan = plan
        self.probe = probe
        self.schedule = schedule
        self.readiness = readiness
        self.mutable_buffer_names = tuple(mutable_buffer_names)
        self.tracker_getter = tracker_getter
        self.cuda_device = cuda_device
        self.samplers = tuple(samplers)
        self.overlap_objects = tuple(overlap_objects)
        self.fatal_abort = fatal_abort
        self.maximum_model_state_bytes = maximum_model_state_bytes
        self.memory_estimate = memory_estimate
        self.preflight_validator = preflight_validator
        self.rng_a: ReplayRngState | None = None
        self.state = TransactionState.READY

    def _guard(self) -> ReplayStateGuard:
        return ReplayStateGuard(
            self.models,
            mutable_buffer_names=self.mutable_buffer_names,
            samplers=self.samplers,
            overlap_objects=self.overlap_objects,
            tracker_getter=self.tracker_getter,
            cuda_device=self.cuda_device,
            maximum_model_state_bytes=self.maximum_model_state_bytes,
        )

    def _fatal(self, error: BaseException) -> None:
        fatal_error = error
        cleanup_error = self._release_local()
        try:
            self.readiness.settle(cleanup_error, "fatal transaction cleanup")
        except BaseException as collective_cleanup_error:
            fatal_error = BaseExceptionGroup(
                "Tier-1 post-commit failure and cleanup failure", (error, collective_cleanup_error)
            )
        self.fatal_abort(fatal_error)
        raise RuntimeError("Tier-1 fatal-abort protocol returned") from error

    def _release_local(self) -> BaseException | None:
        if self.state == TransactionState.CLOSED:
            return None
        failures: list[tuple[str, BaseException]] = []
        for name, operation in (
            ("probe_release", self.probe.release),
            ("plan_release", self.plan.release),
            ("rng_release", lambda: setattr(self, "rng_a", None)),
        ):
            try:
                operation()
            except BaseException as error:
                failures.append((name, error))
        self.state = TransactionState.CLOSED
        return ReplayRestorationError(failures) if failures else None

    def _release_collectively(self, phase: str) -> None:
        cleanup_error = self._release_local()
        try:
            self.readiness.settle(cleanup_error, phase)
        except BaseException as collective_error:
            self.fatal_abort(collective_error)
            raise RuntimeError("Tier-1 fatal-abort protocol returned") from collective_error

    def _prepare_phase(self, phase: str) -> ExitStack:
        """Prepare guard, hooks, modes, and RNG before the last readiness gate."""

        stack = ExitStack()
        error: BaseException | None = None
        try:
            guard = self._guard()
            guard.prepare()
            stack.enter_context(guard)
            stack.enter_context(torch.no_grad())
            stack.enter_context(
                self.probe.capture_pre() if phase == "pre" else self.probe.capture_post()
            )
            assert self.rng_a is not None
            for _name, operation in self.rng_a.restore_stages(
                tracker_getter=self.tracker_getter, cuda_device=self.cuda_device
            ):
                operation()
        except BaseException as caught:
            error = caught
        try:
            self.readiness.settle(error, f"{phase}-schedule state preparation")
        except BaseException as readiness_error:
            cleanup_error: BaseException | None = None
            try:
                stack.close()
            except BaseException as caught:
                cleanup_error = caught
            self.readiness.settle(cleanup_error, f"{phase}-schedule preparation cleanup")
            raise readiness_error
        return stack

    def _run_schedule_phase(self, phase: str) -> None:
        stack = self._prepare_phase(phase)
        error: BaseException | None = None
        try:
            self.schedule.p2p_started = True
            self.schedule.completed = False
            self.schedule(self.plan, self.probe, phase)
            self.schedule.completed = True
        except BaseException as caught:
            error = caught
        try:
            stack.close()
        except BaseException as caught:
            error = (
                caught
                if error is None
                else BaseExceptionGroup(
                    f"Tier-1 {phase} schedule and restoration failed", (error, caught)
                )
            )
        try:
            self.readiness.settle(error, f"{phase}-schedule committed outcome")
        except BaseException as caught:
            self._fatal(caught)

    def run_pre(self) -> None:
        """Run pre-update replay from RNG state A and restore ambient state."""

        if self.state != TransactionState.READY:
            raise RuntimeError("pre replay can run exactly once")
        error: BaseException | None = None
        try:
            if self.preflight_validator is not None:
                self.preflight_validator()
            self.rng_a = ReplayRngState.capture(
                tracker_getter=self.tracker_getter, cuda_device=self.cuda_device
            )
        except BaseException as caught:
            error = caught
        self.readiness.settle(error, "pre-schedule RNG capture")
        assert self.rng_a is not None
        self._run_schedule_phase("pre")
        self.state = TransactionState.PRE_COMPLETE

    def finish(self, *, update_succeeded: bool) -> Any | None:
        """Run post-update replay from A, or release pre state after overflow."""

        if self.state != TransactionState.PRE_COMPLETE or self.rng_a is None:
            raise RuntimeError("post replay requires a completed pre replay")
        if not update_succeeded:
            self._release_collectively("overflow transaction cleanup")
            return None
        self._run_schedule_phase("post")
        result: Any | None = None
        finalize_error: BaseException | None = None
        try:
            result = self.probe.finalize()
        except BaseException as caught:
            finalize_error = caught
        try:
            self.readiness.settle(finalize_error, "post-schedule finalization")
        except BaseException as caught:
            self._fatal(caught)
        self._release_collectively("successful post transaction cleanup")
        return result

    def release(self) -> None:
        """Release plan, hooks, retained responses, and RNG snapshots."""

        self._release_collectively("explicit transaction cleanup")


class Tier1ReplayEngine:
    """Mandatory-preflight factory for an unintegrated Tier-1 transaction."""

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        *,
        readiness: ReadinessConsensus,
        mutable_buffer_names: Sequence[str] = (),
        tracker_getter: Callable[[], Any] | None = None,
        cuda_device: torch.device | int | None = None,
        samplers: Sequence[object] = (),
        overlap_objects: Sequence[object] = (),
        fatal_abort: FatalAbort | None = None,
    ) -> None:
        self.models = tuple(models)
        self.readiness = readiness
        self.mutable_buffer_names = tuple(mutable_buffer_names)
        self.tracker_getter = tracker_getter
        self.cuda_device = cuda_device
        self.samplers = tuple(samplers)
        self.overlap_objects = tuple(overlap_objects)
        self.fatal_abort = fatal_abort or ProductionFatalAbort()

    def prepare(
        self,
        *,
        plan: ReplayPlan,
        probe: ResponseProbe,
        schedule: ReplaySchedule,
        memory_policy: ReplayMemoryPolicy,
        maximum_extra_bytes: int,
        currently_reserved_bytes: int,
        total_device_bytes: int,
    ) -> Tier1ReplayTransaction:
        """Globally settle plan, descriptor, tracker, and mandatory memory readiness."""

        verify_replay_plan_consensus(
            plan,
            binding=self.readiness.binding,
            readiness=self.readiness,
            device=self.readiness.status.device,
        )
        response_minimum: torch.Tensor | None = None
        response_maximum: torch.Tensor | None = None
        error: BaseException | None = None
        try:
            response_hash = torch.tensor(
                list(bytes.fromhex(probe.descriptor_hash)),
                dtype=torch.uint8,
                device=self.readiness.status.device,
            )
            response_minimum = response_hash.clone()
            response_maximum = response_hash.clone()
        except BaseException as caught:
            error = caught
        self.readiness.settle(error, "response descriptor construction")
        assert response_minimum is not None and response_maximum is not None
        if self.readiness.binding.size > 1:
            dist.all_reduce(
                response_minimum, op=dist.ReduceOp.MIN, group=self.readiness.binding.group
            )
            dist.all_reduce(
                response_maximum, op=dist.ReduceOp.MAX, group=self.readiness.binding.group
            )
            error = (
                ValueError("Tier-1 response descriptor hash differs across ranks")
                if not torch.equal(response_minimum, response_maximum)
                else None
            )
            self.readiness.settle(error, "response descriptor identity")
        error = None
        memory_estimate: ReplayMemoryEstimate | None = None
        plan_facts: _ReplayPlanFacts | None = None
        schedule_facts: _ReplayScheduleFacts | None = None
        model_config_facts: tuple[Any, ...] | None = None
        try:
            from megatron.core.transformer.transformer_layer import TransformerLayer

            from .function_response import RESPONSE_FAMILIES, FunctionResponseProbe, ResponseFamily

            if type(probe) is not FunctionResponseProbe:
                raise TypeError("Tier-1 replay requires the exact bounded response probe")
            if probe.expected_hook_calls != plan.num_microbatches:
                raise ValueError("response hook cardinality disagrees with replay plan")
            if not plan.metadata.descriptor_hash:
                raise ValueError("replay plan has no canonical descriptor hash")
            plan_facts = _ReplayPlanFacts.observe(plan)
            if self.tracker_getter is None:
                raise TypeError("Tier-1 replay requires an inspected Megatron RNG tracker")
            tracker = _validated_tracker(self.tracker_getter)
            if self.tracker_getter() is not tracker:
                raise TypeError("Megatron RNG tracker getter is not identity-stable")
            _validate_dense_gpt_models(self.models)
            if type(schedule) is not NonInterleavedReplaySchedule:
                raise TypeError("Tier-1 replay requires the verified noninterleaved schedule")
            schedule_facts = _ReplayScheduleFacts.observe(schedule, self.models, plan)
            memory_policy.validate()
            state_guard = ReplayStateGuard(
                self.models,
                mutable_buffer_names=self.mutable_buffer_names,
                samplers=self.samplers,
                overlap_objects=self.overlap_objects,
                tracker_getter=self.tracker_getter,
                cuda_device=self.cuda_device,
                maximum_model_state_bytes=maximum_extra_bytes,
            )
            state_guard.prepare()
            assert state_guard.rng is not None
            rng_snapshot_bytes = state_guard.rng.tensor_bytes
            model_state_bytes = state_guard.model.tensor_bytes
            with state_guard:
                pass
            model_config = self.models[0].config
            if any(model.config is not model_config for model in self.models):
                raise ValueError("pipeline model chunks have inconsistent GPT configs")
            model_config_facts = (
                id(model_config),
                model_config.num_layers,
                model_config.hidden_size,
                model_config.ffn_hidden_size,
                model_config.num_attention_heads,
                model_config.params_dtype,
                model_config.gated_linear_unit,
                model_config.tensor_model_parallel_size,
                model_config.context_parallel_size,
                model_config.pipeline_model_parallel_size,
            )
            global_layers = getattr(probe, "global_layers", None)
            descriptors = getattr(probe, "descriptors", None)
            if not isinstance(global_layers, int) or not isinstance(descriptors, tuple):
                raise TypeError("response probe lacks verified layer descriptors")
            descriptor_layers = {descriptor.global_layer for descriptor in descriptors}
            local_layers = len(descriptor_layers)
            if local_layers <= 0:
                raise ValueError("response probe has no local dense GPT layers")
            if len(descriptors) != local_layers * len(RESPONSE_FAMILIES) or any(
                {
                    descriptor.family
                    for descriptor in descriptors
                    if descriptor.global_layer == layer
                }
                != set(RESPONSE_FAMILIES)
                for layer in descriptor_layers
            ):
                raise ValueError("response probe does not cover every local response family")
            model_module_ids = {id(module) for model in self.models for module in model.modules()}
            if any(id(descriptor.module) not in model_module_ids for descriptor in descriptors):
                raise ValueError("response probe references a module outside the guarded model")
            transformer_layers = {
                module.layer_number - 1: module
                for model in self.models
                for module in model.modules()
                if type(module) is TransformerLayer
            }
            if set(transformer_layers) != descriptor_layers:
                raise ValueError("response probe layer set disagrees with the dense GPT graph")
            expected_modules = {
                layer: {
                    ResponseFamily.RESIDUAL: module,
                    ResponseFamily.QKV: module.self_attention.linear_qkv,
                    ResponseFamily.ATTN_OUT: module.self_attention.linear_proj,
                    ResponseFamily.FC1: module.mlp.linear_fc1,
                    ResponseFamily.FC2: module.mlp.linear_fc2,
                }
                for layer, module in transformer_layers.items()
            }
            if any(
                descriptor.module
                is not expected_modules[descriptor.global_layer][descriptor.family]
                for descriptor in descriptors
            ):
                raise ValueError("response hooks are not bound to canonical dense GPT modules")
            if probe.device != schedule.probe_device:
                raise ValueError("response probe and replay schedule devices disagree")
            params_dtype = model_config.params_dtype
            if not isinstance(params_dtype, torch.dtype):
                raise TypeError("GPT parameter dtype is not a torch dtype")
            cp_local = plan.metadata.sequence_length // schedule.cp_size
            if (
                model_config.tensor_model_parallel_size != schedule.tp_size
                or model_config.context_parallel_size != schedule.cp_size
            ):
                raise ValueError("schedule topology disagrees with the GPT config")
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            model_parallel_size = _checked_product(
                schedule.tp_size, schedule.cp_size, model_config.pipeline_model_parallel_size
            )
            if world_size % model_parallel_size:
                raise ValueError("world size is not divisible by model parallel topology")
            data_parallel_size = world_size // model_parallel_size
            gated_linear_unit = bool(model_config.gated_linear_unit)
            response_widths = {
                ResponseFamily.RESIDUAL: model_config.hidden_size,
                ResponseFamily.QKV: 3 * model_config.hidden_size // schedule.tp_size,
                ResponseFamily.ATTN_OUT: model_config.hidden_size,
                ResponseFamily.FC1: (
                    (1 + int(gated_linear_unit)) * model_config.ffn_hidden_size // schedule.tp_size
                ),
                ResponseFamily.FC2: model_config.hidden_size,
            }
            local_attention_heads = model_config.num_attention_heads // schedule.tp_size
            probe.bind_preflight(
                selected_row_capacity=plan_facts.selected_tokens,
                response_widths=response_widths,
                response_dtype=params_dtype,
                attention_heads=local_attention_heads,
                attention_key_length=plan.metadata.sequence_length,
            )
            probe.validate_preflight_binding()
            fixed_workspace_bytes = _checked_sum(
                _checked_product(plan.metadata.micro_batch_size, plan.metadata.sequence_length),
                _checked_product(plan.metadata.micro_batch_size, cp_local),
                _checked_product(
                    plan.metadata.micro_batch_size,
                    cp_local // schedule.tp_size if schedule.sequence_parallel else 0,
                ),
            )
            expected_host_bytes = _checked_product(
                plan.num_microbatches,
                _checked_sum(
                    _checked_product(
                        plan.metadata.micro_batch_size,
                        plan.metadata.sequence_length,
                        8 + 8 + 4 + 8 + 1,
                    ),
                    _checked_product(plan.metadata.micro_batch_size, 2, 8),
                ),
            )
            if plan.source_rank and plan_facts.host_tensor_bytes != expected_host_bytes:
                raise ValueError("source replay tensors disagree with fixed-field byte accounting")
            attention_workspace_bytes = _checked_product(
                8,
                plan_facts.selected_tokens,
                local_attention_heads,
                plan.metadata.sequence_length,
                max(torch.empty((), dtype=params_dtype).element_size(), 4),
            )
            actual_config = ReplayMemoryConfig(
                sequence_length=plan.metadata.sequence_length,
                micro_batch_size=plan.metadata.micro_batch_size,
                global_batch_size=plan.metadata.global_selected_tokens,
                data_parallel_size=data_parallel_size,
                tensor_parallel_size=schedule.tp_size,
                pipeline_parallel_size=model_config.pipeline_model_parallel_size,
                context_parallel_size=schedule.cp_size,
                global_layers=global_layers,
                local_layers=local_layers,
                hidden_size=model_config.hidden_size,
                ffn_hidden_size=model_config.ffn_hidden_size,
                num_attention_heads=model_config.num_attention_heads,
                selected_tokens=plan.metadata.global_selected_tokens,
                replay_microbatches=plan.num_microbatches,
                element_size=torch.empty((), dtype=params_dtype).element_size(),
                rng_snapshot_bytes=rng_snapshot_bytes,
                model_state_bytes=model_state_bytes,
                fixed_workspace_bytes=fixed_workspace_bytes,
                alignment=memory_policy.alignment,
                headroom_fraction=memory_policy.headroom_fraction,
                tp_source=True,
                sequence_parallel=schedule.sequence_parallel,
                gated_linear_unit=gated_linear_unit,
                observed_host_replay_bytes=expected_host_bytes,
                observed_packed_statistics_bytes=probe.packed_statistics_bytes,
                observed_reduction_arena_bytes=probe.reduction_arena_bytes,
                observed_accumulator_scratch_bytes=probe.maximum_accumulator_scratch_bytes,
                observed_attention_workspace_bytes=attention_workspace_bytes,
            )
            memory_estimate = estimate_replay_memory(actual_config)
            if plan.num_microbatches and memory_estimate.predicted_reserved_bytes <= 0:
                raise ReplayPreflightError(
                    "nonempty replay plans require a nonzero memory preflight"
                )
            preflight_replay_memory(
                memory_estimate,
                maximum_extra_bytes=maximum_extra_bytes,
                currently_reserved_bytes=currently_reserved_bytes,
                total_device_bytes=total_device_bytes,
            )
        except BaseException as caught:
            error = caught
        self.readiness.settle(error, "transaction preflight")
        assert (
            memory_estimate is not None
            and plan_facts is not None
            and schedule_facts is not None
            and model_config_facts is not None
        )

        def revalidate_preflight() -> None:
            plan_facts.revalidate(plan)
            schedule_facts.revalidate(schedule, self.models, plan)
            probe.validate_preflight_binding()
            current_config = self.models[0].config
            current_facts = (
                id(current_config),
                current_config.num_layers,
                current_config.hidden_size,
                current_config.ffn_hidden_size,
                current_config.num_attention_heads,
                current_config.params_dtype,
                current_config.gated_linear_unit,
                current_config.tensor_model_parallel_size,
                current_config.context_parallel_size,
                current_config.pipeline_model_parallel_size,
            )
            if current_facts != model_config_facts:
                raise RuntimeError("dense GPT configuration changed after memory preflight")

        return Tier1ReplayTransaction(
            models=self.models,
            plan=plan,
            probe=probe,
            schedule=schedule,
            readiness=self.readiness,
            mutable_buffer_names=self.mutable_buffer_names,
            tracker_getter=self.tracker_getter,
            cuda_device=self.cuda_device,
            samplers=self.samplers,
            overlap_objects=self.overlap_objects,
            fatal_abort=self.fatal_abort,
            maximum_model_state_bytes=maximum_extra_bytes,
            memory_estimate=memory_estimate,
            preflight_validator=revalidate_preflight,
        )
