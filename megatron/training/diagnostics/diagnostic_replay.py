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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum
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
_PLAN_MAGIC = 0x5449455231504C4E
_PLAN_VERSION = 1


def _hash64(*values: int) -> int:
    payload = b"".join(
        int(value).to_bytes(8, "little", signed=True) for value in values
    )
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


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
        result[SAMPLE_INDEX_FIELD] = torch.tensor(
            index.sampler_index, dtype=torch.int64
        )
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
        if (
            isinstance(value, torch.Tensor)
            and value.ndim
            and value.shape[0] == self.batch_size
        ):
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
            torch.nonzero(self.loss_mask > 0, as_tuple=False).flatten().tolist()
        )


@dataclass(frozen=True)
class RecordedBatch:
    """Retain references to one raw CPU batch and its stable identities."""

    raw: Mapping[str, Any]
    samples: tuple[RecordedSample, ...]

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "RecordedBatch":
        """Validate a raw pre-broadcast batch and retain only references."""

        if SAMPLE_INDEX_FIELD not in raw or SAMPLE_EPOCH_FIELD not in raw:
            raise ValueError(
                "Tier-1 replay requires stable sample index and epoch fields"
            )
        indices = torch.as_tensor(
            raw[SAMPLE_INDEX_FIELD], dtype=torch.int64, device="cpu"
        ).view(-1)
        epochs = torch.as_tensor(
            raw[SAMPLE_EPOCH_FIELD], dtype=torch.int64, device="cpu"
        ).view(-1)
        if indices.numel() == 0 or indices.numel() != epochs.numel():
            raise ValueError("sample identity fields must have equal nonzero length")
        for value in raw.values():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise ValueError("raw replay batches must be captured on CPU")
        batch_size = indices.numel()
        samples = tuple(
            RecordedSample(
                SampleId(int(epochs[lane]), int(indices[lane])), raw, lane, batch_size
            )
            for lane in range(batch_size)
        )
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError(
                "stable sample identities must be unique within a raw batch"
            )
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

    def __init__(self, source: Iterator[Mapping[str, Any]]) -> None:
        self.source = source
        self.recorded: list[RecordedBatch] = []

    def __iter__(self) -> "ReplayBatchRecorder":
        return self

    def __next__(self) -> Mapping[str, Any]:
        raw = next(self.source)
        self.recorded.append(RecordedBatch.from_raw(raw))
        return raw

    @property
    def host_capture_bytes(self) -> int:
        """Return event-local referenced host tensor bytes."""

        return sum(batch.tensor_bytes for batch in self.recorded)

    def clear(self) -> None:
        """Release references without advancing or replacing the source iterator."""

        self.recorded.clear()


@dataclass(frozen=True)
class SamplePopulation:
    """Stable identity and number of selectable token positions."""

    sample_id: SampleId
    valid_count: int


def local_sample_populations(
    recorded: Sequence[RecordedBatch],
) -> tuple[SamplePopulation, ...]:
    """Return validated local sample populations."""

    samples = tuple(sample for batch in recorded for sample in batch.samples)
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("recorded sample identities must be unique on a data owner")
    return tuple(
        SamplePopulation(sample.sample_id, len(sample.valid_columns))
        for sample in samples
    )


@dataclass(frozen=True)
class CollectiveBinding:
    """Bind a pre-existing process group to its declared fixed size."""

    identity: str
    group: object | None
    size: int

    def validate(self) -> None:
        """Reject a declared topology that disagrees with the real group."""

        if not self.identity or self.size <= 0:
            raise ValueError(
                "collective bindings require an identity and positive size"
            )
        if dist.is_available() and dist.is_initialized():
            if dist.get_world_size(self.group) != self.size:
                raise ValueError(
                    f"{self.identity} binding size disagrees with its process group"
                )
        elif self.size != 1 or self.group is not None:
            raise RuntimeError(
                "multi-rank bindings require initialized torch.distributed"
            )


class ReplayPreflightError(RuntimeError):
    """Raised by a globally settled failure before schedule/P2P entry."""


class ReadinessConsensus:
    """Own a status scalar allocated before any replay-risky phase."""

    def __init__(
        self, binding: CollectiveBinding, device: torch.device | str = "cpu"
    ) -> None:
        binding.validate()
        self.binding = binding
        self.status = torch.ones(1, dtype=torch.int32, device=device)

    def settle(self, error: BaseException | None, phase: str) -> None:
        """Make one local preflight result identical across the bound group."""

        self.status.fill_(0 if error is not None else 1)
        if self.binding.size > 1:
            dist.all_reduce(self.status, op=dist.ReduceOp.MIN, group=self.binding.group)
        if not bool(self.status.item()):
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
        if maximum_local_samples <= 0:
            raise ValueError("population workspace needs a positive local sample cap")
        self.binding = binding
        self.maximum_local_samples = maximum_local_samples
        self.local_count = torch.zeros(1, dtype=torch.int64, device=device)
        self.counts = torch.zeros(binding.size, dtype=torch.int64, device=device)
        self.local = torch.full(
            (maximum_local_samples, 3), -1, dtype=torch.int64, device=device
        )
        self.gathered = torch.full(
            (binding.size * maximum_local_samples, 3),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.schedule_count = torch.zeros(1, dtype=torch.int64, device=device)

    def gather(
        self,
        local: Sequence[SamplePopulation],
        readiness: ReadinessConsensus,
    ) -> tuple[SamplePopulation, ...]:
        """Exchange fixed sample metadata and reject duplicate global identities."""

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
            dist.all_gather_into_tensor(
                self.counts, self.local_count, group=self.binding.group
            )
            dist.all_gather_into_tensor(
                self.gathered, self.local, group=self.binding.group
            )
        else:
            self.counts.copy_(self.local_count)
            self.gathered.copy_(self.local)
        populations = []
        for rank, count in enumerate(self.counts.cpu().tolist()):
            start = rank * self.maximum_local_samples
            for epoch, sample, valid in (
                self.gathered[start : start + count].cpu().tolist()
            ):
                populations.append(SamplePopulation(SampleId(epoch, sample), valid))
        if len({population.sample_id for population in populations}) != len(
            populations
        ):
            raise ValueError(
                "DP population contains duplicate stable sample identities"
            )
        return tuple(populations)

    def maximum_schedule_count(self, local_count: int) -> int:
        """Return the fixed DP maximum schedule length."""

        if local_count < 0:
            raise ValueError("local replay count cannot be negative")
        self.schedule_count.fill_(local_count)
        if self.binding.size > 1:
            dist.all_reduce(
                self.schedule_count, op=dist.ReduceOp.MAX, group=self.binding.group
            )
        return int(self.schedule_count.item())


@dataclass
class ReplayMicrobatch:
    """One fixed-shape replay microbatch on a TP source rank."""

    data: dict[str, Any]
    sample_ids: tuple[SampleId, ...]

    def model_batch(self) -> dict[str, Any]:
        """Return model-path fields, including the diagnostic mask."""

        return {
            key: value
            for key, value in self.data.items()
            if key not in _RESERVED_FIELDS
        }

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
        ):
            digest.update(value.to_bytes(8, "little", signed=True))
        for token in self.selected_tokens:
            for value in (
                token.sample.epoch,
                token.sample.sampler_index,
                token.sequence_column,
            ):
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

    def diagnostic_mask(
        self, microbatch: int, device: torch.device | str
    ) -> torch.Tensor:
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
            self.metadata.micro_batch_size,
            self.metadata.sequence_length,
            0,
            (),
            (),
            (),
        )


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
            if not all(
                isinstance(row, torch.Tensor) and row.shape == first.shape
                for row in rows
            ):
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
            diagnostic_mask[lane, column] = (
                TokenId(sample.sample_id, column) in selected
            )
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
    """Systematically select global positions and return locally owned IDs."""

    if probe_tokens < 0:
        raise ValueError("probe token count must be nonnegative")
    ordered = sorted(
        global_population,
        key=lambda item: (
            _hash64(
                run_seed, event_id, item.sample_id.epoch, item.sample_id.sampler_index
            ),
            item.sample_id,
        ),
    )
    total = sum(item.valid_count for item in ordered)
    selected_count = min(probe_tokens, total)
    if not selected_count:
        return ()
    positions = (
        set(range(total))
        if selected_count == total
        else {
            ((2 * index + 1) * total) // (2 * selected_count)
            for index in range(selected_count)
        }
    )
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
            columns = sorted(
                sample.valid_columns,
                key=lambda column: (
                    _hash64(
                        run_seed,
                        event_id,
                        sample.sample_id.epoch,
                        sample.sample_id.sampler_index,
                        column,
                    ),
                    column,
                ),
            )
            if len(columns) != population.valid_count:
                raise ValueError("local loss mask disagrees with gathered population")
            result.extend(
                TokenId(sample.sample_id, columns[offset]) for offset in offsets
            )
        cursor += population.valid_count
    return tuple(sorted(result))


def build_local_replay_plan(
    recorded: Sequence[RecordedBatch],
    selected_tokens: Sequence[TokenId],
    *,
    micro_batch_size: int,
    target_microbatches: int | None = None,
) -> ReplayPlan:
    """Reconstruct deterministic fixed batches without advancing a sampler."""

    if micro_batch_size <= 0:
        raise ValueError("replay micro batch size must be positive")
    by_id = {sample.sample_id: sample for batch in recorded for sample in batch.samples}
    if len(by_id) != sum(len(batch.samples) for batch in recorded):
        raise ValueError("recorded sample identities must be unique")
    selected_set = set(selected_tokens)
    selected_samples = [
        by_id[sample_id] for sample_id in sorted({t.sample for t in selected_set})
    ]
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
    sequence_length = (
        int(batches[0].data["loss_mask"].shape[1])
        if batches
        else int(next(iter(by_id.values())).loss_mask.numel())
        if by_id
        else 0
    )
    masks = tuple(
        tuple(
            bool(value)
            for value in batch.data[DIAGNOSTIC_MASK_FIELD].reshape(-1).tolist()
        )
        for batch in batches
    )
    metadata = ReplayPlanMetadata(
        micro_batch_size=micro_batch_size,
        sequence_length=sequence_length,
        num_microbatches=len(batches),
        selected_tokens=tuple(selected_tokens),
        ordered_sample_ids=tuple(
            sample for batch in batches for sample in batch.sample_ids
        ),
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
    if probe_tokens <= 0 or not any(
        population.valid_count for population in global_population
    ):
        readiness.settle(
            ReplayPreflightError("Tier-1 replay has no valid tokens"), "selection"
        )
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
            recorded, selected, micro_batch_size=micro_batch_size
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

        return (
            7
            + 3 * self.maximum_tokens
            + 2 * self.maximum_microbatches * self.micro_batch_size
        )

    @property
    def mask_count(self) -> int:
        """Return the fixed uint8 mask tensor length."""

        return self.maximum_microbatches * self.micro_batch_size * self.sequence_length

    def allocate(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate one reusable fixed wire pair."""

        return (
            torch.full((self.integer_count,), -1, dtype=torch.int64, device=device),
            torch.zeros((self.mask_count,), dtype=torch.uint8, device=device),
        )

    def encode(
        self, plan: ReplayPlan, integer: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Encode one source plan into preallocated tensors."""

        metadata = plan.metadata
        if (
            metadata.micro_batch_size != self.micro_batch_size
            or metadata.sequence_length != self.sequence_length
            or metadata.num_microbatches > self.maximum_microbatches
            or len(metadata.selected_tokens) > self.maximum_tokens
        ):
            raise ValueError("replay plan exceeds its fixed codec")
        if integer.numel() != self.integer_count or mask.numel() != self.mask_count:
            raise ValueError("replay codec received incorrectly sized wire tensors")
        integer.fill_(-1)
        mask.zero_()
        integer[:7] = torch.tensor(
            (
                _PLAN_MAGIC,
                _PLAN_VERSION,
                metadata.num_microbatches,
                metadata.micro_batch_size,
                metadata.sequence_length,
                len(metadata.selected_tokens),
                len(metadata.ordered_sample_ids),
            ),
            dtype=torch.int64,
            device=integer.device,
        )
        offset = 7
        for token in metadata.selected_tokens:
            integer[offset : offset + 3] = torch.tensor(
                (token.sample.epoch, token.sample.sampler_index, token.sequence_column),
                dtype=torch.int64,
                device=integer.device,
            )
            offset += 3
        offset = 7 + 3 * self.maximum_tokens
        for sample in metadata.ordered_sample_ids:
            integer[offset : offset + 2] = torch.tensor(
                (sample.epoch, sample.sampler_index),
                dtype=torch.int64,
                device=integer.device,
            )
            offset += 2
        for index, values in enumerate(metadata.diagnostic_masks):
            start = index * self.micro_batch_size * self.sequence_length
            mask[start : start + len(values)] = torch.tensor(
                values, dtype=torch.uint8, device=mask.device
            )

    def decode(self, integer: torch.Tensor, mask: torch.Tensor) -> ReplayPlanMetadata:
        """Validate and decode one fixed wire pair."""

        header = tuple(int(value) for value in integer[:7].cpu().tolist())
        magic, version, count, mbs, sequence, selected_count, sample_count = header
        if magic != _PLAN_MAGIC or version != _PLAN_VERSION:
            raise ValueError("invalid replay plan wire identity")
        if not 0 <= count <= self.maximum_microbatches:
            raise ValueError("invalid replay microbatch count")
        if mbs != self.micro_batch_size or sequence != self.sequence_length:
            raise ValueError("replay plan wire shape disagrees with its codec")
        if not 0 <= selected_count <= self.maximum_tokens:
            raise ValueError("invalid replay selected-token count")
        expected_samples = count * mbs
        if sample_count != expected_samples:
            raise ValueError("replay sample metadata cardinality is inconsistent")
        offset = 7
        tokens = []
        for _ in range(selected_count):
            epoch, sample, column = (
                int(value) for value in integer[offset : offset + 3]
            )
            tokens.append(TokenId(SampleId(epoch, sample), column))
            offset += 3
        offset = 7 + 3 * self.maximum_tokens
        samples = []
        for _ in range(sample_count):
            epoch, sample = (int(value) for value in integer[offset : offset + 2])
            samples.append(SampleId(epoch, sample))
            offset += 2
        width = mbs * sequence
        masks = tuple(
            tuple(
                bool(value)
                for value in mask[index * width : (index + 1) * width].cpu().tolist()
            )
            for index in range(count)
        )
        return ReplayPlanMetadata(
            mbs, sequence, count, tuple(tokens), tuple(samples), masks
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
        global_source = dist.get_global_rank(binding.group, source_group_rank)
        dist.broadcast(integer, src=global_source, group=binding.group)
        dist.broadcast(mask, src=global_source, group=binding.group)
    metadata = codec.decode(integer, mask)
    if source_plan is not None:
        if metadata.descriptor_hash != source_plan.metadata.descriptor_hash:
            raise ReplayPreflightError(
                "source replay plan changed during typed broadcast"
            )
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


def cp_sequence_columns(
    sequence_length: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """Return Megatron's two-chunk zigzag CP column indices."""

    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError("invalid context-parallel topology")
    if sequence_length % (2 * cp_size):
        raise ValueError("sequence length must be divisible by twice the CP size")
    width = sequence_length // (2 * cp_size)
    chunks = (cp_rank, 2 * cp_size - cp_rank - 1)
    return torch.tensor(
        [
            column
            for chunk in chunks
            for column in range(chunk * width, (chunk + 1) * width)
        ],
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
    columns = cp_sequence_columns(
        mask.shape[1], context_parallel_size, context_parallel_rank
    )
    cp_mask = mask.index_select(1, columns.to(mask.device))
    if not sequence_parallel:
        return cp_mask, None
    if (
        tensor_parallel_size <= 0
        or not 0 <= tensor_parallel_rank < tensor_parallel_size
    ):
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
    if not isinstance(states, Mapping) or not all(
        isinstance(name, str) for name in states
    ):
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
        cls,
        *,
        tracker_getter: Callable[[], Any] | None,
        cuda_device: torch.device | int | None,
    ) -> "ReplayRngState":
        """Capture Python, NumPy, Torch, CUDA, and validated Megatron tracker state."""

        tracker = _validated_tracker(tracker_getter)
        states = (
            {
                name: _clone_tracker_value(value)
                for name, value in tracker.get_states().items()
            }
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
            torch.cuda.get_rng_state(cuda_device).clone()
            if torch.cuda.is_available()
            else None
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
            *(
                value
                for value in self.tracker_states.values()
                if isinstance(value, torch.Tensor)
            ),
        ]
        if self.cuda_state is not None:
            values.append(self.cuda_state)
        return sum(value.numel() * value.element_size() for value in values)

    def restore_stages(
        self,
        *,
        tracker_getter: Callable[[], Any] | None,
        cuda_device: torch.device | int | None,
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
                if not torch.equal(
                    torch.cuda.get_rng_state(cuda_device), self.cuda_state
                ):
                    raise RuntimeError("CUDA RNG restoration verification failed")

        def restore_tracker() -> None:
            if tracker_getter is None:
                if self.tracker_states:
                    raise RuntimeError(
                        "captured tracker state has no restoration getter"
                    )
                return
            tracker = _validated_tracker(tracker_getter)
            assert tracker is not None
            tracker.set_states(
                {
                    name: _clone_tracker_value(value)
                    for name, value in self.tracker_states.items()
                }
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
                raise RuntimeError(
                    "Megatron RNG tracker restoration verification failed"
                )

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
        elif isinstance(value, list):
            contents = tuple(cls.capture(item) for item in value)
        elif isinstance(value, dict):
            contents = tuple((key, cls.capture(item)) for key, item in value.items())
        else:
            contents = value
        return cls(value, contents)

    def restore(self) -> Any:
        if isinstance(self.original, torch.Tensor):
            self.original.copy_(self.contents)
        elif isinstance(self.original, list):
            self.original.clear()
            self.original.extend(item.restore() for item in self.contents)
        elif isinstance(self.original, dict):
            self.original.clear()
            self.original.update((key, item.restore()) for key, item in self.contents)
        return self.original

    def verify(self, value: Any) -> bool:
        if value is not self.original:
            return False
        if isinstance(value, torch.Tensor):
            return torch.equal(value, self.contents)
        if isinstance(value, list):
            return len(value) == len(self.contents) and all(
                snapshot.verify(item)
                for snapshot, item in zip(self.contents, value, strict=True)
            )
        if isinstance(value, dict):
            return tuple(value) == tuple(key for key, _ in self.contents) and all(
                snapshot.verify(value[key]) for key, snapshot in self.contents
            )
        return value is self.original or value == self.contents


@dataclass
class _AttributeSnapshot:
    owner: object
    name: str
    value: _ValueSnapshot

    def restore(self) -> None:
        setattr(self.owner, self.name, self.value.restore())

    def verify(self) -> bool:
        return self.value.verify(getattr(self.owner, self.name))


class DenseGPTStateSnapshot:
    """Snapshot declared model buffers, modes, references, and hook maps."""

    _REFERENCE_ATTRIBUTES = (
        "input_tensor",
        "embedding_activation_buffer",
        "grad_output_buffer",
        "rotary_pos_emb_cache",
        "_decoder_hidden_states_cache",
    )
    _HOOK_ATTRIBUTES = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")

    def __init__(
        self, models: Sequence[torch.nn.Module], mutable_buffers: Sequence[str]
    ) -> None:
        self.models = tuple(models)
        self.mutable_buffers = frozenset(mutable_buffers)
        self.modes: list[tuple[torch.nn.Module, bool]] = []
        self.buffers: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        self.attributes: list[_AttributeSnapshot] = []

    def capture(self) -> None:
        """Capture all explicitly admitted mutable model state."""

        self.modes = [
            (module, module.training)
            for model in self.models
            for module in model.modules()
        ]
        found: set[str] = set()
        seen: set[tuple[int, str]] = set()
        for model_index, model in enumerate(self.models):
            for name, buffer in model.named_buffers():
                qualified = f"{model_index}:{name}"
                if name in self.mutable_buffers or qualified in self.mutable_buffers:
                    found.add(name if name in self.mutable_buffers else qualified)
                    self.buffers.append((qualified, buffer, buffer.detach().clone()))
            for module in model.modules():
                for name in (*self._REFERENCE_ATTRIBUTES, *self._HOOK_ATTRIBUTES):
                    identity = (id(module), name)
                    if identity not in seen and hasattr(module, name):
                        seen.add(identity)
                        self.attributes.append(
                            _AttributeSnapshot(
                                module,
                                name,
                                _ValueSnapshot.capture(getattr(module, name)),
                            )
                        )
        missing = self.mutable_buffers - found
        if missing:
            raise ValueError(
                f"declared mutable model buffers were not found: {sorted(missing)}"
            )

    def restore(self) -> None:
        """Restore model state without performing verification."""

        for _name, buffer, snapshot in self.buffers:
            buffer.copy_(snapshot)
        for attribute in self.attributes:
            attribute.restore()
        for module, training in self.modes:
            module.training = training

    def verify(self) -> None:
        """Verify buffer contents, references, hooks, and module modes."""

        for name, buffer, snapshot in self.buffers:
            if not torch.equal(buffer, snapshot):
                raise RuntimeError(f"model buffer restoration failed: {name}")
        if any(module.training != training for module, training in self.modes):
            raise RuntimeError("model training-mode restoration failed")
        if any(not attribute.verify() for attribute in self.attributes):
            raise RuntimeError("model reference/hook restoration failed")

    @property
    def tensor_bytes(self) -> int:
        """Return exact cloned model-buffer storage."""

        return sum(
            snapshot.numel() * snapshot.element_size()
            for _, _, snapshot in self.buffers
        )

    def release(self) -> None:
        """Release all captured model references."""

        self.modes.clear()
        self.buffers.clear()
        self.attributes.clear()


class IdentityStateSnapshot:
    """Snapshot replay iterator positions and sampler identity counters."""

    def __init__(
        self, iterators: Sequence[ReplayIterator], samplers: Sequence[object]
    ) -> None:
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
                    values.append(
                        (sampler, name, copy.deepcopy(getattr(sampler, name)))
                    )
        self.sampler_values = tuple(values)

    def restore(self) -> None:
        """Restore iterator and sampler counters."""

        for iterator, position in zip(
            self.iterators, self.iterator_positions, strict=True
        ):
            iterator.index = position
        for sampler, name, value in self.sampler_values:
            setattr(sampler, name, copy.deepcopy(value))

    def verify(self) -> None:
        """Verify every captured identity counter."""

        if (
            tuple(iterator.index for iterator in self.iterators)
            != self.iterator_positions
        ):
            raise RuntimeError("replay iterator identity restoration failed")
        if any(
            getattr(owner, name) != value for owner, name, value in self.sampler_values
        ):
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
                if hasattr(owner, name):
                    value = getattr(owner, name)
                    if value is not None:
                        raise ValueError(
                            f"Tier-1 replay requires quiescent overlap state: {name}"
                        )
                    self.attributes.append(
                        _AttributeSnapshot(owner, name, _ValueSnapshot.capture(value))
                    )

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
    ) -> None:
        self.model = DenseGPTStateSnapshot(models, mutable_buffer_names)
        self.identity = IdentityStateSnapshot(replay_iterators, samplers)
        self.overlap = OverlapStateSnapshot(overlap_objects)
        self.tracker_getter = tracker_getter
        self.cuda_device = cuda_device
        self.faults = dict(fault_injections or {})
        self.rng: ReplayRngState | None = None

    def __enter__(self) -> "ReplayStateGuard":
        try:
            self.model.capture()
            self.identity.capture()
            self.overlap.capture()
            self.rng = ReplayRngState.capture(
                tracker_getter=self.tracker_getter, cuda_device=self.cuda_device
            )
        except BaseException:
            self.model.release()
            self.overlap.attributes.clear()
            raise
        return self

    def _attempt(
        self,
        name: str,
        operation: Callable[[], None],
        failures: list[tuple[str, BaseException]],
    ) -> None:
        try:
            fault = self.faults.get(name)
            if fault is not None:
                fault()
            operation()
        except BaseException as error:
            failures.append((name, error))

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
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
    selected_tokens: int
    element_size: int
    rng_snapshot_bytes: int
    model_state_bytes: int
    fixed_workspace_bytes: int
    alignment: int = 256
    headroom_fraction: float = 0.1
    tp_source: bool = True
    sequence_parallel: bool = False


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
        config.element_size,
        config.alignment,
    )
    if any(value <= 0 for value in integers) or config.selected_tokens < 0:
        raise ValueError("replay memory dimensions must be positive")
    if not 0 <= config.headroom_fraction <= 1:
        raise ValueError("replay memory headroom fraction must be in [0, 1]")
    if config.sequence_length % (2 * config.context_parallel_size):
        raise ValueError("sequence length must support zigzag CP")
    if config.hidden_size % config.tensor_parallel_size or (
        config.ffn_hidden_size % config.tensor_parallel_size
    ):
        raise ValueError("response widths must be divisible by tensor parallelism")
    if config.global_layers % config.pipeline_parallel_size or (
        config.local_layers != config.global_layers // config.pipeline_parallel_size
    ):
        raise ValueError("standard replay requires uniform physical pipeline layers")
    if (
        config.sequence_parallel
        and (config.sequence_length // config.context_parallel_size)
        % config.tensor_parallel_size
    ):
        raise ValueError("CP-local sequence length must support sequence parallelism")
    local_samples = min(
        _ceil_div(config.global_batch_size, config.data_parallel_size),
        config.selected_tokens,
    )
    microbatches = (
        _ceil_div(local_samples, config.micro_batch_size) if local_samples else 0
    )
    # tokens, labels, position_ids, loss_mask, diagnostic mask, and two identity vectors.
    full_batch = config.micro_batch_size * config.sequence_length * (8 + 8 + 8 + 4 + 1)
    full_batch += config.micro_batch_size * 2 * 8
    host_inputs = microbatches * full_batch if config.tp_source else 0
    cp_local = config.micro_batch_size * (
        config.sequence_length // config.context_parallel_size
    )
    cp_outputs = cp_local * (8 + 8 + 8 + 4 + 1)
    device_inputs = full_batch + cp_outputs
    sp_mask = cp_local // config.tensor_parallel_size if config.sequence_parallel else 0
    local_rows = _ceil_div(
        _ceil_div(config.selected_tokens, config.data_parallel_size),
        config.context_parallel_size,
    )
    feature_widths = (
        config.hidden_size,
        3 * config.hidden_size // config.tensor_parallel_size,
        config.hidden_size,
        config.ffn_hidden_size // config.tensor_parallel_size,
        config.hidden_size,
    )
    response = 0
    max_response = 0
    for family, width in enumerate(feature_widths):
        rows = local_rows
        owner = family in (1, 3) or config.sequence_parallel
        if config.sequence_parallel and family not in (1, 3):
            rows = _ceil_div(rows, config.tensor_parallel_size)
        if owner:
            allocation = config.local_layers * rows * width * config.element_size
            response += allocation
            max_response = max(max_response, rows * width * config.element_size)
    slots = config.global_layers * 5
    packed = _packed_storage_bytes(slots)
    reduction_arena = packed
    accumulator_scratch = PackedSufficientStatistics.scratch_bytes_for_capacity()
    hook_workspace = 2 * max_response
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
    padding = sum((-value) % config.alignment for _, value in terms if value)
    allocated = sum(value for _, value in terms) + padding
    headroom = math.ceil(allocated * config.headroom_fraction)
    return ReplayMemoryEstimate(terms, padding, headroom)


def preflight_replay_memory(
    estimate: ReplayMemoryEstimate,
    *,
    maximum_extra_bytes: int,
    currently_reserved_bytes: int,
    total_device_bytes: int,
) -> None:
    """Reject a replay whose reserved bound exceeds either mandatory limit."""

    if min(maximum_extra_bytes, currently_reserved_bytes, total_device_bytes) < 0:
        raise ValueError("memory preflight values must be nonnegative")
    if estimate.predicted_reserved_bytes > maximum_extra_bytes:
        raise ReplayPreflightError("Tier-1 replay exceeds its configured memory cap")
    if (
        currently_reserved_bytes + estimate.predicted_reserved_bytes
        > total_device_bytes
    ):
        raise ReplayPreflightError("Tier-1 replay exceeds physical device headroom")


class ResponseProbe(Protocol):
    """Narrow response-probe surface consumed by the replay schedule."""

    expected_hook_calls: int
    descriptor_hash: str

    def set_masks(
        self,
        full_mask: torch.Tensor,
        *,
        sequence_parallel_mask: torch.Tensor | None = None,
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
            raise ValueError(
                "Tier-1 replay does not support interleaved/virtual pipeline"
            )
        self.forward_backward_func = forward_backward_func
        self.forward_step_func = forward_step_func
        self.model = list(model)
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

        def replay_forward(
            data_iterator: Iterator[Any] | None, model: torch.nn.Module, *args: Any
        ):
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

            def zero_reducer(
                output_tensor: torch.Tensor,
            ) -> tuple[torch.Tensor, dict[str, Any]]:
                return torch.zeros(
                    (), dtype=torch.float32, device=output_tensor.device
                ), {}

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
            raise RuntimeError(
                "replay schedule executed an unexpected microbatch count"
            )
        self.completed = True


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
        )

    def _fatal(self, error: BaseException) -> None:
        self.release()
        self.fatal_abort(error)
        raise RuntimeError("Tier-1 fatal-abort protocol returned") from error

    def run_pre(self) -> None:
        """Run pre-update replay from RNG state A and restore ambient state."""

        if self.state != TransactionState.READY:
            raise RuntimeError("pre replay can run exactly once")
        error: BaseException | None = None
        try:
            self.rng_a = ReplayRngState.capture(
                tracker_getter=self.tracker_getter, cuda_device=self.cuda_device
            )
        except BaseException as caught:
            error = caught
        self.readiness.settle(error, "pre-schedule RNG capture")
        assert self.rng_a is not None
        schedule_error: BaseException | None = None
        try:
            with self._guard(), torch.no_grad(), self.probe.capture_pre():
                for _name, operation in self.rng_a.restore_stages(
                    tracker_getter=self.tracker_getter, cuda_device=self.cuda_device
                ):
                    operation()
                self.schedule(self.plan, self.probe, "pre")
        except BaseException as caught:
            schedule_error = caught
        if self.schedule.p2p_started:
            if self.schedule.completed:
                try:
                    self.readiness.settle(schedule_error, "pre-schedule restoration")
                except BaseException as caught:
                    self._fatal(caught)
            if schedule_error is not None:
                self._fatal(schedule_error)
        elif schedule_error is not None:
            raise schedule_error
        self.state = TransactionState.PRE_COMPLETE

    def finish(self, *, update_succeeded: bool) -> Any | None:
        """Run post-update replay from A, or release pre state after overflow."""

        if self.state != TransactionState.PRE_COMPLETE or self.rng_a is None:
            raise RuntimeError("post replay requires a completed pre replay")
        if not update_succeeded:
            self.release()
            return None
        try:
            self.schedule.p2p_started = False
            self.schedule.completed = False
            self.readiness.settle(None, "post-schedule readiness")
            result: Any | None = None
            schedule_error: BaseException | None = None
            try:
                with self._guard(), torch.no_grad(), self.probe.capture_post():
                    for _name, operation in self.rng_a.restore_stages(
                        tracker_getter=self.tracker_getter, cuda_device=self.cuda_device
                    ):
                        operation()
                    self.schedule(self.plan, self.probe, "post")
                result = self.probe.finalize()
            except BaseException as caught:
                schedule_error = caught
            if self.schedule.p2p_started:
                if self.schedule.completed:
                    try:
                        self.readiness.settle(
                            schedule_error, "post-schedule restoration"
                        )
                    except BaseException as caught:
                        self._fatal(caught)
                if schedule_error is not None:
                    self._fatal(schedule_error)
            elif schedule_error is not None:
                raise schedule_error
            return result
        finally:
            self.release()

    def release(self) -> None:
        """Release plan, hooks, retained responses, and RNG snapshots."""

        if self.state == TransactionState.CLOSED:
            return
        self.probe.release()
        self.plan.release()
        self.rng_a = None
        self.state = TransactionState.CLOSED


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
        memory_estimate: ReplayMemoryEstimate,
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
                response_minimum,
                op=dist.ReduceOp.MIN,
                group=self.readiness.binding.group,
            )
            dist.all_reduce(
                response_maximum,
                op=dist.ReduceOp.MAX,
                group=self.readiness.binding.group,
            )
            error = (
                ValueError("Tier-1 response descriptor hash differs across ranks")
                if not torch.equal(response_minimum, response_maximum)
                else None
            )
            self.readiness.settle(error, "response descriptor identity")
        error = None
        try:
            if probe.expected_hook_calls != plan.num_microbatches:
                raise ValueError("response hook cardinality disagrees with replay plan")
            if not plan.metadata.descriptor_hash:
                raise ValueError("replay plan has no canonical descriptor hash")
            if self.tracker_getter is None:
                raise TypeError(
                    "Tier-1 replay requires an inspected Megatron RNG tracker"
                )
            tracker = _validated_tracker(self.tracker_getter)
            if self.tracker_getter() is not tracker:
                raise TypeError("Megatron RNG tracker getter is not identity-stable")
            preflight_replay_memory(
                memory_estimate,
                maximum_extra_bytes=maximum_extra_bytes,
                currently_reserved_bytes=currently_reserved_bytes,
                total_device_bytes=total_device_bytes,
            )
        except BaseException as caught:
            error = caught
        self.readiness.settle(error, "transaction preflight")
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
        )
