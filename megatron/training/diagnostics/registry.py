# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed metric descriptors and deterministic Tier-0 registry."""

import hashlib
import json
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Iterable

import torch

from .accumulator import PackedSlots, PackedSufficientStatistics
from .schema import SCHEMA_VERSION


class MetricFamily(StrEnum):
    """Stable logical tensor families used by Tier-0 diagnostics."""

    RESIDUAL = "residual"
    QKV = "qkv"
    ATTN_OUT = "attn_out"
    FC1 = "fc1"
    FC2 = "fc2"
    NORM = "norm"
    EMBEDDING = "embedding"
    OUTPUT = "output"


class PartitionAxis(StrEnum):
    """Axes along which a logical observation is uniquely partitioned."""

    DATA_SAMPLE = "data_sample"
    TENSOR_FEATURE = "tensor_feature"
    PIPELINE_LAYER = "pipeline_layer"
    CONTEXT_SEQUENCE = "context_sequence"
    EXPERT = "expert"
    OPTIMIZER_SHARD = "optimizer_shard"


class ReplicationAxis(StrEnum):
    """Axes along which local values may duplicate one logical observation."""

    DATA = "data"
    TENSOR = "tensor"
    PIPELINE_TIED = "pipeline_tied"
    CONTEXT = "context"
    EXPERT = "expert"


class Ownership(StrEnum):
    """Static rule that selects unique contributors for a descriptor."""

    EVERY_RANK = "every_rank"
    PIPELINE_STAGE = "pipeline_stage"
    TENSOR_PARALLEL_RANK_ZERO = "tensor_parallel_rank_zero"
    AUTHORITATIVE_SHARD = "authoritative_shard"
    TIED_PARAMETER_OWNER = "tied_parameter_owner"


class MaskKind(StrEnum):
    """Mask semantics expected by a metric descriptor."""

    NONE = "none"
    TOKEN = "token"
    SEQUENCE_PARALLEL_TOKEN = "sequence_parallel_token"
    PARAMETER = "parameter"


@dataclass(frozen=True)
class MetricDescriptor:
    """Static topology and packed-slot contract for one logical metric."""

    logical_name: str
    family: MetricFamily
    global_layer: int | None
    partition_axes: tuple[PartitionAxis, ...]
    replication_axes: tuple[ReplicationAxis, ...]
    ownership: Ownership
    mask_kind: MaskKind
    packed_slots: PackedSlots

    def __post_init__(self) -> None:
        if not self.logical_name:
            raise ValueError("a metric descriptor requires a logical name")
        if self.global_layer is not None and self.global_layer < 0:
            raise ValueError("global layer indices must be nonnegative")
        if len(self.partition_axes) != len(set(self.partition_axes)):
            raise ValueError("partition axes must be unique")
        if len(self.replication_axes) != len(set(self.replication_axes)):
            raise ValueError("replication axes must be unique")


class MetricRegistry:
    """A fixed descriptor sequence plus rank-local ownership decisions."""

    def __init__(
        self,
        descriptors: Iterable[MetricDescriptor],
        *,
        local_owners: Iterable[bool] | None = None,
    ) -> None:
        """Validate descriptors and retain neutral slots for non-owning ranks."""

        self.descriptors = tuple(descriptors)
        names = tuple(descriptor.logical_name for descriptor in self.descriptors)
        if len(names) != len(set(names)):
            raise ValueError("metric descriptor logical names must be unique")
        for index, descriptor in enumerate(self.descriptors):
            if descriptor.packed_slots != PackedSlots.for_index(index):
                raise ValueError(
                    "descriptor packed slots must follow canonical registry order"
                )
        owners = (
            tuple(True for _ in self.descriptors)
            if local_owners is None
            else tuple(local_owners)
        )
        if len(owners) != len(self.descriptors):
            raise ValueError("local ownership must have one entry per descriptor")
        self.local_owners = owners
        self._indices = {name: index for index, name in enumerate(names)}

    @property
    def slot_names(self) -> tuple[str, ...]:
        """Return the fixed collective slot order."""

        return tuple(descriptor.logical_name for descriptor in self.descriptors)

    @property
    def descriptor_hash(self) -> str:
        """Return a stable SHA-256 hash of static, rank-independent descriptors."""

        records = []
        for descriptor in self.descriptors:
            records.append(
                {
                    "family": descriptor.family.value,
                    "global_layer": descriptor.global_layer,
                    "logical_name": descriptor.logical_name,
                    "mask_kind": descriptor.mask_kind.value,
                    "ownership": descriptor.ownership.value,
                    "packed_slots": {
                        packed_field.name: getattr(
                            descriptor.packed_slots, packed_field.name
                        )
                        for packed_field in fields(PackedSlots)
                    },
                    "partition_axes": sorted(
                        axis.value for axis in descriptor.partition_axes
                    ),
                    "replication_axes": sorted(
                        axis.value for axis in descriptor.replication_axes
                    ),
                }
            )
        encoded = json.dumps(
            {"descriptors": records, "schema_version": SCHEMA_VERSION},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def new_accumulator(self, device: torch.device | str) -> PackedSufficientStatistics:
        """Allocate neutral packs matching the registry on CPU or CUDA."""

        return PackedSufficientStatistics(self.slot_names, device)

    def owns(self, logical_name: str) -> bool:
        """Return whether this rank contributes to ``logical_name``."""

        return self.local_owners[self._index(logical_name)]

    def add_masked_tensor(
        self,
        accumulator: PackedSufficientStatistics,
        logical_name: str,
        values: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        replication_multiplicity: int = 1,
    ) -> None:
        """Accumulate an owned tensor or leave its neutral slots untouched."""

        index = self._index(logical_name)
        self._validate_accumulator(accumulator)
        if self.local_owners[index]:
            accumulator.add_masked_tensor(
                index,
                values,
                mask=mask,
                replication_multiplicity=replication_multiplicity,
            )

    def add_masked_pair(
        self,
        accumulator: PackedSufficientStatistics,
        logical_name: str,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        replication_multiplicity: int = 1,
    ) -> None:
        """Accumulate an owned pair or leave its neutral slots untouched."""

        index = self._index(logical_name)
        self._validate_accumulator(accumulator)
        if self.local_owners[index]:
            accumulator.add_masked_pair(
                index,
                lhs,
                rhs,
                mask=mask,
                replication_multiplicity=replication_multiplicity,
            )

    def _index(self, logical_name: str) -> int:
        try:
            return self._indices[logical_name]
        except KeyError as error:
            raise KeyError(f"unknown metric descriptor: {logical_name}") from error

    def _validate_accumulator(self, accumulator: PackedSufficientStatistics) -> None:
        if accumulator.slot_names != self.slot_names:
            raise ValueError(
                "accumulator slot order does not match the metric registry"
            )
