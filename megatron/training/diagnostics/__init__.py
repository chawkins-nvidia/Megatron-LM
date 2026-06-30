# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scalable, topology-aware training diagnostic primitives."""

from .accumulator import DerivedStatistic, PackedSlots, PackedSufficientStatistics
from .registry import (
    MaskKind,
    MetricDescriptor,
    MetricFamily,
    MetricRegistry,
    Ownership,
    PartitionAxis,
    ReplicationAxis,
)
from .schema import (
    SCHEMA_VERSION,
    TIER0_KEYS,
    TIER0_PREFIX,
    Tier0Reason,
    Tier0Status,
    assert_payload_schema,
)

__all__ = [
    "DerivedStatistic",
    "MaskKind",
    "MetricDescriptor",
    "MetricFamily",
    "MetricRegistry",
    "Ownership",
    "PackedSlots",
    "PackedSufficientStatistics",
    "PartitionAxis",
    "ReplicationAxis",
    "SCHEMA_VERSION",
    "TIER0_KEYS",
    "TIER0_PREFIX",
    "Tier0Reason",
    "Tier0Status",
    "assert_payload_schema",
]
