# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scalable, topology-aware training diagnostic primitives."""

from .accumulator import DerivedStatistic, PackedSlots, PackedSufficientStatistics
from .registry import (
    DenominatorKind,
    MaskKind,
    MetricDescriptor,
    MetricFamily,
    MetricRegistry,
    NormalizationKind,
    Ownership,
    PartitionAxis,
    ReductionKind,
    ReplicationAxis,
    StatisticKind,
)
from .schema import (
    SCHEMA_PREFIX,
    SCHEMA_VERSION,
    TIER0_KEYS,
    TIER0_METADATA_KEYS,
    TIER0_METRIC_KEYS,
    TIER0_PREFIX,
    Tier0Reason,
    Tier0Status,
    assert_payload_schema,
)

__all__ = [
    "DerivedStatistic",
    "DenominatorKind",
    "MaskKind",
    "MetricDescriptor",
    "MetricFamily",
    "MetricRegistry",
    "NormalizationKind",
    "Ownership",
    "PackedSlots",
    "PackedSufficientStatistics",
    "PartitionAxis",
    "ReductionKind",
    "ReplicationAxis",
    "SCHEMA_PREFIX",
    "SCHEMA_VERSION",
    "StatisticKind",
    "TIER0_KEYS",
    "TIER0_METADATA_KEYS",
    "TIER0_METRIC_KEYS",
    "TIER0_PREFIX",
    "Tier0Reason",
    "Tier0Status",
    "assert_payload_schema",
]
