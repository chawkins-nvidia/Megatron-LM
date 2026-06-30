# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scalable, topology-aware training diagnostic primitives.

Exports are lazy so the machine-readable capability probe can load the schema
without importing PyTorch or the GPU training stack.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "Bf16DistributedOptimizerDiagnosticAdapter": "distributed_optimizer",
    "CanonicalDgradNormalizer": "normalization",
    "CaptureTopology": "capture",
    "DeviceMemoryState": "distributed_optimizer",
    "DerivedStatistic": "accumulator",
    "DenominatorKind": "registry",
    "DistributedOptimizerCapabilityReport": "distributed_optimizer",
    "DistributedOptimizerDiagnosticReason": "distributed_optimizer",
    "DistributedOptimizerDiagnosticUnsupportedError": "distributed_optimizer",
    "DistributedOptimizerEventStatus": "distributed_optimizer",
    "LayerCaptureTarget": "capture",
    "MaskKind": "registry",
    "MetricDescriptor": "registry",
    "MetricFamily": "registry",
    "MetricNormalizationAdapter": "registry",
    "MetricRegistry": "registry",
    "NormalizationKind": "registry",
    "Ownership": "registry",
    "PackedSlots": "accumulator",
    "PackedSufficientStatistics": "accumulator",
    "PartitionAxis": "registry",
    "ProcessGroupIdentity": "accumulator",
    "ReductionBinding": "accumulator",
    "ReductionKind": "accumulator",
    "ReplicationAxis": "registry",
    "SCHEMA_PREFIX": "schema",
    "SCHEMA_VERSION": "schema",
    "SnapshotMemoryError": "distributed_optimizer",
    "SnapshotMemoryEstimate": "distributed_optimizer",
    "SnapshotMemoryMeasurement": "distributed_optimizer",
    "SnapshotMemoryPreflight": "distributed_optimizer",
    "SnapshotMemoryReason": "distributed_optimizer",
    "StagedTokenMask": "capture",
    "StatisticKind": "registry",
    "TIER0_KEYS": "schema",
    "TIER0_METADATA_KEYS": "schema",
    "TIER0_METRIC_KEYS": "schema",
    "TIER0_PREFIX": "schema",
    "Tier0Cadence": "tier0",
    "Tier0Capability": "tier0",
    "Tier0CaptureResult": "capture",
    "Tier0CaptureSession": "capture",
    "Tier0Heartbeat": "tier0",
    "Tier0Reason": "schema",
    "Tier0Status": "schema",
    "TokenLayout": "capture",
    "assert_payload_schema": "schema",
    "build_update_registry": "tier0",
    "discover_layer_capture_targets": "capture",
    "negotiate_tier0_capability": "tier0",
    "pack_valid_token_mask_sideband": "capture",
    "slice_sequence_parallel_mask": "capture",
    "stage_valid_token_mask": "capture",
    "unpack_valid_token_mask_sideband": "capture",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    """Load one public diagnostic symbol on first use."""

    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
