# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Versioned scalar schema for the Tier-0 diagnostic heartbeat."""

from enum import IntEnum
from typing import Mapping

SCHEMA_VERSION = 2
SCHEMA_PREFIX = f"diag/v{SCHEMA_VERSION}/"
TIER0_PREFIX = f"{SCHEMA_PREFIX}t0/"


class Tier0Status(IntEnum):
    """Stable event-level Tier-0 status codes."""

    OK = 0
    INVALID_STATISTICS = 1
    UNSUPPORTED = 2
    RUNTIME_ERROR = 3


class Tier0Reason(IntEnum):
    """Stable reason codes used when a Tier-0 statistic or event is invalid."""

    NONE = 0
    NO_CONTRIBUTORS = 1
    ZERO_DENOMINATOR = 2
    NONFINITE_INPUT = 3
    MASK_MISMATCH = 4
    DESCRIPTOR_MISMATCH = 5
    UNSUPPORTED_LAYOUT = 6
    NONFINITE_ARITHMETIC = 7


_DEPTH_SUMMARIES = ("first", "q1", "middle", "q3", "last", "p10", "p50", "p90")
_MODULE_FAMILIES = ("qkv", "attn_out", "fc1", "fc2")
_LAYERED_UPDATE_FAMILIES = (*_MODULE_FAMILIES, "norm")
_MATRIX_FAMILIES = ("embedding", *_MODULE_FAMILIES, "output")

TIER0_METRIC_KEYS: tuple[str, ...] = (
    *(
        f"{TIER0_PREFIX}activation/residual/rms/{summary}"
        for summary in _DEPTH_SUMMARIES
    ),
    *(f"{TIER0_PREFIX}dgrad/residual/rms/{summary}" for summary in _DEPTH_SUMMARIES),
    *(
        f"{TIER0_PREFIX}dgrad/{family}/{summary}"
        for family in _MODULE_FAMILIES
        for summary in ("p10", "p50", "zero_fraction")
    ),
    *(
        f"{TIER0_PREFIX}update/{family}/{summary}"
        for family in _LAYERED_UPDATE_FAMILIES
        for summary in ("p10", "p50", "p90", "starved_fraction")
    ),
    f"{TIER0_PREFIX}update/embedding/relative_rms",
    f"{TIER0_PREFIX}update/output/relative_rms",
    *(
        f"{TIER0_PREFIX}retention/{family}/{summary}"
        for family in _MATRIX_FAMILIES
        for summary in ("median", "zero_fraction")
    ),
    *(
        f"{TIER0_PREFIX}activation/{family}/max_abs"
        for family in (*_MODULE_FAMILIES, "residual")
    ),
)

TIER0_METADATA_KEYS: tuple[str, ...] = (
    f"{SCHEMA_PREFIX}event/successful_update",
    f"{SCHEMA_PREFIX}event/valid_positions",
    f"{SCHEMA_PREFIX}health/nonfinite_fraction",
    f"{SCHEMA_PREFIX}health/underflow_fraction",
    f"{SCHEMA_PREFIX}status/valid",
    f"{SCHEMA_PREFIX}perf/peak_hbm_bytes_max_rank",
    f"{SCHEMA_PREFIX}perf/latency_ms_median_rank",
    f"{SCHEMA_PREFIX}perf/latency_ms_max_rank",
)

TIER0_KEYS: tuple[str, ...] = (*TIER0_METRIC_KEYS, *TIER0_METADATA_KEYS)


def assert_payload_schema(payload: Mapping[str, object]) -> None:
    """Require a payload to contain exactly the 75 canonical Tier-0 keys.

    Args:
        payload: Mapping that is about to be emitted to a scalar sink.

    Raises:
        ValueError: If any canonical key is missing or any extra key is present.
    """

    expected = set(TIER0_KEYS)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "Tier-0 payload does not match the canonical schema: "
            f"missing={missing}, unexpected={unexpected}"
        )


def assert_tiered_payload_schema(
    payload: Mapping[str, object], *, effective_tier: int
) -> None:
    """Require the exact cumulative 75/105/122-key dense payload surface."""

    if effective_tier not in (0, 1, 2):
        raise ValueError("effective diagnostic tier must be 0, 1, or 2")
    expected = list(TIER0_KEYS)
    if effective_tier >= 1:
        from .function_response import TIER1_KEYS

        expected.extend(TIER1_KEYS)
    if effective_tier >= 2:
        from .secant import TIER2_OUTPUT_KEYS

        expected.extend(TIER2_OUTPUT_KEYS)
    if tuple(payload) != tuple(expected):
        expected_set = set(expected)
        actual_set = set(payload)
        raise ValueError(
            "tiered diagnostic payload does not match the canonical schema: "
            f"missing={sorted(expected_set - actual_set)}, "
            f"unexpected={sorted(actual_set - expected_set)}, "
            f"order_matches={tuple(payload) == tuple(expected)}"
        )
