# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.training.diagnostics.schema import (
    SCHEMA_PREFIX,
    TIER0_KEYS,
    TIER0_METADATA_KEYS,
    TIER0_METRIC_KEYS,
    TIER0_PREFIX,
    Tier0Reason,
    Tier0Status,
    assert_payload_schema,
)


def test_tier0_schema_is_exactly_75_unique_canonical_keys() -> None:
    assert len(TIER0_KEYS) == 75
    assert len(set(TIER0_KEYS)) == 75
    assert len(TIER0_METRIC_KEYS) == 67
    assert len(TIER0_METADATA_KEYS) == 8
    assert all(key.startswith(TIER0_PREFIX) for key in TIER0_METRIC_KEYS)
    assert not any(key.startswith(TIER0_PREFIX) for key in TIER0_METADATA_KEYS)
    assert all(key.startswith(SCHEMA_PREFIX) for key in TIER0_KEYS)
    assert not any(
        key.startswith(("act/", "dgrad/", "dy/", "lin/")) for key in TIER0_KEYS
    )


def test_tier0_metric_keys_match_the_independent_canonical_expansion() -> None:
    prefix = "diag/v2/t0/"
    depth = ("first", "q1", "middle", "q3", "last", "p10", "p50", "p90")
    modules = ("qkv", "attn_out", "fc1", "fc2")
    expected = {
        *(f"{prefix}activation/residual/rms/{summary}" for summary in depth),
        *(f"{prefix}dgrad/residual/rms/{summary}" for summary in depth),
        *(
            f"{prefix}dgrad/{family}/{summary}"
            for family in modules
            for summary in ("p10", "p50", "zero_fraction")
        ),
        *(
            f"{prefix}update/{family}/{summary}"
            for family in (*modules, "norm")
            for summary in ("p10", "p50", "p90", "starved_fraction")
        ),
        f"{prefix}update/embedding/relative_rms",
        f"{prefix}update/output/relative_rms",
        *(
            f"{prefix}retention/{family}/{summary}"
            for family in ("embedding", *modules, "output")
            for summary in ("median", "zero_fraction")
        ),
        *(f"{prefix}activation/{family}/max_abs" for family in (*modules, "residual")),
    }

    assert len(expected) == 67
    assert set(TIER0_METRIC_KEYS) == expected


def test_tier0_metadata_keys_are_the_eight_canonical_base_namespace_keys() -> None:
    expected = {
        "diag/v2/event/successful_update",
        "diag/v2/event/valid_positions",
        "diag/v2/health/nonfinite_fraction",
        "diag/v2/health/underflow_fraction",
        "diag/v2/status/valid",
        "diag/v2/perf/peak_hbm_bytes_max_rank",
        "diag/v2/perf/latency_ms_median_rank",
        "diag/v2/perf/latency_ms_max_rank",
    }

    assert set(TIER0_METADATA_KEYS) == expected
    assert set(TIER0_KEYS) == set(TIER0_METRIC_KEYS) | expected


def test_payload_validation_requires_exact_set_without_aliases() -> None:
    payload = dict.fromkeys(TIER0_KEYS, 0.0)
    assert_payload_schema(payload)

    missing = dict(payload)
    missing.pop(TIER0_KEYS[0])
    with pytest.raises(ValueError, match="missing="):
        assert_payload_schema(missing)

    aliased = dict(payload)
    aliased["act/residual/rms"] = aliased.pop(TIER0_KEYS[0])
    with pytest.raises(ValueError, match="unexpected=.*act/residual/rms"):
        assert_payload_schema(aliased)


def test_status_and_reason_codes_are_stable() -> None:
    assert {status.name: status.value for status in Tier0Status} == {
        "OK": 0,
        "INVALID_STATISTICS": 1,
        "UNSUPPORTED": 2,
        "RUNTIME_ERROR": 3,
    }
    assert {reason.name: reason.value for reason in Tier0Reason} == {
        "NONE": 0,
        "NO_CONTRIBUTORS": 1,
        "ZERO_DENOMINATOR": 2,
        "NONFINITE_INPUT": 3,
        "MASK_MISMATCH": 4,
        "DESCRIPTOR_MISMATCH": 5,
        "UNSUPPORTED_LAYOUT": 6,
        "NONFINITE_ARITHMETIC": 7,
    }
