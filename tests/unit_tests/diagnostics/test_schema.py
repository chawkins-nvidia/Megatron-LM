# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.training.diagnostics.schema import (
    TIER0_KEYS,
    TIER0_PREFIX,
    Tier0Reason,
    Tier0Status,
    assert_payload_schema,
)


def test_tier0_schema_is_exactly_75_unique_canonical_keys() -> None:
    assert len(TIER0_KEYS) == 75
    assert len(set(TIER0_KEYS)) == 75
    assert all(key.startswith(TIER0_PREFIX) for key in TIER0_KEYS)
    assert not any(
        key.startswith(("act/", "dgrad/", "dy/", "lin/")) for key in TIER0_KEYS
    )


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
    }
