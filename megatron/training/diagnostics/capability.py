# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-safe, versioned runtime capability probe for scaling preflight."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from .schema import SCHEMA_PREFIX, SCHEMA_VERSION, TIER0_KEYS

PROBE_VERSION = 1
SUPPORTED_MAX_TIER = 0
SUPPORT_SIGNATURE = (
    "dense_mcore_gpt:local:bf16:chained_distributed_optimizer_fp32_master:"
    "per_token_loss:dp_tp_pp_cp:sequence_parallel_optional:recompute_mcore:"
    "no_te_fp8_fsdp_moe_vpp_mtp_custom_layout_hybrid_cp_param_gather_overlap"
)
CONSUMED_DIAGNOSTIC_CONFIG_FIELDS = (
    "diagnostic_heartbeat",
    "diagnostic_interval",
    "diagnostic_early_updates",
    "diagnostic_unsupported_policy",
    "diagnostic_max_extra_bytes",
    "diagnostic_dgrad_starvation_threshold",
    "diagnostic_update_starvation_threshold",
)
CHECKPOINTED_RUNTIME_FIELDS = (
    "diagnostic_successful_updates",
    "diagnostic_event_id",
)
WRITER_POLICY = {
    "wandb_rank": 0,
    "wandb_calls_per_event": 1,
    "wandb_step": "iteration_plus_one",
    "tensorboard_ownership": "existing",
}


def _runtime_consumers() -> tuple[str | None, str | None]:
    """Verify heartbeat and config consumers from source without importing training."""

    training_dir = Path(__file__).resolve().parents[1]
    tier0_path = Path(__file__).with_name("tier0.py")
    config_path = training_dir / "config" / "training_config.py"
    try:
        tier0_tree = ast.parse(tier0_path.read_text(encoding="utf-8"))
        config_tree = ast.parse(config_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None, None

    heartbeat_present = any(
        isinstance(node, ast.ClassDef) and node.name == "Tier0Heartbeat"
        for node in tier0_tree.body
    )
    config_fields: set[str] = set()
    for node in config_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LoggerConfig":
            config_fields.update(
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            )
    config_present = set(CONSUMED_DIAGNOSTIC_CONFIG_FIELDS).issubset(config_fields)
    return (
        "megatron.training.diagnostics.tier0.Tier0Heartbeat"
        if heartbeat_present
        else None,
        "megatron.training.config.training_config.LoggerConfig"
        if config_present
        else None,
    )


def schema_hash() -> str:
    """Return the stable SHA-256 hash of the ordered canonical key tuple."""

    encoded = json.dumps(TIER0_KEYS, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract_hash() -> str:
    """Return the stable SHA-256 hash of the consumed runtime contract."""

    contract = {
        "checkpointed_runtime_fields": CHECKPOINTED_RUNTIME_FIELDS,
        "consumed_diagnostic_config_fields": CONSUMED_DIAGNOSTIC_CONFIG_FIELDS,
        "probe_version": PROBE_VERSION,
        "schema_hash": schema_hash(),
        "schema_version": SCHEMA_VERSION,
        "support_signature": SUPPORT_SIGNATURE,
        "supported_max_tier": SUPPORTED_MAX_TIER,
        "writer_policy": WRITER_POLICY,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def capability_payload(
    *, source_commit: str | None = None, build_identity: str | None = None
) -> dict[str, object]:
    """Build the stable machine-readable runtime capability payload."""

    heartbeat_consumer, config_consumer = _runtime_consumers()
    return {
        "probe_version": PROBE_VERSION,
        "contract": SCHEMA_PREFIX.rstrip("/"),
        "contract_hash": contract_hash(),
        "schema_version": SCHEMA_VERSION,
        "schema_hash": schema_hash(),
        "schema_key_count": len(TIER0_KEYS),
        "supported_max_tier": SUPPORTED_MAX_TIER,
        "consumed_diagnostic_config_fields": list(CONSUMED_DIAGNOSTIC_CONFIG_FIELDS),
        "checkpointed_runtime_fields": list(CHECKPOINTED_RUNTIME_FIELDS),
        "heartbeat_consumer": heartbeat_consumer,
        "config_consumer": config_consumer,
        "runtime_contract_present": bool(heartbeat_consumer and config_consumer),
        "writer_policy": dict(WRITER_POLICY),
        "support_signature": SUPPORT_SIGNATURE,
        "source_commit": source_commit,
        "build_identity": build_identity,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit one compact JSON object"
    )
    parser.add_argument("--source-commit", default=os.getenv("MEGATRON_SOURCE_COMMIT"))
    parser.add_argument(
        "--build-identity", default=os.getenv("MEGATRON_BUILD_IDENTITY")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the runtime capability probe."""

    args = _parser().parse_args(argv)
    payload = capability_payload(
        source_commit=args.source_commit, build_identity=args.build_identity
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["runtime_contract_present"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
