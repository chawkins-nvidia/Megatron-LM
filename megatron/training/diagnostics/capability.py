# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-safe runtime capability probe for the integrated diag/v2 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROBE_VERSION = 3
SUPPORTED_MAX_TIER = 2
RUNTIME_CAPABILITY_SCHEMA = "diag/v2/runtime-capabilities"
ARTIFACT_SCHEMA_VERSION = 3
ARTIFACT_SCHEMA = "diag/v2/artifact-v3"
LAUNCH_ARTIFACT_SCHEMA = "diag/v2/launch-artifact-v3"
SUPPORT_SIGNATURE = (
    "dense_mcore_gpt:local:bf16:chained_distributed_optimizer_fp32_master:"
    "per_token_loss:dp_tp_pp_cp:sequence_parallel_optional:recompute_mcore:"
    "canonical_gpt_mask_producer:no_te_fp8_fsdp_moe_vpp_mtp_custom_layout:"
    "no_hybrid_cp_param_gather_overlap"
)
CONSUMED_DIAGNOSTIC_CONFIG_FIELDS = (
    "diagnostic_heartbeat",
    "diagnostic_interval",
    "diagnostic_early_updates",
    "diagnostic_unsupported_policy",
    "diagnostic_layer_pattern",
    "diagnostic_include_special_layers",
    "diagnostic_max_extra_bytes",
    "diagnostic_dgrad_starvation_threshold",
    "diagnostic_update_starvation_threshold",
    "diag_schema",
    "diag_enabled",
    "diag_max_tier",
    "diag_require_tier",
    "diag_schedule_mode",
    "diag_early_successful_updates",
    "diag_every_successful_updates",
    "diag_early_consumed_tokens",
    "diag_every_consumed_tokens",
    "diag_retry_after_skipped_update",
    "diag_sample_selector",
    "diag_sample_seed",
    "diag_max_sequences_global",
    "diag_max_positions_per_sequence",
    "diag_max_valid_positions_global",
    "diag_replay_full_sequences",
    "diag_replay_input_bytes_per_rank",
    "diag_all_layer_stats_bytes_per_rank",
    "diag_max_extra_allocated_bytes_per_rank",
    "diag_max_extra_allocated_fraction",
    "diag_max_total_hbm_fraction",
    "diag_min_free_bytes_after_reservation",
    "diag_max_event_artifact_bytes",
    "diag_max_run_artifact_bytes",
    "diag_max_campaign_bytes",
    "diag_include_raw_tokens",
    "diag_include_raw_activations",
    "diag_tier0_optimizer_adapter",
    "diag_tier1_replay_mode",
    "diag_tier1_deterministic_dropout",
    "diag_tier1_equal_pipeline_participation",
    "diag_tier2_midpoint_fraction",
    "diag_tier2_midpoint_tolerance",
    "diag_tier2_min_response_over_replay_floor",
    "diag_tier2_enabled_by_default",
    "diag_capability_policy",
    "diag_allow_vpp",
    "diag_allow_ep",
    "diag_allow_fsdp",
    "diag_allow_fp8_parameters",
    "diag_allow_fp4_parameters",
)
CHECKPOINTED_RUNTIME_FIELDS = (
    "diagnostic_successful_updates",
    "diagnostic_event_id",
    "diagnostic_cumulative_artifact_bytes",
)
SCHEMA_FILES = ("diag_v2.metrics.json", "diag_v2.artifact-v3.schema.json")


def repository_root() -> Path:
    """Return the source checkout containing this module."""

    return Path(__file__).resolve().parents[3]


def schema_directory() -> Path:
    """Return the packaged byte-identical Scaling schema directory."""

    return Path(__file__).with_name("schemas")


def diagnostic_schema_hash(root: Path | None = None) -> str:
    """Hash canonical schema filenames and raw bytes using the Scaling algorithm."""

    digest = hashlib.sha256()
    schema_root = schema_directory() if root is None else root
    for name in SCHEMA_FILES:
        path = schema_root / name
        if not path.is_file():
            raise RuntimeError(f"required diag/v2 schema is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def static_capability_path() -> Path:
    """Return the committed capability document inspected by the launcher."""

    return repository_root() / ".diag_v2_runtime_capabilities.json"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded host-memory chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_hash() -> str:
    """Return the launcher-compatible bundled schema hash."""

    return diagnostic_schema_hash()


def contract_hash() -> str:
    """Return the content identity of the committed static capability file."""

    return sha256_file(static_capability_path())


def verified_source_commit(root: Path | None = None) -> str:
    """Read and verify the checkout's actual git HEAD without caller input."""

    checkout = repository_root() if root is None else root
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("git HEAD did not resolve to a full lowercase commit")
    return commit


def load_static_capability(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally authenticate the committed launcher capability file."""

    capability_path = static_capability_path() if path is None else path
    payload = json.loads(capability_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("runtime capability document must be a JSON object")
    if payload.get("schema") != RUNTIME_CAPABILITY_SCHEMA:
        raise RuntimeError("runtime capability document has the wrong schema")
    if payload.get("schema_hash") != diagnostic_schema_hash():
        raise RuntimeError("runtime capability schema hash disagrees with packaged schemas")
    if payload.get("runtime_fields") != list(CONSUMED_DIAGNOSTIC_CONFIG_FIELDS):
        raise RuntimeError("runtime capability fields disagree with the integrated consumer")
    if payload.get("supported_max_tier") != SUPPORTED_MAX_TIER:
        raise RuntimeError("runtime capability tier disagrees with the integrated consumer")
    if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("runtime capability artifact schema version is not v3")
    if payload.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise RuntimeError("runtime capability artifact schema identity is wrong")
    if payload.get("launch_artifact_schema") != LAUNCH_ARTIFACT_SCHEMA:
        raise RuntimeError("runtime capability launch artifact schema identity is wrong")
    if payload.get("integrated_heartbeat_consumer") is not True:
        raise RuntimeError("runtime capability does not declare an integrated consumer")
    return payload


def capability_payload() -> dict[str, Any]:
    """Bind the static bundle file to the verified runtime checkout identity."""

    static_path = static_capability_path()
    static = load_static_capability(static_path)
    static_hash = sha256_file(static_path)
    return {
        **static,
        "probe_version": PROBE_VERSION,
        "static_capability_sha256": static_hash,
        "build_identity": f"capability-file-sha256:{static_hash}",
        "source_commit": verified_source_commit(),
        "runtime_contract_present": True,
    }


def materialize_capability(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write deterministic capability JSON."""

    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit one compact JSON object")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the runtime capability probe."""

    _parser().parse_args(argv)
    payload = capability_payload()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
