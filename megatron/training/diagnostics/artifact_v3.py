# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Truthful, bounded ``diag/v2/artifact-v3`` event publication.

This module is intentionally CPU-safe and independent of the training hot path.
The diagnostic runtime supplies already-reduced layer statistics, rank evidence,
state digests, and the exact observed collective ledger.  The last world rank
validates those structures, materializes one deterministic three-file artifact,
and performs one W&B artifact call.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .capability import diagnostic_schema_hash, verified_source_commit

ARTIFACT_SCHEMA = "diag/v2/artifact-v3"
ARTIFACT_TYPE = "diagnostic-event-v2"
MAX_EVENT_BYTES = 4 * 1024**2
MAX_EVENT_UNCOMPRESSED_BYTES = 16 * 1024**2
MAX_RUN_BYTES = 8 * 1024**3
TIER0_STATE_REASON = "tier0_population_only_no_replay_state"
MIDPOINT_STATE_REASON = "midpoint_not_required_below_tier2"
COMPONENT_STATE_REASON = "runtime_component_not_applicable"
SECANT_STATE_REASON = "secant_restoration_not_required_below_tier2"
TIER0_IDENTITY_DISCLAIMER = (
    "collision-prone population checksum; does not establish sample or token identity"
)
STATE_COMPONENTS = (
    "model",
    "optimizer",
    "rng",
    "mutable_state",
    "data_iterator",
    "fp8",
    "router",
    "cache",
)
CAPABILITY_ALLOW = {
    "allow_vpp": False,
    "allow_ep": False,
    "allow_fsdp": False,
    "allow_fp8_parameters": False,
    "allow_fp4_parameters": False,
}
LAYER_DTYPES = {
    "global_layer_id": "int32",
    "pp_owner": "int16",
    "family_code": "uint8",
    "metric_code": "uint16",
    "value": "float32",
    "valid": "uint8",
    "count": "int64",
    "sum": "float64",
    "sum_sq": "float64",
    "denominator_sum_sq": "float64",
    "zero_count": "int64",
    "nonfinite_count": "int64",
}
RANK_PERF_DTYPES = {
    "rank": "int32",
    "event_wall_time_ms": "float64",
    "ordinary_step_wall_time_ms": "float64",
    "detected_hbm_capacity_bytes": "float64",
    "pre_event_allocated_bytes": "float64",
    "pre_event_reserved_bytes": "float64",
    "predicted_post_gather_peak_allocated_bytes": "float64",
    "predicted_post_gather_peak_reserved_bytes": "float64",
    "post_gather_peak_allocated_bytes": "float64",
    "post_gather_peak_reserved_bytes": "float64",
    "sink_post_interval_peak_allocated_bytes": "float64",
    "sink_post_interval_peak_reserved_bytes": "float64",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SLURM_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")


@dataclass(frozen=True)
class RuntimeIdentity:
    """Immutable scheduler/W&B identity for every event in one run."""

    run_id: str
    job_name: str
    training_job_id: str
    writer_rank: int
    world_size: int

    def as_manifest(self) -> dict[str, Any]:
        """Return the exact v3 run-identity mapping."""

        return {
            "run_id": self.run_id,
            "job_name": self.job_name,
            "training_job_id": self.training_job_id,
            "writer_rank": self.writer_rank,
        }


@dataclass(frozen=True)
class EventEvidence:
    """Fully reduced evidence consumed by :class:`ArtifactV3Writer`."""

    requested_tier: int
    effective_tier: int
    require_tier: int
    status: str
    status_reason: str | None
    capability_signature: Mapping[str, Any]
    capability_status: Mapping[str, str]
    topology: Mapping[str, Any]
    sampling: Mapping[str, Any]
    digests: Mapping[str, Any]
    metric_enums: Mapping[str, int]
    family_enums: Mapping[str, int]
    layer_arrays: Mapping[str, np.ndarray]
    rank_arrays: Mapping[str, np.ndarray]
    state_snapshots: Mapping[str, Mapping[str, Any]]
    restore_evidence: Mapping[str, Mapping[str, Any]]
    secant_restore_evidence: Mapping[str, Mapping[str, Any]]
    process_groups: Sequence[Mapping[str, Any]]
    collective_contract: Mapping[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file in bounded host-memory chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(json.dumps(list(values), separators=(",", ":")).encode("ascii"))


def membership_digest(ranks: Sequence[int]) -> str:
    """Hash one sorted process-group membership using the v3 encoding."""

    return _sha256_bytes(json.dumps(list(ranks), separators=(",", ":")).encode("ascii"))


def identity_from_runtime(
    wandb_writer: object, *, world_size: int, global_rank: int
) -> RuntimeIdentity:
    """Read immutable run identity on the sole last-world publishing rank.

    Args:
        wandb_writer: Initialized W&B module/facade carrying ``run.id``.
        world_size: Actual distributed world size.
        global_rank: Actual global rank invoking publication.

    Raises:
        RuntimeError: If the caller is not the last world rank or any identity
            is absent, forged, or not scheduler-owned.
    """

    if isinstance(world_size, bool) or world_size < 1:
        raise RuntimeError("diag/v2 publication requires a positive world size")
    if global_rank != world_size - 1:
        raise RuntimeError("diag/v2 artifacts may only be published by world_size-1")
    run = getattr(wandb_writer, "run", None)
    run_id = getattr(run, "id", None)
    environment_run_id = os.environ.get("WANDB_RUN_ID")
    if not isinstance(run_id, str) or not run_id or run_id != environment_run_id:
        raise RuntimeError("initialized W&B run ID must equal WANDB_RUN_ID")
    job_name = os.environ.get("SLURM_JOB_NAME")
    training_job_id = os.environ.get("SLURM_JOB_ID")
    if not isinstance(job_name, str) or not job_name:
        raise RuntimeError("SLURM_JOB_NAME must be set inside the allocation")
    if not isinstance(training_job_id, str) or not _SLURM_JOB_ID.fullmatch(training_job_id):
        raise RuntimeError("SLURM_JOB_ID must be an immutable scheduler job ID")
    return RuntimeIdentity(
        run_id=run_id,
        job_name=job_name,
        training_job_id=training_job_id,
        writer_rank=global_rank,
        world_size=world_size,
    )


def repro_from_runtime() -> dict[str, str]:
    """Read and validate the canonical cross-repository repro identity."""

    environments = {
        "scaling_commit": "DIAG_V2_SCALING_COMMIT",
        "megatron_commit": "DIAG_V2_MEGATRON_COMMIT",
        "resolved_config_sha256": "DIAG_V2_RESOLVED_CONFIG_SHA256",
        "scaling_bundle_sha256": "DIAG_V2_SCALING_BUNDLE_SHA256",
        "megatron_bundle_sha256": "DIAG_V2_MEGATRON_BUNDLE_SHA256",
    }
    repro = {name: os.environ.get(environment) for name, environment in environments.items()}
    missing = [environments[name] for name, value in repro.items() if not value]
    if missing:
        raise RuntimeError("diag/v2 launcher provenance is missing: " + ", ".join(missing))
    result = {name: str(value) for name, value in repro.items()}
    for name in ("scaling_commit", "megatron_commit"):
        if not _GIT_SHA.fullmatch(result[name]):
            raise RuntimeError(f"{environments[name]} must be a full lowercase git SHA")
    for name in ("resolved_config_sha256", "scaling_bundle_sha256", "megatron_bundle_sha256"):
        if not _SHA256.fullmatch(result[name]):
            raise RuntimeError(f"{environments[name]} must be a lowercase SHA-256")
    if result["megatron_commit"] != verified_source_commit():
        raise RuntimeError("DIAG_V2_MEGATRON_COMMIT disagrees with the running checkout")
    return result


def optimizer_chain_signature(optimizer: object) -> dict[str, Any]:
    """Derive the only admitted one-child optimizer chain without relabeling.

    The supported shape is exactly ``ChainedOptimizer -> DistributedOptimizer
    -> Adam|AdamW|FusedAdam``.  Direct, empty, multi-child, and unknown chains
    fail closed.
    """

    if type(optimizer).__name__ != "ChainedOptimizer":
        raise RuntimeError("diagnostics require an actual ChainedOptimizer root")
    children = getattr(optimizer, "chained_optimizers", None)
    if not isinstance(children, (list, tuple)) or len(children) != 1:
        raise RuntimeError("ChainedOptimizer must contain exactly one child")
    distributed = children[0]
    if type(distributed).__name__ != "DistributedOptimizer":
        raise RuntimeError("diagnostics require a DistributedOptimizer child")
    leaf = getattr(distributed, "optimizer", None)
    leaf_name = type(leaf).__name__
    if leaf_name not in {"Adam", "AdamW", "FusedAdam"}:
        raise RuntimeError(f"unsupported optimizer leaf {leaf_name!r}")
    return {
        "name": "ChainedOptimizer",
        "children": [
            {"name": "DistributedOptimizer", "children": [{"name": leaf_name, "children": []}]}
        ],
    }


def build_runtime_signature(
    optimizer: object,
    *,
    precision: str,
    dp: int,
    tp: int,
    pp: int,
    cp: int,
    ep: int = 1,
    vpp: int = 1,
    moe: bool = False,
    fsdp: bool = False,
    overlap_param_gather: bool = False,
    parameter_cache_mode: str = "none",
) -> dict[str, Any]:
    """Build an exact v3 runtime signature from observed runtime objects."""

    dimensions = {"dp": dp, "tp": tp, "pp": pp, "cp": cp, "ep": ep, "vpp": vpp}
    if any(isinstance(value, bool) or value < 1 for value in dimensions.values()):
        raise RuntimeError("diagnostic topology dimensions must be positive integers")
    if not precision or not parameter_cache_mode:
        raise RuntimeError("precision and parameter cache mode must be nonempty")
    return {
        "backend": "megatron",
        "optimizer_chain": optimizer_chain_signature(optimizer),
        "precision": precision,
        **dimensions,
        "moe": bool(moe),
        "fsdp": bool(fsdp),
        "overlap_param_gather": bool(overlap_param_gather),
        "parameter_cache_mode": parameter_cache_mode,
    }


def unavailable_state_snapshot(reason: str) -> dict[str, Any]:
    """Return one explicitly inapplicable v3 state snapshot."""

    if not reason:
        raise ValueError("an inapplicable state snapshot requires a reason")
    return {
        "applicable": False,
        "reason": reason,
        "model_sha256_by_rank": [],
        "optimizer_sha256_by_rank": [],
        "model_sha256": None,
        "optimizer_sha256": None,
        "model_consensus_sha256_by_rank": [],
        "optimizer_consensus_sha256_by_rank": [],
    }


def complete_state_snapshot(
    *, model_sha256_by_rank: Sequence[str], optimizer_sha256_by_rank: Sequence[str]
) -> dict[str, Any]:
    """Build one all-rank state snapshot and its consensus evidence."""

    model = list(model_sha256_by_rank)
    optimizer = list(optimizer_sha256_by_rank)
    if not model or len(model) != len(optimizer):
        raise ValueError("state snapshot must cover the same nonempty rank set")
    if any(not _SHA256.fullmatch(value) for value in (*model, *optimizer)):
        raise ValueError("state snapshot values must be lowercase SHA-256 digests")
    model_digest = _rank_digest(model)
    optimizer_digest = _rank_digest(optimizer)
    return {
        "applicable": True,
        "reason": None,
        "model_sha256_by_rank": model,
        "optimizer_sha256_by_rank": optimizer,
        "model_sha256": model_digest,
        "optimizer_sha256": optimizer_digest,
        "model_consensus_sha256_by_rank": [model_digest] * len(model),
        "optimizer_consensus_sha256_by_rank": [optimizer_digest] * len(model),
    }


def unavailable_restore_evidence(reason: str) -> dict[str, Any]:
    """Return one explicitly inapplicable v3 restoration record."""

    if not reason:
        raise ValueError("inapplicable restoration evidence requires a reason")
    return {
        "applicable": False,
        "reason": reason,
        "before_sha256_by_rank": [],
        "after_sha256_by_rank": [],
        "before_sha256": None,
        "after_sha256": None,
        "before_consensus_sha256_by_rank": [],
        "after_consensus_sha256_by_rank": [],
    }


def complete_restore_evidence(
    *, before_sha256_by_rank: Sequence[str], after_sha256_by_rank: Sequence[str]
) -> dict[str, Any]:
    """Build exact bitwise restoration evidence, rejecting any mismatch."""

    before = list(before_sha256_by_rank)
    after = list(after_sha256_by_rank)
    if not before or before != after:
        raise ValueError("restoration evidence requires identical nonempty before/after digests")
    if any(not _SHA256.fullmatch(value) for value in before):
        raise ValueError("restoration evidence values must be lowercase SHA-256 digests")
    digest = _rank_digest(before)
    return {
        "applicable": True,
        "reason": None,
        "before_sha256_by_rank": before,
        "after_sha256_by_rank": after,
        "before_sha256": digest,
        "after_sha256": digest,
        "before_consensus_sha256_by_rank": [digest] * len(before),
        "after_consensus_sha256_by_rank": [digest] * len(before),
    }


def tier0_state_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return truthful Tier-0 snapshot, normal-restore, and secant-restore maps."""

    snapshots = {
        name: unavailable_state_snapshot(TIER0_STATE_REASON) for name in ("pre", "post", "midpoint")
    }
    restore = {name: unavailable_restore_evidence(TIER0_STATE_REASON) for name in STATE_COMPONENTS}
    secant = {name: unavailable_restore_evidence(TIER0_STATE_REASON) for name in STATE_COMPONENTS}
    return snapshots, restore, secant


def tier0_sampling_from_rank_evidence(
    rank_evidence: Sequence[Sequence[float]],
    *,
    seed: int,
    mask_shape: Sequence[int],
    descriptor_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert the heartbeat's fixed 11-field host rows to Tier-0 sampling evidence."""

    rows = np.asarray(rank_evidence, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 11:
        raise ValueError("Tier-0 rank evidence must have fixed shape [world, 11]")
    if list(map(int, rows[:, 0])) != list(range(len(rows))):
        raise ValueError("Tier-0 rank evidence must be ordered by exact world rank")
    shape = [int(value) for value in mask_shape]
    if not shape or any(value < 1 for value in shape):
        raise ValueError("Tier-0 mask shape must contain positive dimensions")
    populations = list(map(int, rows[:, 10]))
    checksums = list(map(float, rows[:, 9]))
    if any(value < 0 for value in populations) or any(
        not math.isfinite(value) for value in checksums
    ):
        raise ValueError("Tier-0 population/checksum evidence is invalid")
    if not _SHA256.fullmatch(descriptor_sha256):
        raise ValueError("Tier-0 descriptor must be a lowercase SHA-256")
    sampling = {
        "selector": "tier0_mask_population_checksum_v1",
        "seed": int(seed),
        "mask_shape": shape,
        "global_mask_population": sum(populations),
        "mask_population_by_rank": populations,
        "global_mask_checksum": math.fsum(checksums),
        "mask_checksum_by_rank": checksums,
        "checksum_algorithm": "weighted_f32_summed_f64_v1",
        "descriptor_sha256": descriptor_sha256,
        "identity_disclaimer": TIER0_IDENTITY_DISCLAIMER,
    }
    return sampling, {"descriptor_sha256_by_rank": [descriptor_sha256] * len(rows)}


def identity_sampling_evidence(
    *,
    seed: int,
    selected_sample_id_hashes: Sequence[str],
    valid_position_count: int,
    selected_sample_ids_sha256: str,
    valid_token_ids_sha256: str,
    descriptor_sha256_by_rank: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build Tier-1/2 exact selected-sample and valid-token identity evidence."""

    selected = list(selected_sample_id_hashes)
    descriptor = list(descriptor_sha256_by_rank)
    all_hashes = (*selected, *descriptor, selected_sample_ids_sha256, valid_token_ids_sha256)
    if any(not _SHA256.fullmatch(value) for value in all_hashes):
        raise ValueError("identity sampling requires lowercase SHA-256 values")
    if len(selected) != len(set(selected)) or valid_position_count < 1 or not descriptor:
        raise ValueError("identity sampling selection/count/descriptor evidence is invalid")
    sampling = {
        "selector": "global_topk_hash_v1",
        "seed": int(seed),
        "selected_sample_id_hashes": selected,
        "valid_position_count": int(valid_position_count),
        "selected_sample_ids_sha256": selected_sample_ids_sha256,
        "valid_token_ids_sha256": valid_token_ids_sha256,
    }
    digests = {
        "descriptor_sha256_by_rank": descriptor,
        "selected_sample_ids_sha256_by_rank": [selected_sample_ids_sha256] * len(descriptor),
        "valid_token_ids_sha256_by_rank": [valid_token_ids_sha256] * len(descriptor),
    }
    return sampling, digests


def expected_layer_pairs(
    effective_tier: int, *, attention_available: bool, moe_available: bool
) -> tuple[tuple[str, str], ...]:
    """Return the exact all-layer row identities required by Scaling v3."""

    pairs = {
        ("residual", "activation_rms"),
        ("residual", "activation_max_abs"),
        ("residual", "dgrad_rms"),
        ("norm", "update_relative_rms"),
        *(
            (family, metric)
            for family in ("qkv", "attn_out", "fc1", "fc2")
            for metric in ("activation_max_abs", "dgrad_rms", "update_relative_rms", "retention")
        ),
    }
    if effective_tier >= 1:
        pairs.update(
            (family, metric)
            for family in ("residual", "qkv", "attn_out", "fc1", "fc2")
            for metric in ("dy_rel", "response_starved")
        )
        if attention_available:
            pairs.update(
                ("attention", metric)
                for metric in (
                    "logit_abs_p50",
                    "logit_abs_p90",
                    "entropy_p10",
                    "entropy_p50",
                    "collapse_fraction",
                )
            )
    if effective_tier >= 2:
        pairs.update(
            (family, metric)
            for family in ("residual", "qkv", "attn_out", "fc1", "fc2")
            for metric in (
                "true_response",
                "secant_error",
                "secant_cosine",
                "realized_midpoint_fraction",
                "replay_floor",
            )
        )
    if moe_available:
        pairs.update(
            ("moe", metric)
            for metric in (
                "load_cv",
                "min_token_share",
                "starved_expert_count",
                "drop_fraction",
                "router_logit_max",
                "expert_bias_range",
                "update_starved_fraction",
                "valid",
            )
        )
    return tuple(sorted(pairs))


def build_layer_metric_arrays(
    rows: Mapping[tuple[int, str, str], Mapping[str, float | int]],
    *,
    num_layers: int,
    layer_pp_owners: Sequence[int],
    effective_tier: int,
    attention_available: bool = False,
    moe_available: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
    """Encode exact per-layer sufficient statistics into bounded v3 arrays."""

    owners = list(map(int, layer_pp_owners))
    if num_layers < 1 or len(owners) != num_layers:
        raise ValueError("layer owners must contain one entry per model layer")
    pairs = expected_layer_pairs(
        effective_tier, attention_available=attention_available, moe_available=moe_available
    )
    identities = [
        (layer, family, metric) for layer in range(num_layers) for family, metric in pairs
    ]
    if set(rows) != set(identities):
        raise ValueError("layer evidence identities must exactly cover every model layer")
    families = {name: index for index, name in enumerate(sorted({pair[0] for pair in pairs}))}
    metrics = {name: index for index, name in enumerate(sorted({pair[1] for pair in pairs}))}
    arrays: dict[str, np.ndarray] = {
        "global_layer_id": np.asarray([item[0] for item in identities], dtype=np.int32),
        "pp_owner": np.asarray([owners[item[0]] for item in identities], dtype=np.int16),
        "family_code": np.asarray([families[item[1]] for item in identities], dtype=np.uint8),
        "metric_code": np.asarray([metrics[item[2]] for item in identities], dtype=np.uint16),
        "value": np.empty(len(identities), dtype=np.float32),
        "valid": np.empty(len(identities), dtype=np.uint8),
        "count": np.empty(len(identities), dtype=np.int64),
        "sum": np.empty(len(identities), dtype=np.float64),
        "sum_sq": np.empty(len(identities), dtype=np.float64),
        "denominator_sum_sq": np.empty(len(identities), dtype=np.float64),
        "zero_count": np.empty(len(identities), dtype=np.int64),
        "nonfinite_count": np.empty(len(identities), dtype=np.int64),
    }
    fields = (
        "value",
        "valid",
        "count",
        "sum",
        "sum_sq",
        "denominator_sum_sq",
        "zero_count",
        "nonfinite_count",
    )
    for index, identity in enumerate(identities):
        row = rows[identity]
        if set(row) != set(fields):
            raise ValueError(f"layer evidence fields are not exact for {identity}")
        for field in fields:
            arrays[field][index] = row[field]
    return arrays, metrics, families


def rank_perf_arrays(rows: Sequence[Mapping[str, float | int]]) -> dict[str, np.ndarray]:
    """Encode one exact v3 rank-performance row per world rank."""

    if not rows:
        raise ValueError("rank performance evidence cannot be empty")
    if any(set(row) != set(RANK_PERF_DTYPES) for row in rows):
        raise ValueError("rank performance fields must exactly match artifact-v3")
    if [int(row["rank"]) for row in rows] != list(range(len(rows))):
        raise ValueError("rank performance rows must be ordered by exact world rank")
    return {
        name: np.asarray([row[name] for row in rows], dtype=dtype)
        for name, dtype in RANK_PERF_DTYPES.items()
    }


def tier0_rank_perf_from_heartbeat(
    rank_evidence: Sequence[Sequence[float]], *, writer_rank: int
) -> dict[str, np.ndarray]:
    """Truthfully adapt the heartbeat's fixed host rows without inventing all-rank peaks."""

    source = np.asarray(rank_evidence, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 11:
        raise ValueError("Tier-0 rank evidence must have fixed shape [world, 11]")
    world_size = len(source)
    if list(map(int, source[:, 0])) != list(range(world_size)):
        raise ValueError("Tier-0 rank evidence must exactly cover ordered world ranks")
    if not 0 <= writer_rank < world_size:
        raise ValueError("Tier-0 writer rank is outside the world")
    unavailable = np.full(world_size, np.nan, dtype=np.float64)
    event = unavailable.copy()
    ordinary = unavailable.copy()
    sink_allocated = unavailable.copy()
    sink_reserved = unavailable.copy()
    event[writer_rank] = source[writer_rank, 1]
    ordinary[writer_rank] = source[writer_rank, 2]
    sink_allocated[writer_rank] = source[writer_rank, 7]
    sink_reserved[writer_rank] = source[writer_rank, 8]
    return {
        "rank": source[:, 0].astype(np.int32),
        "event_wall_time_ms": event,
        "ordinary_step_wall_time_ms": ordinary,
        "detected_hbm_capacity_bytes": source[:, 3].astype(np.float64),
        "pre_event_allocated_bytes": source[:, 4].astype(np.float64),
        "pre_event_reserved_bytes": source[:, 5].astype(np.float64),
        "predicted_post_gather_peak_allocated_bytes": unavailable.copy(),
        "predicted_post_gather_peak_reserved_bytes": unavailable.copy(),
        "post_gather_peak_allocated_bytes": unavailable.copy(),
        "post_gather_peak_reserved_bytes": unavailable.copy(),
        "sink_post_interval_peak_allocated_bytes": sink_allocated,
        "sink_post_interval_peak_reserved_bytes": sink_reserved,
    }


def tier0_collective_operations(
    num_layers: int, world_size: int
) -> dict[str, list[dict[str, Any]]]:
    """Return the exact fixed Tier-0 world ledger declared by the v3 contract."""

    if num_layers < 1 or world_size < 1:
        raise ValueError("Tier-0 collective ledger requires positive layer/world sizes")
    return {
        "world": [
            {
                "name": "all_reduce",
                "phase": "mask_population",
                "count": 1,
                "bytes": 96 * 5 * num_layers,
            },
            {
                "name": "all_reduce",
                "phase": "mask_checksum",
                "count": 1,
                "bytes": 96 * 10 * num_layers,
            },
            {"name": "all_reduce", "phase": "descriptor_digest", "count": 1, "bytes": 96 * 10},
            {"name": "all_gather", "phase": "rank_summary", "count": 1, "bytes": 88 * world_size},
        ]
    }


def build_process_group_evidence(
    *,
    topology: Mapping[str, int],
    membership_ranks_by_group: Mapping[str, Sequence[Sequence[int]]],
    operations_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Bind exact observed memberships and operation ledgers for active groups."""

    world_size = math.prod(int(topology[name]) for name in ("dp", "tp", "pp", "cp"))
    expected = {"world"} | {
        name for name in ("dp", "tp", "pp", "cp", "ep") if int(topology[name]) > 1
    }
    if set(membership_ranks_by_group) != expected or set(operations_by_group) != expected:
        raise ValueError("process-group membership and ledgers must exactly cover active groups")
    groups: list[dict[str, Any]] = []
    for name in ("world", "dp", "tp", "pp", "cp", "ep"):
        if name not in expected:
            continue
        memberships = [list(map(int, members)) for members in membership_ranks_by_group[name]]
        if len(memberships) != world_size:
            raise ValueError(f"{name} membership evidence must contain one row per rank")
        operations = [dict(operation) for operation in operations_by_group[name]]
        groups.append(
            {
                "name": name,
                "membership_ranks_by_rank": memberships,
                "membership_sha256_by_rank": [
                    membership_digest(members) for members in memberships
                ],
                "operations": operations,
            }
        )
    return groups


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": _sha256_bytes(np.ascontiguousarray(value).tobytes(order="C")),
    }


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, value in sorted(arrays.items()):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _npz_uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(info.file_size for info in archive.infolist())


def _validate_array_set(
    arrays: Mapping[str, np.ndarray], expected: Mapping[str, str], label: str
) -> int:
    if set(arrays) != set(expected):
        raise ValueError(f"{label} array names are not exact")
    lengths: set[int] = set()
    for name, dtype in expected.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.ndim != 1 or str(value.dtype) != dtype:
            raise ValueError(f"{label}:{name} must be a one-dimensional {dtype} array")
        lengths.add(len(value))
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise ValueError(f"{label} arrays must have one common positive row count")
    return next(iter(lengths))


def _validate_tier_payload(payload: Mapping[str, object], effective_tier: int) -> None:
    try:
        from .schema import assert_tiered_payload_schema
    except ImportError:
        assert_tiered_payload_schema = None
    if assert_tiered_payload_schema is not None:
        assert_tiered_payload_schema(payload, effective_tier=effective_tier)
        return
    # Standalone branch fallback; the integration branch supplies the canonical
    # schema validator above.  Keep the fallback exact so this module is testable
    # before the two owned branches are combined.
    from .function_response import TIER1_KEYS
    from .schema import TIER0_KEYS
    from .secant import TIER2_OUTPUT_KEYS

    expected = TIER0_KEYS
    if effective_tier >= 1:
        expected = (*expected, *TIER1_KEYS)
    if effective_tier >= 2:
        expected = (*expected, *TIER2_OUTPUT_KEYS)
    if tuple(payload) != tuple(expected):
        raise ValueError("diagnostic scalar payload keys/order do not match effective tier")


def _validate_rank_hashes(
    values: object, *, world_size: int, label: str, consensus: bool = False
) -> list[str]:
    if (
        not isinstance(values, list)
        or len(values) != world_size
        or any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in values)
    ):
        raise ValueError(f"{label} must contain one SHA-256 per rank")
    if consensus and len(set(values)) != 1:
        raise ValueError(f"{label} must reach rank consensus")
    return values


def _validate_snapshot(
    snapshot: Mapping[str, Any], *, expected: bool, reason: str | None, world_size: int, label: str
) -> None:
    fields = {
        "applicable",
        "reason",
        "model_sha256_by_rank",
        "optimizer_sha256_by_rank",
        "model_sha256",
        "optimizer_sha256",
        "model_consensus_sha256_by_rank",
        "optimizer_consensus_sha256_by_rank",
    }
    if (
        set(snapshot) != fields
        or snapshot["applicable"] is not expected
        or snapshot["reason"] != reason
    ):
        raise ValueError(f"{label} snapshot applicability/reason/fields are not exact")
    for component in ("model", "optimizer"):
        values = snapshot[f"{component}_sha256_by_rank"]
        global_hash = snapshot[f"{component}_sha256"]
        consensus_values = snapshot[f"{component}_consensus_sha256_by_rank"]
        if not expected:
            if values or consensus_values or global_hash is not None:
                raise ValueError(f"inapplicable {label}.{component} snapshot must be empty")
            continue
        values = _validate_rank_hashes(
            values, world_size=world_size, label=f"{label}.{component}_sha256_by_rank"
        )
        if global_hash != _rank_digest(values):
            raise ValueError(f"{label}.{component} global digest is wrong")
        consensus_values = _validate_rank_hashes(
            consensus_values,
            world_size=world_size,
            label=f"{label}.{component}_consensus_sha256_by_rank",
            consensus=True,
        )
        if consensus_values[0] != global_hash:
            raise ValueError(f"{label}.{component} consensus digest is wrong")


def _validate_restore(
    evidence: Mapping[str, Any], *, expected: bool, reason: str | None, world_size: int, label: str
) -> None:
    fields = {
        "applicable",
        "reason",
        "before_sha256_by_rank",
        "after_sha256_by_rank",
        "before_sha256",
        "after_sha256",
        "before_consensus_sha256_by_rank",
        "after_consensus_sha256_by_rank",
    }
    if (
        set(evidence) != fields
        or evidence["applicable"] is not expected
        or evidence["reason"] != reason
    ):
        raise ValueError(f"{label} restore applicability/reason/fields are not exact")
    before = evidence["before_sha256_by_rank"]
    after = evidence["after_sha256_by_rank"]
    before_consensus = evidence["before_consensus_sha256_by_rank"]
    after_consensus = evidence["after_consensus_sha256_by_rank"]
    if not expected:
        if any(
            (
                before,
                after,
                before_consensus,
                after_consensus,
                evidence["before_sha256"] is not None,
                evidence["after_sha256"] is not None,
            )
        ):
            raise ValueError(f"inapplicable {label} restore evidence must be empty")
        return
    before = _validate_rank_hashes(
        before, world_size=world_size, label=f"{label}.before_sha256_by_rank"
    )
    after = _validate_rank_hashes(
        after, world_size=world_size, label=f"{label}.after_sha256_by_rank"
    )
    if before != after:
        raise ValueError(f"{label} was not restored bitwise")
    digest = _rank_digest(before)
    if evidence["before_sha256"] != digest or evidence["after_sha256"] != digest:
        raise ValueError(f"{label} global restore digest is wrong")
    before_consensus = _validate_rank_hashes(
        before_consensus,
        world_size=world_size,
        label=f"{label}.before_consensus_sha256_by_rank",
        consensus=True,
    )
    after_consensus = _validate_rank_hashes(
        after_consensus,
        world_size=world_size,
        label=f"{label}.after_consensus_sha256_by_rank",
        consensus=True,
    )
    if before_consensus[0] != digest or after_consensus[0] != digest:
        raise ValueError(f"{label} restore consensus digest is wrong")


def _validate_evidence(evidence: EventEvidence, identity: RuntimeIdentity) -> None:
    tiers = (evidence.requested_tier, evidence.effective_tier, evidence.require_tier)
    if any(isinstance(value, bool) or value not in (0, 1, 2) for value in tiers):
        raise ValueError("requested/effective/required tiers must be exactly 0, 1, or 2")
    if (
        evidence.effective_tier > evidence.requested_tier
        or evidence.effective_tier < evidence.require_tier
    ):
        raise ValueError("effective diagnostic tier contradicts requested/required tiers")
    if evidence.status not in {"ok", "degraded", "unsupported", "budget_exceeded", "invalid"}:
        raise ValueError("diagnostic event status is unknown")
    if evidence.status == "ok" and evidence.status_reason is not None:
        raise ValueError("ok diagnostic events require status_reason=None")
    if evidence.status != "ok" and not evidence.status_reason:
        raise ValueError("non-ok diagnostic events require a nonempty status reason")
    if evidence.effective_tier == 0 and evidence.status != "invalid":
        raise ValueError("Tier-0 artifact-v3 evidence must truthfully be invalid")
    if evidence.effective_tier < evidence.requested_tier and evidence.status == "ok":
        raise ValueError("a tier downgrade cannot claim status=ok")
    signature = dict(evidence.capability_signature)
    expected_signature = {
        "backend",
        "optimizer_chain",
        "precision",
        "dp",
        "tp",
        "pp",
        "cp",
        "ep",
        "vpp",
        "moe",
        "fsdp",
        "overlap_param_gather",
        "parameter_cache_mode",
    }
    if set(signature) != expected_signature or signature.get("backend") != "megatron":
        raise ValueError("capability signature fields/backend are not exact")
    chain = signature["optimizer_chain"]
    try:
        root = chain["name"]
        root_children = chain["children"]
        distributed = root_children[0]
        leaf = distributed["children"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("optimizer signature is not an exact three-level chain") from error
    if (
        set(chain) != {"name", "children"}
        or root != "ChainedOptimizer"
        or len(root_children) != 1
        or set(distributed) != {"name", "children"}
        or distributed["name"] != "DistributedOptimizer"
        or len(distributed["children"]) != 1
        or set(leaf) != {"name", "children"}
        or leaf["name"] not in {"Adam", "AdamW", "FusedAdam"}
        or leaf["children"] != []
    ):
        raise ValueError("optimizer signature is not the admitted one-child chain")
    topology = dict(evidence.topology)
    expected_topology = {"dp", "tp", "pp", "cp", "ep", "vpp", "num_layers", "layer_pp_owners"}
    if set(topology) != expected_topology:
        raise ValueError("topology fields are not exact")
    for name in ("dp", "tp", "pp", "cp", "ep", "vpp"):
        if topology[name] != signature[name]:
            raise ValueError(f"topology.{name} disagrees with capability signature")
    world_size = math.prod(int(topology[name]) for name in ("dp", "tp", "pp", "cp"))
    if world_size != identity.world_size:
        raise ValueError("event topology disagrees with immutable runtime world size")
    if len(topology["layer_pp_owners"]) != int(topology["num_layers"]):
        raise ValueError("layer PP ownership must cover every model layer")
    layer_count = _validate_array_set(evidence.layer_arrays, LAYER_DTYPES, "layer_metrics")
    rank_count = _validate_array_set(evidence.rank_arrays, RANK_PERF_DTYPES, "rank_perf")
    if rank_count != world_size or list(map(int, evidence.rank_arrays["rank"])) != list(
        range(world_size)
    ):
        raise ValueError("rank_perf must exactly cover 0..world_size-1")
    if set(evidence.capability_status) != {"attention", "moe"} or any(
        state not in {"available", "unsupported", "not_applicable"}
        for state in evidence.capability_status.values()
    ):
        raise ValueError("capability status fields/states are not exact")
    if signature["moe"] is False and evidence.capability_status["moe"] != "not_applicable":
        raise ValueError("dense runtime must mark MoE metrics not_applicable")
    if evidence.effective_tier == 0 and evidence.capability_status["attention"] == "available":
        raise ValueError("Tier-0 cannot claim attention metrics")
    for label, enum in (("metric", evidence.metric_enums), ("family", evidence.family_enums)):
        if (
            not enum
            or any(not isinstance(name, str) or not name for name in enum)
            or any(isinstance(code, bool) or code < 0 for code in enum.values())
            or len(set(enum.values())) != len(enum)
        ):
            raise ValueError(f"{label} enum dictionary is invalid")
    if set(map(int, evidence.layer_arrays["metric_code"])) != set(evidence.metric_enums.values()):
        raise ValueError("metric enum codes disagree with layer rows")
    if set(map(int, evidence.layer_arrays["family_code"])) != set(evidence.family_enums.values()):
        raise ValueError("family enum codes disagree with layer rows")
    metric_by_code = {code: name for name, code in evidence.metric_enums.items()}
    family_by_code = {code: name for name, code in evidence.family_enums.items()}
    expected_pairs = set(
        expected_layer_pairs(
            evidence.effective_tier,
            attention_available=evidence.capability_status["attention"] == "available",
            moe_available=evidence.capability_status["moe"] == "available",
        )
    )
    expected_identities = {
        (layer, family, metric)
        for layer in range(int(topology["num_layers"]))
        for family, metric in expected_pairs
    }
    actual_identities: set[tuple[int, str, str]] = set()
    for index in range(layer_count):
        layer = int(evidence.layer_arrays["global_layer_id"][index])
        if not 0 <= layer < int(topology["num_layers"]):
            raise ValueError("layer row has an out-of-bounds global layer")
        if int(evidence.layer_arrays["pp_owner"][index]) != int(topology["layer_pp_owners"][layer]):
            raise ValueError("layer row disagrees with declared PP ownership")
        try:
            family = family_by_code[int(evidence.layer_arrays["family_code"][index])]
            metric = metric_by_code[int(evidence.layer_arrays["metric_code"][index])]
        except KeyError as error:
            raise ValueError("layer row references an unknown enum code") from error
        identity_row = (layer, family, metric)
        if identity_row in actual_identities:
            raise ValueError("layer evidence contains a duplicate identity")
        actual_identities.add(identity_row)
        valid = int(evidence.layer_arrays["valid"][index])
        count = int(evidence.layer_arrays["count"][index])
        nonfinite = int(evidence.layer_arrays["nonfinite_count"][index])
        denominator = float(evidence.layer_arrays["denominator_sum_sq"][index])
        if valid not in (0, 1) or count < 0 or nonfinite < 0 or nonfinite > count:
            raise ValueError("layer evidence population fields are invalid")
        if evidence.status == "ok" and not valid:
            raise ValueError("ok event cannot contain invalid layer rows")
        if valid and (
            count < 1
            or nonfinite != 0
            or not math.isfinite(float(evidence.layer_arrays["value"][index]))
            or not math.isfinite(float(evidence.layer_arrays["sum"][index]))
            or not math.isfinite(float(evidence.layer_arrays["sum_sq"][index]))
            or not math.isfinite(denominator)
            or denominator < 0
        ):
            raise ValueError("valid layer evidence contains invalid statistics")
    if actual_identities != expected_identities:
        raise ValueError("layer metric identities do not exactly match the effective tier")
    ranks = evidence.rank_arrays
    sink_fields = (
        "sink_post_interval_peak_allocated_bytes",
        "sink_post_interval_peak_reserved_bytes",
    )
    for name in sink_fields:
        for rank, value in enumerate(ranks[name]):
            if rank == identity.writer_rank:
                if not math.isfinite(float(value)) or value < 0:
                    raise ValueError("sink memory evidence is missing on writer rank")
            elif not math.isnan(float(value)):
                raise ValueError("sink memory evidence must remain writer-only")
    if evidence.effective_tier == 0:
        if evidence.sampling.get("selector") != "tier0_mask_population_checksum_v1":
            raise ValueError("Tier-0 sampling selector is not truthful")
        if evidence.sampling.get("identity_disclaimer") != TIER0_IDENTITY_DISCLAIMER:
            raise ValueError("Tier-0 sampling must carry the collision disclaimer")
        if set(evidence.digests) != {"descriptor_sha256_by_rank"}:
            raise ValueError("Tier-0 digests must not claim sample/token identity")
        unavailable = (
            "predicted_post_gather_peak_allocated_bytes",
            "predicted_post_gather_peak_reserved_bytes",
            "post_gather_peak_allocated_bytes",
            "post_gather_peak_reserved_bytes",
        )
        if any(np.any(~np.isnan(evidence.rank_arrays[name])) for name in unavailable):
            raise ValueError("Tier-0 must not invent all-rank post-gather memory")
        for name in ("event_wall_time_ms", "ordinary_step_wall_time_ms"):
            for rank, value in enumerate(ranks[name]):
                if rank == identity.writer_rank:
                    if not math.isfinite(float(value)) or value <= 0:
                        raise ValueError("Tier-0 sink latency must be positive")
                elif not math.isnan(float(value)):
                    raise ValueError("Tier-0 latency evidence must remain writer-only")
    else:
        if evidence.sampling.get("selector") != "global_topk_hash_v1":
            raise ValueError("Tier-1/2 sampling must use exact global identity")
        if set(evidence.digests) != {
            "descriptor_sha256_by_rank",
            "selected_sample_ids_sha256_by_rank",
            "valid_token_ids_sha256_by_rank",
        }:
            raise ValueError("Tier-1/2 identity digests are not exact")
        full_rank_fields = set(RANK_PERF_DTYPES) - {
            "rank",
            "sink_post_interval_peak_allocated_bytes",
            "sink_post_interval_peak_reserved_bytes",
        }
        if any(np.any(~np.isfinite(evidence.rank_arrays[name])) for name in full_rank_fields):
            raise ValueError("Tier-1/2 rank performance evidence must cover all ranks")
        if np.any(ranks["event_wall_time_ms"] <= 0) or np.any(
            ranks["ordinary_step_wall_time_ms"] <= 0
        ):
            raise ValueError("Tier-1/2 rank latency must be positive")
        if np.any(ranks["detected_hbm_capacity_bytes"] <= 0):
            raise ValueError("detected HBM capacity must be positive")
        if np.any(
            ranks["post_gather_peak_allocated_bytes"] < ranks["pre_event_allocated_bytes"]
        ) or np.any(ranks["post_gather_peak_reserved_bytes"] < ranks["pre_event_reserved_bytes"]):
            raise ValueError("post-gather peak cannot be below pre-event memory")
    if set(evidence.state_snapshots) != {"pre", "post", "midpoint"}:
        raise ValueError("state snapshot phase names are not exact")
    for phase, snapshot in evidence.state_snapshots.items():
        expected_snapshot = evidence.effective_tier >= 1 and (
            phase in {"pre", "post"} or evidence.effective_tier == 2
        )
        reason = (
            TIER0_STATE_REASON
            if evidence.effective_tier == 0
            else (None if expected_snapshot else MIDPOINT_STATE_REASON)
        )
        _validate_snapshot(
            snapshot, expected=expected_snapshot, reason=reason, world_size=world_size, label=phase
        )
    for field in (evidence.restore_evidence, evidence.secant_restore_evidence):
        if set(field) != set(STATE_COMPONENTS):
            raise ValueError("restoration component names are not exact")
    runtime_components = {
        "model": True,
        "optimizer": True,
        "rng": True,
        "mutable_state": True,
        "data_iterator": True,
        "fp8": signature["precision"] == "fp8" or signature["parameter_cache_mode"] == "fp8",
        "router": bool(signature["moe"]),
        "cache": signature["parameter_cache_mode"] != "none",
    }
    for field_name, evidence_map, secant in (
        ("restore_evidence", evidence.restore_evidence, False),
        ("secant_restore_evidence", evidence.secant_restore_evidence, True),
    ):
        enabled = evidence.effective_tier == 2 if secant else evidence.effective_tier >= 1
        for component, component_present in runtime_components.items():
            expected_restore = enabled and component_present
            if evidence.effective_tier == 0:
                reason = TIER0_STATE_REASON
            elif secant and not expected_restore:
                reason = SECANT_STATE_REASON
            elif not expected_restore:
                reason = COMPONENT_STATE_REASON
            else:
                reason = None
            _validate_restore(
                evidence_map[component],
                expected=expected_restore,
                reason=reason,
                world_size=world_size,
                label=f"{field_name}.{component}",
            )
    group_names = [group.get("name") for group in evidence.process_groups]
    expected_groups = {"world"} | {
        name for name in ("dp", "tp", "pp", "cp", "ep") if int(topology[name]) > 1
    }
    if len(group_names) != len(set(group_names)) or set(group_names) != expected_groups:
        raise ValueError("process-group evidence does not exactly cover active groups")
    operations = evidence.collective_contract.get("operations_by_group")
    if not isinstance(operations, Mapping) or set(operations) != expected_groups:
        raise ValueError("collective contract does not exactly cover active groups")
    for group in evidence.process_groups:
        name = str(group["name"])
        if list(group.get("operations", [])) != list(operations[name]):
            raise ValueError(f"{name} observed operations disagree with declared ledger")
        memberships = group.get("membership_ranks_by_rank")
        hashes = group.get("membership_sha256_by_rank")
        if (
            not isinstance(memberships, list)
            or not isinstance(hashes, list)
            or len(memberships) != world_size
            or len(hashes) != world_size
        ):
            raise ValueError(f"{name} process-group evidence must cover every rank")
        for rank, (members, digest) in enumerate(zip(memberships, hashes, strict=True)):
            group_size = world_size if name == "world" else int(topology[name])
            if (
                members != sorted(members)
                or len(members) != group_size
                or len(set(members)) != group_size
                or rank not in members
                or any(member < 0 or member >= world_size for member in members)
                or digest != membership_digest(members)
            ):
                raise ValueError(f"{name} process-group membership is forged or asymmetric")
            for member in members:
                if memberships[member] != members:
                    raise ValueError(f"{name} process-group membership is asymmetric")
    if evidence.effective_tier == 0:
        if evidence.collective_contract.get("name") != "tier0_mask_population_checksum_v1":
            raise ValueError("Tier-0 collective contract name is wrong")
        world_ops = next(group for group in evidence.process_groups if group["name"] == "world")[
            "operations"
        ]
        reductions = [operation for operation in world_ops if operation["name"] == "all_reduce"]
        gathers = [operation for operation in world_ops if operation["name"] == "all_gather"]
        if (
            len(world_ops) != 4
            or len(reductions) != 3
            or any(operation["count"] != 1 for operation in reductions)
            or sum(operation["bytes"] for operation in reductions)
            != 96 * (15 * int(topology["num_layers"]) + 10)
            or len(gathers) != 1
            or gathers[0]["count"] != 1
            or gathers[0]["bytes"] != 88 * world_size
        ):
            raise ValueError("Tier-0 world collective ledger is not exact")
    elif (
        evidence.collective_contract.get("name") != "fixed_global_topk_hash_v1"
        or not operations["world"]
    ):
        raise ValueError("Tier-1/2 collective contract is not fixed/nonempty")
    if evidence.collective_contract.get("per_layer_collectives") is not False:
        raise ValueError("per-layer diagnostic collectives are forbidden")


class ArtifactV3Writer:
    """Materialize and publish one bounded v3 artifact per successful event."""

    def __init__(
        self,
        root: Path,
        *,
        identity: RuntimeIdentity,
        repro: Mapping[str, str],
        wandb_writer: object | None,
        cumulative_bytes: int = 0,
        max_run_bytes: int = MAX_RUN_BYTES,
    ) -> None:
        self.root = root
        self.identity = identity
        self.repro = dict(repro)
        self.wandb_writer = wandb_writer
        self.cumulative_bytes = int(cumulative_bytes)
        self.max_run_bytes = min(int(max_run_bytes), MAX_RUN_BYTES)
        if set(self.repro) != {
            "scaling_commit",
            "megatron_commit",
            "resolved_config_sha256",
            "scaling_bundle_sha256",
            "megatron_bundle_sha256",
        }:
            raise ValueError("event repro fields are not exact")

    @classmethod
    def from_runtime(
        cls,
        args: Any,
        wandb_writer: object,
        *,
        world_size: int,
        global_rank: int,
        cumulative_bytes: int,
    ) -> "ArtifactV3Writer":
        """Construct the sole last-world writer from allocation-owned identity."""

        identity = identity_from_runtime(
            wandb_writer, world_size=world_size, global_rank=global_rank
        )
        save = getattr(args, "save", None)
        if not isinstance(save, str) or not save:
            raise RuntimeError("diag/v2 artifact publication requires args.save")
        return cls(
            Path(save) / "diagnostics" / "diag-v2-v3",
            identity=identity,
            repro=repro_from_runtime(),
            wandb_writer=wandb_writer,
            cumulative_bytes=cumulative_bytes,
            max_run_bytes=int(getattr(args, "diag_max_run_artifact_bytes", MAX_RUN_BYTES)),
        )

    def write(
        self,
        *,
        event_id: int,
        successful_update: int,
        consumed_tokens: int,
        scalar_payload: Mapping[str, object],
        evidence: EventEvidence,
    ) -> tuple[Path, int]:
        """Validate, atomically materialize, and log one exact event artifact."""

        for name, value in (("event_id", event_id), ("successful_update", successful_update)):
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(consumed_tokens, bool) or consumed_tokens < 0:
            raise ValueError("consumed_tokens must be a nonnegative integer")
        _validate_tier_payload(scalar_payload, evidence.effective_tier)
        status_value = scalar_payload.get("diag/v2/status/valid")
        if evidence.effective_tier == 0 and status_value != 0:
            raise ValueError("Tier-0 artifact-v3 scalar status must be invalid")
        _validate_evidence(evidence, self.identity)
        target = self.root / f"event-{event_id:08d}"
        if target.exists():
            raise RuntimeError(f"diagnostic event artifact already exists: {target}")
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=self.root))
        try:
            layer_arrays = {
                name: np.ascontiguousarray(value) for name, value in evidence.layer_arrays.items()
            }
            rank_arrays = {
                name: np.ascontiguousarray(value) for name, value in evidence.rank_arrays.items()
            }
            _write_deterministic_npz(temporary / "layer_metrics.npz", layer_arrays)
            _write_deterministic_npz(temporary / "rank_perf.npz", rank_arrays)
            manifest: dict[str, Any] = {
                "schema": ARTIFACT_SCHEMA,
                "successful_update": successful_update,
                "consumed_tokens": consumed_tokens,
                "event_id": event_id,
                "requested_tier": evidence.requested_tier,
                "effective_tier": evidence.effective_tier,
                "require_tier": evidence.require_tier,
                "status": evidence.status,
                "status_reason": evidence.status_reason,
                "run_identity": self.identity.as_manifest(),
                "capability_signature": dict(evidence.capability_signature),
                "capability_status": dict(evidence.capability_status),
                "capability_allow": dict(CAPABILITY_ALLOW),
                "topology": dict(evidence.topology),
                "sampling": dict(evidence.sampling),
                "digests": dict(evidence.digests),
                "enums": {
                    "metric": dict(evidence.metric_enums),
                    "family": dict(evidence.family_enums),
                },
                "files": {
                    name: {
                        "sha256": sha256_file(temporary / name),
                        "arrays": {
                            key: _array_descriptor(value) for key, value in sorted(arrays.items())
                        },
                    }
                    for name, arrays in (
                        ("layer_metrics.npz", layer_arrays),
                        ("rank_perf.npz", rank_arrays),
                    )
                },
                "state_snapshots": {
                    name: dict(value) for name, value in evidence.state_snapshots.items()
                },
                "restore_evidence": {
                    name: dict(value) for name, value in evidence.restore_evidence.items()
                },
                "secant_restore_evidence": {
                    name: dict(value) for name, value in evidence.secant_restore_evidence.items()
                },
                "process_groups": [dict(group) for group in evidence.process_groups],
                "collective_contract": {
                    "name": evidence.collective_contract["name"],
                    "per_layer_collectives": False,
                    "operations_by_group": {
                        name: [dict(operation) for operation in operations]
                        for name, operations in evidence.collective_contract[
                            "operations_by_group"
                        ].items()
                    },
                },
                "artifact_bytes": {"compressed": 0, "uncompressed": 0},
                "repro": self.repro,
            }
            manifest_path = temporary / "manifest.json"
            for _ in range(10):
                encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                manifest_path.write_bytes(encoded)
                compressed = sum(path.stat().st_size for path in temporary.iterdir())
                uncompressed = len(encoded) + sum(
                    _npz_uncompressed_bytes(temporary / name)
                    for name in ("layer_metrics.npz", "rank_perf.npz")
                )
                sizes = {"compressed": compressed, "uncompressed": uncompressed}
                if manifest["artifact_bytes"] == sizes:
                    break
                manifest["artifact_bytes"] = sizes
            else:
                raise RuntimeError("artifact byte declarations did not stabilize")
            if compressed > MAX_EVENT_BYTES or uncompressed > MAX_EVENT_UNCOMPRESSED_BYTES:
                raise RuntimeError("diagnostic event artifact exceeds its hard byte cap")
            if self.cumulative_bytes + compressed > self.max_run_bytes:
                raise RuntimeError("diagnostic cumulative artifact budget exceeded")
            os.replace(temporary, target)
            try:
                if self.wandb_writer is not None:
                    artifact = self.wandb_writer.Artifact(
                        name=f"diag-v2-{self.identity.run_id}", type=ARTIFACT_TYPE
                    )
                    artifact.add_dir(str(target))
                    self.wandb_writer.run.log_artifact(artifact)
            except Exception:
                _remove_artifact_directory(target)
                raise
            self.cumulative_bytes += compressed
            return target, compressed
        except Exception:
            if temporary.exists():
                _remove_artifact_directory(temporary)
            raise


def _remove_artifact_directory(directory: Path) -> None:
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        else:
            path.rmdir()
    directory.rmdir()


def assert_schema_bundle_identity() -> None:
    """Fail if the packaged metric/event schema pair is not the v3 contract."""

    expected = "9ca44a0a719e3a096cded95ab64d1cc6e5d3787ff4e8d47e37b069c1f8f684b6"
    if diagnostic_schema_hash() != expected:
        raise RuntimeError("packaged diag/v2 artifact-v3 schema identity is wrong")
