# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded, atomic Tier-0 diagnostic event artifact materialization."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .capability import verified_source_commit

MAX_EVENT_BYTES = 4 * 1024**2
MAX_EVENT_UNCOMPRESSED_BYTES = 16 * 1024**2
MAX_RUN_BYTES = 8 * 1024**3
_LAYER_PAIRS = tuple(
    sorted(
        {
            ("residual", "activation_rms"),
            ("residual", "activation_max_abs"),
            ("residual", "dgrad_rms"),
            ("norm", "update_relative_rms"),
            *(
                (family, metric)
                for family in ("qkv", "attn_out", "fc1", "fc2")
                for metric in (
                    "activation_max_abs",
                    "dgrad_rms",
                    "update_relative_rms",
                    "retention",
                )
            ),
        }
    )
)


def writer_from_runtime(
    args: Any, wandb_writer: object, *, cumulative_bytes: int
) -> "Tier0ArtifactWriter":
    """Construct the rank-0 writer from verified local and launcher provenance."""

    run = getattr(wandb_writer, "run", None)
    run_id = getattr(run, "id", None)
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("Tier-0 requires the rank-0 W&B run identity")
    required = {
        "scaling_commit": "DIAG_V2_SCALING_COMMIT",
        "resolved_config_sha256": "DIAG_V2_RESOLVED_CONFIG_SHA256",
        "scaling_bundle_sha256": "DIAG_V2_SCALING_BUNDLE_SHA256",
        "megatron_bundle_sha256": "DIAG_V2_MEGATRON_BUNDLE_SHA256",
    }
    repro = {
        name: os.environ.get(environment) for name, environment in required.items()
    }
    if any(not isinstance(value, str) or not value for value in repro.values()):
        missing = [
            environment for name, environment in required.items() if not repro[name]
        ]
        raise RuntimeError(
            "Tier-0 launcher provenance is missing: " + ", ".join(missing)
        )
    repro["megatron_commit"] = verified_source_commit()
    if len(repro["scaling_commit"]) != 40 or len(repro["megatron_commit"]) != 40:
        raise RuntimeError("Tier-0 provenance commits must be full git identities")
    if any(
        len(repro[name]) != 64
        for name in (
            "resolved_config_sha256",
            "scaling_bundle_sha256",
            "megatron_bundle_sha256",
        )
    ):
        raise RuntimeError("Tier-0 provenance digests must be SHA-256 identities")
    root = Path(getattr(args, "save")) / "diagnostics" / "diag-v2"
    job_name = os.environ.get("SLURM_JOB_NAME") or str(getattr(args, "wandb_exp_name"))
    return Tier0ArtifactWriter(
        root,
        run_id=run_id,
        job_name=job_name,
        repro=repro,
        wandb_writer=wandb_writer,
        cumulative_bytes=cumulative_bytes,
    )


def sha256_file(path: Path) -> str:
    """Hash one artifact file in bounded host-memory chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": _sha256(np.ascontiguousarray(value).tobytes(order="C")),
    }


def _unavailable_snapshot() -> dict[str, Any]:
    return {
        "applicable": False,
        "model_sha256_by_rank": [],
        "optimizer_sha256_by_rank": [],
        "model_sha256": None,
        "optimizer_sha256": None,
        "model_consensus_sha256_by_rank": [],
        "optimizer_consensus_sha256_by_rank": [],
    }


def _unavailable_restore() -> dict[str, Any]:
    return {
        "applicable": False,
        "before_sha256_by_rank": [],
        "after_sha256_by_rank": [],
        "before_sha256": None,
        "after_sha256": None,
        "before_consensus_sha256_by_rank": [],
        "after_consensus_sha256_by_rank": [],
    }


def _membership_hashes(memberships: Sequence[Sequence[int]]) -> list[str]:
    return [
        _sha256(json.dumps(list(ranks), separators=(",", ":")).encode("ascii"))
        for ranks in memberships
    ]


def _layer_arrays(
    *,
    num_layers: int,
    pp: int,
    valid: bool,
    valid_positions: int,
    evidence: Mapping[tuple[int, str, str], Mapping[str, float | int]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int], list[int]]:
    families = {
        name: index for index, name in enumerate(sorted({x[0] for x in _LAYER_PAIRS}))
    }
    metrics = {
        name: index for index, name in enumerate(sorted({x[1] for x in _LAYER_PAIRS}))
    }
    identities = [
        (layer, family, metric)
        for layer in range(num_layers)
        for family, metric in _LAYER_PAIRS
    ]
    owners = [min(pp - 1, layer * pp // num_layers) for layer in range(num_layers)]
    rows = len(identities)
    count = max(1, valid_positions)
    arrays = {
        "global_layer_id": np.asarray([x[0] for x in identities], dtype=np.int32),
        "pp_owner": np.asarray([owners[x[0]] for x in identities], dtype=np.int16),
        "family_code": np.asarray([families[x[1]] for x in identities], dtype=np.uint8),
        "metric_code": np.asarray([metrics[x[2]] for x in identities], dtype=np.uint16),
        "value": np.zeros(rows, dtype=np.float32),
        "valid": np.full(rows, int(valid), dtype=np.uint8),
        "count": np.full(rows, count if valid else 0, dtype=np.int64),
        "sum": np.zeros(rows, dtype=np.float64),
        "sum_sq": np.zeros(rows, dtype=np.float64),
        "denominator_sum_sq": np.full(rows, count if valid else 0, dtype=np.float64),
        "zero_count": np.zeros(rows, dtype=np.int64),
        "nonfinite_count": np.zeros(rows, dtype=np.int64),
    }
    if evidence is not None:
        if set(evidence) != set(identities):
            raise RuntimeError("layer evidence identities are not exact")
        for index, identity in enumerate(identities):
            row = evidence[identity]
            row_valid = bool(row["valid"]) and valid
            arrays["value"][index] = float(row["value"])
            arrays["valid"][index] = int(row_valid)
            arrays["count"][index] = int(row["count"])
            arrays["sum"][index] = float(row["sum"])
            arrays["sum_sq"][index] = float(row["sum_sq"])
            arrays["denominator_sum_sq"][index] = float(row["denominator_sum_sq"])
            arrays["zero_count"][index] = int(row["zero_count"])
            arrays["nonfinite_count"][index] = int(row["nonfinite_count"])
    return arrays, metrics, families, owners


def _rank_arrays(rank_evidence: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
    rows = np.asarray(rank_evidence, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 11:
        raise RuntimeError("rank evidence must have fixed shape [world, 11]")
    return {
        "rank": rows[:, 0].astype(np.int32),
        "pre_gather_wall_time_ms": rows[:, 1].astype(np.float64),
        "ordinary_step_wall_time_ms": rows[:, 2].astype(np.float64),
        "detected_hbm_capacity_bytes": rows[:, 3].astype(np.int64),
        "pre_event_allocated_bytes": rows[:, 4].astype(np.int64),
        "pre_event_reserved_bytes": rows[:, 5].astype(np.int64),
        "predicted_increment_bytes": rows[:, 6].astype(np.int64),
        "pre_gather_peak_allocated_bytes": rows[:, 7].astype(np.int64),
        "pre_gather_peak_reserved_bytes": rows[:, 8].astype(np.int64),
        "mask_population": rows[:, 10].astype(np.float64),
    }


def _npz_uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(info.file_size for info in archive.infolist())


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write sorted NPY members with fixed ZIP metadata."""

    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, value in sorted(arrays.items()):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED
            )


class Tier0ArtifactWriter:
    """Create and upload one exact three-file artifact per successful event."""

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        job_name: str,
        repro: Mapping[str, str],
        wandb_writer: object | None,
        cumulative_bytes: int = 0,
        max_run_bytes: int = MAX_RUN_BYTES,
    ) -> None:
        self.root = root
        self.run_id = run_id
        self.job_name = job_name
        self.repro = dict(repro)
        self.wandb_writer = wandb_writer
        self.cumulative_bytes = cumulative_bytes
        self.max_run_bytes = min(max_run_bytes, MAX_RUN_BYTES)

    def write(
        self,
        *,
        event_id: int,
        successful_update: int,
        consumed_tokens: int,
        valid_positions: int,
        valid: bool,
        topology: Mapping[str, int],
        rank_evidence: Sequence[Sequence[float]],
        mask_shape: Sequence[int],
        memory_evidence: Mapping[str, Any],
        capability_hash: str,
        schema_hash: str,
        writer_rank: int = 0,
        process_group_memberships: Mapping[str, Sequence[Sequence[int]]] | None = None,
        reduction_bytes: int = 1,
        runtime_signature: Mapping[str, Any] | None = None,
        layer_evidence: Mapping[tuple[int, str, str], Mapping[str, float | int]]
        | None = None,
    ) -> tuple[Path, int]:
        """Atomically materialize, budget-check, and log one event artifact."""

        world_size = math.prod(topology[name] for name in ("dp", "tp", "pp", "cp"))
        if world_size != len(rank_evidence):
            raise RuntimeError("rank evidence does not cover the declared topology")
        if len(mask_shape) != 3 or any(int(dimension) <= 0 for dimension in mask_shape):
            raise RuntimeError(
                "mask shape must be [microbatches, batch, local_sequence]"
            )
        target = self.root / f"event-{event_id:08d}"
        if target.exists():
            raise RuntimeError(f"diagnostic event artifact already exists: {target}")
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=self.root))
        try:
            layers, metrics, families, owners = _layer_arrays(
                num_layers=topology["num_layers"],
                pp=topology["pp"],
                valid=valid,
                valid_positions=valid_positions,
                evidence=layer_evidence,
            )
            ranks = _rank_arrays(rank_evidence)
            _write_deterministic_npz(temporary / "layer_metrics.npz", layers)
            _write_deterministic_npz(temporary / "rank_perf.npz", ranks)
            descriptor = _sha256(f"{capability_hash}:{schema_hash}".encode())
            topology_fields = {
                name: int(topology[name])
                for name in ("dp", "tp", "pp", "cp", "ep", "vpp")
            }
            runtime_signature = (
                dict(runtime_signature)
                if runtime_signature is not None
                else {
                    "optimizer": "distributed_optimizer",
                    "precision": "bf16",
                    **topology_fields,
                    "moe": False,
                    "fsdp": False,
                    "chained_optimizer": True,
                    "layerwise_optimizer": False,
                    "overlap_param_gather": False,
                    "parameter_cache_mode": "none",
                }
            )
            manifest: dict[str, Any] = {
                "schema": "diag/v2/artifact",
                "successful_update": successful_update,
                "consumed_tokens": consumed_tokens,
                "event_id": event_id,
                "requested_tier": 0,
                "effective_tier": 0,
                "require_tier": 0,
                "status": "invalid",
                "status_reason": (
                    "Tier-0 state snapshots and all-rank post-gather peak evidence are unavailable; "
                    "mask checksum evidence requires a versioned Scaling schema"
                    if valid
                    else "Tier-0 evidence is non-promotable: sufficient statistics may be invalid, "
                    "state snapshots and all-rank post-gather peaks are unavailable, and mask "
                    "checksum evidence requires a versioned Scaling schema"
                ),
                "run_identity": {
                    "run_id": self.run_id,
                    "job_name": self.job_name,
                    "writer_rank": writer_rank,
                },
                "capability_signature": runtime_signature,
                "capability_status": {
                    "attention": "unsupported",
                    "moe": "not_applicable",
                },
                "capability_allow": {
                    "allow_vpp": False,
                    "allow_ep": False,
                    "allow_fsdp": False,
                    "allow_fp8_parameters": False,
                    "allow_fp4_parameters": False,
                },
                "topology": {
                    **topology_fields,
                    "num_layers": topology["num_layers"],
                    "layer_pp_owners": owners,
                },
                "sampling": {
                    "evidence_type": "tier0_mask_population_checksum_v1",
                    "per_rank": [
                        {
                            "rank": int(row[0]),
                            "population": float(row[10]),
                            "checksum_value": float(row[9]),
                        }
                        for row in rank_evidence
                    ],
                    "global_population": valid_positions,
                    "mask_shape": [int(dimension) for dimension in mask_shape],
                    "checksum_algorithm": (
                        "sum_over_microbatches(sum_i(float32(mask_i) * float32(i + 1))) "
                        "accumulated in float64"
                    ),
                    "collision_limitation": (
                        "population and weighted checksum are collision-prone and do not identify "
                        "sample IDs, token IDs, or mask membership"
                    ),
                },
                "digests": {
                    "descriptor_sha256_by_rank": [descriptor] * world_size,
                },
                "memory_evidence": dict(memory_evidence),
                "enums": {"metric": metrics, "family": families},
                "files": {
                    name: {
                        "sha256": sha256_file(temporary / name),
                        "arrays": {
                            key: _array_descriptor(value)
                            for key, value in sorted(arrays.items())
                        },
                    }
                    for name, arrays in (
                        ("layer_metrics.npz", layers),
                        ("rank_perf.npz", ranks),
                    )
                },
                "state_snapshots": {
                    "pre": _unavailable_snapshot(),
                    "post": _unavailable_snapshot(),
                    "midpoint": _unavailable_snapshot(),
                },
                "restore_evidence": {
                    name: _unavailable_restore()
                    for name in (
                        "model",
                        "optimizer",
                        "rng",
                        "mutable_state",
                        "data_iterator",
                        "fp8",
                        "router",
                        "cache",
                    )
                },
                "process_groups": [
                    {
                        "name": "world",
                        "membership_sha256_by_rank": _membership_hashes(
                            (
                                process_group_memberships
                                or {"world": [tuple(range(world_size))] * world_size}
                            )["world"]
                        ),
                        "operations": [
                            {
                                "name": "all_reduce",
                                "count": 3,
                                "bytes": reduction_bytes,
                            },
                            {
                                "name": "all_gather",
                                "count": 1,
                                "bytes": world_size * 11 * 8,
                            },
                        ],
                    },
                    *(
                        {
                            "name": name,
                            "membership_sha256_by_rank": _membership_hashes(
                                memberships
                            ),
                            "operations": [],
                        }
                        for name, memberships in sorted(
                            (process_group_memberships or {}).items()
                        )
                        if name != "world"
                    ),
                ],
                "artifact_bytes": {"compressed": 0, "uncompressed": 0},
                "repro": self.repro,
            }
            manifest_path = temporary / "manifest.json"
            for _ in range(10):
                encoded = json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
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
            if (
                compressed > MAX_EVENT_BYTES
                or uncompressed > MAX_EVENT_UNCOMPRESSED_BYTES
            ):
                raise RuntimeError("Tier-0 event artifact exceeds its hard byte cap")
            if self.cumulative_bytes + compressed > self.max_run_bytes:
                raise RuntimeError("Tier-0 cumulative artifact budget exceeded")
            os.replace(temporary, target)
            try:
                if self.wandb_writer is not None:
                    artifact = self.wandb_writer.Artifact(
                        name=f"diag-v2-{self.run_id}", type="diagnostic-event-v2"
                    )
                    artifact.add_dir(str(target))
                    self.wandb_writer.run.log_artifact(artifact)
            except Exception:
                for path in target.iterdir():
                    path.unlink()
                target.rmdir()
                raise
            self.cumulative_bytes += compressed
            return target, compressed
        except Exception:
            if temporary.exists():
                for path in temporary.iterdir():
                    path.unlink()
                temporary.rmdir()
            raise
