# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from megatron.training.diagnostics.artifact_v3 import (
    COMPONENT_STATE_REASON,
    SECANT_STATE_REASON,
    ArtifactV3Writer,
    EventEvidence,
    RuntimeIdentity,
    build_layer_metric_arrays,
    build_process_group_evidence,
    complete_restore_evidence,
    complete_state_snapshot,
    identity_sampling_evidence,
    rank_perf_arrays,
    tier0_collective_operations,
    tier0_rank_perf_from_heartbeat,
    tier0_sampling_from_rank_evidence,
    tier0_state_evidence,
    unavailable_restore_evidence,
)
from megatron.training.diagnostics.capability import (
    CONSUMED_DIAGNOSTIC_CONFIG_FIELDS,
    capability_payload,
    diagnostic_schema_hash,
    load_static_capability,
    verified_source_commit,
)
from megatron.training.diagnostics.function_response import TIER1_KEYS
from megatron.training.diagnostics.schema import TIER0_KEYS
from megatron.training.diagnostics.secant import TIER2_OUTPUT_KEYS


class _Artifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.directory: str | None = None

    def add_dir(self, directory: str) -> None:
        self.directory = directory


class _Run:
    def __init__(self, run_id: str = "run-209") -> None:
        self.id = run_id
        self.calls: list[_Artifact] = []

    def log_artifact(self, artifact: _Artifact) -> None:
        self.calls.append(artifact)


class _Wandb:
    Artifact = _Artifact

    def __init__(self, run_id: str = "run-209") -> None:
        self.run = _Run(run_id)


def _signature() -> dict[str, object]:
    return {
        "backend": "megatron",
        "optimizer_chain": {
            "name": "ChainedOptimizer",
            "children": [
                {"name": "DistributedOptimizer", "children": [{"name": "AdamW", "children": []}]}
            ],
        },
        "precision": "bf16",
        "dp": 1,
        "tp": 1,
        "pp": 1,
        "cp": 1,
        "ep": 1,
        "vpp": 1,
        "moe": False,
        "fsdp": False,
        "overlap_param_gather": False,
        "parameter_cache_mode": "none",
    }


def _layer_evidence(tier: int, *, attention: bool) -> tuple[dict, dict, dict]:
    from megatron.training.diagnostics.artifact_v3 import expected_layer_pairs

    rows = {
        (0, family, metric): {
            "value": 0.5,
            "valid": 1,
            "count": 4,
            "sum": 2.0,
            "sum_sq": 1.0,
            "denominator_sum_sq": 4.0,
            "zero_count": 0,
            "nonfinite_count": 0,
        }
        for family, metric in expected_layer_pairs(
            tier, attention_available=attention, moe_available=False
        )
    }
    return build_layer_metric_arrays(
        rows, num_layers=1, layer_pp_owners=[0], effective_tier=tier, attention_available=attention
    )


def _repro() -> dict[str, str]:
    return {
        "scaling_commit": "a" * 40,
        "megatron_commit": "b" * 40,
        "resolved_config_sha256": "c" * 64,
        "scaling_bundle_sha256": "d" * 64,
        "megatron_bundle_sha256": "e" * 64,
    }


def test_capability_is_v3_tier2_and_consumes_the_complete_flattened_contract() -> None:
    static = load_static_capability()
    payload = capability_payload()
    assert static["artifact_schema_version"] == 3
    assert static["artifact_schema"] == "diag/v2/artifact-v3"
    assert static["launch_artifact_schema"] == "diag/v2/launch-artifact-v3"
    assert static["supported_max_tier"] == 2
    assert static["schema_hash"] == diagnostic_schema_hash()
    assert diagnostic_schema_hash() == (
        "9ca44a0a719e3a096cded95ab64d1cc6e5d3787ff4e8d47e37b069c1f8f684b6"
    )
    assert static["runtime_fields"] == list(CONSUMED_DIAGNOSTIC_CONFIG_FIELDS)
    assert len(static["runtime_fields"]) == 50
    assert "diagnostic_layer_pattern" in static["runtime_fields"]
    assert "diagnostic_include_special_layers" in static["runtime_fields"]
    assert sum(name.startswith("diag_") for name in static["runtime_fields"]) == 41
    assert payload["source_commit"] == verified_source_commit()


def test_tier0_event_is_truthful_invalid_and_logs_one_v3_artifact(tmp_path: Path) -> None:
    wandb = _Wandb()
    identity = RuntimeIdentity("run-209", "job-r4", "422999", 0, 1)
    rank_source = [[0, 10, 25, 1000, 90, 100, 20, 100, 120, 7, 4]]
    sampling, digests = tier0_sampling_from_rank_evidence(
        rank_source, seed=209, mask_shape=[1, 1, 4], descriptor_sha256="1" * 64
    )
    layer, metrics, families = _layer_evidence(0, attention=False)
    snapshots, restore, secant = tier0_state_evidence()
    operations = tier0_collective_operations(1, 1)
    groups = build_process_group_evidence(
        topology={"dp": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1},
        membership_ranks_by_group={"world": [[0]]},
        operations_by_group=operations,
    )
    evidence = EventEvidence(
        requested_tier=0,
        effective_tier=0,
        require_tier=0,
        status="invalid",
        status_reason="tier0 evidence is ingestion-only",
        capability_signature=_signature(),
        capability_status={"attention": "unsupported", "moe": "not_applicable"},
        topology={
            "dp": 1,
            "tp": 1,
            "pp": 1,
            "cp": 1,
            "ep": 1,
            "vpp": 1,
            "num_layers": 1,
            "layer_pp_owners": [0],
        },
        sampling=sampling,
        digests=digests,
        metric_enums=metrics,
        family_enums=families,
        layer_arrays=layer,
        rank_arrays=tier0_rank_perf_from_heartbeat(rank_source, writer_rank=0),
        state_snapshots=snapshots,
        restore_evidence=restore,
        secant_restore_evidence=secant,
        process_groups=groups,
        collective_contract={
            "name": "tier0_mask_population_checksum_v1",
            "per_layer_collectives": False,
            "operations_by_group": operations,
        },
    )
    payload = dict.fromkeys(TIER0_KEYS, 0.0)
    payload["diag/v2/status/valid"] = 0
    writer = ArtifactV3Writer(tmp_path, identity=identity, repro=_repro(), wandb_writer=wandb)
    directory, compressed = writer.write(
        event_id=1,
        successful_update=1,
        consumed_tokens=1024,
        scalar_payload=payload,
        evidence=evidence,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema"] == "diag/v2/artifact-v3"
    assert manifest["status"] == "invalid"
    assert manifest["run_identity"]["training_job_id"] == "422999"
    assert "selected_sample_ids_sha256" not in json.dumps(manifest)
    assert compressed == writer.cumulative_bytes
    assert len(wandb.run.calls) == 1
    assert wandb.run.calls[0].name == "diag-v2-run-209"
    assert wandb.run.calls[0].type == "diagnostic-event-v2"
    with np.load(directory / "rank_perf.npz", allow_pickle=False) as rank:
        assert set(rank.files) == {
            "rank",
            "event_wall_time_ms",
            "ordinary_step_wall_time_ms",
            "detected_hbm_capacity_bytes",
            "pre_event_allocated_bytes",
            "pre_event_reserved_bytes",
            "predicted_post_gather_peak_allocated_bytes",
            "predicted_post_gather_peak_reserved_bytes",
            "post_gather_peak_allocated_bytes",
            "post_gather_peak_reserved_bytes",
            "sink_post_interval_peak_allocated_bytes",
            "sink_post_interval_peak_reserved_bytes",
        }
        assert np.isnan(rank["post_gather_peak_allocated_bytes"]).all()


def test_tier2_event_requires_complete_state_memory_identity_and_secant_restore(
    tmp_path: Path,
) -> None:
    identity = RuntimeIdentity("run-tier2", "job-r4", "423000", 0, 1)
    layer, metrics, families = _layer_evidence(2, attention=True)
    sampling, digests = identity_sampling_evidence(
        seed=209,
        selected_sample_id_hashes=["1" * 64],
        valid_position_count=4,
        selected_sample_ids_sha256="2" * 64,
        valid_token_ids_sha256="3" * 64,
        descriptor_sha256_by_rank=["4" * 64],
    )
    snapshots = {
        phase: complete_state_snapshot(
            model_sha256_by_rank=[salt * 64],
            optimizer_sha256_by_rank=[format(int(salt, 16) + 8, "x") * 64],
        )
        for phase, salt in (("pre", "1"), ("post", "2"), ("midpoint", "3"))
    }
    normal: dict[str, dict] = {}
    secant: dict[str, dict] = {}
    for index, name in enumerate(
        ("model", "optimizer", "rng", "mutable_state", "data_iterator"), 5
    ):
        digest = format(index, "064x")
        normal[name] = complete_restore_evidence(
            before_sha256_by_rank=[digest], after_sha256_by_rank=[digest]
        )
        secant[name] = complete_restore_evidence(
            before_sha256_by_rank=[digest], after_sha256_by_rank=[digest]
        )
    for name in ("fp8", "router", "cache"):
        normal[name] = unavailable_restore_evidence(COMPONENT_STATE_REASON)
        secant[name] = unavailable_restore_evidence(SECANT_STATE_REASON)
    rows = [
        {
            "rank": 0,
            "event_wall_time_ms": 20.0,
            "ordinary_step_wall_time_ms": 10.0,
            "detected_hbm_capacity_bytes": 1000.0,
            "pre_event_allocated_bytes": 100.0,
            "pre_event_reserved_bytes": 120.0,
            "predicted_post_gather_peak_allocated_bytes": 150.0,
            "predicted_post_gather_peak_reserved_bytes": 180.0,
            "post_gather_peak_allocated_bytes": 150.0,
            "post_gather_peak_reserved_bytes": 180.0,
            "sink_post_interval_peak_allocated_bytes": 150.0,
            "sink_post_interval_peak_reserved_bytes": 180.0,
        }
    ]
    operations = {
        "world": [
            {"name": "all_gather", "phase": "global_topk", "count": 1, "bytes": 88},
            {"name": "all_reduce", "phase": "fixed_statistics", "count": 1, "bytes": 4096},
        ]
    }
    groups = build_process_group_evidence(
        topology={"dp": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1},
        membership_ranks_by_group={"world": [[0]]},
        operations_by_group=operations,
    )
    evidence = EventEvidence(
        requested_tier=2,
        effective_tier=2,
        require_tier=0,
        status="ok",
        status_reason=None,
        capability_signature=_signature(),
        capability_status={"attention": "available", "moe": "not_applicable"},
        topology={
            "dp": 1,
            "tp": 1,
            "pp": 1,
            "cp": 1,
            "ep": 1,
            "vpp": 1,
            "num_layers": 1,
            "layer_pp_owners": [0],
        },
        sampling=sampling,
        digests=digests,
        metric_enums=metrics,
        family_enums=families,
        layer_arrays=layer,
        rank_arrays=rank_perf_arrays(rows),
        state_snapshots=snapshots,
        restore_evidence=normal,
        secant_restore_evidence=secant,
        process_groups=groups,
        collective_contract={
            "name": "fixed_global_topk_hash_v1",
            "per_layer_collectives": False,
            "operations_by_group": operations,
        },
    )
    payload = dict.fromkeys((*TIER0_KEYS, *TIER1_KEYS, *TIER2_OUTPUT_KEYS), 0.5)
    payload["diag/v2/status/valid"] = 1
    writer = ArtifactV3Writer(tmp_path, identity=identity, repro=_repro(), wandb_writer=None)
    directory, _ = writer.write(
        event_id=1,
        successful_update=100,
        consumed_tokens=4096,
        scalar_payload=payload,
        evidence=evidence,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["effective_tier"] == 2
    assert manifest["capability_status"]["attention"] == "available"
    assert all(item["applicable"] for item in manifest["state_snapshots"].values())
    assert all(
        manifest["secant_restore_evidence"][name]["applicable"]
        for name in normal
        if name not in {"fp8", "router", "cache"}
    )


def test_optimizer_signature_rejects_direct_and_multichild_shapes() -> None:
    from megatron.training.diagnostics.artifact_v3 import optimizer_chain_signature

    AdamW = type("AdamW", (), {})
    DistributedOptimizer = type("DistributedOptimizer", (), {})
    ChainedOptimizer = type("ChainedOptimizer", (), {})
    distributed = DistributedOptimizer()
    distributed.optimizer = AdamW()
    chained = ChainedOptimizer()
    chained.chained_optimizers = [distributed]
    assert optimizer_chain_signature(chained)["children"][0]["name"] == "DistributedOptimizer"
    chained.chained_optimizers.append(distributed)
    try:
        optimizer_chain_signature(chained)
    except RuntimeError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("multi-child optimizer chain was accepted")
    try:
        optimizer_chain_signature(distributed)
    except RuntimeError as error:
        assert "ChainedOptimizer" in str(error)
    else:
        raise AssertionError("direct distributed optimizer was accepted")


def test_final_launch_publisher_uses_actual_scheduler_identity_and_logs_once(
    tmp_path: Path, monkeypatch
) -> None:
    from megatron.training.diagnostics import launch_artifact

    source = tmp_path / "staged"
    source.mkdir()
    megatron_commit = verified_source_commit()
    scaling_commit = "a" * 40
    resolved = source / "resolved.yaml"
    megatron = source / "megatron.yaml"
    slurm = source / "rendered.sbatch"
    scaling_bundle = source / f"scaling-{scaling_commit[:12]}.snapshot.bundle"
    megatron_bundle = source / f"megatron-lm-{megatron_commit[:12]}.snapshot.bundle"
    resolved.write_text("r: 4\nslurm:\n  job_name: job-r4\nenvironment:\n  WANDB_RUN_ID: run-209\n")
    megatron.write_text("model: r4\n")
    slurm.write_text("#!/bin/bash\n")
    scaling_bundle.write_bytes(b"scaling bundle")
    megatron_bundle.write_bytes(b"megatron bundle")
    values = {
        "WANDB_RUN_ID": "run-209",
        "SLURM_JOB_NAME": "job-r4",
        "SLURM_JOB_ID": "423001_7",
        "DIAG_V2_RUNG": "4",
        "DIAG_V2_RESOLVED_CONFIG_PATH": str(resolved),
        "DIAG_V2_MEGATRON_CONFIG_PATH": str(megatron),
        "DIAG_V2_RENDERED_SBATCH_PATH": str(slurm),
        "DIAG_V2_SCALING_BUNDLE_PATH": str(scaling_bundle),
        "DIAG_V2_MEGATRON_BUNDLE_PATH": str(megatron_bundle),
        "DIAG_V2_SCALING_COMMIT": scaling_commit,
        "DIAG_V2_MEGATRON_COMMIT": megatron_commit,
        "DIAG_V2_RESOLVED_CONFIG_SHA256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "DIAG_V2_SCALING_BUNDLE_SHA256": hashlib.sha256(scaling_bundle.read_bytes()).hexdigest(),
        "DIAG_V2_MEGATRON_BUNDLE_SHA256": hashlib.sha256(megatron_bundle.read_bytes()).hexdigest(),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    verified: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        launch_artifact,
        "_verify_complete_bundle",
        lambda path, commit: verified.append((path, commit)),
    )
    wandb = _Wandb()
    output = tmp_path / "output"
    assert (
        launch_artifact.publish_final_launch_config(
            wandb, output_root=output, world_size=2, global_rank=0
        )
        is None
    )
    directory = launch_artifact.publish_final_launch_config(
        wandb, output_root=output, world_size=2, global_rank=1
    )
    assert directory is not None
    manifest = json.loads((directory / "scaling_manifest.json").read_text())["diag_v2_launch"]
    assert manifest["schema"] == "diag/v2/launch-artifact-v3"
    assert manifest["training_job_id"] == "423001_7"
    assert manifest["writer_rank"] == 1
    assert manifest["publication"] == "trainer_inside_allocation"
    assert set(manifest["files"]) == {
        "resolved/r4.yaml",
        "megatron/r4.yaml",
        "slurm/r4.sbatch.txt",
        f"code_bundle/{scaling_bundle.name}",
        f"code_bundle/{megatron_bundle.name}",
    }
    assert verified == [(scaling_bundle, scaling_commit), (megatron_bundle, megatron_commit)]
    assert len(wandb.run.calls) == 1
    assert wandb.run.calls[0].name == "config-run-209"
    assert wandb.run.calls[0].type == "launch-config"


def test_digest_helper_matches_raw_bytes(tmp_path: Path) -> None:
    path = tmp_path / "bytes"
    path.write_bytes(b"publisher")
    from megatron.training.diagnostics.artifact_v3 import sha256_file

    assert sha256_file(path) == hashlib.sha256(b"publisher").hexdigest()
