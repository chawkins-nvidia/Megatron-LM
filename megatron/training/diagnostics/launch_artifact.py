# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Final trainer-inside-allocation ``launch-artifact-v3`` publication."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .artifact_v3 import (
    _GIT_SHA,
    _SHA256,
    RuntimeIdentity,
    identity_from_runtime,
    repro_from_runtime,
    sha256_file,
)

LAUNCH_SCHEMA = "diag/v2/launch-artifact-v3"
LAUNCH_TYPE = "launch-config"
_RUNG = re.compile(r"^[1-9][0-9]*$")
_PATH_ENVIRONMENTS = {
    "resolved": "DIAG_V2_RESOLVED_CONFIG_PATH",
    "megatron": "DIAG_V2_MEGATRON_CONFIG_PATH",
    "slurm": "DIAG_V2_RENDERED_SBATCH_PATH",
    "scaling_bundle": "DIAG_V2_SCALING_BUNDLE_PATH",
    "megatron_bundle": "DIAG_V2_MEGATRON_BUNDLE_PATH",
}


def _run_git(args: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or f"git {' '.join(args)} failed")
    return completed.stdout


def _verify_complete_bundle(path: Path, expected_commit: str) -> None:
    """Prove an isolated bundle reconstructs the advertised commit and tree."""

    if not _GIT_SHA.fullmatch(expected_commit):
        raise RuntimeError("bundle commit must be a full lowercase git SHA")
    with tempfile.TemporaryDirectory(prefix="diag-v2-bundle-") as directory:
        repository = Path(directory) / "repo.git"
        _run_git(["init", "--bare", str(repository)], cwd=Path(directory))
        _run_git(["bundle", "unbundle", str(path)], cwd=repository)
        _run_git(["cat-file", "-e", f"{expected_commit}^{{commit}}"], cwd=repository)
        _run_git(["cat-file", "-e", f"{expected_commit}^{{tree}}"], cwd=repository)
        _run_git(["fsck", "--strict", "--no-dangling"], cwd=repository)


def _regular_source(environment: str) -> Path:
    raw = os.environ.get(environment)
    if not isinstance(raw, str) or not raw or raw != os.path.abspath(raw):
        raise RuntimeError(f"{environment} must be a canonical absolute path")
    path = Path(raw)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError(f"{environment} does not exist: {path}") from error
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"{environment} must be a regular non-symlink file")
    return path


def _validate_resolved_identity(path: Path, *, identity: RuntimeIdentity, rung: int) -> None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"resolved config is invalid: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("resolved config must be a mapping")
    if value.get("r") != rung:
        raise RuntimeError("resolved config rung disagrees with DIAG_V2_RUNG")
    slurm = value.get("slurm")
    environment = value.get("environment")
    if not isinstance(slurm, dict) or slurm.get("job_name") != identity.job_name:
        raise RuntimeError("resolved config job name disagrees with SLURM_JOB_NAME")
    if not isinstance(environment, dict) or environment.get("WANDB_RUN_ID") != identity.run_id:
        raise RuntimeError("resolved config W&B run ID disagrees with initialized run")


def _copy_file(source: Path, destination: Path) -> dict[str, int | str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)
    return {"sha256": sha256_file(destination), "bytes": destination.stat().st_size}


def publish_final_launch_config(
    wandb_writer: object, *, output_root: Path, world_size: int, global_rank: int
) -> Path | None:
    """Publish exactly one canonical final launch artifact on the last rank.

    Non-writer ranks return without touching W&B or launcher paths.  The writer
    obtains scheduler identity from the live allocation, verifies all staged
    inputs and both isolated Git bundles, then uploads ``config-<run_id>`` once.
    """

    if global_rank != world_size - 1:
        return None
    identity = identity_from_runtime(wandb_writer, world_size=world_size, global_rank=global_rank)
    rung_text = os.environ.get("DIAG_V2_RUNG")
    if not isinstance(rung_text, str) or not _RUNG.fullmatch(rung_text):
        raise RuntimeError("DIAG_V2_RUNG must be a canonical positive decimal")
    rung = int(rung_text)
    sources = {
        name: _regular_source(environment) for name, environment in _PATH_ENVIRONMENTS.items()
    }
    repro = repro_from_runtime()
    expected_basenames = {
        "scaling_bundle": f"scaling-{repro['scaling_commit'][:12]}.snapshot.bundle",
        "megatron_bundle": f"megatron-lm-{repro['megatron_commit'][:12]}.snapshot.bundle",
    }
    for name, basename in expected_basenames.items():
        if sources[name].name != basename:
            raise RuntimeError(f"{name} basename does not bind its full commit")
    expected_hashes = {
        "resolved": repro["resolved_config_sha256"],
        "scaling_bundle": repro["scaling_bundle_sha256"],
        "megatron_bundle": repro["megatron_bundle_sha256"],
    }
    for name, expected in expected_hashes.items():
        if not _SHA256.fullmatch(expected) or sha256_file(sources[name]) != expected:
            raise RuntimeError(f"{name} bytes disagree with expected launcher digest")
    _validate_resolved_identity(sources["resolved"], identity=identity, rung=rung)
    _verify_complete_bundle(sources["scaling_bundle"], repro["scaling_commit"])
    _verify_complete_bundle(sources["megatron_bundle"], repro["megatron_commit"])

    target = output_root / f"config-{identity.run_id}"
    if target.exists():
        raise RuntimeError(f"final launch artifact already exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=output_root))
    canonical_sources = {
        f"resolved/r{rung}.yaml": sources["resolved"],
        f"megatron/r{rung}.yaml": sources["megatron"],
        f"slurm/r{rung}.sbatch.txt": sources["slurm"],
        f"code_bundle/{sources['scaling_bundle'].name}": sources["scaling_bundle"],
        f"code_bundle/{sources['megatron_bundle'].name}": sources["megatron_bundle"],
    }
    try:
        files = {
            name: _copy_file(source, temporary / name) for name, source in canonical_sources.items()
        }
        launch = {
            "schema": LAUNCH_SCHEMA,
            "run_id": identity.run_id,
            "rung": rung,
            "job_name": identity.job_name,
            "training_job_id": identity.training_job_id,
            "backend": "megatron",
            "world_size": identity.world_size,
            "writer_rank": identity.writer_rank,
            "publication": "trainer_inside_allocation",
            "files": files,
            "repro": repro,
        }
        (temporary / "scaling_manifest.json").write_text(
            json.dumps({"diag_v2_launch": launch}, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        try:
            artifact = wandb_writer.Artifact(name=f"config-{identity.run_id}", type=LAUNCH_TYPE)
            artifact.add_dir(str(target))
            wandb_writer.run.log_artifact(artifact)
        except Exception:
            shutil.rmtree(target)
            raise
        return target
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def publish_final_launch_config_from_runtime(
    args: Any, wandb_writer: object, *, world_size: int, global_rank: int
) -> Path | None:
    """Publish the final launch artifact below the trainer's save root."""

    save = getattr(args, "save", None)
    if not isinstance(save, str) or not save:
        raise RuntimeError("final launch publication requires args.save")
    return publish_final_launch_config(
        wandb_writer,
        output_root=Path(save) / "diagnostics" / "launch-config-v3",
        world_size=world_size,
        global_rank=global_rank,
    )
