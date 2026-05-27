# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path
from typing import Tuple

from megatron.training.global_vars import get_wandb_writer
from megatron.training.utils import print_rank_last


def _get_wandb_artifact_tracker_filename(save_dir: str) -> Path:
    """Wandb artifact tracker file records the latest artifact wandb entity and project"""
    return Path(save_dir) / "latest_wandb_artifact_path.txt"


def _get_artifact_name_and_version(save_dir: Path, checkpoint_path: Path) -> Tuple[str, str]:
    return save_dir.stem, checkpoint_path.stem


def on_save_checkpoint_success(checkpoint_path: str, tracker_filename: str, save_dir: str, iteration: int) -> None:
    """Function to be called after checkpointing succeeds and checkpoint is persisted for logging it as an artifact in W&B

    Args:
        checkpoint_path (str): path of the saved checkpoint
        tracker_filename (str): path of the tracker filename for the checkpoint iteration
        save_dir (str): path of the root save folder for all checkpoints
        iteration (int): iteration of the checkpoint
    """

    # # CHAWKINS-ONLY-FINAL-ARTIFACT: only register the FINAL checkpoint as a wandb
    # artifact, not every save_interval save. Each artifact call costs
    # tens of seconds of wandb-client overhead; intermediate saves are
    # scratch for resume and don't need a stable artifact version.
    from megatron.training import get_args
    try:
        args = get_args()
    except Exception:
        args = None
    if args is not None:
        target_iter = None
        if getattr(args, 'train_iters', None):
            target_iter = int(args.train_iters)
        elif getattr(args, 'train_samples', None) and getattr(args, 'global_batch_size', None):
            target_iter = (int(args.train_samples) + int(args.global_batch_size) - 1) // int(args.global_batch_size)
        if target_iter is not None and iteration < target_iter:
            return

    wandb_writer = get_wandb_writer()

    # # CHAWKINS-NOOP-WANDB: also reject the case where wandb_writer is truthy
    # but wandb.run has already been cleared (e.g. after wandb.finish()).
    if wandb_writer and getattr(wandb_writer, "run", None) is not None:
        metadata = {"iteration": iteration}
        artifact_name, artifact_version = _get_artifact_name_and_version(Path(save_dir), Path(checkpoint_path))
        artifact = wandb_writer.Artifact(artifact_name, type="model", metadata=metadata)
        # wandb's artifact.add_reference requires absolute paths
        checkpoint_path = str(Path(checkpoint_path).resolve())
        # CHAWKINS-REFERENCE-ONLY: the checkpoint payload is left on
        # lustre and registered as a `file://` reference (no bytes
        # uploaded to W&B). The tracker file is intentionally NOT
        # added: its only useful field (the iteration) is already
        # captured in the artifact's `metadata={"iteration": ...}`
        # and in its version name (checkpoint_path.stem), so
        # uploading the tracker would duplicate non-essential bytes.
        artifact.add_reference(f"file://{checkpoint_path}", checksum=False)
        wandb_writer.run.log_artifact(artifact, aliases=[artifact_version])
        wandb_tracker_filename = _get_wandb_artifact_tracker_filename(save_dir)
        wandb_tracker_filename.write_text(f"{wandb_writer.run.entity}/{wandb_writer.run.project}")


def on_load_checkpoint_success(checkpoint_path: str, load_dir: str) -> None:
    """Function to be called after succesful loading of a checkpoint, for aggregation and logging it to W&B

    Args:
        checkpoint_path (str): path of the loaded checkpoint
        load_dir (str): path of the root save folder for all checkpoints
        iteration (int): iteration of the checkpoint
    """

    wandb_writer = get_wandb_writer()
    
    if wandb_writer:
        try:
            artifact_name, artifact_version = _get_artifact_name_and_version(Path(load_dir), Path(checkpoint_path))
            wandb_tracker_filename = _get_wandb_artifact_tracker_filename(load_dir)
            artifact_path = ""
            if wandb_tracker_filename.is_file():
                artifact_path = wandb_tracker_filename.read_text().strip()
                artifact_path = f"{artifact_path}/"
            wandb_writer.run.use_artifact(f"{artifact_path}{artifact_name}:{artifact_version}")
        except Exception:
            print_rank_last(f"  failed to find checkpoint {checkpoint_path} in wandb")
