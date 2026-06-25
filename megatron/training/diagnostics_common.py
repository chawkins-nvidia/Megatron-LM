# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Shared sink for the stability diagnostics (#118): δy probe, spectral, attn-entropy.

One small, non-leaky layer shared by the diagnostic loggers (design decision D13). It does NOT
unify the per-diagnostic reduction *math* — each logger computes its own scalars — it only
provides (a) the common ``save_grads`` write + W&B flush and (b) a sharding-aware schema contract
(decision D2): a per-stream ``meta.json`` recording the parallel layout and how each key reduces
across ranks, written *now* while the writer is single-rank so TP/PP/DP>1 aggregation is
well-defined later instead of silently wrong.

Reduction tags (the legend in ``meta.json``) describe how a per-rank scalar combines across ranks:

- ``max``    — global = max over ranks (σ_max, max|x|).
- ``sos+n``  — RMS-type: global = sqrt(sum_r ss_r / sum_r n_r). At a single rank we store the final
               RMS value directly; when TP/DP>1 the emitter must instead store sum-of-squares under
               ``<key>`` and the element count under ``<key>::n`` (documented TODO — not built on the
               1-GPU dev box because it cannot be exercised here; the contract is fixed regardless).
- ``replica``— value is identical on every rank (already-reduced / DP-replicated); take any (rank 0).
- ``gather`` — per-shard quantity; global aggregation concatenates shards (forward-compat only).
"""

import json
import os

import torch

from .checkpointing import save_grads

# Reduction-tag vocabulary (stored in meta.json; consumed by the scaling-repo aggregator/fitter).
REDUCE_MAX = "max"
REDUCE_SOS_N = "sos+n"
REDUCE_REPLICA = "replica"
REDUCE_GATHER = "gather"


def current_parallel_layout() -> dict:
    """Best-effort {axis: {rank, size}} for tp/pp/dp/ep; identity when mpu is uninitialised.

    Recorded in meta.json so a later multi-rank aggregator knows the sharding the file was written
    under. On the single-GPU dev box every size is 1 (pure identity)."""
    try:
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            raise RuntimeError
        return {
            "tp": {"rank": mpu.get_tensor_model_parallel_rank(),
                   "size": mpu.get_tensor_model_parallel_world_size()},
            "pp": {"rank": mpu.get_pipeline_model_parallel_rank(),
                   "size": mpu.get_pipeline_model_parallel_world_size()},
            "dp": {"rank": mpu.get_data_parallel_rank(),
                   "size": mpu.get_data_parallel_world_size()},
            "ep": {"rank": mpu.get_expert_model_parallel_rank(),
                   "size": mpu.get_expert_model_parallel_world_size()},
        }
    except Exception:
        return {ax: {"rank": 0, "size": 1} for ax in ("tp", "pp", "dp", "ep")}


def write_stream_meta(save_dir: str, stream: str, reductions: dict) -> None:
    """Write/refresh ``{save_dir}/{stream}/meta.json`` (the schema contract).

    ``reductions`` maps a key *suffix* (the part after the last ``::``, e.g. ``sigma_max_dW``,
    ``abs_max``, or ``""`` for the primary value) to one of the REDUCE_* tags. Written from the
    same rank that owns the save (``expert_data_parallel_rank()==0``)."""
    try:
        from megatron.core import parallel_state as mpu

        if mpu.get_expert_data_parallel_rank() != 0:
            return
    except Exception:
        pass
    if save_dir is None:
        return
    d = os.path.join(save_dir, stream)
    os.makedirs(d, exist_ok=True)
    meta = {
        "stream": stream,
        "schema_version": 1,
        "value_dtype": "float32-0d",
        "parallel_layout": current_parallel_layout(),
        "reductions": reductions,
        "note": "0-d CPU scalars under dict[model_chunk][<module>/<io>[::<stat>]]. Tags say how "
                "each key reduces across ranks; sos+n RMS keys store the final RMS at single rank.",
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def maybe_log_wandb(state: dict, iteration: int, prefix: str) -> None:
    """Mirror dgrad_logging._maybe_log_wandb: push 0-d scalars to the shared wandb.run, swallow
    any failure so a logging hiccup never kills training. No-op when wandb is absent (the dev box)."""
    try:
        import wandb  # type: ignore
    except ImportError:
        return
    if getattr(wandb, "run", None) is None:
        return
    scalars = {}
    for chunk_name, mods in state.items():
        for key, tensor in mods.items():
            try:
                val = float(tensor)
            except Exception:
                continue
            if val != val:  # NaN
                continue
            scalars[f"{prefix}/{chunk_name}/{key}"] = val
    if scalars:
        try:
            wandb.log(scalars, step=iteration)
        except Exception:
            pass


def save_diag_state(save_dir: str, stream: str, iteration: int, state: dict,
                    reductions: dict, wandb_prefix: str | None = None) -> None:
    """Persist a diagnostic state dict (the common path for every stream).

    Writes ``{save_dir}/{stream}/iter_{iteration:07d}/mp_rank_*.pth`` via the existing ``save_grads``
    (per-rank, only from ``expert_data_parallel_rank()==0``), refreshes the stream ``meta.json``
    contract, and flushes scalars to W&B. ``state`` is ``dict[model_chunk][key]=0-d CPU tensor``."""
    if not state:
        return
    save_grads(save_dir, state, iteration, stream)
    write_stream_meta(save_dir, stream, reductions)
    maybe_log_wandb(state, iteration, wandb_prefix or stream)
