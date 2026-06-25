# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""δy activation-update probe (#118, PLAN §3.2 Test 2; design 10 + DESIGN_DECISIONS D3/D4/D13/D14).

Measures, on a FIXED frozen probe batch bracketing one ``optimizer.step()``, how much the model's
activations move from a single weight update. Two distinct, separately-saved signals (decision D3):

1. ``dy_local`` — per ``LINEAR_TYPES`` module, the LOCAL update with the module INPUT held frozen:
   ``ΔRMS_local = ‖ module_{t+1}(x_l) − y_t ‖_RMS`` where ``x_l`` is the pass-1-cached input and
   ``y_t`` the pass-1-cached output. For a Linear this is exactly ``‖δW·x‖`` (Everett ``r_l``).
   This is a per-module RECOMPUTE on the cached input — NO second full forward is needed for it.
2. ``dy_resid`` — per layer, the PROPAGATED residual-stream change: a SECOND full forward at
   θ_{t+1} on the same frozen batch, diffing each layer's ``post_mixer_residual`` against pass 1.
   This is PLAN §0's "step-to-step change in the residual stream" desideratum.
3. relative versions ``dy_local_rel`` / ``dy_resid_rel`` = ``ΔRMS / RMS(y_t or residual_t)`` — the
   scale-free feature-learning gate (decision D4). All diffs are computed in fp32.

Off the training hot path (structural, not best-effort): both passes run under ``torch.no_grad()``
with ``model.eval()`` (dropout off; prior mode restored), never call ``backward``, and never advance
the training data iterator (the probe owns a separate frozen batch). Hooks are registered and
removed within each pass, so non-probe steps and the training fwd/bwd carry zero probe overhead.

Memory is bounded to ≈ one full activation set (NOT two): pass 1 caches ``x_l``/``y_t`` per module;
pass 2 (dy_local) recomputes one module at a time and frees that module's cache as it is consumed
(peak ≈ pass-1 cache + one transient recompute). dy_resid only retains one residual tensor per
layer, diffed and freed during the second full forward.

Reused by import from ``activation_logging`` (single source of truth — do NOT duplicate):
``LINEAR_TYPES``, ``_register_hooks``, ``_rms``, ``_iter_tensors``, ``unwrap_model``, and the
post-mixer-residual matching logic. The save path is the shared D13 sink ``save_diag_state``.

Public API (mirrors the module-function style of ``activation_logging``):

    capture_probe_batch(model, batch)            # freeze the fixed probe batch ONCE (idempotent)
    run_probe_forward(model) -> output           # reusable no_grad forward-only driver on the batch
    probe_pre(model)                             # PASS 1: cache x_l, y_t, residual_t
    probe_post(model, save_dir, iteration)       # dy_local recompute + PASS 2 fwd (dy_resid) + save
    reset_probe()                                # drop frozen batch + residue (test/teardown hook)

# TODO(trainer-wiring, megatron/training/training.py — DO NOT edit those shared files here):
#   The trainer must, on a diagnostic step (gated by e.g. ``save_probe_interval``), call:
#     * capture_probe_batch(model, probe_batch)  -- once; idempotent (e.g. from a held-out iterator)
#     * probe_pre(model)                          -- BEFORE optimizer.step(): ~training.py:2052
#                                                    (after grad reduction, while θ is still θ_t)
#     * optimizer.step()
#     * probe_post(model, args.save, iteration+1) -- AFTER  optimizer.step(), BEFORE the LR
#                                                    scheduler step: ~training.py:2083 (θ is θ_{t+1})
#   Other diagnostics (e.g. attention-entropy) that want to ride the SAME frozen forward may register
#   their own hooks on ``model`` and then call ``run_probe_forward(model)`` directly.
"""

import logging
from collections import defaultdict

import torch

from .activation_logging import (
    LINEAR_TYPES,
    _iter_tensors,
    _register_hooks,
    _rms,
)
from .diagnostic_layer_selection import make_layer_name_filter
from .diagnostics_common import REDUCE_SOS_N, save_diag_state
from .utils import unwrap_model

logger = logging.getLogger(__name__)

_STREAM = "delta_y"
_WANDB_PREFIX = "dy"


def _rms_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Elementwise RMS in fp32, returned as a 0-d CPU tensor (the δy reduction).

    Reuses ``activation_logging._rms`` (sqrt(mean(x*x))); the ``.float()`` is the D6/§6
    bf16-cancellation guard — when δy is tiny, subtracting two close bf16 tensors loses
    precision, so diffs are always upcast to fp32 before the RMS reduction.
    """
    return _rms(tensor.detach().float()).cpu()


def _delta_rms(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """ΔRMS = ‖new − old‖_RMS computed in fp32 (the core δy reduction helper)."""
    return _rms_fp32(new.detach().float() - old.detach().float())


def _merge_with_named_aliases(local: dict, resid: dict) -> dict:
    """Merge δy streams while preserving explicit ``dy_local``/``dy_resid`` aliases.

    The original W&B/PTH surface keeps local module outputs and propagated residual deltas under
    their natural activation names (for example ``linear_qkv/output0::rel`` and
    ``post_mixer_residual::rel``). Keep those keys for compatibility, and add namespaced aliases so
    downstream smoke gates can assert that both conceptual streams reached W&B directly.
    """
    state: defaultdict = defaultdict(dict)
    for chunk_name, mods in local.items():
        for key, tensor in mods.items():
            state[chunk_name][key] = tensor
            state[chunk_name][f"dy_local/{key}"] = tensor
    for chunk_name, mods in resid.items():
        for key, tensor in mods.items():
            state[chunk_name][key] = tensor
            state[chunk_name][f"dy_resid/{key}"] = tensor
    return state


def _first_output_tensor(output):
    """First activation tensor of a module output (``output0`` in ``activation_logging`` keys).

    Mirrors ``activation_logging``'s ``_iter_tensors(output, "output0")`` first-yield convention so
    the δy keys line up 1:1 with the coordinate-check (Test 1) keys for the same module.
    """
    for _suffix, tensor in _iter_tensors(output, "output0"):
        return tensor
    return None


class ProbeLogger:
    """Singleton owning the frozen probe batch and the two-pass δy state.

    State held between ``probe_pre`` and ``probe_post``:
      * ``_xl[chunk][name]``  — cached module INPUT  x_l (detached, on device) for dy_local recompute
      * ``_yt[chunk][name]``  — cached module OUTPUT y_t (detached, on device)
      * ``_resid_t[chunk][layer_key]`` — cached pass-1 post_mixer_residual for dy_resid
    These are freed as consumed in ``probe_post`` so peak extra memory ≈ one activation set.
    """

    def __init__(self):
        self._batch = None  # frozen probe batch tuple (persists across diagnostic steps)
        self._layer_name_filter = None
        self._reset_pass_state()

    def _reset_pass_state(self):
        self._xl: defaultdict = defaultdict(dict)
        self._yt: defaultdict = defaultdict(dict)
        self._resid_t: defaultdict = defaultdict(dict)

    # -- frozen batch -------------------------------------------------------

    def capture_probe_batch(self, model, batch) -> None:
        """Freeze ``batch`` ONCE (idempotent). ``batch`` is the get_batch tuple
        ``(tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params)``.

        Subsequent calls are no-ops so the frozen batch is identical step-to-step (low-variance,
        comparable δy across the width/depth grid). The caller is responsible for moving tensors to
        the right device before/at capture; we store the tuple as given.
        """
        if self._batch is not None:
            return
        self._batch = tuple(batch)

    def run_probe_forward(self, model):
        """Reusable forward-only ``no_grad`` driver on the frozen batch (``model.eval()`` around it,
        prior training/eval mode restored). Returns the model output. Other diagnostics may register
        their own hooks on ``model`` BEFORE calling this to ride the same frozen forward.

        Replays the exact ``pretrain_gpt.forward_step`` call shape:
        ``model(tokens, position_ids, attention_mask, labels=, loss_mask=, packed_seq_params=)``.
        """
        assert self._batch is not None, "capture_probe_batch must be called before run_probe_forward"
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = self._batch
        chunk = unwrap_model(model[0]) if isinstance(model, (list, tuple)) else unwrap_model(model)
        prev_training = chunk.training
        chunk.eval()
        try:
            with torch.no_grad():
                output = chunk(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=labels,
                    loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params,
                )
        finally:
            chunk.train(prev_training)
        return output

    # -- pass 1: cache x_l, y_t, residual_t ---------------------------------

    def _make_cache_hook(self, chunk_name, module_name):
        xl, yt = self._xl, self._yt

        def hook(_module, args, kwargs, output):
            input_tuple = args if isinstance(args, tuple) else (args,)
            # CLONE (not just detach): the cached input/output share storage with the live
            # activation buffer, and Megatron's RowParallelLinear / residual / fused paths
            # mutate those buffers IN PLACE downstream. Between pass-1 capture and the pass-2
            # recompute in probe_post, an in-place mutation would corrupt a detach-only cache —
            # verified: without the clone the zero-LR control left a ~6e-2 dy_local on exactly
            # the RowParallelLinear modules (linear_proj / linear_fc2) whose inputs are rewritten
            # in place, while ColumnParallelLinear (qkv/fc1) read fresh layernorm outputs and were
            # already 0. Cloning makes the frozen-input recompute exact (dy_local==0 at LR=0).
            if input_tuple and isinstance(input_tuple[0], torch.Tensor):
                xl[chunk_name][module_name] = input_tuple[0].detach().clone()
            y = _first_output_tensor(output)
            if isinstance(y, torch.Tensor):
                yt[chunk_name][module_name] = y.detach().clone()

        return hook

    def _make_resid_pre_hook(self, chunk_name, layer_key):
        resid_t = self._resid_t

        def pre_hook(_module, args):
            if args and isinstance(args[0], torch.Tensor):
                resid_t[chunk_name][layer_key] = args[0].detach()

        return pre_hook

    def _make_resid_out_hook(self, chunk_name, layer_key):
        resid_t = self._resid_t

        def fwd_hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                resid_t[chunk_name][layer_key] = tensor.detach()

        return fwd_hook

    def _register_resid_hooks(self, model, *, pre_factory, out_factory, name_filter=None):
        """Register post-mixer-residual hooks, mirroring
        ``ActivationLogger._register_post_attn_residual_hooks``: a pre-hook on each
        ``*.layers.N.pre_mlp_layernorm`` (transformer layers), and a forward hook on Mamba layers
        that have a ``mixer`` but no ``pre_mlp_layernorm``."""
        import re

        handles = []
        for chunk_id, model_chunk in enumerate(model):
            chunk_name = f"model_chunk{chunk_id}"
            unwrapped = unwrap_model(model_chunk)
            for module_name, module in unwrapped.named_modules():
                m = re.fullmatch(r"(.*\.layers\.\d+)\.pre_mlp_layernorm", module_name)
                if m is not None:
                    if name_filter is not None and not name_filter(m.group(1)):
                        continue
                    handles.append(
                        module.register_forward_pre_hook(pre_factory(chunk_name, m.group(1)))
                    )
                    continue
                m2 = re.fullmatch(r"(.*\.layers\.\d+)", module_name)
                if (
                    m2 is not None
                    and hasattr(module, "mixer")
                    and not hasattr(module, "pre_mlp_layernorm")
                ):
                    if name_filter is not None and not name_filter(m2.group(1)):
                        continue
                    handles.append(
                        module.register_forward_hook(out_factory(chunk_name, m2.group(1)))
                    )
        return handles

    def probe_pre(self, model, args=None):
        """PASS 1 (before ``optimizer.step()``): cache each module's input x_l + output y_t and each
        layer's pre-step residual, via one ``no_grad`` forward-only replay of the frozen batch."""
        assert self._batch is not None, "capture_probe_batch must be called before probe_pre"
        self._reset_pass_state()
        name_filter = make_layer_name_filter(model, args, stream="probe")
        self._layer_name_filter = name_filter
        handles = _register_hooks(model, LINEAR_TYPES, self._make_cache_hook, name_filter=name_filter)
        handles += self._register_resid_hooks(
            model,
            pre_factory=self._make_resid_pre_hook,
            out_factory=self._make_resid_out_hook,
            name_filter=name_filter,
        )
        try:
            self.run_probe_forward(model)
        finally:
            for h in handles:
                h.remove()

    # -- pass 2: dy_local recompute + dy_resid full forward + save ----------

    def _compute_dy_local(self, model) -> dict:
        """Per LINEAR_TYPES module, recompute ``module(x_l)`` with the NEW weights (θ_{t+1}) on the
        FROZEN cached input and diff vs the cached output y_t → ΔRMS_local (+ relative). One module
        at a time; each module's cached x_l / y_t is freed as it is consumed (peak ≈ one recompute).
        """
        state: defaultdict = defaultdict(dict)
        chunk_modules = {}
        # Force eval mode for the recompute (mirrors ``run_probe_forward``): ``y_t`` was captured
        # under ``model.eval()`` in pass 1, so the recompute MUST also be in eval mode or any
        # train-mode-only behavior (dropout / stochastic depth) manufactures a SPURIOUS dy_local
        # floor that does NOT vanish at LR=0 (verified: without this, the zero-LR control left a
        # ~2e-2 dy_local while dy_resid — which forces eval — was exactly 0). Prior mode restored.
        prev_modes = []
        for chunk_id, model_chunk in enumerate(model):
            chunk_name = f"model_chunk{chunk_id}"
            unwrapped = unwrap_model(model_chunk)
            chunk_modules[chunk_name] = dict(unwrapped.named_modules())
            prev_modes.append((unwrapped, unwrapped.training))
            unwrapped.eval()

        try:
            with torch.no_grad():
                for chunk_name in list(self._xl.keys()):
                    mods = chunk_modules.get(chunk_name, {})
                    for module_name in list(self._xl[chunk_name].keys()):
                        x_l = self._xl[chunk_name].pop(module_name)
                        y_t = self._yt[chunk_name].pop(module_name, None)
                        module = mods.get(module_name)
                        if module is None or y_t is None:
                            continue
                        y_new = _first_output_tensor(module(x_l))
                        if not isinstance(y_new, torch.Tensor):
                            continue
                        delta = _delta_rms(y_new, y_t)
                        base = _rms_fp32(y_t)
                        key = f"{module_name}/output0"
                        state[chunk_name][key] = delta
                        state[chunk_name][f"{key}::rel"] = (
                            delta / base if float(base) > 0 else torch.tensor(0.0)
                        )
                        del x_l, y_t, y_new  # drop references so memory drains as we go
        finally:
            for unwrapped, was_training in prev_modes:
                unwrapped.train(was_training)
        return state

    def _compute_dy_resid(self, model) -> dict:
        """SECOND full forward at θ_{t+1}; diff each layer's post_mixer_residual vs the pass-1 cache
        → ΔRMS_resid (+ relative). Streaming: the pass-2 residual hook diffs against the cached
        residual_t and frees it immediately (peak ≈ one residual tensor above the pass-1 cache)."""
        state: defaultdict = defaultdict(dict)
        resid_t = self._resid_t

        def diff_pre_factory(chunk_name, layer_key):
            def pre_hook(_module, args):
                if args and isinstance(args[0], torch.Tensor):
                    self._record_resid(state, chunk_name, layer_key, args[0])

            return pre_hook

        def diff_out_factory(chunk_name, layer_key):
            def fwd_hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor):
                    self._record_resid(state, chunk_name, layer_key, tensor)

            return fwd_hook

        handles = self._register_resid_hooks(
            model,
            pre_factory=diff_pre_factory,
            out_factory=diff_out_factory,
            name_filter=self._layer_name_filter,
        )
        try:
            self.run_probe_forward(model)
        finally:
            for h in handles:
                h.remove()
        # Sanity: every cached residual_t should have been consumed.
        leftover = sum(len(v) for v in resid_t.values())
        if leftover:
            logger.warning("probe: %d post_mixer_residual cache entries unconsumed", leftover)
        self._resid_t = defaultdict(dict)
        return state

    def _record_resid(self, state, chunk_name, layer_key, new_tensor):
        old = self._resid_t[chunk_name].pop(layer_key, None)
        if old is None:
            return
        delta = _delta_rms(new_tensor, old)
        base = _rms_fp32(old)
        key = f"{layer_key}/post_mixer_residual"
        state[chunk_name][key] = delta
        state[chunk_name][f"{key}::rel"] = (
            delta / base if float(base) > 0 else torch.tensor(0.0)
        )
        del old

    def probe_post(self, model, save_dir, iteration) -> None:
        """PASS 2 (after ``optimizer.step()``): compute dy_local (per-module recompute on cached
        inputs) and dy_resid (second full forward), merge, save via the shared D13 sink, clear
        per-step state. The frozen batch persists."""
        local = self._compute_dy_local(model)
        resid = self._compute_dy_resid(model)
        state = _merge_with_named_aliases(local, resid)
        # Both ΔRMS and the ::rel companion reduce as RMS-type (sum-of-squares) across ranks (D2).
        save_diag_state(
            save_dir,
            stream=_STREAM,
            iteration=iteration,
            state=state,
            reductions={"": REDUCE_SOS_N, "rel": REDUCE_SOS_N},
            wandb_prefix=_WANDB_PREFIX,
        )
        self._reset_pass_state()
        self._layer_name_filter = None

    def reset(self) -> None:
        self._batch = None
        self._layer_name_filter = None
        self._reset_pass_state()


# --- latest-training-batch stash (single source of truth across __main__ and the package) ---
# pretrain_*.get_batch lives in the __main__ script, while the trainer imports this module as
# ``megatron.training.probe_logging`` — two distinct module objects if the stash lived in the
# script. Keeping it HERE means both the producer (get_batch) and the consumer (train_step) read
# the SAME module global. Only the diagnostic path touches it (gated by --save-*-interval).
_LAST_TRAINING_BATCH = None


def set_last_training_batch(batch) -> None:
    """Stash the most recent full training batch (6-tuple) for the δy/entropy probe."""
    global _LAST_TRAINING_BATCH
    _LAST_TRAINING_BATCH = batch


def get_last_training_batch():
    """Return the most recent full training batch stashed by ``set_last_training_batch`` (or None)."""
    return _LAST_TRAINING_BATCH


_PROBE: ProbeLogger | None = None


def _get_probe() -> ProbeLogger:
    global _PROBE
    if _PROBE is None:
        _PROBE = ProbeLogger()
    return _PROBE


# -- module-function public API (mirrors activation_logging) ----------------

def capture_probe_batch(model, batch) -> None:
    _get_probe().capture_probe_batch(model, batch)


def run_probe_forward(model):
    return _get_probe().run_probe_forward(model)


def probe_pre(model, args=None) -> None:
    _get_probe().probe_pre(model, args)


def probe_post(model, save_dir, iteration) -> None:
    _get_probe().probe_post(model, save_dir, iteration)


def reset_probe() -> None:
    if _PROBE is not None:
        _PROBE.reset()
