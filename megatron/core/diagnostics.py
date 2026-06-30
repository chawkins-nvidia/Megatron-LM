# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Execution identity shared by core diagnostic instrumentation."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import torch

_DIAGNOSTIC_MICROBATCH_ID: ContextVar[int | None] = ContextVar(
    "diagnostic_microbatch_id", default=None
)
_DIAGNOSTIC_RECOMPUTE: ContextVar[bool] = ContextVar("diagnostic_recompute", default=False)
_DIAGNOSTIC_GLOBAL_VALID_TOKENS: ContextVar["torch.Tensor | None"] = ContextVar(
    "diagnostic_global_valid_tokens", default=None
)


def set_diagnostic_microbatch_id(index: int | None) -> None:
    """Set the microbatch identity observed by diagnostic hooks.

    Args:
        index: Schedule-local microbatch index, or ``None`` outside a microbatch.
    """

    if index is not None and index < 0:
        raise ValueError("diagnostic microbatch indices must be nonnegative")
    _DIAGNOSTIC_MICROBATCH_ID.set(index)


def get_diagnostic_microbatch_id() -> int | None:
    """Return the current schedule-local diagnostic microbatch index."""

    return _DIAGNOSTIC_MICROBATCH_ID.get()


@contextmanager
def diagnostic_recompute(microbatch_id: int) -> Iterator[None]:
    """Mark a backward activation rerun with its original microbatch identity.

    Args:
        microbatch_id: Microbatch whose forward graph is being reconstructed.

    Yields:
        Control while recomputation identity is active.
    """

    if microbatch_id < 0:
        raise ValueError("diagnostic microbatch indices must be nonnegative")
    microbatch_token = _DIAGNOSTIC_MICROBATCH_ID.set(microbatch_id)
    recompute_token = _DIAGNOSTIC_RECOMPUTE.set(True)
    try:
        yield
    finally:
        _DIAGNOSTIC_RECOMPUTE.reset(recompute_token)
        _DIAGNOSTIC_MICROBATCH_ID.reset(microbatch_token)


def is_diagnostic_recompute() -> bool:
    """Return whether execution is a backward activation recomputation."""

    return _DIAGNOSTIC_RECOMPUTE.get()


def set_diagnostic_global_valid_tokens(tokens: "torch.Tensor | None") -> None:
    """Retain the token scalar finalized by the normal gradient path.

    Args:
        tokens: Globally pooled valid-token tensor, or ``None`` between attempts.
    """

    _DIAGNOSTIC_GLOBAL_VALID_TOKENS.set(tokens)


def get_diagnostic_global_valid_tokens() -> "torch.Tensor | None":
    """Return the globally pooled valid-token tensor from gradient finalization."""

    return _DIAGNOSTIC_GLOBAL_VALID_TOKENS.get()
