# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed post-reduction normalization for Tier-0 captured gradients."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .accumulator import PackedSufficientStatistics
from .registry import NormalizationKind

if TYPE_CHECKING:
    from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler


@dataclass(frozen=True)
class CanonicalDgradNormalizer:
    """Normalize raw activation dgrad to the global valid-token mean loss.

    The adapter is deliberately post-reduction: raw square sums are pooled
    across unequal microbatches/ranks first, then divided by the square of the
    constant loss scale and globally pooled valid-token count.

    Attributes:
        loss_scale: Configured constant loss scale, or a scalar tensor carrying it.
    """

    loss_scale: int | float | torch.Tensor = 1.0

    kind = NormalizationKind.LOSS_SCALE_AND_GLOBAL_VALID_TOKENS
    identity = "canonical_dgrad_global_valid_tokens_v1"

    def __post_init__(self) -> None:
        """Reject nonconstant, malformed host-side scale declarations."""

        if isinstance(self.loss_scale, torch.Tensor):
            if self.loss_scale.numel() != 1:
                raise ValueError("constant loss scale must be scalar")
        elif (
            not isinstance(self.loss_scale, (int, float))
            or isinstance(self.loss_scale, bool)
            or not math.isfinite(self.loss_scale)
            or self.loss_scale <= 0
        ):
            raise ValueError("constant loss scale must be finite and positive")

    @classmethod
    def from_grad_scaler(
        cls, grad_scaler: "ConstantGradScaler | DynamicGradScaler | None"
    ) -> "CanonicalDgradNormalizer":
        """Construct from the supported Megatron loss-scaling configuration.

        Args:
            grad_scaler: ``None`` or a constant scaler. Dynamic scaling fails closed.

        Returns:
            Canonical normalizer bound to the configured constant scale.

        Raises:
            ValueError: If dynamic or unknown loss scaling is configured.
        """

        from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler

        if grad_scaler is None:
            return cls()
        if isinstance(grad_scaler, DynamicGradScaler):
            raise ValueError("canonical dgrad does not support dynamic loss scaling")
        if not isinstance(grad_scaler, ConstantGradScaler):
            raise ValueError("canonical dgrad requires a constant loss scaler")
        return cls(grad_scaler.scale.detach())

    def normalize_(
        self, accumulator: PackedSufficientStatistics, slot: int, global_valid_tokens: torch.Tensor
    ) -> None:
        """Normalize one reduced raw-dgrad slot in place on its device.

        Args:
            accumulator: Reduced packed sufficient statistics.
            slot: Zero-based packed descriptor slot.
            global_valid_tokens: Globally pooled valid-token scalar.
        """

        slots = accumulator.slots(slot)
        device = accumulator.sum_pack.device
        loss_scale = torch.as_tensor(self.loss_scale, dtype=torch.float64, device=device)
        valid_tokens = global_valid_tokens.detach().to(dtype=torch.float64, device=device)
        denominator = loss_scale * valid_tokens
        valid = torch.isfinite(denominator) & (denominator > 0)
        safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
        inverse = safe_denominator.reciprocal()
        inverse_squared = inverse.square()

        accumulator.sum_pack[slots.sum].mul_(inverse)
        for packed_slot in (slots.sumsq, slots.dot, slots.lhs_sumsq, slots.rhs_sumsq):
            accumulator.sum_pack[packed_slot].mul_(inverse_squared)
        accumulator.max_pack[slots.maximum].mul_(inverse.to(dtype=torch.float32))
        accumulator.min_pack[slots.minimum].mul_(inverse.to(dtype=torch.float32))
        accumulator.sum_pack[slots.nonfinite_arithmetic].add_(
            (~valid).to(dtype=accumulator.sum_pack.dtype)
        )
