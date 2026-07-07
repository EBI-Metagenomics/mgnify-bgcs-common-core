"""Focal loss for the SanntiS detection ANN.

Replaces the custom duration-robust loss (RLF) used in the paper with an
already-implemented alternative, as prescribed by
``06-baseline-strategy.md`` §3.3. Focal loss reweights per-position
binary cross-entropy by ``(1 - p_t)**gamma`` so that easy background positions
contribute less as training progresses — same intent as RLF (absorb class-count
and duration imbalance across BGC classes) but with a well-supported
reference implementation.

The implementation follows Lin et al. 2017 ("Focal Loss for Dense Object
Detection"). ``gamma=2.0`` matches the RLF gamma used in the paper. The
optional ``alpha`` balances positive vs. negative positions; leave it at
``None`` to disable class-balance reweighting and rely on ``gamma`` alone.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Reduction = Literal["none", "mean", "sum"]


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
    reduction: Reduction = "mean",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-position focal loss on sigmoid logits.

    Args:
        logits: raw pre-sigmoid outputs, arbitrary shape (typically
            ``(batch, timesteps)`` for the detector).
        targets: 0/1 tensor with the same shape as ``logits``.
        gamma: focusing parameter. ``2.0`` matches the paper's RLF gamma.
        alpha: optional positive-class weight in ``[0, 1]``. If given, the
            per-position loss is scaled by ``alpha`` for positives and
            ``1 - alpha`` for negatives (the standard focal-loss convention).
        reduction: ``"none"`` returns the per-position loss;
            ``"mean"`` / ``"sum"`` reduce. When a ``mask`` is provided,
            ``"mean"`` averages only over unmasked positions.
        mask: optional 0/1 tensor with the same shape as ``logits`` marking
            valid positions. Padded positions in 200-timestep windows should
            be masked out.

    Returns:
        Scalar (``mean``/``sum``) or per-position (``none``) loss.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != targets shape "
            f"{tuple(targets.shape)}"
        )

    ce = F.binary_cross_entropy_with_logits(
        logits, targets.to(dtype=logits.dtype), reduction="none"
    )
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ((1.0 - p_t) ** gamma) * ce

    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != logits shape "
                f"{tuple(logits.shape)}"
            )
        mask_f = mask.to(dtype=loss.dtype)
        loss = loss * mask_f
        if reduction == "mean":
            denom = mask_f.sum().clamp_min(1.0)
            return loss.sum() / denom
        if reduction == "sum":
            return loss.sum()
        return loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class FocalLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`sigmoid_focal_loss`.

    Use this in a trainer when you want the loss to appear in
    ``model.state_dict()`` graphs and TensorBoard-style module trees, or when
    you want the loss's hyperparameters to be part of the pickled trainer.
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float | None = None,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction: Reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return sigmoid_focal_loss(
            logits,
            targets,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
            mask=mask,
        )
