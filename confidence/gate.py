"""Hard gating based on render-mask consistency."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mask_confidence import binary_mask_iou


@dataclass(frozen=True)
class MaskGateResult:
    """Per-observation mask confidence and acceptance decision."""

    mask_iou: torch.Tensor
    accepted: torch.Tensor


def apply_mask_confidence_gate(
    segmentation_masks: torch.Tensor,
    rendered_masks: torch.Tensor,
    *,
    enabled: bool,
    confidence_threshold: float,
    mask_binary_threshold: float = 0.5,
) -> MaskGateResult:
    """Apply an inclusive hard threshold to render-mask IoU.

    When disabled, all observations are accepted. IoU is still returned so the
    caller can log confidence in an explicitly requested analysis run.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            f"confidence_threshold must be in [0, 1], got {confidence_threshold}."
        )
    mask_iou = binary_mask_iou(
        segmentation_masks,
        rendered_masks,
        threshold=mask_binary_threshold,
    )
    accepted = mask_iou >= confidence_threshold
    if not enabled:
        accepted = torch.ones_like(accepted, dtype=torch.bool)
    return MaskGateResult(mask_iou=mask_iou, accepted=accepted)
