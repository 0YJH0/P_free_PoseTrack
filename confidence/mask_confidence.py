"""Render-mask consistency confidence."""

from __future__ import annotations

import torch


def binary_mask_iou(
    segmentation_masks: torch.Tensor,
    rendered_masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute one binary intersection-over-union score per observation.

    Args:
        segmentation_masks: Image-segmentation masks with shape ``[N, H, W]``.
        rendered_masks: Geometry-rendered masks with the same shape.
        threshold: Values greater than or equal to this value are foreground.
        eps: Numerical stabilizer. Observations with an empty union receive 0.

    Returns:
        A tensor of shape ``[N]`` containing values in ``[0, 1]``.
    """
    if segmentation_masks.shape != rendered_masks.shape:
        raise ValueError(
            "Segmentation and rendered masks must have identical shapes, got "
            f"{tuple(segmentation_masks.shape)} and {tuple(rendered_masks.shape)}."
        )
    if segmentation_masks.ndim != 3:
        raise ValueError(
            f"Expected masks with shape [N, H, W], got {tuple(segmentation_masks.shape)}."
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"mask threshold must be in [0, 1], got {threshold}.")

    segmentation = segmentation_masks >= threshold
    rendered = rendered_masks >= threshold
    reduce_dims = (1, 2)
    intersection = (segmentation & rendered).sum(dim=reduce_dims, dtype=torch.float32)
    union = (segmentation | rendered).sum(dim=reduce_dims, dtype=torch.float32)
    scores = intersection / union.clamp_min(eps)
    return torch.where(union > 0, scores, torch.zeros_like(scores))
