"""Multi-view observability gates for local mesh deformation."""

from __future__ import annotations

import torch


def vertex_gradient_consensus(
    gradients: torch.Tensor,
    agreement_threshold: float = 0.60,
    keep_fraction: float = 0.30,
    magnitude_epsilon: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a vertex mask supported by consistent multi-view gradients.

    ``gradients`` has shape ``[views, vertices, 3]``. Agreement is the norm of
    the mean unit direction, while support is the number of nonzero views.
    """
    if gradients.ndim != 3 or gradients.shape[-1] != 3:
        raise ValueError("gradients must have shape [views, vertices, 3]")
    if gradients.shape[0] < 2:
        raise ValueError("at least two views are required")
    magnitudes = torch.linalg.vector_norm(gradients, dim=-1)
    visible = magnitudes > float(magnitude_epsilon)
    directions = gradients / magnitudes.clamp_min(float(magnitude_epsilon)).unsqueeze(-1)
    directions = directions * visible.unsqueeze(-1)
    support = visible.sum(0)
    agreement = torch.linalg.vector_norm(directions.sum(0), dim=-1) / support.clamp_min(1)
    strength = (magnitudes * visible).sum(0) / support.clamp_min(1)
    supported = support >= 2
    eligible = supported & (agreement >= float(agreement_threshold))
    mask = torch.zeros_like(eligible)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_indices.numel():
        keep = max(1, int(round(float(keep_fraction) * gradients.shape[1])))
        keep = min(keep, eligible_indices.numel())
        ranking = strength[eligible_indices] * agreement[eligible_indices]
        selected = eligible_indices[torch.topk(ranking, keep).indices]
        mask[selected] = True
    return mask, agreement, support
