"""Training-free uncertainty calibration and guarded soft publication for V6."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .joint_pose_geometry import (
    per_observation_boundary_loss,
    per_observation_iou,
    temporal_pose_loss,
)


@dataclass(frozen=True)
class RiskCalibratedConfig:
    log_variance_radius: float = 2.0
    uncertainty_prior_weight: float = 0.10
    cvar_quantile: float = 0.90
    cvar_weight: float = 0.35
    uncertainty_weight: float = 0.25
    boundary_weight: float = 0.25
    pose_weight: float = 0.10
    temporal_weight: float = 0.05
    geometry_weight: float = 0.10
    risk_margin: float = 0.001
    tail_tolerance: float = 0.01
    alpha_grid: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


class BoundedLogVariance(nn.Module):
    """Per-observation uncertainty bounded around a causal reliability prior."""

    def __init__(
        self,
        prior_weights: torch.Tensor,
        radius: float = 2.0,
        optimizable: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        prior = prior_weights.detach().float().clamp(0.05, 1.0)
        self.register_buffer("prior", -torch.log(prior))
        mask = torch.ones_like(prior)
        if optimizable is not None:
            if optimizable.shape != prior.shape:
                raise ValueError("optimizable mask must match prior weights")
            mask = optimizable.to(prior)
        self.register_buffer("optimizable", mask)
        self.raw = nn.Parameter(torch.zeros_like(prior))
        self.radius = float(radius)

    def forward(self) -> torch.Tensor:
        return self.prior + self.optimizable * self.radius * torch.tanh(self.raw)

    def precision(self) -> torch.Tensor:
        return torch.exp(-self()).clamp(0.01, 20.0)

    def prior_penalty(self) -> torch.Tensor:
        return ((self() - self.prior).square() * self.optimizable).sum() / (
            self.optimizable.sum().clamp_min(1.0)
        )


def heteroscedastic_silhouette_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    boundary_weight: float = 0.10,
) -> torch.Tensor:
    residual = (1.0 - per_observation_iou(prediction, target)) + float(
        boundary_weight
    ) * per_observation_boundary_loss(prediction, target)
    log_variance = log_variance.to(residual).reshape(-1)
    if log_variance.shape != residual.shape:
        raise ValueError("one log variance is required per observation")
    return (torch.exp(-log_variance) * residual + log_variance).mean()


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.reshape(-1).clamp_min(0.0)
    return weights.sum().square() / weights.square().sum().clamp_min(1e-8)


def weighted_cvar(
    losses: torch.Tensor, weights: torch.Tensor, quantile: float = 0.90
) -> torch.Tensor:
    losses, weights = losses.reshape(-1), weights.to(losses).reshape(-1).clamp_min(0.0)
    if losses.numel() == 0 or losses.shape != weights.shape:
        raise ValueError("losses and weights must be non-empty and aligned")
    order = torch.argsort(losses, descending=True)
    sorted_losses, sorted_weights = losses[order], weights[order]
    tail_mass = max(1.0 - float(quantile), 1e-3) * weights.sum().clamp_min(1e-8)
    cumulative = torch.cumsum(sorted_weights, dim=0)
    previous = cumulative - sorted_weights
    included = (tail_mass - previous).clamp(min=0.0)
    included = torch.minimum(included, sorted_weights)
    return (included * sorted_losses).sum() / included.sum().clamp_min(1e-8)


@dataclass(frozen=True)
class GuardedRisk:
    total: float
    weighted_mean: float
    cvar: float
    uncertainty: float
    boundary: float
    pose_correction: float
    temporal: float
    geometry_step_ratio: float
    effective_samples: float


def evaluate_guarded_risk(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prior_weights: torch.Tensor,
    twists: torch.Tensor,
    poses: torch.Tensor,
    geometry_step_ratio: float,
    config: RiskCalibratedConfig,
) -> GuardedRisk:
    with torch.no_grad():
        losses = 1.0 - per_observation_iou(prediction, target)
        weights = prior_weights.to(losses).reshape(-1).clamp(0.05, 1.0)
        mean = (weights * losses).sum() / weights.sum().clamp_min(1e-8)
        cvar = weighted_cvar(losses, weights, config.cvar_quantile)
        centered = losses - mean
        variance = (weights * centered.square()).sum() / weights.sum().clamp_min(1e-8)
        n_eff = effective_sample_size(weights)
        uncertainty = torch.sqrt(variance / n_eff.clamp_min(1.0))
        boundary = (
            weights * per_observation_boundary_loss(prediction, target)
        ).sum() / weights.sum().clamp_min(1e-8)
        pose = torch.linalg.vector_norm(twists, dim=1).mean()
        temporal = temporal_pose_loss(poses)
        total = (
            mean
            + config.cvar_weight * cvar
            + config.uncertainty_weight * uncertainty
            + config.boundary_weight * boundary
            + config.pose_weight * pose
            + config.temporal_weight * temporal
            + config.geometry_weight * float(geometry_step_ratio)
        )
    return GuardedRisk(
        total=float(total), weighted_mean=float(mean), cvar=float(cvar),
        uncertainty=float(uncertainty), boundary=float(boundary),
        pose_correction=float(pose), temporal=float(temporal),
        geometry_step_ratio=float(geometry_step_ratio), effective_samples=float(n_eff),
    )


def guarded_risk_is_safe(
    old: GuardedRisk, new: GuardedRisk, config: RiskCalibratedConfig
) -> bool:
    return (
        new.total <= old.total - config.risk_margin
        and new.cvar <= old.cvar + config.tail_tolerance
    )


def select_publication_alpha(
    old: GuardedRisk,
    candidates: dict[float, GuardedRisk],
    config: RiskCalibratedConfig,
) -> tuple[float, GuardedRisk]:
    """Select the largest predeclared safe step, not the best observed step."""
    selected_alpha, selected_risk = 0.0, old
    for alpha in sorted(config.alpha_grid):
        if alpha <= 0.0 or alpha not in candidates:
            continue
        candidate = candidates[alpha]
        if guarded_risk_is_safe(old, candidate, config):
            selected_alpha, selected_risk = float(alpha), candidate
    return selected_alpha, selected_risk


def select_minimum_risk_alpha(
    old: GuardedRisk,
    candidates: dict[float, GuardedRisk],
    config: RiskCalibratedConfig,
) -> tuple[float, GuardedRisk]:
    """Select the safest minimum-risk step using a strictly future batch.

    Ties prefer the smaller geometry change.  Alpha zero is an exact rollback.
    """
    safe = [
        (float(alpha), candidate)
        for alpha, candidate in candidates.items()
        if alpha > 0.0 and guarded_risk_is_safe(old, candidate, config)
    ]
    if not safe:
        return 0.0, old
    return min(safe, key=lambda item: (item[1].total, item[0]))
