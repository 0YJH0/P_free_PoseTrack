"""Core losses and risk validation for training-free joint optimization.

The module deliberately has no dependency on BIT's renderer.  Callers provide
rendered silhouettes, keeping the geometry representation and renderer
replaceable and making the numerical parts independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class JointOptimizationConfig:
    outer_steps: int = 3
    pose_steps: int = 20
    geometry_steps: int = 60
    pose_lr: float = 2e-3
    geometry_lr: float = 1e-2
    max_rotation_deg: float = 5.0
    max_translation_ratio: float = 0.03
    pose_prior_weight: float = 0.05
    temporal_weight: float = 0.02
    boundary_weight: float = 0.10
    laplacian_weight: float = 0.10
    geometry_prior_weight: float = 0.02
    risk_pose_weight: float = 0.10
    risk_temporal_weight: float = 0.05
    risk_boundary_weight: float = 0.25
    risk_geometry_weight: float = 0.10
    risk_margin: float = 0.002
    tail_tolerance: float = 0.01


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map for twists ordered as [rx, ry, rz, tx, ty, tz]."""
    if twist.shape[-1] != 6:
        raise ValueError("SE(3) twists must have a final dimension of six")
    omega, velocity = twist[..., :3], twist[..., 3:]
    theta2 = (omega * omega).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    omega_hat = _skew(omega)
    omega_hat2 = omega_hat @ omega_hat

    small = theta2 < 1e-8
    a_exact = torch.sin(theta) / theta.clamp_min(1e-8)
    b_exact = (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-8)
    c_exact = (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(1e-8)
    a = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, a_exact)
    b = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, b_exact)
    c = torch.where(small, 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0, c_exact)

    eye = torch.eye(3, dtype=twist.dtype, device=twist.device).expand(omega_hat.shape)
    rotation = eye + a[..., None] * omega_hat + b[..., None] * omega_hat2
    left_jacobian = eye + b[..., None] * omega_hat + c[..., None] * omega_hat2
    translation = (left_jacobian @ velocity.unsqueeze(-1)).squeeze(-1)
    result = torch.eye(4, dtype=twist.dtype, device=twist.device).expand(
        twist.shape[:-1] + (4, 4)
    ).clone()
    result[..., :3, :3] = rotation
    result[..., :3, 3] = translation
    return result


class PoseDeltaParameterization(nn.Module):
    """Bounded SE(3) increments with explicit fixed-observation support."""

    def __init__(
        self,
        count: int,
        object_scale: float | torch.Tensor,
        max_rotation_deg: float = 5.0,
        max_translation_ratio: float = 0.03,
        optimizable: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(count, 6, device=device, dtype=dtype))
        mask = torch.ones(count, 1, device=device, dtype=dtype)
        if optimizable is not None:
            if optimizable.shape != (count,):
                raise ValueError("optimizable mask must contain one value per pose")
            mask = optimizable.to(device=device, dtype=dtype).reshape(count, 1)
        self.register_buffer("optimizable", mask)
        self.register_buffer(
            "object_scale", torch.as_tensor(object_scale, device=device, dtype=dtype).reshape(())
        )
        self.max_rotation_rad = float(max_rotation_deg) * torch.pi / 180.0
        self.max_translation_ratio = float(max_translation_ratio)

    def twists(self) -> torch.Tensor:
        bounded = torch.tanh(self.raw)
        rotation = bounded[:, :3] * self.max_rotation_rad
        translation = (
            bounded[:, 3:] * self.max_translation_ratio * self.object_scale.clamp_min(1e-8)
        )
        return torch.cat((rotation, translation), dim=1) * self.optimizable

    def forward(self, base_poses: torch.Tensor) -> torch.Tensor:
        return compose_pose_deltas(base_poses, self.twists())


def compose_pose_deltas(base_poses: torch.Tensor, twists: torch.Tensor) -> torch.Tensor:
    if base_poses.ndim != 3 or base_poses.shape[-2:] != (4, 4):
        raise ValueError("base poses must have shape [N, 4, 4]")
    if twists.shape != (len(base_poses), 6):
        raise ValueError("one six-dimensional twist is required per pose")
    return se3_exp(twists.to(base_poses)) @ base_poses


def per_observation_iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("silhouettes must have identical [N, H, W] shapes")
    dims = (1, 2)
    intersection = (prediction * target).sum(dims)
    union = (prediction + target - prediction * target).sum(dims).clamp_min(1e-6)
    return intersection / union


def weighted_iou_loss(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    iou = per_observation_iou(prediction, target)
    weights = weights.to(iou).reshape(-1)
    if weights.numel() != iou.numel():
        raise ValueError("one weight is required per observation")
    return ((1.0 - iou) * weights).sum() / weights.sum().clamp_min(1e-6)


def _sobel_edges(masks: torch.Tensor) -> torch.Tensor:
    kernel_x = masks.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4.0
    kernel_y = kernel_x.transpose(0, 1)
    kernels = torch.stack((kernel_x, kernel_y)).unsqueeze(1)
    gradient = F.conv2d(masks.unsqueeze(1), kernels, padding=1)
    return torch.sqrt((gradient * gradient).sum(1) + 1e-8)


def per_observation_boundary_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("silhouettes must have identical [N, H, W] shapes")
    difference = (_sobel_edges(prediction) - _sobel_edges(target)).abs()
    return difference.flatten(1).mean(1)


def boundary_alignment_loss(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    losses = per_observation_boundary_loss(prediction, target)
    weights = weights.to(losses).reshape(-1)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def pose_prior_loss(twists: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    squared = (twists * twists).sum(1)
    if weights is None:
        return squared.mean()
    weights = weights.to(squared).reshape(-1)
    return (squared * weights).sum() / weights.sum().clamp_min(1e-6)


def temporal_pose_loss(poses: torch.Tensor) -> torch.Tensor:
    """Robust second difference of translation and rotation matrices."""
    if len(poses) < 3:
        return poses.sum() * 0.0
    translations = poses[:, :3, 3]
    translation_acceleration = translations[2:] - 2 * translations[1:-1] + translations[:-2]
    rotations = poses[:, :3, :3]
    rotation_acceleration = rotations[2:] - 2 * rotations[1:-1] + rotations[:-2]
    residual = torch.cat(
        (translation_acceleration, rotation_acceleration.flatten(1)), dim=1
    )
    # Smooth L1 remains stable when genuine fast motion is present.
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="mean")


@dataclass(frozen=True)
class TrackingRisk:
    total: float
    mean_iou_loss: float
    tail_iou_loss: float
    boundary_loss: float
    pose_correction: float
    temporal_loss: float
    geometry_step_ratio: float


def evaluate_tracking_risk(
    prediction: torch.Tensor,
    target: torch.Tensor,
    twists: torch.Tensor,
    corrected_poses: torch.Tensor,
    geometry_step_ratio: float,
    config: JointOptimizationConfig,
) -> TrackingRisk:
    with torch.no_grad():
        iou_losses = 1.0 - per_observation_iou(prediction, target)
        boundary = per_observation_boundary_loss(prediction, target).mean()
        pose = torch.linalg.vector_norm(twists, dim=1).mean()
        temporal = temporal_pose_loss(corrected_poses)
        mean_iou = iou_losses.mean()
        quantile = torch.quantile(iou_losses, 0.9) if len(iou_losses) > 1 else iou_losses[0]
        # The caller orders the untouched causal holdout last. Protect both the
        # empirical tail and that held-out observation from regression.
        tail = torch.maximum(quantile, iou_losses[-1])
        total = (
            mean_iou
            + config.risk_boundary_weight * boundary
            + config.risk_pose_weight * pose
            + config.risk_temporal_weight * temporal
            + config.risk_geometry_weight * float(geometry_step_ratio)
        )
    return TrackingRisk(
        total=float(total), mean_iou_loss=float(mean_iou), tail_iou_loss=float(tail),
        boundary_loss=float(boundary), pose_correction=float(pose),
        temporal_loss=float(temporal), geometry_step_ratio=float(geometry_step_ratio),
    )


def risk_accepts_update(
    old: TrackingRisk, new: TrackingRisk, config: JointOptimizationConfig
) -> bool:
    return (
        new.total <= old.total - config.risk_margin
        and new.tail_iou_loss <= old.tail_iou_loss + config.tail_tolerance
    )
