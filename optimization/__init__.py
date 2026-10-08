"""Training-free joint pose/geometry optimization primitives."""

from .joint_pose_geometry import (
    JointOptimizationConfig,
    PoseDeltaParameterization,
    TrackingRisk,
    boundary_alignment_loss,
    compose_pose_deltas,
    evaluate_tracking_risk,
    pose_prior_loss,
    risk_accepts_update,
    se3_exp,
    temporal_pose_loss,
    weighted_iou_loss,
)
from .risk_calibrated_joint import (
    BoundedLogVariance,
    GuardedRisk,
    RiskCalibratedConfig,
    effective_sample_size,
    evaluate_guarded_risk,
    guarded_risk_is_safe,
    heteroscedastic_silhouette_loss,
    select_publication_alpha,
    select_minimum_risk_alpha,
    weighted_cvar,
)
from .observable_deformation import vertex_gradient_consensus

__all__ = [
    "JointOptimizationConfig", "PoseDeltaParameterization", "TrackingRisk",
    "boundary_alignment_loss", "compose_pose_deltas", "evaluate_tracking_risk",
    "pose_prior_loss", "risk_accepts_update", "se3_exp", "temporal_pose_loss",
    "weighted_iou_loss",
    "BoundedLogVariance", "GuardedRisk", "RiskCalibratedConfig",
    "effective_sample_size", "evaluate_guarded_risk", "guarded_risk_is_safe",
    "heteroscedastic_silhouette_loss", "select_publication_alpha",
    "select_minimum_risk_alpha",
    "weighted_cvar",
    "vertex_gradient_consensus",
]
