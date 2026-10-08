from __future__ import annotations

import math
import unittest

import torch

from optimization import (
    JointOptimizationConfig,
    PoseDeltaParameterization,
    TrackingRisk,
    boundary_alignment_loss,
    compose_pose_deltas,
    risk_accepts_update,
    se3_exp,
    temporal_pose_loss,
    weighted_iou_loss,
)


class JointPoseGeometryTest(unittest.TestCase):
    def test_se3_identity_and_translation(self) -> None:
        twists = torch.zeros(2, 6)
        twists[1, 3:] = torch.tensor([1.0, 2.0, 3.0])
        transforms = se3_exp(twists)
        self.assertTrue(torch.allclose(transforms[0], torch.eye(4)))
        self.assertTrue(torch.allclose(transforms[1, :3, 3], twists[1, 3:]))

    def test_se3_rotation_is_orthonormal_and_differentiable(self) -> None:
        twist = torch.tensor([[0.2, -0.1, 0.3, 0.0, 0.0, 0.0]], requires_grad=True)
        transform = se3_exp(twist)
        rotation = transform[0, :3, :3]
        self.assertTrue(torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-5))
        transform.sum().backward()
        self.assertTrue(torch.isfinite(twist.grad).all())

    def test_pose_parameterization_anchors_and_bounds(self) -> None:
        poses = torch.eye(4).repeat(2, 1, 1)
        parameterization = PoseDeltaParameterization(
            2, object_scale=2.0, max_rotation_deg=5.0,
            max_translation_ratio=0.03, optimizable=torch.tensor([False, True]),
        )
        with torch.no_grad():
            parameterization.raw.fill_(100.0)
        twists = parameterization.twists()
        self.assertTrue(torch.equal(twists[0], torch.zeros(6)))
        self.assertLessEqual(float(twists[1, :3].abs().max()), math.radians(5.0) + 1e-6)
        self.assertLessEqual(float(twists[1, 3:].abs().max()), 0.06 + 1e-6)
        self.assertTrue(torch.allclose(parameterization(poses)[0], poses[0]))

    def test_composition_uses_left_increment(self) -> None:
        poses = torch.eye(4).repeat(1, 1, 1)
        poses[0, 0, 3] = 2.0
        twists = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        corrected = compose_pose_deltas(poses, twists)
        self.assertAlmostEqual(float(corrected[0, 0, 3]), 3.0, places=6)

    def test_losses_prefer_matching_silhouettes(self) -> None:
        target = torch.zeros(2, 8, 8)
        target[:, 2:6, 2:6] = 1.0
        weights = torch.tensor([1.0, 0.5])
        perfect_iou = weighted_iou_loss(target, target, weights)
        empty_iou = weighted_iou_loss(torch.zeros_like(target), target, weights)
        perfect_boundary = boundary_alignment_loss(target, target, weights)
        empty_boundary = boundary_alignment_loss(torch.zeros_like(target), target, weights)
        self.assertLess(float(perfect_iou), float(empty_iou))
        self.assertLess(float(perfect_boundary), float(empty_boundary))

    def test_temporal_loss_detects_acceleration(self) -> None:
        smooth = torch.eye(4).repeat(3, 1, 1)
        smooth[:, 0, 3] = torch.tensor([0.0, 1.0, 2.0])
        abrupt = smooth.clone()
        abrupt[2, 0, 3] = 4.0
        self.assertLess(float(temporal_pose_loss(smooth)), float(temporal_pose_loss(abrupt)))

    def test_risk_acceptance_requires_mean_gain_and_safe_tail(self) -> None:
        config = JointOptimizationConfig(risk_margin=0.01, tail_tolerance=0.02)
        old = TrackingRisk(1.0, 0.4, 0.5, 0.1, 0.0, 0.0, 0.0)
        safe = TrackingRisk(0.9, 0.3, 0.51, 0.1, 0.0, 0.0, 0.0)
        unsafe = TrackingRisk(0.8, 0.3, 0.6, 0.1, 0.0, 0.0, 0.0)
        self.assertTrue(risk_accepts_update(old, safe, config))
        self.assertFalse(risk_accepts_update(old, unsafe, config))


if __name__ == "__main__":
    unittest.main()
