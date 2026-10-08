from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from confidence import (
    GradientConsistency,
    MultiSignalConfidence,
    apply_mask_confidence_gate,
    binary_mask_iou,
)
from logging_utils import ConfidenceCsvLogger, ConfidenceRow


class MaskConfidenceTest(unittest.TestCase):
    def test_binary_mask_iou_known_cases(self) -> None:
        segmentation = torch.tensor(
            [[[1, 1], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]],
            dtype=torch.float32,
        )
        rendered = torch.tensor(
            [[[1, 1], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 0]]],
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.equal(binary_mask_iou(segmentation, rendered), torch.tensor([1.0, 0.0, 0.0]))
        )

    def test_binary_mask_iou_partial_overlap(self) -> None:
        segmentation = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.float32)
        rendered = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.float32)
        self.assertAlmostEqual(binary_mask_iou(segmentation, rendered).item(), 1.0 / 3.0)

    def test_gate_is_inclusive_and_can_be_disabled(self) -> None:
        mask = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.float32)
        enabled = apply_mask_confidence_gate(
            mask, mask, enabled=True, confidence_threshold=1.0
        )
        disabled = apply_mask_confidence_gate(
            mask,
            torch.zeros_like(mask),
            enabled=False,
            confidence_threshold=1.0,
        )
        self.assertEqual(enabled.accepted.tolist(), [True])
        self.assertEqual(disabled.accepted.tolist(), [True])

    def test_shape_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            binary_mask_iou(torch.zeros(1, 2, 2), torch.zeros(2, 2, 2))

    def test_confidence_csv_logger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confidence.csv"
            logger = ConfidenceCsvLogger(path)
            logger.append([ConfidenceRow(7, 0.75, True, True, True)])
            logger.append([ConfidenceRow(9, 0.25, True, False, False)])
            with path.open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual([record["frame_id"] for record in records], ["7", "9"])
            self.assertEqual(len(records), 2)
            logger.reset()
            self.assertFalse(path.exists())

    def test_multisignal_confidence_penalizes_pose_and_visibility(self) -> None:
        evaluator = MultiSignalConfidence(adaptive=False)
        segmentation = torch.tensor(
            [[[1, 1], [0, 0]], [[1, 0], [0, 0]]], dtype=torch.float32
        )
        rendered = segmentation.clone()
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[1, 0, 3] = 1.0
        result = evaluator.evaluate(segmentation, rendered, poses, [0, 1])
        self.assertEqual(result.confidence.shape, (2,))
        self.assertGreater(result.pose_score[0].item(), result.pose_score[1].item())
        self.assertTrue(torch.all((result.weight >= 0) & (result.weight <= 1)))

    def test_adaptive_threshold_is_causal_and_bounded(self) -> None:
        evaluator = MultiSignalConfidence(
            base_threshold=0.65, adaptive=True, threshold_min=0.5, threshold_max=0.8
        )
        masks = torch.ones(6, 4, 4)
        poses = torch.eye(4).repeat(6, 1, 1)
        result = evaluator.evaluate(masks, masks, poses, list(range(6)))
        for threshold in result.threshold[:5]:
            self.assertAlmostEqual(threshold.item(), 0.65, places=6)
        self.assertGreaterEqual(result.threshold[5].item(), 0.5 - 1e-6)
        self.assertLessEqual(result.threshold[5].item(), 0.8 + 1e-6)

    def test_gradient_consistency_downweights_conflicting_update(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        losses = torch.stack(
            ((parameter - 1.0) ** 2, (parameter - 1.0) ** 2, (parameter + 1.0) ** 2)
        )
        result = GradientConsistency(alpha=0.5).evaluate(
            losses,
            [parameter],
            reference_count=1,
            prior_candidate_weights=torch.ones(2),
        )
        self.assertAlmostEqual(result.scores[0].item(), 1.0, places=6)
        self.assertAlmostEqual(result.scores[1].item(), 0.0, places=6)
        self.assertGreater(result.fused_weights[0], result.fused_weights[1])


if __name__ == "__main__":
    unittest.main()
