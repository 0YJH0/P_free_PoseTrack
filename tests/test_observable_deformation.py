from __future__ import annotations

import unittest
import torch

from optimization import vertex_gradient_consensus


class ObservableDeformationTest(unittest.TestCase):
    def test_consistent_multiview_vertex_is_selected(self) -> None:
        gradients = torch.zeros(3, 4, 3)
        gradients[:, 0, 0] = 1.0
        gradients[0, 1, 0], gradients[1, 1, 0] = 1.0, -1.0
        mask, agreement, support = vertex_gradient_consensus(
            gradients, agreement_threshold=0.6, keep_fraction=0.5
        )
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertEqual(int(support[0]), 3)
        self.assertAlmostEqual(float(agreement[0]), 1.0)

    def test_single_view_support_is_rejected(self) -> None:
        gradients = torch.zeros(2, 2, 3)
        gradients[0, 0, 2] = 2.0
        mask, _, support = vertex_gradient_consensus(gradients)
        self.assertFalse(mask.any())
        self.assertEqual(int(support[0]), 1)

    def test_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            vertex_gradient_consensus(torch.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
