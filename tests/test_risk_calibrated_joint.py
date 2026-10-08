from __future__ import annotations

import unittest

import torch

from optimization import (
    BoundedLogVariance,
    GuardedRisk,
    RiskCalibratedConfig,
    effective_sample_size,
    guarded_risk_is_safe,
    heteroscedastic_silhouette_loss,
    select_publication_alpha,
    select_minimum_risk_alpha,
    weighted_cvar,
)


def risk(total: float, cvar: float | None = None) -> GuardedRisk:
    value = total if cvar is None else cvar
    return GuardedRisk(total, total, value, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0)


class RiskCalibratedJointTest(unittest.TestCase):
    def test_log_variance_is_bounded_and_anchorable(self) -> None:
        module = BoundedLogVariance(
            torch.tensor([1.0, 0.25]), radius=1.5,
            optimizable=torch.tensor([False, True]),
        )
        with torch.no_grad():
            module.raw.fill_(100.0)
        values = module()
        self.assertAlmostEqual(float(values[0]), 0.0, places=6)
        self.assertLessEqual(float(values[1] - module.prior[1]), 1.5 + 1e-6)
        self.assertGreater(float(module.prior_penalty()), 0.0)

    def test_heteroscedastic_loss_is_differentiable(self) -> None:
        target = torch.zeros(2, 8, 8)
        target[:, 2:6, 2:6] = 1.0
        prediction = target.clone().requires_grad_(True)
        log_variance = torch.zeros(2, requires_grad=True)
        loss = heteroscedastic_silhouette_loss(prediction, target, log_variance)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.isfinite(log_variance.grad).all())

    def test_effective_sample_size_penalizes_concentration(self) -> None:
        uniform = effective_sample_size(torch.ones(4))
        concentrated = effective_sample_size(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(uniform), 4.0)
        self.assertAlmostEqual(float(concentrated), 1.0)

    def test_weighted_cvar_tracks_worst_tail(self) -> None:
        losses = torch.tensor([0.1, 0.2, 0.3, 0.9])
        value = weighted_cvar(losses, torch.ones(4), 0.75)
        self.assertAlmostEqual(float(value), 0.9, places=6)

    def test_guard_rejects_tail_regression(self) -> None:
        config = RiskCalibratedConfig(risk_margin=0.01, tail_tolerance=0.02)
        old = risk(1.0, 0.5)
        self.assertTrue(guarded_risk_is_safe(old, risk(0.9, 0.51), config))
        self.assertFalse(guarded_risk_is_safe(old, risk(0.8, 0.6), config))

    def test_soft_publication_selects_largest_safe_alpha(self) -> None:
        config = RiskCalibratedConfig(risk_margin=0.01, tail_tolerance=0.02)
        old = risk(1.0, 0.5)
        alpha, selected = select_publication_alpha(
            old,
            {0.125: risk(0.95, 0.50), 0.5: risk(0.90, 0.51), 1.0: risk(0.80, 0.60)},
            config,
        )
        self.assertEqual(alpha, 0.5)
        self.assertEqual(selected.total, 0.90)

    def test_alpha_zero_is_exact_rollback(self) -> None:
        config = RiskCalibratedConfig(risk_margin=0.01)
        old = risk(1.0)
        alpha, selected = select_publication_alpha(old, {1.0: risk(1.1)}, config)
        self.assertEqual(alpha, 0.0)
        self.assertIs(selected, old)

    def test_shadow_selects_future_minimum_risk_not_largest_step(self) -> None:
        config = RiskCalibratedConfig(risk_margin=0.01, tail_tolerance=0.02)
        old = risk(1.0, 0.5)
        alpha, selected = select_minimum_risk_alpha(
            old,
            {0.25: risk(0.80, 0.50), 0.5: risk(0.85, 0.49), 1.0: risk(0.70, 0.60)},
            config,
        )
        self.assertEqual(alpha, 0.25)
        self.assertEqual(selected.total, 0.80)

    def test_shadow_tie_prefers_smaller_step(self) -> None:
        config = RiskCalibratedConfig(risk_margin=0.01)
        old = risk(1.0)
        alpha, _ = select_minimum_risk_alpha(
            old, {0.5: risk(0.8), 0.125: risk(0.8)}, config
        )
        self.assertEqual(alpha, 0.125)


if __name__ == "__main__":
    unittest.main()
