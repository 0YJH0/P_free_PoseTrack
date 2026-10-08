# Frozen RBOT short-panel protocol

Freeze date: 2026-10-06 (Asia/Shanghai), before inspecting any short-panel
aggregate result.

The original 72-unit run was stopped during reference generation after
systematic host-memory failures on high-histogram objects (for example,
`benchviseblue` requested about 24 GB in one allocation). Partial full-cohort
outputs are retained as technical diagnostics and are not confirmation results.

## Frozen panel

- Objects: `ape`, `cube`, `duck`, `squirrel`.
- Conditions: `a_regular`, `b_dynamiclight`, `c_noisy`, `d_occlusion`.
- Methods: frozen baseline and frozen v5 (`joint_risk`).
- One independent run, all 1,001 frames per object-condition unit.
- Pairing: identical frozen references and identical initial-mesh SHA-256 for
  baseline and v5 within every unit.
- Primary metric: RBOT 5-degree/5-cm success rate.
- Secondary metrics: mean rotation error, mean translation error, failures,
  runtime, candidate updates, and accepted joint updates.

The panel contains 16 paired units and is an exploratory resource-screening
experiment, not a substitute for the complete external confirmation.

## Frozen interpretation rule

- Promising generalization: v5 is higher in at least 10 of 16 units and its
  macro-average success-rate change is greater than +1 percentage point.
- Predominantly null/degrading: v5 wins at most 7 units and its macro-average
  change is non-positive. Expensive repeats are then stopped and development
  moves to v6 on non-RBOT development data.
- Conditional robustness: the overall rule is not met, but the mean change is
  positive and directionally consistent within one or more adverse conditions
  (`b_dynamiclight`, `c_noisy`, or `d_occlusion`). The claim is restricted to
  those conditions and requires later independent confirmation.
- Otherwise the outcome is inconclusive; report effect sizes and do not tune v5
  on these RBOT outcomes.

No v5 threshold, optimizer setting, risk weight, acceptance rule, or reference
selection setting may be changed using this panel.
