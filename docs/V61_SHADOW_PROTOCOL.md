# V6.1 frozen causal-shadow pilot protocol

Frozen before any V6.1 run on 2026-10-06.

## Method

`risk_calibrated_shadow` never evaluates and publishes a geometry proposal on
the observations that optimized it.  Online update 1 creates a shadow proposal
and keeps the live mesh unchanged.  Online update 2 compares the old mesh and
the predeclared alpha grid on the newly arrived, causally future batch.  Among
steps passing the fixed total-risk and CVaR guards, it publishes the
minimum-risk step; ties select the smaller step.  Otherwise alpha zero performs
an exact rollback.  No future pose correction is fitted during this comparison.

## Frozen development panel

- Objects: `ape`, `cube`
- Conditions: `a_regular`, `c_noisy`, `d_occlusion`
- Arms: frozen-initial-geometry `baseline`, `v5`, `v61`
- One deterministic full-sequence run per cell for architecture screening
- Existing baseline/V5 artifacts may be reused when hashes and configs match

The untouched objects `bakingsoda`, `broccolisoup`, `clown`, `glue`, and
`koalacandy` remain unavailable for design or threshold selection.

## Fixed gates for advancing to blind confirmation

Relative to V5, V6.1 must have mean success-rate gain at least 2 percentage
points, remain positive after removing its largest gain, have no cell worse by
more than 2 points, and publish a nonzero prospective alpha in at least half of
eligible runs.  Runtime is reported separately and must not exceed 1.5x V5 for
the panel mean.  Failure stops repeated/full experiments and triggers redesign.
