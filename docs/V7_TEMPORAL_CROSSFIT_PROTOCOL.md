# V7 frozen temporal cross-fit shadow protocol

Frozen before any V7 run on 2026-10-06.

## Hypothesis and method

V6.1 showed that every same-batch proposal increased risk on the next batch.
V7 therefore changes proposal generation rather than weakening the guard.
Candidate observations are kept in temporal order.  The earlier half is
available to joint pose--geometry fitting; the later half receives exactly zero
geometry weight and has frozen pose correction.  The fixed alpha grid is first
screened on this later internal set.  Only internally safe alphas enter the
unchanged V6.1 shadow state.  A distinct next update batch then performs the
final prospective risk/CVaR test and selects its minimum-risk safe alpha.

This is a two-stage temporal cross-fit: internal forward validation followed by
external causal confirmation.  Alpha zero remains exact rollback.  The split is
50/50 (floor for fitting, remainder for validation); risk margin, CVaR tolerance,
alpha grid, optimizer, and maximum vertex step are inherited unchanged.

## Frozen development panel and advancement rule

- Objects: `ape`, `cube`
- Conditions: `a_regular`, `c_noisy`, `d_occlusion`
- Primary comparator: V5; baseline, V6, and V6.1 are descriptive
- One deterministic full sequence per cell
- Untouched objects remain sealed

Advance only if mean V7--V5 is at least +2 percentage points, the mean after
removing the largest gain is positive, no cell loses more than 2 points,
at least half of runs retain a nonzero internal alpha, at least one-third make a
nonzero prospective commit, and mean runtime is at most 1.5x V5.  Otherwise no
blind/full experiment is permitted.
