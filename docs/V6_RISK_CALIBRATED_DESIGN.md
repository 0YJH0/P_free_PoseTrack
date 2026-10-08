# V6: uncertainty-calibrated pose–geometry optimization with soft publication

Status: architecture and experiment design, 2026-10-06. V5 remains frozen and
is not modified. V6 is a new method with a separate mode and results tree.

## 1. Motivation from the frozen V5 audit

The RBOT short panel showed a directional gain in 12/16 units, concentrated in
noise and occlusion, but also exposed an attribution problem. Several improved
units accepted no joint geometry proposal. V5 uses one fused confidence, fixed
loss weights, a hard maximum vertex step, and binary whole-mesh publication.
A useful update can therefore be discarded in full, and an observed gain
cannot always be attributed to joint optimization.

V6 addresses this mechanism rather than tuning V5 thresholds on RBOT.

## 2. Risk-Calibrated Soft Joint Optimization (RCS-JO)

For causal observations `i=1..N`, optimize pose increments `xi_i`, mesh
displacement `dV`, and observation log variances `s_i`:

```text
T'_i = exp(xi_i^) T_i
V'   = V + dV
```

Each residual vector contains silhouette IoU loss, boundary distance,
visible-area inconsistency, pose innovation, temporal acceleration, and
geometry-gradient disagreement. Residuals are standardized using only causal
history medians and MADs.

### 2.1 Heteroscedastic observation reliability

V6 replaces one heuristic weight with an uncertainty-aware objective:

```text
L_obs = sum_i [exp(-s_i) * rho(r_i) + s_i]
        + lambda_s * sum_i (s_i - s_prior_i)^2
```

`s_prior_i` is determined from the robust multi-signal score. The bounded
`s_i` and prior penalty prevent the trivial solution of declaring every hard
observation uncertain. No learned confidence network is required.

The optimizer alternates pose, uncertainty, and geometry blocks. The first
reference and a causal holdout remain fixed. Pose and mesh trust radii adapt
only from past accepted risk ratios.

### 2.2 Distributionally guarded tracking risk

Candidate quality is evaluated on untouched causal validation observations:

```text
R = weighted_mean(loss)
    + beta_tail * CVaR_0.9(loss)
    + beta_unc * sqrt(weighted_variance(loss) / N_eff)
    + pose, temporal, boundary, and geometry penalties.
```

The variance term guards small or low-effective-sample batches. Validation uses
frozen reliability priors, so optimized uncertainty cannot hide a bad update.

### 2.3 Continuous soft publication

V5 publishes all or none of a candidate. V6 evaluates the causal path

```text
V(alpha) = V_old + alpha * dV,  alpha in [0, 1].
```

The largest safe `alpha` whose guarded risk improves is published; otherwise
`alpha=0`. Pose corrections are used inside candidate optimization and cannot
rewrite past tracking outputs. The predeclared alpha grid is
`{0, 1/8, 1/4, 1/2, 3/4, 1}`. A continuous backtracking variant is secondary
and cannot be selected from blind-confirmation outcomes.

### 2.4 Contribution accounting

Every update logs uncertainty priors and estimates, effective sample size,
CVaR, pose-only/geometry-only/full risks, proposed and published vertex steps,
selected alpha, risk improvement, rejection reason, and counterfactual risks
at alpha zero and one. This directly separates confidence, pose, and geometry
contributions.

## 3. Implementation boundaries

Create without changing frozen V5 behavior:

- `optimization/risk_calibrated_joint.py` for robust scaling, bounded log
  variance, CVaR/effective-sample risk, and soft-publication selection;
- mode `risk_calibrated_soft` in `examples/run_example.py`;
- V6-only fields appended to `logging_utils/confidence_csv.py`;
- a separate `results/paper_v6/` tree and experiment arms;
- tests for uncertainty degeneracy, risk monotonicity, alpha selection, causal
  holdout isolation, and exact rollback at alpha zero.

V5 numerical code, parameters, configs, hashes, and results remain unchanged.

## 4. Leakage-controlled data split

The inspected RBOT short panel (`ape`, `cube`, `duck`, `squirrel`) becomes V6
development diagnostic data and cannot support independent V6 confirmation.

Development evidence consists of the historical four-object MOPED development
cohort and this RBOT panel, with leave-one-object-out reporting.

After V6 freeze, blind RBOT confirmation uses `bakingsoda`, `broccolisoup`,
`clown`, `glue`, and `koalacandy` under all four conditions: 20 untouched
tracking units. These passed reference-generation feasibility, but their
tracking metrics have not been inspected. Objects with systematic reference
OOM are fixed technical exclusions; no outcome-dependent replacement is
allowed.

## 5. Development experiments and ablations

All comparisons share references and initial geometry:

1. Baseline.
2. Frozen V5.
3. V6-U: uncertainty calibration only, hard V5 publication.
4. V6-S: fixed V5 confidence, guarded soft publication.
5. V6-R: uncertainty plus guarded risk, hard publication.
6. Full V6: uncertainty, guarded risk, and soft publication.

Primary development metric is macro tracking success. Secondary metrics are
adverse-condition success, ADD/AUC where available, worst-condition result,
published-update rate, effective sample size, and runtime.

Architecture-selection gates fixed before implementation results:

- full V6 improves macro success over V5 by at least 2 percentage points;
- regular-condition success decreases by no more than 1 point;
- at least half of units publish a nonzero geometry step;
- improvement remains positive after removing the largest-gain unit;
- runtime is no more than 1.5 times V5.

If full V6 fails, report it and use ablations only for diagnosis. Do not select
a component using blind objects.

## 6. Frozen confirmation plan

After architecture selection, freeze code hashes and parameters. On 20 blind
RBOT units run baseline, frozen V5, and frozen V6 from the same artifact. The
primary contrast is V6 versus baseline; V6 versus V5 is secondary. Begin with
one run. If the primary direction is favorable, add two repeats and fit a mixed
model with method fixed and object plus object-condition grouping factors.

Report paired effects and confidence intervals, condition interactions, run
variance, failures, alpha distribution, runtime, and exclusions. A blind null
result is final and cannot be used to revise V6.

## 7. Paper claim boundary

The intended contribution is a training-free uncertainty-calibrated optimizer
that jointly reasons about pose, geometry, and reliability, then continuously
controls how much geometry is safely published according to causal tail risk.
Overall accuracy claims require blind confirmation; development results support
mechanism analysis only.
