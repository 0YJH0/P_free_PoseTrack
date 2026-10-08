# V5 frozen development experiment

## Protocol

V5 (`joint_risk`) was run with its implementation-time default parameters on
all 21 technically valid sequences from the four-object development cohort.
No value from the previously inspected six-object confirmation cohort was used
for parameter selection. Baseline, V4, and V5 were each evaluated in three
independent processes, yielding 189 observations. All initial-geometry hashes
matched within every sequence.

This remains an exploratory development experiment, not independent
confirmation.

## Mean performance

| Method | Mean ADD (mm) | Mean AUC | Mean runtime (s) |
|---|---:|---:|---:|
| Baseline | 23.0195 | 76.9665 | 70.47 |
| V4 | 22.8383 | 77.3581 | 71.27 |
| V5 | **21.6128** | **78.5117** | 87.14 |

The mixed-effects V5-minus-baseline ADD estimate was **-1.4068 mm** (95% CI
[-2.7402, -0.0733], p=0.0387). V5-minus-V4 was **-1.2256 mm** (95% CI
[-2.0125, -0.4387], p=0.00227). Corresponding AUC estimates were +1.5451 and
+1.1536. ADD versus baseline is the designated primary development contrast;
V5-versus-V4 and AUC results are secondary and descriptive.

The result is evidence that joint optimization is worth independent testing,
not a paper-level superiority claim: architecture design and evaluation use
the same four-object development cohort, and only four object clusters are
available.

## Repeatability

| Method | Mean within-sequence ADD SD | Median SD | Mean run range | Maximum range |
|---|---:|---:|---:|---:|
| Baseline | 2.1009 | 0.2754 | 3.7758 | 35.5827 |
| V4 | 0.5518 | 0.0640 | 0.9770 | 16.2077 |
| V5 | **0.3409** | **0.0000** | **0.6161** | **5.7931** |

V5 has the best point estimates, but paired stability tests are not
significant. Relative to baseline, V5 has lower SD on 11 sequences, higher on
7, and ties on 3 (one-sided Wilcoxon p=0.1061). Relative to V4 the counts are
9/6/6 (p=0.4670). Object-hierarchical variance-ratio intervals are extremely
wide and cross one. Stability improvement must therefore be described as a
trend, not a confirmed effect.

## `toy_plane/03`

| Method | Replicate 0 | Replicate 1 | Replicate 2 | Mean ADD |
|---|---:|---:|---:|---:|
| Baseline | 12.9501 | 45.1770 | 48.5328 | 35.5533 |
| V4 | 30.9997 | 30.9997 | 30.9997 | 30.9997 |
| V5 | 30.9997 | 30.9997 | 30.9997 | 30.9997 |

V5 does not repair the V4 mechanism on this sequence. Its two proposed
updates had normalized maximum vertex steps 0.2091 and 0.2133, both exceeding
the frozen 0.15 safety bound, so both were rolled back. The second proposal did
reduce aggregate tracking risk (0.4554 to 0.4353) and guarded tail loss (0.4735
to 0.4318), but could not pass the independent geometry safety constraint.

The case exposes a clear next hypothesis: a scalar global maximum-step limit
may be too coarse for joint optimization. This observation must not be used to
tune and re-report the same cohort. Any replacement, such as spatially adaptive
trust regions or step projection, requires a newly declared development and
confirmation cycle.

## Computational cost

V5 increases mean runtime by 21.5% relative to baseline and 22.3% relative to
V4. The accuracy point estimate must therefore be presented together with this
cost.

Machine-readable results:

- `results/paper_moped/analysis/v5_development_summary.json`
- `results/paper_moped/analysis/v5_development_pairs.csv`
- `results/paper_moped/analysis/v5_three_replicate_mixed_effects.json`
- `results/paper_moped/analysis/v5_three_method_repeated_runs.csv`

