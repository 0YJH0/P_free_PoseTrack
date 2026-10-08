# Three-replicate stability and mixed-effects analysis

## Protocol

Baseline and v4 were each run in three independent processes on all 21
included sequences, producing 126 observations. Frozen initial meshes, seeds,
input data, and resource settings were identical. Repetition measures
implementation and closed-loop variability. ADD was primary; an ADD
equivalence margin of ±2 mm was fixed before the three-replicate analysis.

## Mean performance and equivalence

The Gaussian mixed model contains method as a fixed effect and random
intercepts for object and sequence nested within object.

| Outcome | v4 − baseline | 95% CI | p-value |
|---|---:|---:|---:|
| ADD (mm) | -0.1812 | [-1.4834, 1.1210] | 0.7851 |
| AUC | +0.3915 | [-0.8575, 1.6406] | 0.5390 |

The ADD interval lies wholly inside ±2 mm, supporting equivalent mean accuracy
under the prespecified practical margin. It does not support superior mean
accuracy.

## Repeatability

Mean within-sequence ADD SD fell from 2.1009 mm for baseline to 0.5518 mm for
v4; median SD fell from 0.2754 to 0.0640 mm. The pooled within-sequence variance
ratio is 6.03 in favor of v4. V4 had lower ADD SD on 16 sequences, higher SD on
2, and tied on 3 (one-sided paired Wilcoxon p=0.00385; sign-test p=0.000656).

Mean run range fell from 3.7758 to 0.9770 mm, and the 95th-percentile range from
14.6852 to 1.6225 mm. AUC repeatability gives the same conclusion: 14 lower-SD,
2 higher-SD, and 5 tied sequences (Wilcoxon p=0.00655).

The object-hierarchical bootstrap interval for the pooled ADD variance ratio is
very wide and crosses one because only four objects are available. Thus the
sequence-level stability result is strong within this benchmark, while
cross-object generalization remains limited and should be confirmed on another
dataset.

## Supported paper claim

Under the four-object resource-constrained MOPED protocol, v4 preserves mean
tracking accuracy within ±2 mm while significantly reducing sequence-level
run-to-run variability. The defensible contribution is reliable and
repeatable online geometry adaptation, not higher average accuracy.

Machine-readable output is in
`results/paper_moped/analysis/mixed_effects_three_replicates.json`; all 126
observations are in `results/paper_moped/analysis/repeated_runs.csv`.
