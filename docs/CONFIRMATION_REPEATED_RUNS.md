# Three-run independent confirmation

## Design

Baseline and frozen v4 were each run in three independent processes on all 33
sequences from the six confirmation objects, yielding 198 complete
observations. Every replicate reused the same per-sequence initial mesh and all
hash checks passed. Two baseline attempts terminated because the non-atomic
baseline mesh path exposed an empty model to C++; both were rerun unchanged and
the failed attempts were retained in their logs but excluded because they had
no evaluable endpoint.

## Mean performance

The mixed-effects model, with method as a fixed effect and object/sequence
random intercepts, estimates the v4-minus-baseline ADD effect as **-1.8075 mm**
(95% normal-approximation CI **[-5.1263, 1.5112]**, p=0.2858). The interval
crosses both zero and the prespecified -2 mm equivalence boundary. Thus the
three-run confirmation supports neither ADD superiority nor ±2 mm equivalence.

The corresponding AUC effect is +2.4016 (95% CI [0.0504, 4.7528], unadjusted
p=0.0453). AUC is secondary; Holm correction across ADD and AUC gives p=0.0906,
so it is suggestive rather than confirmatory.

## Stability

Mean within-sequence ADD SD decreased from 4.2131 mm for baseline to 2.9753 mm
for v4. The pooled variance ratio was 2.0737 in favor of v4, but its
object/sequence hierarchical-bootstrap 95% interval was [0.2362, 17.2409]. V4
had lower SD on 13 sequences, higher SD on 11, and tied on 9; one-sided paired
Wilcoxon p=0.3524 and sign-test p=0.4194. Mean run range decreased from 7.4636
to 5.3611 mm, while the 95th-percentile range decreased from 40.1455 to
23.3178 mm.

These point estimates favor v4, but the variance interval crosses one and both
paired tests are non-significant. The strong stability gain observed on the
four-object development cohort therefore **does not statistically replicate
across the six confirmation objects**. Instability is highly sequence-specific;
for example, v4 `pouch/00` spans 3.7749–72.8432 mm.

## Paper-level conclusion

Across independent objects and repeated runs, v4 has a favorable but uncertain
ADD point estimate and a nominal AUC gain. It cannot currently support claims
of universal accuracy superiority, equivalence, or general stability
improvement. The defensible contribution is a confidence-aware rollback design
with strong development-cohort repeatability, promising average external
performance, and clearly characterized object-dependent failure modes.

Machine-readable results are
`results/paper_moped/analysis/confirmation_mixed_effects_three_replicates.json`
and `results/paper_moped/analysis/confirmation_repeated_runs.csv`.
