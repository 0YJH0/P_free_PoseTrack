# Confidence-gating paired experiment results

## Locked protocol

Threshold selection used sequence `00` from each of four MOPED objects. The
held-out test used the remaining 17 technically valid sequences. Baseline and
confidence-gated runs reused byte-identical initial geometry per sequence.
ADD (mm; lower is better) was the primary endpoint and ADD AUC (higher is
better) was secondary.

`black_drill` was excluded as a complete object before held-out testing after
the gated validation process repeatedly failed with C++ `std::bad_alloc` at
both 128 and 64 histogram caps. No threshold-specific successful subset was
retained. This limits generalization to the four evaluated objects.

## Validation threshold selection

| Threshold | Macro paired ADD delta (mm) | Macro paired AUC delta |
|---:|---:|---:|
| 0.70 | +0.2955 | -0.2901 |
| 0.75 | +0.4754 | -0.4601 |
| **0.80** | **-2.5895** | **+2.5737** |
| 0.85 | -1.5331 | +1.5389 |

Threshold 0.80 was locked before test aggregation.

## Held-out test result

Across 17 sequences from four objects, baseline mean ADD was 21.0505 mm and
the confidence-gated mean was 22.4223 mm. The sequence-weighted difference was
+1.3718 mm (worse), while the prespecified object-macro difference was +1.5112
mm with a 95% hierarchical-bootstrap interval of [-2.6852, 8.9607] mm.
Wilcoxon p=0.9032 and the exact sign-test p=0.7905 (6 improvements, 8
degradations, 3 ties).

Mean AUC changed from 78.9408 to 77.7113. The object-macro AUC difference was
-1.3943 with a 95% hierarchical-bootstrap interval of [-8.9063, 2.7275]. The
mean accepted-candidate rate was 96.67%.

## Statistical conclusion

The held-out experiment does **not** support an improvement from the current
hard confidence gate. The point estimates mildly favor the baseline, but the
intervals include zero and the paired tests are not significant. This is not
evidence that the entire tracking architecture is poor; it specifically shows
that the current mask-IoU hard gate rejects too few candidates and does not
produce a robust gain under this resource-constrained four-object protocol.
Further method development should be evaluated with a new validation/test
cycle rather than tuning against these held-out outcomes.

Machine-readable results are in
`results/paper_moped/analysis/selected_threshold.json` and
`results/paper_moped/analysis/test_summary_gate_t080.json`.
