# V6 initial paired-pilot results

Date: 2026-10-06. This is a six-unit development pilot, not confirmation.
Baseline and frozen V5 values were reused from the paired RBOT short panel; V6
used the same frozen reference data and initial-mesh hashes.

| Condition | Object | Baseline | V5 | V6 | V6−baseline | V6−V5 | V6/V5 runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| regular | ape | 79.52% | 79.22% | 79.22% | -0.30 pp | 0.00 pp | 1.15× |
| regular | cube | 79.62% | 75.82% | 81.62% | +2.00 pp | +5.79 pp | 6.22× |
| noisy | ape | 60.94% | 71.43% | 61.44% | +0.50 pp | -9.99 pp | 3.58× |
| noisy | cube | 17.48% | 17.88% | 17.88% | +0.40 pp | 0.00 pp | 1.46× |
| occlusion | ape | 54.65% | 68.53% | 52.15% | -2.50 pp | -16.38 pp | 10.14× |
| occlusion | cube | 16.38% | 66.73% | 66.73% | +50.35 pp | 0.00 pp | 3.64× |

Aggregate results:

- V6 versus baseline: mean +8.41 pp, median +0.45 pp, 4/6 wins.
- Removing the largest-gain unit: mean +0.02 pp versus baseline.
- V6 versus V5: mean -3.43 pp, median 0.00 pp, 1 win, 2 losses, 3 ties.
- Removing the largest V6-over-V5 gain: mean -5.27 pp.
- Aggregate runtime ratio relative to V5: 4.36×.

The predeclared architecture gates therefore fail: V6 does not improve V5 by
2 pp, the leave-largest-gain-out effect is negative, and runtime exceeds 1.5×.
The current implementation must not proceed to blind or full comparison.

## Mechanistic finding

Soft publication was active, selecting nonzero alpha in 4/6 units. However,
small local guarded-risk reductions did not predict downstream tracking:

- noisy/ape accepted alpha 1.0 once but lost 9.99 pp versus V5;
- occlusion/ape accepted alpha 1.0 and 0.75 but lost 16.38 pp versus V5;
- the largest observed local risk improvements were only a few thousandths.

The failure is therefore not merely a threshold issue. Immediate publication
uses a short same-batch surrogate that can approve geometry harmful to future
frames. Choosing the largest safe alpha further amplifies this mismatch.

## Required redesign before another pilot

The next candidate should keep uncertainty calibration but replace immediate
publication with a causal shadow mesh:

1. optimize a candidate and keep it out of the live tracker;
2. evaluate old and shadow meshes on the next causal keyframe batch;
3. commit only the risk-minimizing alpha after prospective evidence;
4. expire the candidate if evidence is insufficient or tail risk worsens;
5. cache shared renders and use successive halving over alpha to control cost.

This prospective delayed-commit mechanism directly targets final tracking risk
and is more defensible than lowering the current risk margin. It requires a new
pilot and is not yet the final paper architecture.
