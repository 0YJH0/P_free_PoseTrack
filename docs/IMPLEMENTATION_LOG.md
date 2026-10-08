# Implementation Log

## 2026-10-06 — V6 initial pilot rejected

- Implemented and tested heteroscedastic confidence, CVaR/effective-sample
  risk, and guarded soft geometry publication.
- Six paired RBOT development units completed without technical failures.
- V6 lost 3.43 percentage points on average to frozen V5 and cost 4.36× its
  runtime; see `docs/V6_PILOT_RESULTS.md`.
- Rejected this V6 form before blind testing. The next design must use
  prospective shadow-mesh validation rather than same-batch immediate commit.

## 2026-10-06 — V6 architecture design

- Kept V5 frozen and specified V6 as a separate risk-calibrated soft joint
  optimizer; see `docs/V6_RISK_CALIBRATED_DESIGN.md`.
- Reclassified the inspected four-object RBOT panel as V6 development data.
- Reserved five technically feasible, tracking-uninspected RBOT objects for a
  later blind comparison of baseline, frozen V5, and frozen V6.
- Predeclared component ablations, selection gates, contribution accounting,
  and the mixed-effects confirmation protocol before implementation.

## 2026-09-30 — Official baseline import and analysis

- Changed files: `docs/BIT_CODEBASE_ANALYSIS.md`,
  `docs/MINIMAL_MODIFICATION_PLAN.md`, `docs/METHOD_DESIGN.md`,
  `docs/EXPERIMENT_PLAN.md`, `docs/IMPLEMENTATION_LOG.md`.
- Purpose: map the official BIT implementation before algorithmic changes and
  define the smallest correct Render-Mask Confidence Gating experiment.
- Implementation: documented Python/C++ entry points, dataset loaders,
  segmentation, pose tracking, keyframe selection, geometry update/loss,
  rendering, evaluation, process-boundary data structures, and integration
  risks.
- Potential risk: analysis is static; runtime behavior still depends on CUDA,
  Qt/OpenGL, C++ dependencies, SAM checkpoint, and benchmark data.
- Test result: documentation paths and cited symbols were checked against
  official commit `6dc95feb51d0b96d96adc009d408fc4948b06fdc`.

No tracking result or benchmark metric has been generated.

## 2026-09-30 — Render-mask confidence gating v1

- Changed files: `confidence/*`, `logging_utils/*`,
  `examples/communication.py`, `examples/run_example.py`,
  `config/config_generator.py`, experiment configs, and CPU tests.
- Purpose: implement configurable hard gating using pre-update rendered-mask
  IoU while preserving accept-all baseline behavior.
- Implementation: added per-observation IoU, hard gate, frame-ID propagation,
  CSV logging, headless visualization control, consume-once disk communication,
  and Python/C++ process lifecycle handling.
- Potential risk: BIT's differentiable initial geometry is not fully
  deterministic; one integration run produced no valid online keyframes after
  initialization. Full baseline/confidence comparison therefore requires
  repeated controlled runs and artifact preservation.
- Test result: five CPU unit tests pass. CUDA/Qt-offscreen integration produced
  three mesh updates and completed MOPED `black_drill/evaluation/00`. The run
  logged six real online candidates; all six passed threshold 0.5. Official
  evaluation output was ADD 20.5422 mm and reported ADD AUC 79.4459. These are
  single-run engineering results, not aggregate or paper claims.

## 2026-09-30 — Baseline single-sequence verification

- Config: `config/moped/black_drill/evaluation/00_baseline.yml` with confidence
  gating disabled.
- Test result: the C++ tracker completed the sequence and reported ADD 15.9126
  mm and ADD AUC 84.0946. Two geometry updates completed; a final partial set
  of five candidates did not reach the configured six-frame update batch.
- Comparison warning: the baseline and confidence runs followed different
  keyframe/update trajectories due to differentiable-renderer randomness, so
  their single-run metrics are not a controlled ablation and must not be used
  as evidence of improvement or degradation.

## 2026-09-30 — Frozen-initial-geometry paired experiment

- Both arms used the byte-identical initial geometry artifact
  `artifacts/initial_geometry/black_drill_eval00_seed42.obj` (SHA-256
  `55f63cc3416bfd08159991379c82556fe356784c688c0acde4d9a25b4210c130`).
  The first geometry stage copied this artifact directly and performed no
  optimization.
- To isolate the decision itself, both arms computed render-mask IoU through
  the same code path. The baseline accepted all candidates; the experimental
  arm applied the only semantic change, `confidence_threshold: 0.8`.
- The first online candidate batch was exactly paired: frames 0, 15, and 41
  had IoUs 0.848816, 0.824392, and 0.723823 in both arms. The gate rejected
  frame 41; subsequent candidate trajectories therefore diverged by design.
- Baseline: 6/6 candidates accepted, ADD 21.0991 mm, ADD AUC 78.8919.
- Confidence gate: 4/6 candidates accepted, ADD 18.2342 mm, ADD AUC 81.7432.
- Paired single-sequence delta: ADD improved by 2.8649 mm (13.58% lower) and
  AUC increased by 2.8513 points. This is a controlled engineering result for
  one object/sequence, not a statistical or paper-level claim.
- Unit/static verification: Python compilation passed and all five CPU unit
  tests passed. Both GPU tracker runs returned code 0 with zero reported
  failures.

## 2026-10-01 — Leakage-controlled multi-sequence experiment

- Frozen initial geometry was prepared for every included sequence, and all
  analyzed baseline/method pairs have matching SHA-256 digests.
- Validation compared thresholds 0.70, 0.75, 0.80, and 0.85 on sequence `00`
  from four objects. Threshold 0.80 was selected by the prespecified macro ADD
  criterion (-2.5895 mm; macro AUC delta +2.5737).
- `black_drill` was excluded as a whole before held-out aggregation because
  gated validation repeatedly raised C++ `std::bad_alloc` at both 128 and 64
  histogram caps. Successful threshold-specific fragments were not used.
- The locked 0.80 arm completed all 17 held-out sequences. Baseline versus
  gated mean ADD was 21.0505 versus 22.4223 mm; macro paired delta +1.5112 mm,
  hierarchical-bootstrap 95% CI [-2.6852, 8.9607], Wilcoxon p=0.9032.
- Baseline versus gated mean AUC was 78.9408 versus 77.7113; macro paired delta
  -1.3943, 95% CI [-8.9063, 2.7275]. The current hard gate is therefore not
  supported as an improvement.

## 2026-10-01 — Multi-signal adaptive soft confidence v2

- Added causal fusion of mask IoU, normalized pose innovation, visible-area
  reliability, and temporal IoU consistency in `confidence/multisignal.py`.
- Added robust median/MAD adaptive thresholds and sigmoid observation weights;
  weights now act directly on per-observation silhouette IoU loss.
- Extended CSV logs with all components, fused confidence, active threshold,
  and applied loss weight. Legacy baseline and hard-gate modes remain usable.
- Exploratory validation at base threshold 0.65 completed on four sequences.
  All ADD deltas favored v2: -0.2987, -0.8536, -0.0593, and -3.6534 mm; macro
  delta -1.2163 mm. Macro AUC delta was +1.2220. Across 18 candidates, weights
  ranged 0.1181–0.9847 (mean 0.7724), providing more discrimination than v1.
- These are exploratory validation results, not a new held-out paper claim.
  The previously inspected test split cannot independently confirm v2.

## 2026-10-01 — Gradient-consistent confidence v3 and full development reuse

- Added per-observation geometry-gradient consistency against reference and
  causal historical directions, plus a direct v2/v3 development analyzer.
- Reused all 21 sequences as requested. v2 mean/macro ADD deltas were
  +1.4472/+1.4877 mm; v3 reduced them to +0.7516/+0.8751 mm. Both methods won
  12, lost 6, and tied 3 sequences, but neither was significant.
- A single sensitive sequence, `toy_plane/03`, degraded by +38.3149 mm (v2)
  and +35.2208 mm (v3). Raising the v3 minimum weight to 0.75 still produced
  +34.5043 mm, rejecting simple clipping as a sufficient safeguard.
- Result: v3 improves failure severity but does not support a positive paper
  claim. A post-update accept/rollback mechanism is required next.

## 2026-10-01 — Causal holdout and geometry trust-region rollback v4

- Added update-level validation on references plus the lowest-confidence
  leave-one-out candidate, a 15%-of-object-scale maximum vertex step, atomic
  mesh publication, and exact rollback to the previous mesh.
- Completed all 21 paired development sequences: 12 updates accepted and 15
  rolled back. Mean/macro ADD deltas were +0.6446/+0.6603 mm, 95% hierarchical
  CI [-1.0820, 3.5323], Wilcoxon p=0.6095. This is the best of the proposed
  variants but still does not outperform BIT overall.
- Rejected updates can still yield different tracking metrics from baseline,
  demonstrating material C++ run-to-run/closed-loop nondeterminism. Repeated
  process replicates are required before attributing small deltas to a method.

## 2026-10-01 — Two-replicate variance and mixed-effects analysis

- Completed baseline and v4 twice on every included sequence (84 observations).
- Mixed model with method fixed effect and object/sequence random intercepts:
  ADD effect +0.3144 mm, 95% CI [-1.2333, 1.8620], p=0.6905; AUC effect
  -0.0662, CI [-1.5729, 1.4404], p=0.9313.
- Mean within-sequence ADD SD was 2.0271 mm for baseline and 0.1213 mm for v4.
  The current evidence supports increased repeatability, not increased mean
  accuracy. Only two repeats are available, so variance reduction is preliminary.

## 2026-10-02 — Three-replicate paper-level stability analysis

- Completed the third independent baseline/v4 run: 126 total observations.
- Mixed-effect ADD estimate (v4 minus baseline) was -0.1812 mm, 95% CI
  [-1.4834, 1.1210], inside the prespecified ±2 mm equivalence margin.
- Mean within-sequence ADD SD fell from 2.1009 to 0.5518 mm. V4 reduced SD on
  16/18 non-tied sequences (one-sided Wilcoxon p=0.00385; sign p=0.000656).
- Mean and 95th-percentile run ranges fell from 3.7758/14.6852 mm to
  0.9770/1.6225 mm. Evidence supports repeatability with equivalent mean
  accuracy, not an average-accuracy improvement.
- The four-object hierarchical variance-ratio interval remains wide; a second
  dataset is required for a strong cross-object generalization claim.

## 2026-10-02 — Independent six-object confirmation

- Froze v4 and threshold 0.65, then evaluated all 33 sequences from six MOPED
  objects unused during development. All paired initial-mesh hashes matched.
- Mean ADD improved from 32.0188 to 29.5121 mm (-2.5067 mm, 7.83%); the
  object-macro delta was -2.4909 mm with hierarchical-bootstrap 95% CI
  [-7.9555, 3.1188]. Five of six object means improved.
- Sequence-level ADD outcomes were 15 wins, 8 losses, and 10 ties (Wilcoxon
  p=0.0327; sign p=0.2100). The primary object-cluster interval crosses zero,
  so this is promising external evidence rather than a conclusive superiority
  result.
- Tail behavior remains a limitation: the worst degradation was +49.2376 mm,
  concentrated in `cheezit/02`; it was retained in every aggregate.

## 2026-10-02 — Three-run confirmation and failure mechanism

- Completed baseline/v4 three times on all 33 confirmation sequences (198
  valid observations); all initial-geometry hashes matched.
- Mixed-effect ADD estimate was -1.8075 mm, 95% CI [-5.1263, 1.5112],
  p=0.2858. Neither superiority nor ±2 mm equivalence was established.
- Mean within-sequence ADD SD was 4.2131 mm for baseline and 2.9753 mm for v4,
  but the pooled variance-ratio interval [0.2362, 17.2409] crossed one and the
  paired Wilcoxon p=0.3524. Development-cohort stability did not replicate.
- `cheezit/02` reproduced at baseline mean 68.3995 versus v4 119.3350 mm. V4
  rejected both proposed updates and retained the initial mesh exactly, while
  baseline published two deformed meshes. The failure is a stable conservative
  rollback failure, not a one-run fluctuation.

## 2026-10-02 — V5 joint pose–geometry optimizer

- Added bounded SE(3) pose corrections and three-block alternating pose/mesh
  optimization with silhouette, boundary, pose-prior, temporal, Laplacian, and
  geometry-trust losses.
- Replaced V4's conjunctive IoU decision in the new `joint_risk` arm with an
  aggregate tracking-risk decrease plus a 90th-percentile tail guard. V4 and
  baseline execution paths remain unchanged.
- Added 7 numerical tests (15 total). A `duplo_dude/00` GPU smoke test completed
  with one rejected and one accepted geometry update, ADD 15.8742 mm. This is
  not a tuned or confirmatory result.

## 2026-10-03 — Full V5 development cohort and three replicates

- Completed V5 on all 21 development sequences three times. Combined with the
  existing baseline/V4 replicates, the comparison contains 189 observations
  with matching initial-geometry hashes.
- Mean ADD was 23.0195/22.8383/21.6128 mm for baseline/V4/V5. Mixed-effect V5
  effects were -1.4068 mm versus baseline (95% CI [-2.7402, -0.0733], p=0.0387)
  and -1.2256 mm versus V4 (CI [-2.0125, -0.4387], p=0.00227).
- Mean within-sequence ADD SD was 2.1009/0.5518/0.3409 mm, but paired stability
  tests for V5 were not significant. Mean runtime rose to 87.14 s (+21.5%
  versus baseline).
- `toy_plane/03` remained exactly equal to V4 at 30.9997 mm in all runs: both
  V5 updates exceeded the frozen 0.15 vertex-step bound and were rolled back.
  V5 improves aggregate development estimates but does not fix this mechanism.
