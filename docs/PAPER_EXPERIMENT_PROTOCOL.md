# Paper Experiment Protocol

## Scope

The main statistical experiment uses four MOPED objects: `duplo_dude`,
`rinse_aid`, `toy_plane`, and `vim_mug`. We extend evaluation from one example
trajectory to every technically valid evaluation sequence for those objects. Every
comparison is paired at the sequence level and uses a byte-identical frozen
initial geometry artifact.

Two sequences (`duplo_dude/03` and `vim_mug/03`) are excluded before running
either method because their distributed mask streams are incomplete and the
unmodified BIT reference generator deterministically segfaults. The exclusion
is based on input integrity and preparation failure, not outcome values. This
leaves 21 sequences across four objects after the object-level `black_drill`
feasibility exclusion documented below.

## Leakage-controlled split

- Validation: sequence `00` for every object (4 sequences).
- Held-out test: every remaining technically valid sequence (17 sequences).
- Candidate hard-gating thresholds: 0.70, 0.75, 0.80, and 0.85.
- Select one threshold by validation macro object mean paired ADD difference;
  use macro AUC difference only as a tie-breaker.
- Lock the selected threshold before any aggregate held-out test analysis.

This split keeps all objects represented during threshold calibration while
preventing the exact test trajectories from selecting the operating point.
It is a pragmatic benchmark split, not an official MOPED training/test split.

## Strict pairing

For every object/sequence pair:

1. Generate initial geometry once with seed 42.
2. Save its SHA-256 digest and immutable OBJ artifact.
3. Run baseline and confidence-gated variants from that same artifact.
4. Make both variants compute mask IoU through the same code path.
5. Change only whether the configured threshold filters online observations.

Runs whose initial-geometry digests differ are excluded as protocol
violations, not treated as ordinary failures.

## Outcomes and statistical units

- Primary outcome: per-sequence ADD error in millimetres (lower is better).
- Secondary outcome: official BIT ADD AUC (higher is better).
- Experimental unit: an evaluation sequence, paired between methods.
- Report sequence-weighted means and object-macro paired differences.
- Report a two-sided paired Wilcoxon signed-rank test and exact sign test.
- Report a 95% hierarchical bootstrap confidence interval by resampling
  objects, then sequences within objects (20,000 samples, seed 20250930).
- Report wins/ties/losses and the accepted-candidate rate.

ADD is the prespecified primary endpoint. AUC is secondary; its unadjusted
p-value is descriptive unless corrected together with any additional tested
outcomes.

## Reproducibility

Raw runs preserve resolved YAML, stdout/stderr, confidence CSV, metrics JSON,
runtime, seed, threshold, initial-geometry digest, and generated meshes under
`results/paper_moped/raw`. The drivers are resumable and do not overwrite a
completed run unless `--force` is supplied.

The final experiment runtime uses `BIT_TCLC_MAX_HISTS=128`, one OpenCV/OMP
thread, and an 8 GB experiment swap file in every paired arm. `black_drill` is
excluded as a whole before test evaluation because its confidence-gated
validation run repeatedly terminates with `std::bad_alloc` at both 128 and 64
histograms; no successful threshold-specific subset is retained. This is an
implementation-feasibility exclusion and limits the inference population to
`duplo_dude`, `rinse_aid`, `toy_plane`, and `vim_mug`. The bundled C++
tracker otherwise reaches roughly 14.5 GB resident memory and its default
parallel path also intermittently throws `std::bad_alloc` before any confidence
decision. These constraints produce a stable resource-constrained BIT baseline
on the 16 GB host. They change tracker capacity relative to the paper's default,
so conclusions are explicitly limited to this controlled baseline. Runtime
settings are serialized in each metrics file. Diagnostic runs with other
settings are retained in `*_diagnostic` directories and excluded from
inference.

The single-object pilot results under other result directories are excluded
from threshold selection and final inference.

## Post-development independent confirmation

After freezing v4 (`gradient_rollback`, base threshold 0.65), a separate
confirmation cohort was defined from six objects not used for architecture or
threshold development: `cheezit`, `duster`, `graphics_card`, `orange_drill`,
`pouch`, and `remote`. All 33 available sequences are analyzed without outcome
exclusions. The primary confirmation estimate is the object-macro paired ADD
difference with a 95% hierarchical bootstrap that resamples objects and then
sequences. Sequence-level Wilcoxon and sign tests are sensitivity analyses.
The fixed ±2 mm margin is reported descriptively; the confirmation experiment
targets superiority and is not re-tuned if its interval crosses zero.
