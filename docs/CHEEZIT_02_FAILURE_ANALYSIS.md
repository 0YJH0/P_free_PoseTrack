# `cheezit/02` read-only failure analysis

This analysis was performed without changing a threshold, parameter, sample,
or reported outcome.

## Reproducibility of the failure

| Method | Replicate 0 ADD | Replicate 1 ADD | Replicate 2 ADD | Mean ADD |
|---|---:|---:|---:|---:|
| Baseline | 70.0974 | 68.0806 | 67.0205 | 68.3995 |
| v4 | 119.3350 | 119.3350 | 119.3350 | 119.3350 |

All six runs used the same initial-geometry SHA-256 digest. The v4 penalty is
therefore systematic for this sequence: +50.9355 mm using three-run means.

## Mechanism

V4 proposed two online geometry updates and rejected both:

1. Reference IoU decreased 0.629939→0.623726, although holdout IoU increased
   0.548980→0.562192; vertex-step ratio was 0.073158.
2. Reference IoU decreased 0.629939→0.594051, although holdout IoU increased
   0.419446→0.460815; vertex-step ratio was 0.155397, also exceeding the fixed
   0.15 trust-region limit.

Consequently all three v4 runtime meshes are byte-identical to the frozen
initial mesh. Baseline publishes two deformed meshes: relative to the initial
mesh, their mean vertex displacements are 0.07134 and 0.11037 model units and
maximum displacements are 0.20883 and 0.39171.

The supported causal interpretation is therefore not that v4 accepted a bad
shape. Its conservative conjunctive trust rule rejected geometry changes that
were beneficial for downstream pose tracking on this sequence. Local
silhouette validation and final trajectory ADD are misaligned here. This is a
documented limitation, not a reason to tune on or exclude `cheezit/02`.

Machine-readable evidence is in
`results/paper_moped/analysis/cheezit_02_failure.json`.

