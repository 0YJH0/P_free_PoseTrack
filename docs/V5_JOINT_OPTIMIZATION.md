# V5 joint pose–geometry optimization

## Scope

V5 adds a new `joint_risk` arm while preserving baseline and v4 behavior. It
is training-free and causal: only observations already available at an online
update are used. The lowest-confidence candidate is held out from optimization
and used only for update validation.

## Optimization

Each observation receives a bounded left-multiplicative SE(3) correction

\[
T_i' = \exp(\delta\xi_i^\wedge)T_i.
\]

Rotation corrections are bounded to 5 degrees per axis and translations to 3%
of the current object scale per axis. The first reference pose and causal
holdout are fixed, preventing unconstrained pose/geometry co-adaptation.

The optimizer alternates three times between:

1. 20 pose-only steps with weighted silhouette IoU, differentiable boundary
   alignment, pose prior, and robust temporal acceleration losses;
2. 60 geometry-only steps with weighted silhouette IoU, boundary alignment,
   Laplacian regularization, and displacement from the pre-update mesh.

Confidence and per-observation geometry-gradient consistency initialize the
observation weights. V4 remains available as `gradient_rollback` for a strict
ablation.

## Tracking-risk validation

V5 replaces the conjunctive reference/holdout IoU rule with an aggregate risk
containing mean silhouette loss, 90th-percentile silhouette loss, boundary
error, required pose correction, temporal acceleration, and normalized mesh
step. A candidate is published only if aggregate risk improves by the frozen
margin and the tail loss does not exceed its tolerance. The existing 15%
maximum vertex step remains a hard safety constraint.

Every candidate log now includes pre/post tracking risk, post-update tail loss,
and mean pose correction. Mesh publication remains atomic.

## Initial smoke test

`duplo_dude/00` completed end-to-end. The first proposed update was rejected
because risk increased from 0.398632 to 0.400467. The second was accepted after
risk decreased from 0.418171 to 0.395049 and its guarded tail IoU loss fell
from 0.425676 to 0.386390. The two mean pose-correction norms were 0.001487 and
0.003755, confirming that the pose block was active. ADD was 15.8742 mm,
compared with 19.2707 for the stored canonical baseline and 17.8215 for v4.

This is an implementation smoke test, not an accuracy claim or parameter
selection result. The previously inspected six-object confirmation cohort must
not be used to tune V5. A new frozen development/confirmation cycle is needed.
The subsequent full three-replicate development analysis is reported in
`docs/V5_DEVELOPMENT_RESULTS.md` and supersedes this smoke result for aggregate
development conclusions.

## Entry points

```bash
python experiments/run_moped_paired.py joint-all \
  --cohort development --threshold 0.65
```

Core numerical code is in `optimization/joint_pose_geometry.py`; orchestration
and differentiable rendering remain in `examples/run_example.py`.
