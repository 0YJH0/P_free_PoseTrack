# Experiment Plan

## Phase 1: baseline and mask-confidence gate

Run the same public benchmark sequence twice with identical initialization,
reference observations, seed, tracker settings, geometry iterations, and
evaluation protocol:

1. official behavior with `use_confidence_gate: false`;
2. Render-Mask Confidence Gating with an explicitly recorded threshold.

Record the resolved config, dataset/object/sequence, seed, runtime, pose
metrics, candidate and accepted keyframe counts, geometry update count, and
per-candidate confidence CSV. No numerical result is reported until the run
actually finishes.

Required sanity endpoints are accept-all (`threshold = 0`) and reject-all
(`threshold > 1`) modes. Reject-all must continue pose tracking without a
C++/Python synchronization deadlock.

Later phases will add confidence correlation analysis, geometry-pollution
analysis, component ablations, and RBOT robustness conditions only after this
phase is reproducible.

