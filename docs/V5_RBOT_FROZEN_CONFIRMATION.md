# Frozen v5 external-confirmation protocol

Freeze date: 2026-10-03 (Asia/Shanghai)

This document was written before downloading or inspecting RBOT results. RBOT is
an external, previously unused dataset for this project. No v5 parameter may be
changed after this freeze on the basis of RBOT outcomes. Any technical adapter
needed to load or score RBOT must be method-invariant and documented separately.

## Frozen implementation

- `optimization/joint_pose_geometry.py`: `8fc46d8c0f11f8d205853658640a0f1cc4dadaea5e81982b3813451f18b056dd`
- `examples/run_example.py`: `a7da75dbbe263b697c128dccf9e17ef2960b1f1fe8ca659905c75c2114afa224`
- `experiments/run_moped_paired.py`: `98fdaaefa2076616c325efad9e4a76d9d106f538e005d5bb3bbb00766c5fc078`
- `experiments/run_replicates.py`: `ba83ce5c11d95a0b51623132e864dc8c74058868cc5de761981aada5dd5fdbbc`
- Combined working-tree patch SHA-256 for the four files: `caa07a26ddc7577beab8a9efeb32502a740484358b76baf99f798208c7194fc4`

Frozen v5 defaults are those resolved by `examples/run_example.py`: three outer
steps, 20 pose steps, 60 geometry steps, pose/geometry learning rates 0.002/0.01,
5-degree rotation and 0.03-scale translation bounds, risk margin 0.002, tail
tolerance 0.01, and maximum vertex-step ratio 0.15. All remaining loss and risk
weights are frozen at the defaults in that file.

## Confirmation population and comparisons

Use all 18 RBOT objects and all four official conditions (`a_regular`,
`b_dynamiclight`, `c_noisy`, and `d_occlusion`) for which the downloaded dataset
passes integrity checks. Run baseline, frozen v4, and frozen v5 from identical
initial geometry, with three independent repeats per method-condition-object.

The primary comparison is frozen v5 versus baseline over the complete RBOT
population. Frozen v5 versus v4 and condition-specific results are secondary.
Report paired effects with object and sequence/condition grouping, confidence
intervals, run variance, failures, and runtime. The exact benchmark metric must
be implemented from the RBOT ground truth and fixed before aggregate method
results are inspected. Technical smoke tests may use one object/condition but
must not be used for parameter selection.

## Leakage rule

RBOT outcomes cannot be used to alter v5, thresholds, loss weights, optimizer
steps, stopping rules, or publication gates. A failed confirmation remains a
reportable result. Any later v6 is a new method and requires another untouched
confirmation set.
