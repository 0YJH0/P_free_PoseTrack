# Minimal Modification Plan: Render-Mask Confidence Gating

## 1. Experimental objective

Implement only the first controlled comparison:

```text
Original BIT
vs.
BIT + Render-Mask Confidence Gating
```

For each geometry-update candidate observation (i):

```text
C_mask(i) = IoU(M_seg(i), M_render(i))
accept(i) = !use_confidence_gate OR C_mask(i) >= confidence_threshold
```

Rejected observations continue through pose tracking but contribute no
silhouette supervision to that geometry update. No pose confidence, combined
confidence, soft weighting, mask refinement, feature model, or learned module
is included in this phase.

## 2. Why gating belongs in Python for v1

The confidence requires the final SAM mask. In the official pipeline SAM runs
only after C++ has selected and serialized a candidate batch. The existing
`cmc/mask` file is an image-derived tracker confidence mask, not the C++
rendered silhouette.

The minimal correct implementation is therefore:

1. retain the original C++ viewpoint candidate selection;
2. receive the candidate observations in Python;
3. run SAM exactly as the baseline does;
4. render the current mesh once under `torch.no_grad()` with the same pose,
   intrinsics, coordinate transforms, and output size used by geometry loss;
5. compute binary IoU per observation;
6. select accepted observations before constructing the optimization batch;
7. optimize only accepted observations plus accepted immutable references;
8. always publish a model artifact so the C++ process cannot deadlock.

This location makes `M_render` numerically identical in meaning and alignment
to the silhouette used by the baseline geometry loss.

## 3. New files

### `confidence/__init__.py`

Exports the confidence API without adding dependencies.

### `confidence/mask_confidence.py`

Add a small, independently testable function:

```python
def binary_mask_iou(
    segmentation_masks: torch.Tensor,
    rendered_masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return one IoU in [0, 1] per aligned observation."""
```

Inputs are aligned `[N, H, W]` masks. Empty-union behavior must be explicit;
for gating, returning zero is conservative and avoids accepting an observation
with neither a valid segmentation nor render.

### `confidence/gate.py`

Add a typed result object and gate:

```python
@dataclass(frozen=True)
class MaskGateResult:
    mask_iou: torch.Tensor
    accepted: torch.Tensor

def apply_mask_confidence_gate(..., enabled: bool, threshold: float) -> MaskGateResult:
    ...
```

When disabled, every observation is accepted. This invariant is essential for
baseline equivalence.

### `logging_utils/confidence_csv.py`

Append one row per C++ candidate to a CSV with the required schema:

```text
frame_id,mask_iou,is_candidate_keyframe,is_accepted_keyframe,geometry_update
```

The logger should buffer rows and write only at update boundaries to avoid
measurably perturbing tracking.

`frame_id` is not currently retained by Python `DataGroups`; see required
official-file changes below.

### `tests/test_mask_confidence.py`

CPU-only unit tests for perfect overlap, disjoint masks, partial overlap,
threshold equality, disabled-gate behavior, shape validation, and empty masks.

## 4. Minimal official-file changes

### A. `config/config_generator.py`

Why: generated per-sequence YAML is the single config shared by both
processes.

Add, for both RBOT and MOPED:

```yaml
confidence:
    use_confidence_gate: false
    confidence_threshold: 0.5
    mask_binary_threshold: 0.5
    log_csv: "<sequence output>/confidence.csv"
```

The exact default threshold is an experiment setting, not a claimed optimal
value. `use_confidence_gate: false` must preserve the original baseline path.

### B. `tracker/interface/communication.h`

Why: Python needs the actual source frame IDs for required per-frame logging.

Extend `LocalStorageCommunicator` with an `index` output directory and a small
integer writer, or encode the source index in a dedicated file. Do not infer
frame ID from array position because reference frames and nonuniform candidate
selection break that correspondence.

Input: `index` already passed to `SentDatas2Py`.

Output: one indexed metadata file consumed alongside pose/K/ROI files.

### C. `tracker/interface/communication.cpp`

Why: create/clear the `index` metadata directory and serialize the source
frame ID in `SentDatas2Py`.

No selection or tracker behavior changes are required in C++ for v1.

### D. `examples/communication.py`

Why: carry frame identity through the Python process and prevent rejected
observations from re-entering through cached base data.

Changes:

- add `frame_ids_` to `DataGroups`;
- scan and parse the new index files;
- add a method such as `select(indices) -> DataGroups` that filters every
  field consistently;
- cache only the accepted reference/base subset;
- validate equal lengths across all fields before optimization.

Input: the existing disk protocol plus frame IDs.

Output: aligned, filterable `DataGroups`.

### E. `examples/run_example.py`

Why: this is the only location where aligned SAM silhouettes, current mesh,
poses, intrinsics, and the differentiable renderer coexist.

Minimal changes:

1. Load confidence config with safe defaults (`False`, threshold from config).
2. Refactor only the repeated render preparation into a helper that returns
   alpha silhouettes; do not duplicate the pipeline.
3. Before creating the Adam optimizer, render the current mesh under
   `torch.no_grad()` and compute per-observation IoUs.
4. Apply the gate consistently to SAM masks and every pose/intrinsic/data
   field used by geometry optimization.
5. Keep references according to an explicit policy: pseudo/externally
   generated reference observations are immutable baseline initialization and
   should not be logged as online candidate keyframes; online candidates are
   gated. The code must preserve their provenance rather than relying on list
   position after repeated updates.
6. If the gate is disabled, execute the baseline tensors and loss unchanged.
7. If no online candidate is accepted, skip Adam and write an unchanged copy
   of the current mesh as `_realtime_N.obj`; this acknowledges the update
   round and prevents the waiting C++ process from deadlocking.
8. Write CSV rows after knowing whether an optimization actually ran. For an
   accepted candidate, `geometry_update` is true only when its observation was
   included in a completed optimization.

No changes to `neg_iou_loss` are needed for hard gating. It should remain
unchanged to isolate selection effects.

## 5. Files that do not need algorithmic changes in v1

- `tracker/example/eg_BIT.cpp`: retain original viewpoint novelty selection
  and tracking. Optionally add diagnostics only; no confidence decision occurs
  here in v1.
- `tracker/interface/tracker_summer/tracker_summer.cc`: it already supplies the
  tracked pose and renderer state; do not modify pose optimization.
- `examples/sam_utils.py`: retain official SAM prompting and mask generation.
- `soft_renderer/*`: use the existing renderer without changing CUDA kernels.
- `tracker/tracker/summer/*`: do not instrument pose residuals yet.
- evaluation functions: no metric definition changes in this experiment.

## 6. Baseline and confidence configs

The generated sequence config should be the authoritative runtime config.
For convenient experiment presets, add small overlay/example configs only if
the loader supports deterministic composition; otherwise duplicating absolute
dataset sequence YAMLs would harm reproducibility.

Required semantic presets are:

- baseline: `use_confidence_gate: false`;
- mask-gate: `use_confidence_gate: true` with an explicitly recorded
  `confidence_threshold`.

Every result directory must preserve the resolved YAML used for that run.

## 7. Logging semantics

One row corresponds to an online frame that passed the original BIT viewpoint
novelty condition and was sent to Python.

- `frame_id`: original dataset sequence index.
- `mask_iou`: IoU of aligned SAM and current-geometry silhouettes before the
  current update.
- `is_candidate_keyframe`: always true for these online rows; the explicit
  field makes later logs compatible with all-frame logging.
- `is_accepted_keyframe`: gate decision, or true for every candidate when the
  gate is disabled.
- `geometry_update`: true only if the observation actually contributed to a
  completed geometry optimization.

Atomic replacement or flush-at-round-boundary should be used so interrupted
runs leave a parseable CSV.

## 8. Verification sequence

### Static and CPU tests

1. Run formatting/import checks on new modules.
2. Run `tests/test_mask_confidence.py` without CUDA.
3. Test `DataGroups.select` with synthetic arrays and verify alignment.
4. Test CSV schema and append/resume behavior in a temporary directory.
5. Generate one MOPED and one RBOT config and validate OpenCV YAML plus Python
   YAML parsing.

### Integration smoke test

After dependencies, SAM checkpoint, compiled tracker, and a dataset sequence
are available:

1. Run baseline with confidence disabled and compare candidate count, geometry
   update count, and pose outputs against the untouched official commit.
2. Run confidence mode on the same sequence and seed.
3. Verify all IoUs lie in `[0, 1]` and CSV frame IDs match C++ source indices.
4. Force threshold `0.0`: decisions should match accept-all baseline.
5. Force threshold above `1.0`: no online observation should enter geometry
   optimization, tracking must continue, and the process must not deadlock.
6. Confirm rejected observations are absent from every later geometry batch.
7. Record runtime and resolved config; do not report tracking metrics until the
   full sequence has actually completed.

## 9. Smallest practical change set

The first implementation needs five official integration files and four new
support/test files:

```text
modify  config/config_generator.py
modify  tracker/interface/communication.h
modify  tracker/interface/communication.cpp
modify  examples/communication.py
modify  examples/run_example.py

add     confidence/__init__.py
add     confidence/mask_confidence.py
add     confidence/gate.py
add     logging_utils/confidence_csv.py
add     tests/test_mask_confidence.py
```

If logging is temporarily allowed to use the communication filename as the
frame ID, the two C++ communication edits can be avoided, but that assumption
is fragile and should not be used for paper experiments. The recommended
minimal research-quality implementation carries frame identity explicitly.

## 10. Deferred work

Explicitly deferred until baseline and hard mask gating run end-to-end:

- pose/visibility/feature confidence;
- combined confidence;
- reliability-aware C++ keyframe scoring;
- soft geometry-loss weighting;
- geometry-guided mask refinement;
- pose-optimizer residual instrumentation;
- RBOT protocol completion and broader ablations.

