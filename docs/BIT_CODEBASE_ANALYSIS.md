# BIT Official Codebase Analysis

## 1. Scope and revision

This document analyzes the official implementation of **Prior-free 3D Object
Tracking (BIT, CVPR 2025)** at commit
`6dc95feb51d0b96d96adc009d408fc4948b06fdc` (2025-06-14).

The repository is a two-process system:

- C++ performs model-based pose tracking, viewpoint-based keyframe selection,
  rendering, dataset iteration, and final evaluation.
- Python runs SAM and differentiable mesh optimization.
- The two processes exchange observations and meshes through files under each
  sequence's `cmc/` directory.

No confidence-aware change has been made while producing this analysis.

## 2. Executable entry points

### Tracking entry point

`run_example.sh` invokes:

```text
python examples/run_example.py <dataset> <generated-config.yml>
```

`examples/run_example.py::main` is the top-level Python controller. It loads
the YAML config, initializes the differentiable renderer, mesh, and SAM, then
starts `tracker/bin/eg_BIT` as a subprocess. The C++ executable's source is
`tracker/example/eg_BIT.cpp::main`.

The runtime call chain is:

```text
run_example.sh
  -> examples/run_example.py::main
       -> subprocess tracker/bin/eg_BIT <same YAML>
            -> tracker/example/eg_BIT.cpp::main
```

### Reference-frame entry point

`gen_refers.sh` invokes `tracker/bin/gen_references`, whose source is
`tracker/example/gen_references.cpp`. Reference generation is a prerequisite
of the current implementation even when only the first frame is used as a
pseudo-reference.

### Build and config entry points

- `setup_cpp.sh`: configures and compiles the C++ tracker with CMake.
- `gen_configs.sh` / `config/config_generator.py`: create RBOT/MOPED sequence
  YAML files from paths in `config/dataset.yml`.
- `setup.py`: builds the CUDA extension used by `soft_renderer`.

## 3. Dataset loading and configuration

There is no Python `Dataset` class in the online BIT path. Dataset sequences
are consumed directly by C++.

`tracker/example/eg_BIT.cpp::main` reads the `dataset` field and branches:

- **RBOT**: `LoadPoses`, `LoadImages`, `LoadCamera`, and `LoadObjModel` load
  poses, RGB frames, intrinsics, and the object mesh.
- **MOPED**: `LoadCameraFromJson`, `NormalizeMopedModel`, `LoadMopedPoses`,
  `LoadImages`, `LoadBody2World`, and `LoadRegistration` additionally reconcile
  MOPED coordinate systems.

These loader/evaluation utilities are declared in
`tracker/ar_utils/data_io/data_loader.h` and implemented in
`tracker/ar_utils/data_io/data_loader.cpp`.

`config/config_generator.py::GenRbotYmls4Summer` and
`GenMopedConfigs` define the effective experiment configuration. Important
fields include dataset paths, image size, viewpoint `angle`, reference and
estimated image counts, number of mesh deformation rounds, differentiable
rendering settings, SAM settings, tracker executable, and communication root.

The official generator currently emits only RBOT `a_regular` configs. The
other RBOT conditions (`b_dynamiclight`, `c_noisy`, `d_occlusion`) are present
only in commented legacy generator calls and need explicit support before the
robustness experiments.

## 4. End-to-end data flow

For every image, `tracker/example/eg_BIT.cpp::main` fills a `StdData` object and
calls:

```text
PushData
  -> SystemManager::PushData
  -> TrackerManager::SetData
ProcessData
  -> SystemManager::ProcessData
  -> TrackerManager::ProcessData
  -> SummerTracker::Estimate
  -> summer::Tracker::EstimatePoses
  -> SLCTracker::Track / RunIteration
GetResult
  -> SystemManager::GetResult
```

After tracking, C++ computes the current view direction. If the frame is not a
reference and its view is sufficiently novel, it serializes a `DataGroup` to
disk. Python polls those files, loads a batch, segments the corresponding RGB
images with SAM, and optimizes the mesh against their silhouettes. The newly
saved OBJ is detected by C++, rescaled, and installed into a newly initialized
SLOT/Summer tracker while preserving the current pose.

```text
RGB frame
  -> C++ SLOT/Summer pose tracking
  -> viewpoint novelty test
  -> cmc/{img,prob,mask,pose,K,originRoi,targetRoi,targetSize}
  -> Python SAM segmentation
  -> soft-render silhouette optimization
  -> cmc/deformed_model/_realtime_N.obj
  -> C++ ChangeModelScale + ChangeModels
  -> subsequent pose tracking
```

## 5. Segmentation chain

Segmentation has two related but distinct components.

1. `summer::SLCTracker::prob_map` in
   `tracker/tracker/summer/src/mbt/tracker_slc.cc` obtains a foreground
   probability map from the tracker's foreground/background histograms.
2. `SummerTracker::ComDataGroup` in
   `tracker/interface/tracker_summer/tracker_summer.cc` uses the rendered model
   ROI to crop/recenter that probability map and creates an image-derived
   confidence mask through `Prompter::ComConfidenceMaskFromProb`.
3. C++ writes RGB, probability map, confidence mask, pose, intrinsics, and ROI
   metadata through `LocalStorageCommunicator::SentDatas2Py` in
   `tracker/interface/communication.cpp`.
4. Python reads them through
   `examples/communication.py::LocalStorageCommunication.parse_data` into a
   `DataGroups` object.
5. `examples/sam_utils.py::segment` runs `SamPredictor.predict` using the
   tracker-derived ROI as a box prompt, takes `masks[0]`, and maps it into the
   recentered target coordinate system with `ResizeMask`.
6. `examples/run_example.py::main` flips, pads, resizes, and converts the SAM
   masks to `[N, 4, H, W]` float tensors for geometry optimization.

Important caveat: SAM's returned `scores` are currently discarded, and
`masks[0]` is selected without consulting those scores.

## 6. Pose-tracking chain

The public interface is in `tracker/interface/`:

- `SystemManager` dispatches calls.
- `TrackerManager` owns the selected tracker implementation.
- `SummerTracker` adapts BIT data to the bundled SLOT/Summer tracker.

The actual optimization begins at
`summer::Tracker::EstimatePoses` in
`tracker/tracker/summer/src/mbt/tracker.cc`, which builds a three-level image
pyramid and calls the virtual `Track` implementation.

`summer::SLCTracker::Track` in `tracker_slc.cc` performs a fixed coarse-to-fine
sequence of `RunIteration` calls. Its contour objective is assembled in
`SLCTracker::ComputeJac`, which accumulates `J^T M` and weighted `J^T J` terms
from foreground/background probabilities sampled along contour-normal search
lines.

The official public result object exposes the resulting pose, success flag,
probability map, render mask, ROI, and overlay, but it does **not** expose a
final scalar optimizer residual, residual decrease, or convergence flag.
Consequently, a future pose-confidence component must first instrument the
existing optimizer rather than invent a proxy unrelated to its internals.

Initialization uses the first-frame ground-truth pose in
`SummerTracker::StartTracking`. On an `IsSuccess` failure,
`SummerTracker::Estimate` resets the internal model pose to ground truth after
capturing `currPose`; this benchmark-only recovery behavior must be explicitly
accounted for when interpreting robustness results.

## 7. Keyframe selection

Online candidate selection is implemented directly in
`tracker/example/eg_BIT.cpp`, not in a dedicated keyframe class.

- `calculateAngle` computes the angular difference between view vectors.
- `DifferEnough` rejects a zero view and requires the new view to differ by at
  least config field `angle` from **every** stored template view.
- In `main`, the candidate condition is
  `!communicator->InRefers(index) && DifferEnough(view, templateViews, angle)`.
- Accepted candidates are written with `SentDatas2Py`, and their view is
  immediately appended to `templateViews`.

The current selection uses viewpoint novelty only. It has no observation
quality test. A fixed number of candidates (`estimated_imgs_nums`) triggers
each Python geometry update.

## 8. Geometry generation and update

Geometry generation is implemented in `examples/run_example.py`.

- `Model.__init__` loads `sphere_2600_unit.obj` or a previously generated OBJ.
- Learnable parameters are per-vertex `displace` and a global `center`.
- `Model.forward` generates vertices and returns the mesh plus Laplacian and
  flatten regularizers.
- `SoftRasterizer` renders the batch using observation-specific poses and
  intrinsics set through `sr.LookAt`.
- Adam performs `iteration_nums_in_each_deform` iterations.
- The resulting mesh is written as `_realtime_N.obj`.

For later update rounds, Python initializes `Model(save_obj_path)` from the
previously optimized mesh and combines the newly received observations with
the original base/reference `DataGroups` via
`LocalStorageCommunication.add_new_datagroups_2_base`.

C++ detects a new OBJ in `GetModelFromPy`, rescales it in
`UpdateModel -> ChangeModelScale`, and calls
`SystemManager::ChangeModels -> TrackerManager::ChangeModels ->
SummerTracker::ChangeModels`. The latter reconstructs the tracked object and
tracker, restores the current pose, and reinitializes tracking on the current
frame.

## 9. Geometry loss

The active online loss is in `examples/run_example.py`:

```text
loss = neg_iou_loss(images_pred[:, 3], images_gt[:, 3])
       + 0.1 * laplacian_loss
```

`neg_iou_loss` computes `1 - mean(IoU)` over the batch. The predicted alpha
channel is the differentiably rendered silhouette, and `images_gt` is the SAM
silhouette. `flatten_loss` is computed by `Model.forward` but is not included
in the active loss.

The current `neg_iou_loss` reduces all observations to one scalar before
backpropagation. Soft per-observation confidence weighting will therefore
eventually require an unreduced IoU loss, but that is outside the first hard
gating experiment.

## 10. Rendered-mask sources

There are two render paths:

1. **C++ tracking renderer**:
   `summer::Renderer::DrawMask` in
   `tracker/tracker/summer/src/mbt/renderer.cc` calls `RenderSilhouette` and
   downloads `Renderer::MASK`. `SummerTracker::Estimate` stores this in
   `TrackingResult.mask`.
2. **Python differentiable renderer**:
   `examples/run_example.py` calls `rasterizer(mesh)`; alpha channel
   `images_pred[:, 3]` is the silhouette used by the geometry loss.

Crucial distinction: `LocalStorageCommunicator::SentDatas2Py` writes
`trackingResult->dataGroup.mask`, not `trackingResult->mask`. The former comes
from `ComConfidenceMaskFromProb` and is **not** the geometry-rendered mask.
Therefore files currently found in `cmc/mask/` must not be labeled
`M_render`.

For the first Render-Mask Confidence Gating experiment, the least invasive and
coordinate-consistent source is a no-gradient Python render of the current
mesh immediately before optimization, using the same transformed poses,
intrinsics, padding, flip, and resolution as the SAM targets.

## 11. Evaluation metrics

Evaluation is embedded at the end of `tracker/example/eg_BIT.cpp::main` and is
currently explicit only for MOPED:

- `ComADD` computes mean ADD.
- `ComAddAUC` computes ADD AUC for configured distance ranges.
- Implementations are in `tracker/ar_utils/data_io/data_loader.cpp`.

Per-frame results are also written through `SaveTrackingResult` in the same
utility module. RBOT-specific success-rate evaluation is not implemented in
the visible `eg_BIT.cpp` path. The code loads RBOT ground truth and uses it for
success checks/recovery, but does not print a final RBOT metric block analogous
to MOPED. A reproducible RBOT evaluator must therefore be added or matched to
the benchmark protocol before reporting RBOT numbers.

## 12. Core data structures and process boundary

### C++ structures

Defined in `tracker/definition/global_definition.h`:

- `StdData`: source/frame index, RGB/depth/mask, GT pose, pose scale, timestamp.
- `CamParams`: image dimensions and pinhole intrinsics.
- `MeshModel`: model ID and OBJ path.
- `DataGroup`: RGB, tracker probability, confidence mask, adjusted intrinsics,
  tracked pose, source/target ROIs, target size, and normalized view vector.
- `TrackingResult`: pose, GT, masks/probability/overlay, ROI, `DataGroup`, and
  tracking status fields.
- `ProcessResult`: pointer to `TrackingResult`.

### Python structures

`examples/communication.py::DataGroups` holds lists/arrays of RGB images,
probability maps, confidence masks, 4x4 poses, compact intrinsics, origin ROIs,
target ROIs, and target sizes.

### Disk protocol

`tracker/interface/communication.{h,cpp}` writes indexed files under:

```text
cmc/img
cmc/prob
cmc/mask
cmc/pose
cmc/K
cmc/originRoi
cmc/targetRoi
cmc/targetSize
cmc/deformed_model
```

Poses sent to Python have translation divided by the initial object's longest
bounding-box side, matching the normalized mesh scale. Python discovers a
batch when enough `K` files exist; C++ waits for a new mesh after the expected
candidate count is reached.

## 13. Baseline risks relevant to the proposed work

1. C++/Python synchronization is count-based and blocking. Rejecting all
   observations must still produce a next model artifact (or introduce an
   explicit acknowledgement), otherwise C++ deadlocks waiting for a mesh.
2. Candidate views are appended in C++ before Python can compute SAM/render
   IoU. A rejected frame therefore still suppresses nearby later viewpoints in
   the baseline selector unless the protocol is expanded. The first minimal
   experiment can retain this behavior for isolation, but it must be logged
   and documented.
3. Python's persistent `baseDataGroups_` currently includes the whole first
   received group. Confidence filtering must ensure rejected observations do
   not later re-enter geometry optimization through this cache.
4. The C++ render mask and image-derived confidence mask are currently easy to
   confuse due to naming. Confidence code must use explicit names.
5. No pose optimizer quality scalar is exposed today.
6. Baseline visualization uses `cv2.imshow`/Qt and may fail in a headless
   environment; headless support should be treated as reproducibility work,
   not an algorithmic modification.
7. The generated config contains absolute dataset paths but has no experiment
   result schema, seed field, or confidence settings yet.

