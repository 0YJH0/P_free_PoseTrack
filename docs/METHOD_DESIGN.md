# Method Design

## Current phase

The official BIT geometry update uses tracked poses and segmented silhouettes:

```text
T_i, M_i -> geometry optimization
```

The first controlled extension is hard Render-Mask Confidence Gating:

```text
C_mask(i) = IoU(M_seg(i), M_render(i))
T_i, M_i, C_mask(i) -> accept/reject -> geometry optimization
```

`M_render` is produced from the current pre-update geometry using the same
pose, intrinsics, coordinate conversion, and rasterizer as the geometry loss.
The gate is disabled by default and does not affect pose tracking.

## v2 multi-signal soft confidence

The v2 observation confidence combines four causal signals: render-mask IoU;
pose innovation from depth-normalized translation and frame-gap-normalized
rotation; visibility from silhouette-area agreement and rendered image
fraction; and temporal IoU consistency against an exponential moving average
containing only earlier candidates.

The scores use a weighted geometric mean (default 0.40/0.20/0.20/0.20), so a
near-zero reliability component cannot be hidden by one high component. A
causal robust threshold is computed as the clipped history median minus 1.5
scaled MAD, falling back to the configured base threshold until five prior
observations exist. `sigmoid((confidence-threshold)/temperature)` produces the
soft weight applied directly to each silhouette IoU loss term; reference
observations retain weight 1. Legacy disabled and hard-IoU modes remain
available for ablation. Every component, threshold, and applied weight is
written to CSV.

Mask refinement remains outside the current scope.

## v3 gradient-consistent influence

For each candidate, v3 differentiates its silhouette loss with respect to the
current geometry parameters. The normalized gradient is compared with an
anchor formed from reference-observation gradients and a causal EMA of earlier
candidate directions. Cosine similarity is mapped to [0, 1] and geometrically
fused with the v2 soft weight. This directly estimates whether an observation
supports or conflicts with the reliable geometry update direction, without
training or labels.

## v4 causal update acceptance

The lowest-confidence candidate in each batch is withheld from optimization.
After the tentative update, v4 requires both reference and holdout silhouette
IoU to remain within a configured tolerance and caps the maximum vertex step
at 15% of the current object bounding-box diagonal. Failed updates atomically
publish the previous mesh; successful updates are also atomically published to
prevent the tracker from observing partially written OBJ files.
