# V8 frozen observable-deformation protocol

Frozen before any V8 run on 2026-10-06.

V8 retains V7's fixed 50/50 temporal cross-fit and next-batch shadow
confirmation.  It changes only the proposal parameterization.  Before joint
optimization, per-view silhouette gradients are measured on at most the eight
highest-weight fitting observations.  A vertex is deformable only when it has
nonzero support in at least two views, directional agreement at least 0.60, and
lies in the strongest 30% of all vertices by agreement-weighted gradient
magnitude.  The global mesh center is frozen so pose error cannot be represented
as a geometry translation.  The mask is fixed for the complete update.

Development panel, comparators, alpha grid, risk thresholds, optimizer settings,
and advancement criteria are identical to V7.  In addition, the active vertex
fraction and gradient agreement are reported. Untouched RBOT objects remain
sealed. No V8 threshold may be changed after observing panel results.
