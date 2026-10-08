# Independent object-level confirmation of v4

## Frozen design

The v4 architecture and threshold (0.65) were frozen after development on
`duplo_dude`, `rinse_aid`, `toy_plane`, and `vim_mug`. Confirmation uses six
previously unused MOPED objects: `cheezit`, `duster`, `graphics_card`,
`orange_drill`, `pouch`, and `remote`. All 33 available sequences were included.
Every baseline/v4 pair reused a byte-identical initial mesh; all 33 SHA-256
checks passed. No confirmation result was used to change the method, threshold,
or exclusions.

## Results

| Endpoint | Baseline | v4 | Sequence-weighted change | Object-macro change (95% hierarchical CI) |
|---|---:|---:|---:|---:|
| ADD, mm (lower is better) | 32.0188 | 29.5121 | -2.5067 | -2.4909 [-7.9555, 3.1188] |
| ADD AUC (higher is better) | 71.0756 | 73.5293 | +2.4537 | +2.4002 [-0.7062, 5.1164] |

V4 improved mean ADD for five of six objects. The object-level mean changes
were +4.0050 mm (`cheezit`), -0.1001 (`duster`), -4.2933
(`graphics_card`), -2.5797 (`orange_drill`), -4.2725 (`pouch`), and -7.7049
(`remote`). At sequence level there were 15 improvements, 8 degradations, and
10 exact ties. The paired Wilcoxon sensitivity analysis gave p=0.0327 for ADD,
but the sign test gave p=0.2100.

The tail is important: the largest improvement was 43.2392 mm and the largest
degradation was 49.2376 mm. Three sequences degraded by more than 2 mm and two
by more than 10 mm. The large positive failure is concentrated in
`cheezit/02`; it must remain in all primary aggregates.

## Defensible conclusion

The independent cohort provides encouraging external evidence: mean ADD fell
by 2.51 mm (7.83%) and five of six object means improved. It does **not** yet
establish a statistically conclusive cross-object mean-accuracy gain because
the primary object-cluster bootstrap interval crosses zero. The sequence-level
Wilcoxon result is supportive but cannot replace object-level inference because
sequences from the same object are correlated.

Together with the three-run development experiment, the strongest current
claim is: v4 materially improves run-to-run repeatability while preserving
mean accuracy on the development cohort, and shows a promising but
object-heterogeneous mean-accuracy improvement on six independently held-out
objects. It is not yet defensible to claim universal accuracy superiority.

Machine-readable outputs are
`results/paper_moped/analysis/confirmation_summary.json` and
`results/paper_moped/analysis/confirmation_pairs_gradient_rollback.csv`.

This single-run result is superseded for inferential claims by the three-run
analysis in `docs/CONFIRMATION_REPEATED_RUNS.md`. In particular, repeated runs
reduce the estimated ADD effect to -1.8075 mm and show that the development-set
stability advantage does not significantly generalize.
