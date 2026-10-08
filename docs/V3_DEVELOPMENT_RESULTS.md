# v2/v3 reused-data development benchmark

## Scope

All 21 available sequences from `duplo_dude`, `rinse_aid`, `toy_plane`, and
`vim_mug` are reused here for method development. This maximizes evidence for
failure analysis, but it is not an independent confirmation set. Every arm
uses the same frozen initial mesh as its paired BIT baseline.

## Methods

- **BIT baseline:** uniform observation influence.
- **v2 fusion soft:** causal fusion of mask, pose, visibility, and temporal
  signals with adaptive soft weights.
- **v3 gradient soft:** v2 confidence fused with cosine consistency between
  each candidate geometry gradient and the reference/history gradient anchor.

## Full 21-sequence results

| Method | Mean ADD (mm) ↓ | Mean delta | Macro delta | 95% hierarchical CI | Wins/losses/ties |
|---|---:|---:|---:|---:|---:|
| BIT baseline | 22.4507 | — | — | — | — |
| v2 fusion soft | 23.8979 | +1.4472 | +1.4877 | [-1.0470, 7.4929] | 12/6/3 |
| v3 gradient soft | 23.2023 | +0.7516 | +0.8751 | [-1.8910, 6.6442] | 12/6/3 |
| v4 gradient rollback | 23.0953 | +0.6446 | +0.6603 | [-1.0820, 3.5323] | 11/7/3 |

For v3, Wilcoxon p=0.2837 and sign-test p=0.2379. Mean AUC changes from
77.5355 to 76.8091; macro AUC delta is -0.8500. Neither v2 nor v3 supports a
statistically reliable aggregate improvement.

The median ADD delta is favorable (-0.0593 mm for v2 and -0.2286 mm for v3),
but `toy_plane/03` has a catastrophic +38.3149/+35.2208 mm delta and dominates
the mean. A conservative v3 trust-floor pilot (minimum weight 0.75) still has
+34.5043 mm on this sequence, so simple weight clipping does not solve the
closed-loop trajectory instability.

## Conclusion

Gradient consistency is a meaningful diagnostic and reduces v2's average
damage, but the current weighted-update architecture is not ready as a claimed
improvement. The next architectural step must control the *whole geometry
update* (rollback/trust-region acceptance using a post-update validation
criterion), rather than only reweighting observations inside an update.

## v4 update-level verification

The update-level controller reserves the lowest-confidence candidate as a
causal holdout, checks reference and holdout IoU before/after optimization, and
rejects a mesh whose maximum vertex step exceeds 15% of the current object
bounding-box diagonal. Meshes are published atomically. Across 21 sequences it
accepted 12 updates and rolled back 15. It further reduced average and macro
damage to +0.6446 and +0.6603 mm, but remained non-significant (Wilcoxon
p=0.6095) and did not beat BIT overall. Thus the code-level safeguard works,
but it is not yet an accuracy improvement.

Machine-readable statistics are in
`results/paper_moped/analysis/development_v2_v3.json`.
