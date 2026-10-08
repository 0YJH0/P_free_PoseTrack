#!/usr/bin/env python3
"""Variance components and mixed-effects inference for repeated paired runs."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/paper_moped/raw"
OUT = ROOT / "results/paper_moped/analysis"
COHORTS = {
    "development": ({"duplo_dude", "rinse_aid", "toy_plane", "vim_mug"}, 21),
    "confirmation": ({"cheezit", "duster", "graphics_card", "orange_drill", "pouch", "remote"}, 33),
}
REPLICATES = [(0, ""), (1, "_r01"), (2, "_r02")]


def load(arm: str, method: str, replicate: int, objects: set[str], expected: int) -> list[dict]:
    rows = []
    for path in sorted(RAW.glob(f"{arm}/*/*/metrics.json")):
        record = json.loads(path.read_text())
        if record["object"] in objects and record["status"] == "complete":
            rows.append({
                "method": method, "replicate": replicate,
                "object": record["object"], "sequence": record["sequence"],
                "add_mm": record["add_mm"], "auc": record["add_auc"],
                "initial_geometry_sha256": record.get("initial_geometry_sha256"),
            })
    if len(rows) != expected:
        raise SystemExit(f"{arm}: expected {expected} complete records, got {len(rows)}")
    return rows


def mixed_model(rows: list[dict], outcome: str) -> dict:
    y = np.asarray([row[outcome] for row in rows], float)
    method = np.asarray([row["method"] == "v4" for row in rows], float)
    x = np.column_stack((np.ones(len(rows)), method))
    objects = sorted({row["object"] for row in rows})
    sequences = sorted({(row["object"], row["sequence"]) for row in rows})
    zo = np.asarray([[row["object"] == item for item in objects] for row in rows], float)
    zs = np.asarray([[(row["object"], row["sequence"]) == item for item in sequences] for row in rows], float)

    def fit_at(log_vars):
        vo, vs, ve = np.exp(log_vars)
        v = vo * (zo @ zo.T) + vs * (zs @ zs.T) + ve * np.eye(len(y))
        sign, logdet = np.linalg.slogdet(v)
        if sign <= 0: return 1e100, None
        vi = np.linalg.inv(v)
        xtvi = x.T @ vi
        cov_beta = np.linalg.inv(xtvi @ x)
        beta = cov_beta @ xtvi @ y
        residual = y - x @ beta
        objective = logdet + residual @ vi @ residual
        return objective, (beta, cov_beta)

    initial = np.log([max(np.var(y) * 0.1, 1e-4), max(np.var(y) * 0.5, 1e-4), max(np.var(y) * 0.4, 1e-4)])
    result = optimize.minimize(lambda value: fit_at(value)[0], initial, method="L-BFGS-B")
    beta, cov_beta = fit_at(result.x)[1]
    se = float(np.sqrt(cov_beta[1, 1]))
    z = float(beta[1] / se)
    return {
        "method_effect_v4_minus_baseline": float(beta[1]),
        "standard_error": se,
        "normal_approx_ci95": [float(beta[1] - 1.96 * se), float(beta[1] + 1.96 * se)],
        "z": z, "p_two_sided": float(2 * stats.norm.sf(abs(z))),
        "variance_object": float(np.exp(result.x[0])),
        "variance_sequence_within_object": float(np.exp(result.x[1])),
        "variance_run_residual": float(np.exp(result.x[2])),
        "optimizer_success": bool(result.success),
    }


def stability_summary(rows: list[dict], outcome: str, samples: int = 20000) -> dict:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["object"], row["sequence"]), []).append(row[outcome])
    sequence_keys = sorted({(row["object"], row["sequence"]) for row in rows})
    variances = {
        method: {
            (obj, seq): float(np.var(grouped[(method, obj, seq)], ddof=1))
            for obj, seq in sequence_keys
        }
        for method in ("baseline", "v4")
    }
    ratio = sum(variances["baseline"].values()) / max(sum(variances["v4"].values()), 1e-12)
    keys = sorted(variances["baseline"])
    baseline_sd = np.asarray([np.sqrt(variances["baseline"][key]) for key in keys])
    v4_sd = np.asarray([np.sqrt(variances["v4"][key]) for key in keys])
    sd_difference = baseline_sd - v4_sd
    nonzero = sd_difference[~np.isclose(sd_difference, 0.0)]
    wilcoxon = stats.wilcoxon(nonzero, alternative="greater", method="auto")
    wins = int(np.sum(nonzero > 0))
    sign_test = stats.binomtest(wins, len(nonzero), 0.5, alternative="greater")
    rng = np.random.default_rng(20261002)
    by_object = {}
    for obj, seq in variances["baseline"]:
        by_object.setdefault(obj, []).append(seq)
    by_object = {obj: sorted(seqs) for obj, seqs in by_object.items()}
    objects = sorted(by_object)
    boot = []
    for _ in range(samples):
        numerator = denominator = 0.0
        for obj in rng.choice(objects, len(objects), replace=True):
            seqs = by_object[obj]
            for seq in rng.choice(seqs, len(seqs), replace=True):
                numerator += variances["baseline"][(obj, seq)]
                denominator += variances["v4"][(obj, seq)]
        boot.append(numerator / max(denominator, 1e-12))
    ranges = {}
    for method in ("baseline", "v4"):
        values = [
            max(grouped[(method, obj, seq)]) - min(grouped[(method, obj, seq)])
            for obj, seq in sequence_keys
        ]
        ranges[method] = {
            "mean_range": float(np.mean(values)),
            "p95_range": float(np.percentile(values, 95)),
            "max_range": float(max(values)),
        }
    return {
        "pooled_within_sequence_variance_ratio_baseline_over_v4": float(ratio),
        "variance_ratio_hierarchical_bootstrap_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "probability_variance_ratio_gt_1": float(np.mean(np.asarray(boot) > 1.0)),
        "median_within_sequence_sd": {
            "baseline": float(np.median(baseline_sd)), "v4": float(np.median(v4_sd))
        },
        "sequences_lower_sd_v4": wins,
        "sequences_higher_sd_v4": int(len(nonzero) - wins),
        "sequences_sd_tied": int(len(sd_difference) - len(nonzero)),
        "paired_wilcoxon_baseline_sd_greater_p": float(wilcoxon.pvalue),
        "paired_sign_test_baseline_sd_greater_p": float(sign_test.pvalue),
        "run_range": ranges,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=sorted(COHORTS), default="development")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    objects, expected = COHORTS[args.cohort]
    rows = []
    for replicate, suffix in REPLICATES:
        rows += load("baseline" + suffix, "baseline", replicate, objects, expected)
        rows += load("gradient_rollback" + suffix, "v4", replicate, objects, expected)
    hashes = {}
    for row in rows:
        hashes.setdefault((row["object"], row["sequence"]), set()).add(row["initial_geometry_sha256"])
    hashes_match = all(len(values) == 1 and None not in values for values in hashes.values())
    if not hashes_match:
        raise SystemExit("Strict pairing violation: replicate initial geometry hashes differ")
    summary = {
        "cohort": args.cohort,
        "n_objects": len(objects),
        "n_sequences": expected,
        "n_observations": len(rows),
        "n_replicates_per_method": len(REPLICATES),
        "all_initial_geometry_hashes_match": hashes_match,
    }
    for outcome in ("add_mm", "auc"):
        summary[outcome] = mixed_model(rows, outcome)
        method_sds = {}
        for method in ("baseline", "v4"):
            grouped = {}
            for row in rows:
                if row["method"] == method:
                    grouped.setdefault((row["object"], row["sequence"]), []).append(row[outcome])
            method_sds[method] = float(np.mean([np.std(value, ddof=1) for value in grouped.values()]))
        summary[outcome]["mean_within_sequence_sd"] = method_sds
        summary[outcome]["stability"] = stability_summary(rows, outcome, args.bootstrap_samples)
        if outcome == "add_mm":
            lower, upper = summary[outcome]["normal_approx_ci95"]
            margin = 2.0
            summary[outcome]["equivalence_margin_mm"] = margin
            summary[outcome]["equivalent_within_margin"] = lower > -margin and upper < margin
    raw_p = {name: summary[name]["p_two_sided"] for name in ("add_mm", "auc")}
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        value = min(1.0, raw_p[name] * (len(ordered) - rank))
        running = max(running, value)
        adjusted[name] = running
    summary["holm_adjusted_p_across_add_and_auc"] = adjusted
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "repeated_runs" if args.cohort == "development" else "confirmation_repeated_runs"
    summary_name = (
        "mixed_effects_three_replicates.json" if args.cohort == "development"
        else "confirmation_mixed_effects_three_replicates.json"
    )
    with (OUT / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUT / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
