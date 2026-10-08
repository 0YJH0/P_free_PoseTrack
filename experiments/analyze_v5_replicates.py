#!/usr/bin/env python3
"""Three-replicate mixed-effects analysis for frozen V5 development runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analyze_replicates import mixed_model, stability_summary


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/paper_moped/raw"
OUT = ROOT / "results/paper_moped/analysis"
OBJECTS = {"duplo_dude", "rinse_aid", "toy_plane", "vim_mug"}
REPLICATES = [(0, ""), (1, "_r01"), (2, "_r02")]
ARMS = {"baseline": "baseline", "v4": "gradient_rollback", "v5": "joint_risk"}


def load(method: str, replicate: int, suffix: str) -> list[dict]:
    arm = ARMS[method] + suffix
    rows = []
    for path in sorted((RAW / arm).glob("*/*/metrics.json")):
        record = json.loads(path.read_text())
        if record["object"] in OBJECTS and record.get("status") == "complete":
            rows.append({
                "method": method,
                "replicate": replicate,
                "object": record["object"],
                "sequence": record["sequence"],
                "add_mm": record["add_mm"],
                "auc": record["add_auc"],
                "runtime_seconds": record["runtime_seconds"],
                "initial_geometry_sha256": record["initial_geometry_sha256"],
            })
    if len(rows) != 21:
        raise SystemExit(f"{arm}: expected 21 complete records, got {len(rows)}")
    return rows


def contrast(all_rows: list[dict], comparator: str, outcome: str) -> dict:
    selected = []
    for row in all_rows:
        if row["method"] not in {comparator, "v5"}:
            continue
        mapped = dict(row)
        mapped["method"] = "baseline" if row["method"] == comparator else "v4"
        selected.append(mapped)
    result = mixed_model(selected, outcome)
    result["contrast"] = f"v5_minus_{comparator}"
    if outcome == "add_mm":
        lower, upper = result["normal_approx_ci95"]
        result["equivalent_within_2mm"] = lower > -2.0 and upper < 2.0
    return result


def method_stability(rows: list[dict], outcome: str) -> dict:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["object"], row["sequence"]), []).append(
            row[outcome]
        )
    result = {}
    for method in ARMS:
        values = [
            np.asarray(grouped[(method, obj, seq)], float)
            for obj, seq in sorted({(r["object"], r["sequence"]) for r in rows})
        ]
        sds = np.asarray([np.std(value, ddof=1) for value in values])
        ranges = np.asarray([np.ptp(value) for value in values])
        result[method] = {
            "mean_within_sequence_sd": float(sds.mean()),
            "median_within_sequence_sd": float(np.median(sds)),
            "mean_range": float(ranges.mean()),
            "p95_range": float(np.percentile(ranges, 95)),
            "max_range": float(ranges.max()),
        }
    return result


def pairwise_stability(rows: list[dict], comparator: str, outcome: str) -> dict:
    selected = []
    for row in rows:
        if row["method"] not in {comparator, "v5"}:
            continue
        mapped = dict(row)
        mapped["method"] = "baseline" if row["method"] == comparator else "v4"
        selected.append(mapped)
    result = stability_summary(selected, outcome, samples=50000)
    result["ratio_interpretation"] = f"{comparator}_variance_over_v5_variance"
    return result


def main() -> None:
    rows = []
    for replicate, suffix in REPLICATES:
        for method in ARMS:
            rows.extend(load(method, replicate, suffix))
    hashes = {}
    for row in rows:
        hashes.setdefault((row["object"], row["sequence"]), set()).add(
            row["initial_geometry_sha256"]
        )
    if not all(len(values) == 1 for values in hashes.values()):
        raise SystemExit("Strict pairing violation: initial geometry hashes differ")

    summary = {
        "scope": "three-replicate frozen development comparison; not independent confirmation",
        "n_objects": 4,
        "n_sequences": 21,
        "n_replicates_per_method": 3,
        "n_observations": len(rows),
        "all_initial_geometry_hashes_match": True,
        "mean_by_method": {},
    }
    for method in ARMS:
        subset = [row for row in rows if row["method"] == method]
        summary["mean_by_method"][method] = {
            "add_mm": float(np.mean([row["add_mm"] for row in subset])),
            "auc": float(np.mean([row["auc"] for row in subset])),
            "runtime_seconds": float(np.mean([row["runtime_seconds"] for row in subset])),
        }
    for outcome in ("add_mm", "auc"):
        summary[outcome] = {
            "v5_vs_baseline": contrast(rows, "baseline", outcome),
            "v5_vs_v4": contrast(rows, "v4", outcome),
            "stability_by_method": method_stability(rows, outcome),
            "stability_v5_vs_baseline": pairwise_stability(rows, "baseline", outcome),
            "stability_v5_vs_v4": pairwise_stability(rows, "v4", outcome),
        }
    toy = [
        row for row in rows if row["object"] == "toy_plane" and row["sequence"] == "03"
    ]
    summary["toy_plane_03"] = {
        method: {
            "values_add_mm": [row["add_mm"] for row in toy if row["method"] == method],
            "mean_add_mm": float(np.mean([
                row["add_mm"] for row in toy if row["method"] == method
            ])),
        }
        for method in ARMS
    }
    summary["multiplicity_note"] = (
        "ADD v5-vs-baseline is primary. V5-vs-v4 and all AUC tests are secondary; "
        "unadjusted p-values are descriptive."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "v5_three_method_repeated_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUT / "v5_three_replicate_mixed_effects.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
