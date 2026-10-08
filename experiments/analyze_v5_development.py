#!/usr/bin/env python3
"""Frozen-parameter V5 development comparison against baseline and V4."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analyze_moped import ANALYSIS_ROOT, RAW_ROOT, paired_inference


OBJECTS = {"duplo_dude", "rinse_aid", "toy_plane", "vim_mug"}
ARMS = ("baseline", "gradient_rollback", "joint_risk")


def load(arm: str) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in sorted((RAW_ROOT / arm).glob("*/*/metrics.json")):
        record = json.loads(path.read_text())
        if record["object"] in OBJECTS and record.get("status") == "complete":
            rows[(record["object"], record["sequence"])] = record
    if len(rows) != 21:
        raise SystemExit(f"{arm}: expected 21 complete development records, got {len(rows)}")
    return rows


def update_counts(object_name: str, sequence: str) -> tuple[int, int]:
    path = RAW_ROOT / "joint_risk" / object_name / sequence / "confidence.csv"
    if not path.exists():
        return 0, 0
    with path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    updates = {}
    for row in records:
        key = (
            row.get("pre_tracking_risk"), row.get("post_tracking_risk"),
            row.get("max_vertex_step_ratio"),
        )
        updates[key] = row.get("update_accepted") == "True"
    return sum(updates.values()), len(updates) - sum(updates.values())


def main() -> None:
    data = {arm: load(arm) for arm in ARMS}
    rows = []
    for key in sorted(data["baseline"]):
        object_name, sequence = key
        baseline, v4, v5 = (data[arm][key] for arm in ARMS)
        accepted, rejected = update_counts(object_name, sequence)
        rows.append({
            "object": object_name,
            "sequence": sequence,
            "baseline_add_mm": baseline["add_mm"],
            "v4_add_mm": v4["add_mm"],
            "v5_add_mm": v5["add_mm"],
            "v5_minus_baseline_add_mm": v5["add_mm"] - baseline["add_mm"],
            "v5_minus_v4_add_mm": v5["add_mm"] - v4["add_mm"],
            "baseline_auc": baseline["add_auc"],
            "v4_auc": v4["add_auc"],
            "v5_auc": v5["add_auc"],
            "v5_minus_baseline_auc": v5["add_auc"] - baseline["add_auc"],
            "v5_minus_v4_auc": v5["add_auc"] - v4["add_auc"],
            "baseline_runtime_seconds": baseline["runtime_seconds"],
            "v4_runtime_seconds": v4["runtime_seconds"],
            "v5_runtime_seconds": v5["runtime_seconds"],
            "v5_updates_accepted": accepted,
            "v5_updates_rejected": rejected,
            "all_initial_geometry_hashes_match": len({
                baseline.get("initial_geometry_sha256"),
                v4.get("initial_geometry_sha256"),
                v5.get("initial_geometry_sha256"),
            }) == 1,
        })
    if not all(row["all_initial_geometry_hashes_match"] for row in rows):
        raise SystemExit("Strict pairing violation: initial geometry hashes differ")

    bootstrap_samples = 50000
    baseline_add = paired_inference(
        rows, "v5_minus_baseline_add_mm", "negative", bootstrap_samples
    )
    v4_add = paired_inference(rows, "v5_minus_v4_add_mm", "negative", bootstrap_samples)
    baseline_auc = paired_inference(
        rows, "v5_minus_baseline_auc", "positive", bootstrap_samples
    )
    v4_auc = paired_inference(rows, "v5_minus_v4_auc", "positive", bootstrap_samples)
    runtime = np.asarray([row["v5_runtime_seconds"] for row in rows])
    baseline_runtime = np.asarray([row["baseline_runtime_seconds"] for row in rows])
    v4_runtime = np.asarray([row["v4_runtime_seconds"] for row in rows])
    toy = next(row for row in rows if row["object"] == "toy_plane" and row["sequence"] == "03")
    payload = {
        "scope": "frozen-parameter exploratory development cohort; not independent confirmation",
        "n_objects": 4,
        "n_sequences": 21,
        "all_initial_geometry_hashes_match": True,
        "mean_add_mm": {
            "baseline": float(np.mean([row["baseline_add_mm"] for row in rows])),
            "v4": float(np.mean([row["v4_add_mm"] for row in rows])),
            "v5": float(np.mean([row["v5_add_mm"] for row in rows])),
        },
        "v5_vs_baseline_add": baseline_add,
        "v5_vs_v4_add": v4_add,
        "v5_vs_baseline_auc": baseline_auc,
        "v5_vs_v4_auc": v4_auc,
        "runtime": {
            "baseline_mean_seconds": float(baseline_runtime.mean()),
            "v4_mean_seconds": float(v4_runtime.mean()),
            "v5_mean_seconds": float(runtime.mean()),
            "v5_over_baseline_ratio": float(runtime.mean() / baseline_runtime.mean()),
            "v5_over_v4_ratio": float(runtime.mean() / v4_runtime.mean()),
        },
        "v5_updates": {
            "accepted": int(sum(row["v5_updates_accepted"] for row in rows)),
            "rejected": int(sum(row["v5_updates_rejected"] for row in rows)),
        },
        "toy_plane_03": toy,
        "interpretation": (
            "V5 improves development point estimates but remains non-significant. "
            "toy_plane/03 is unchanged from V4 because both V5 proposals exceeded "
            "the frozen 0.15 vertex-step safety bound and were rolled back."
        ),
    }
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_ROOT / "v5_development_pairs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (ANALYSIS_ROOT / "v5_development_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
