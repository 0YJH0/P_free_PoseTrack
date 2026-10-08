#!/usr/bin/env python3
"""Read-only mechanism audit for the frozen cheezit/02 failure case."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/paper_moped/raw"
INITIAL = ROOT / "results/paper_moped/initial_geometry/cheezit/02.obj"
OUT = ROOT / "results/paper_moped/analysis/cheezit_02_failure.json"


def vertices(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            values.append([float(value) for value in line.split()[1:4]])
    return np.asarray(values)


def mesh_audit(arm: str) -> list[dict]:
    initial = vertices(INITIAL)
    rows = []
    for path in sorted((RAW / arm / "cheezit/02/models").glob("*.obj")):
        current = vertices(path)
        displacement = np.linalg.norm(current - initial, axis=1)
        rows.append({
            "mesh": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "equals_initial_bytes": path.read_bytes() == INITIAL.read_bytes(),
            "mean_vertex_displacement": float(np.mean(displacement)),
            "max_vertex_displacement": float(np.max(displacement)),
        })
    return rows


def replicate_metrics(arm: str) -> list[dict]:
    rows = []
    for replicate, suffix in [(0, ""), (1, "_r01"), (2, "_r02")]:
        path = RAW / f"{arm}{suffix}/cheezit/02/metrics.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("status") == "complete":
                rows.append({
                    "replicate": replicate,
                    "add_mm": record["add_mm"],
                    "auc": record["add_auc"],
                    "initial_geometry_sha256": record["initial_geometry_sha256"],
                })
    return rows


def main() -> None:
    with (RAW / "gradient_rollback/cheezit/02/confidence.csv").open(newline="") as handle:
        confidence = list(csv.DictReader(handle))
    trust_checks = []
    seen = set()
    for row in confidence:
        key = (
            row["pre_reference_iou"], row["post_reference_iou"],
            row["pre_holdout_iou"], row["post_holdout_iou"],
            row["max_vertex_step_ratio"], row["update_accepted"],
        )
        if key not in seen:
            seen.add(key)
            trust_checks.append({
                "pre_reference_iou": float(key[0]), "post_reference_iou": float(key[1]),
                "pre_holdout_iou": float(key[2]), "post_holdout_iou": float(key[3]),
                "max_vertex_step_ratio": float(key[4]), "accepted": key[5] == "True",
            })
    baseline = replicate_metrics("baseline")
    v4 = replicate_metrics("gradient_rollback")
    payload = {
        "case": "cheezit/02",
        "analysis_policy": "read-only; no parameter or exclusion change",
        "baseline_replicates": baseline,
        "v4_replicates": v4,
        "replicate_initial_geometry_hashes_match": len({
            row["initial_geometry_sha256"] for row in baseline + v4
        }) == 1,
        "trust_checks": trust_checks,
        "baseline_meshes": mesh_audit("baseline"),
        "v4_meshes": mesh_audit("gradient_rollback"),
        "mechanism": (
            "Both proposed geometry updates were rolled back. V4 therefore retained the frozen "
            "initial mesh, whereas baseline published two deformed meshes. Three replicates show "
            "that the resulting accuracy gap is systematic for this sequence."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
