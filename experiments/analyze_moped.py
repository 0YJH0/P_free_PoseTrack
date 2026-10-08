#!/usr/bin/env python3
"""Aggregate paired MOPED runs with leakage-safe selection and inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "paper_moped"
RAW_ROOT = RESULT_ROOT / "raw"
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
ANALYSIS_OBJECTS = {"duplo_dude", "rinse_aid", "toy_plane", "vim_mug"}
CONFIRMATION_OBJECTS = {
    "cheezit", "duster", "graphics_card", "orange_drill", "pouch", "remote"
}
EXPECTED_VALIDATION_PAIRS = 4
EXPECTED_TEST_PAIRS = 17


def load_arm(arm: str) -> dict[tuple[str, str], dict]:
    records = {}
    for path in sorted((RAW_ROOT / arm).glob("*/*/metrics.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "complete" or record.get("add_mm") is None:
            continue
        records[(record["object"], record["sequence"])] = record
    return records


def paired_records(arm: str, split: str) -> list[dict]:
    baseline = load_arm("baseline")
    method = load_arm(arm)
    rows = []
    for key in sorted(set(baseline) & set(method)):
        object_name, sequence = key
        if object_name not in ANALYSIS_OBJECTS:
            continue
        if split == "validation" and sequence != "00":
            continue
        if split == "test" and sequence == "00":
            continue
        base, gate = baseline[key], method[key]
        rows.append({
            "object": object_name,
            "sequence": sequence,
            "baseline_add_mm": base["add_mm"],
            "method_add_mm": gate["add_mm"],
            "delta_add_mm": gate["add_mm"] - base["add_mm"],
            "baseline_auc": base["add_auc"],
            "method_auc": gate["add_auc"],
            "delta_auc": gate["add_auc"] - base["add_auc"],
            "baseline_candidates": base.get("candidate_keyframes"),
            "method_candidates": gate.get("candidate_keyframes"),
            "method_accepted": gate.get("accepted_keyframes"),
            "initial_geometry_sha256_match": (
                base.get("initial_geometry_sha256") == gate.get("initial_geometry_sha256")
            ),
        })
    return rows


def mean(values) -> float:
    return float(np.mean(list(values)))


def macro_delta(rows: list[dict], field: str) -> float:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["object"]].append(row[field])
    return mean(mean(values) for values in grouped.values())


def hierarchical_bootstrap(rows: list[dict], field: str, samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["object"]].append(row[field])
    objects = sorted(grouped)
    estimates = []
    for _ in range(samples):
        sampled_object_means = []
        for object_name in rng.choices(objects, k=len(objects)):
            values = grouped[object_name]
            sampled = rng.choices(values, k=len(values))
            sampled_object_means.append(mean(sampled))
        estimates.append(mean(sampled_object_means))
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def paired_inference(rows: list[dict], field: str, favorable: str, bootstrap_samples: int) -> dict:
    values = np.asarray([row[field] for row in rows], dtype=float)
    nonzero = values[~np.isclose(values, 0.0)]
    if len(nonzero):
        wilcoxon = stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
        favorable_count = int(np.sum(nonzero < 0 if favorable == "negative" else nonzero > 0))
        sign = stats.binomtest(favorable_count, len(nonzero), 0.5, alternative="two-sided")
    else:
        wilcoxon = sign = None
        favorable_count = 0
    lower, upper = hierarchical_bootstrap(rows, field, bootstrap_samples, seed=20250930)
    return {
        "n_sequences": len(values),
        "mean_sequence_delta": mean(values),
        "median_sequence_delta": float(np.median(values)),
        "macro_object_delta": macro_delta(rows, field),
        "macro_object_delta_ci95_hierarchical_bootstrap": [lower, upper],
        "wilcoxon_two_sided_statistic": float(wilcoxon.statistic) if wilcoxon else None,
        "wilcoxon_two_sided_p": float(wilcoxon.pvalue) if wilcoxon else None,
        "sign_test_two_sided_p": float(sign.pvalue) if sign else None,
        "favorable": favorable_count,
        "unfavorable": int(len(nonzero) - favorable_count),
        "ties": int(len(values) - len(nonzero)),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_threshold(arms: list[str]) -> None:
    summaries = []
    for arm in arms:
        rows = paired_records(arm, "validation")
        if len(rows) != EXPECTED_VALIDATION_PAIRS:
            raise SystemExit(
                f"Incomplete validation arm {arm}: expected {EXPECTED_VALIDATION_PAIRS}, got {len(rows)}"
            )
        summaries.append({
            "arm": arm,
            "n": len(rows),
            "macro_delta_add_mm": macro_delta(rows, "delta_add_mm"),
            "macro_delta_auc": macro_delta(rows, "delta_auc"),
            "all_initial_geometry_hashes_match": all(
                row["initial_geometry_sha256_match"] for row in rows
            ),
        })
    if not summaries:
        raise SystemExit("No complete validation pairs found")
    summaries.sort(key=lambda row: (row["macro_delta_add_mm"], -row["macro_delta_auc"], row["arm"]))
    chosen = summaries[0]["arm"]
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    write_rows(ANALYSIS_ROOT / "validation_threshold_selection.csv", summaries)
    payload = {
        "selection_split": "sequence 00 from each of 4 analysis objects",
        "primary_selection_metric": "macro object mean paired ADD delta (lower is better)",
        "tie_breaker": "macro object mean paired AUC delta (higher is better)",
        "chosen_arm": chosen,
        "candidates": summaries,
    }
    (ANALYSIS_ROOT / "selected_threshold.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def analyze_test(arm: str, bootstrap_samples: int) -> None:
    rows = paired_records(arm, "test")
    if not rows:
        raise SystemExit(f"No complete test pairs found for {arm}")
    if len(rows) != EXPECTED_TEST_PAIRS:
        raise SystemExit(
            f"Incomplete test arm {arm}: expected {EXPECTED_TEST_PAIRS}, got {len(rows)}"
        )
    if not all(row["initial_geometry_sha256_match"] for row in rows):
        raise SystemExit("Strict pairing violation: initial geometry hashes differ")
    acceptance = [
        row["method_accepted"] / row["method_candidates"]
        for row in rows
        if row["method_candidates"]
    ]
    summary = {
        "arm": arm,
        "split": "held-out sequences != 00",
        "n_objects": len({row["object"] for row in rows}),
        "n_sequences": len(rows),
        "all_initial_geometry_hashes_match": True,
        "baseline_mean_add_mm": mean(row["baseline_add_mm"] for row in rows),
        "method_mean_add_mm": mean(row["method_add_mm"] for row in rows),
        "baseline_mean_auc": mean(row["baseline_auc"] for row in rows),
        "method_mean_auc": mean(row["method_auc"] for row in rows),
        "mean_acceptance_rate": mean(acceptance) if acceptance else None,
        "add_mm": paired_inference(rows, "delta_add_mm", "negative", bootstrap_samples),
        "auc": paired_inference(rows, "delta_auc", "positive", bootstrap_samples),
        "multiple_testing_note": "ADD is primary; AUC is secondary and descriptive unless Holm-adjusted.",
    }
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    write_rows(ANALYSIS_ROOT / f"test_pairs_{arm}.csv", rows)
    (ANALYSIS_ROOT / f"test_summary_{arm}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def analyze_development(arms: list[str], bootstrap_samples: int) -> None:
    summaries = []
    for arm in arms:
        rows = paired_records(arm, "all")
        if len(rows) != EXPECTED_VALIDATION_PAIRS + EXPECTED_TEST_PAIRS:
            raise SystemExit(f"Incomplete development arm {arm}: got {len(rows)} pairs")
        summaries.append({
            "arm": arm,
            "n_objects": len({row["object"] for row in rows}),
            "n_sequences": len(rows),
            "baseline_mean_add_mm": mean(row["baseline_add_mm"] for row in rows),
            "method_mean_add_mm": mean(row["method_add_mm"] for row in rows),
            "baseline_mean_auc": mean(row["baseline_auc"] for row in rows),
            "method_mean_auc": mean(row["method_auc"] for row in rows),
            "add_mm": paired_inference(rows, "delta_add_mm", "negative", bootstrap_samples),
            "auc": paired_inference(rows, "delta_auc", "positive", bootstrap_samples),
            "all_initial_geometry_hashes_match": all(
                row["initial_geometry_sha256_match"] for row in rows
            ),
        })
        write_rows(ANALYSIS_ROOT / f"development_pairs_{arm}.csv", rows)
    payload = {
        "scope": "reused 21-sequence development benchmark; not independent confirmation",
        "methods": summaries,
    }
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_ROOT / "development_v2_v3.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def analyze_confirmation(arm: str, bootstrap_samples: int) -> None:
    """Analyze the frozen, never-tuned-on object confirmation cohort."""
    baseline = load_arm("baseline")
    method = load_arm(arm)
    rows = []
    for key in sorted(set(baseline) & set(method)):
        object_name, sequence = key
        if object_name not in CONFIRMATION_OBJECTS:
            continue
        base, candidate = baseline[key], method[key]
        rows.append({
            "object": object_name,
            "sequence": sequence,
            "baseline_add_mm": base["add_mm"],
            "method_add_mm": candidate["add_mm"],
            "delta_add_mm": candidate["add_mm"] - base["add_mm"],
            "baseline_auc": base["add_auc"],
            "method_auc": candidate["add_auc"],
            "delta_auc": candidate["add_auc"] - base["add_auc"],
            "baseline_runtime_seconds": base.get("runtime_seconds"),
            "method_runtime_seconds": candidate.get("runtime_seconds"),
            "initial_geometry_sha256_match": (
                base.get("initial_geometry_sha256") == candidate.get("initial_geometry_sha256")
            ),
        })
    if len(rows) != 33:
        raise SystemExit(f"Incomplete independent confirmation: expected 33 pairs, got {len(rows)}")
    if not all(row["initial_geometry_sha256_match"] for row in rows):
        raise SystemExit("Strict pairing violation: initial geometry hashes differ")

    object_summaries = []
    for object_name in sorted(CONFIRMATION_OBJECTS):
        subset = [row for row in rows if row["object"] == object_name]
        object_summaries.append({
            "object": object_name,
            "n_sequences": len(subset),
            "baseline_mean_add_mm": mean(row["baseline_add_mm"] for row in subset),
            "method_mean_add_mm": mean(row["method_add_mm"] for row in subset),
            "delta_add_mm": mean(row["delta_add_mm"] for row in subset),
            "delta_auc": mean(row["delta_auc"] for row in subset),
        })

    add = paired_inference(rows, "delta_add_mm", "negative", bootstrap_samples)
    auc = paired_inference(rows, "delta_auc", "positive", bootstrap_samples)
    add_ci = add["macro_object_delta_ci95_hierarchical_bootstrap"]
    deltas = np.asarray([row["delta_add_mm"] for row in rows])
    summary = {
        "arm": arm,
        "scope": "preregistered-style independent confirmation on six objects unused in development",
        "n_objects": 6,
        "n_sequences": 33,
        "all_initial_geometry_hashes_match": True,
        "baseline_mean_add_mm": mean(row["baseline_add_mm"] for row in rows),
        "method_mean_add_mm": mean(row["method_add_mm"] for row in rows),
        "baseline_mean_auc": mean(row["baseline_auc"] for row in rows),
        "method_mean_auc": mean(row["method_auc"] for row in rows),
        "add_mm": add,
        "auc": auc,
        "tail_risk": {
            "largest_improvement_mm": float(-np.min(deltas)),
            "largest_degradation_mm": float(np.max(deltas)),
            "degradations_over_2mm": int(np.sum(deltas > 2.0)),
            "degradations_over_10mm": int(np.sum(deltas > 10.0)),
        },
        "equivalence_margin_mm": 2.0,
        "macro_add_equivalent_at_95pct": bool(add_ci[0] >= -2.0 and add_ci[1] <= 2.0),
        "object_summaries": object_summaries,
        "inference_note": (
            "The object-cluster hierarchical bootstrap is primary. Sequence-level Wilcoxon and sign "
            "tests are sensitivity analyses because sequences within an object are not independent."
        ),
        "data_use_note": "No threshold, architecture, or exclusion was changed after viewing this cohort.",
    }
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    write_rows(ANALYSIS_ROOT / f"confirmation_pairs_{arm}.csv", rows)
    (ANALYSIS_ROOT / "confirmation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "test", "development", "confirmation"])
    parser.add_argument("--arms", nargs="+", default=["gate_t070", "gate_t075", "gate_t080", "gate_t085"])
    parser.add_argument("--arm")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    if args.stage == "select":
        select_threshold(args.arms)
    elif args.stage == "test":
        arm = args.arm
        if arm is None:
            selected = json.loads((ANALYSIS_ROOT / "selected_threshold.json").read_text())
            arm = selected["chosen_arm"]
        analyze_test(arm, args.bootstrap_samples)
    elif args.stage == "development":
        arms = args.arms
        if arms == ["gate_t070", "gate_t075", "gate_t080", "gate_t085"]:
            arms = ["fusion_soft_adaptive", "gradient_soft"]
        analyze_development(arms, args.bootstrap_samples)
    else:
        analyze_confirmation(args.arm or "gradient_rollback", args.bootstrap_samples)


if __name__ == "__main__":
    main()
