#!/usr/bin/env python3
"""Resumable, leakage-controlled paired MOPED experiment driver."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config" / "moped"
GENERATED_ROOT = ROOT / "config" / "paper_moped"
RESULT_ROOT = ROOT / "results" / "paper_moped"
ARTIFACT_ROOT = RESULT_ROOT / "initial_geometry"
DEVELOPMENT_OBJECTS = {"duplo_dude", "rinse_aid", "toy_plane", "vim_mug"}
CONFIRMATION_OBJECTS = {"cheezit", "duster", "graphics_card", "orange_drill", "pouch", "remote"}
# These archive sequences have incomplete mask streams and deterministically
# crash the official reference generator before preparation can complete.
TECHNICAL_EXCLUSIONS = {("duplo_dude", "03"), ("vim_mug", "03")}

ADD_RE = re.compile(r"ADD Error in mm:\s*([0-9.eE+-]+)")
AUC_RE = re.compile(r"ADD AUC Score in mm:\s*([0-9.eE+-]+)")
FAIL_RE = re.compile(r"fail nums\s+(\d+)")


def sequences(allowed_objects=DEVELOPMENT_OBJECTS) -> list[tuple[str, str, Path]]:
    items = []
    for object_dir in sorted(CONFIG_ROOT.iterdir()):
        if object_dir.name not in allowed_objects:
            continue
        evaluation = object_dir / "evaluation"
        if not evaluation.is_dir():
            continue
        for config_path in sorted(evaluation.glob("[0-9][0-9].yml")):
            if (object_dir.name, config_path.stem) in TECHNICAL_EXCLUSIONS:
                continue
            items.append((object_dir.name, config_path.stem, config_path))
    return items


def replace(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}:\s*.*$")
    if not pattern.search(text):
        raise ValueError(f"Missing config key: {key}")
    return pattern.sub(f"{key}: {value}", text, count=1)


def replace_indented(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^    {re.escape(key)}:\s*.*$")
    if not pattern.search(text):
        raise ValueError(f"Missing confidence config key: {key}")
    return pattern.sub(f"    {key}: {value}", text, count=1)


def make_config(
    source: Path,
    destination: Path,
    run_name: str,
    log_csv: Path,
    gate: bool,
    threshold: float,
    confidence_mode: str,
    initial_geometry: Path | None,
    export_geometry: Path | None = None,
    stop_after_initial: bool = False,
) -> None:
    text = source.read_text()
    text = replace(text, "modelName", f'"{run_name}"')
    text = replace_indented(text, "use_confidence_gate", "True" if gate else "False")
    text = replace_indented(text, "confidence_threshold", f"{threshold:.6f}")
    text = replace_indented(text, "log_csv", f'"{log_csv}"')
    confidence_settings = (
        f"    mode: {confidence_mode}\n"
        f"    base_threshold: {threshold:.6f}\n"
        "    adaptive_threshold: True\n"
        "    adaptive_k: 1.5\n"
        "    threshold_min: 0.45\n"
        "    threshold_max: 0.85\n"
        "    soft_temperature: 0.08\n"
        "    min_weight: 0.05\n"
        "    history_size: 32\n"
        "    fusion_weights: [0.40, 0.20, 0.20, 0.20]\n"
        "    pose_translation_scale: 0.03\n"
        "    pose_rotation_scale_deg: 10.0\n"
        "    temporal_scale: 0.15\n"
        "    min_visible_fraction: 0.02\n"
        "    gradient_alpha: 0.35\n"
        "    gradient_history_momentum: 0.8\n"
        "    gradient_min_weight: 0.75\n"
        "    rollback_tolerance: 0.002\n"
        "    max_vertex_step_ratio: 0.15\n"
        "    joint_outer_steps: 3\n"
        "    joint_pose_steps: 20\n"
        "    joint_geometry_steps: 60\n"
        "    joint_pose_lr: 0.002\n"
        "    joint_geometry_lr: 0.01\n"
        "    joint_max_rotation_deg: 5.0\n"
        "    joint_max_translation_ratio: 0.03\n"
        "    joint_pose_prior_weight: 0.05\n"
        "    joint_temporal_weight: 0.02\n"
        "    joint_boundary_weight: 0.10\n"
        "    joint_laplacian_weight: 0.10\n"
        "    joint_geometry_prior_weight: 0.02\n"
        "    joint_risk_pose_weight: 0.10\n"
        "    joint_risk_temporal_weight: 0.05\n"
        "    joint_risk_boundary_weight: 0.25\n"
        "    joint_risk_geometry_weight: 0.10\n"
        "    joint_risk_margin: 0.002\n"
        "    joint_tail_tolerance: 0.01\n"
        "    v6_log_variance_radius: 2.0\n"
        "    v6_uncertainty_prior_weight: 0.10\n"
        "    v6_cvar_quantile: 0.90\n"
        "    v6_cvar_weight: 0.35\n"
        "    v6_uncertainty_weight: 0.25\n"
        "    v6_risk_boundary_weight: 0.25\n"
        "    v6_risk_pose_weight: 0.10\n"
        "    v6_risk_temporal_weight: 0.05\n"
        "    v6_risk_geometry_weight: 0.10\n"
        "    v6_risk_margin: 0.001\n"
        "    v6_tail_tolerance: 0.01\n"
    )
    text = re.sub(
        r"(?m)^(    mask_binary_threshold:\s*.*\n)",
        r"\1" + confidence_settings,
        text,
        count=1,
    )
    additions = [
        f'initial_geometry_path: "{initial_geometry or ""}"',
        f'initial_geometry_export_path: "{export_geometry or ""}"',
        f"stop_after_initial_geometry: {'True' if stop_after_initial else 'False'}",
    ]
    marker = re.compile(r"(?m)^(template_mesh:\s*.*)$")
    text = marker.sub(r"\1\n" + "\n".join(additions), text, count=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_output_dir(source: Path) -> Path:
    for line in source.read_text().splitlines():
        if line.startswith("output_dir:"):
            return Path(line.split(":", 1)[1].strip().strip('"'))
    raise ValueError(f"No output_dir in {source}")


def clear_runtime_meshes(source: Path) -> None:
    output_dir = sequence_output_dir(source)
    for path in output_dir.glob("_realtime_*.obj"):
        path.unlink()
    rescaled = output_dir.parent / "deformed_model_rescaled"
    for path in rescaled.glob("model*.obj"):
        path.unlink()


def run_one(
    object_name: str,
    sequence: str,
    source: Path,
    arm: str,
    gate: bool,
    threshold: float,
    prepare: bool,
    timeout: int,
    force: bool,
    confidence_mode: str = "legacy_hard",
) -> bool:
    result_dir = RESULT_ROOT / "raw" / arm / object_name / sequence
    metrics_path = result_dir / "metrics.json"
    artifact = ARTIFACT_ROOT / object_name / f"{sequence}.obj"
    tclc_max_hists = int(os.environ.get("BIT_TCLC_MAX_HISTS", "512"))
    omp_num_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    malloc_arena_max = int(os.environ.get("MALLOC_ARENA_MAX", "0"))
    opencv_threads = int(os.environ.get("OPENCV_FOR_THREADS_NUM", "0"))
    if metrics_path.exists() and not force:
        metrics = json.loads(metrics_path.read_text())
        has_required_output = prepare or (
            metrics.get("add_mm") is not None and metrics.get("add_auc") is not None
        )
        runtime_matches = prepare or (
            metrics.get("tclc_max_hists") == tclc_max_hists
            and metrics.get("omp_num_threads") == omp_num_threads
            and metrics.get("malloc_arena_max") == malloc_arena_max
            and metrics.get("opencv_threads") == opencv_threads
        )
        if metrics.get("status") == "complete" and has_required_output and runtime_matches:
            print(f"skip complete: {arm}/{object_name}/{sequence}", flush=True)
            return True

    if not prepare and not artifact.exists():
        raise FileNotFoundError(f"Missing frozen initial geometry: {artifact}")

    result_dir.mkdir(parents=True, exist_ok=True)
    confidence_csv = result_dir / "confidence.csv"
    config_path = GENERATED_ROOT / arm / object_name / f"{sequence}.yml"
    run_name = f"paper-{arm}-{object_name}-{sequence}"
    make_config(
        source=source,
        destination=config_path,
        run_name=run_name,
        log_csv=confidence_csv,
        gate=gate,
        threshold=threshold,
        confidence_mode=confidence_mode,
        initial_geometry=None if prepare else artifact,
        export_geometry=artifact if prepare else None,
        stop_after_initial=prepare,
    )
    shutil.copy2(config_path, result_dir / "config.yml")
    clear_runtime_meshes(source)

    command = [
        sys.executable,
        "examples/run_example.py",
        "paper_moped",
        f"{arm}/{object_name}/{sequence}.yml",
    ]
    env = os.environ.copy()
    env.update({"QT_QPA_PLATFORM": "offscreen", "PYTHONUNBUFFERED": "1"})
    started = time.time()
    status = "failed"
    return_code = None
    output = ""
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return_code = completed.returncode
        output = completed.stdout
        status = "complete" if return_code == 0 else "failed"
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") if isinstance(error.stdout, str) else ""
        status = "timeout"

    duration = time.time() - started
    (result_dir / "run.log").write_text(output)
    add_matches = ADD_RE.findall(output)
    auc_matches = AUC_RE.findall(output)
    fail_matches = FAIL_RE.findall(output)
    candidate_count = accepted_count = None
    mean_observation_weight = min_observation_weight = None
    if confidence_csv.exists():
        with confidence_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        candidate_count = len(rows)
        accepted_count = sum(row["is_accepted_keyframe"] == "True" for row in rows)
        logged_weights = [
            float(row["observation_weight"])
            for row in rows
            if row.get("observation_weight") not in {None, ""}
        ]
        if logged_weights:
            mean_observation_weight = sum(logged_weights) / len(logged_weights)
            min_observation_weight = min(logged_weights)

    metrics = {
        "status": status,
        "dataset": "MOPED",
        "object": object_name,
        "sequence": sequence,
        "arm": arm,
        "seed": 42,
        "tclc_max_hists": tclc_max_hists,
        "omp_num_threads": omp_num_threads,
        "malloc_arena_max": malloc_arena_max,
        "opencv_threads": opencv_threads,
        "confidence_gate": gate,
        "confidence_mode": confidence_mode,
        "confidence_threshold": threshold,
        "initial_geometry_sha256": sha256(artifact) if artifact.exists() else None,
        "candidate_keyframes": candidate_count,
        "accepted_keyframes": accepted_count,
        "mean_observation_weight": mean_observation_weight,
        "min_observation_weight": min_observation_weight,
        "add_mm": float(add_matches[-1]) if add_matches else None,
        "add_auc": float(auc_matches[-1]) if auc_matches else None,
        "failures": int(fail_matches[-1]) if fail_matches else None,
        "runtime_seconds": duration,
        "return_code": return_code,
    }
    if not prepare and (metrics["add_mm"] is None or metrics["add_auc"] is None):
        metrics["status"] = status = "failed"
    if prepare and not artifact.exists():
        metrics["status"] = status = "failed"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    if not prepare:
        model_dir = result_dir / "models"
        model_dir.mkdir(exist_ok=True)
        for model_path in sorted(sequence_output_dir(source).glob("_realtime_*.obj")):
            shutil.copy2(model_path, model_dir / model_path.name)

    ok = status == "complete" and (prepare or (metrics["add_mm"] is not None and metrics["add_auc"] is not None))
    print(
        f"{status}: {arm}/{object_name}/{sequence} "
        f"ADD={metrics['add_mm']} AUC={metrics['add_auc']} time={duration:.1f}s",
        flush=True,
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=[
            "prepare", "baseline", "validation", "test", "fusion-validation",
            "fusion-test", "gradient-validation", "gradient-all", "all",
            "gradient-trust-all",
            "rollback-all",
            "joint-all",
        ],
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--replicate", type=int, default=0,
        help="Independent-process replicate ID; zero preserves canonical arm names.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--objects",
        nargs="+",
        help="Restrict execution to named objects (useful for fixed per-object resource strata).",
    )
    parser.add_argument(
        "--cohort", choices=["development", "confirmation", "all"], default="development",
        help="Object cohort; confirmation objects were not used during method development.",
    )
    args = parser.parse_args()

    allowed = {
        "development": DEVELOPMENT_OBJECTS,
        "confirmation": CONFIRMATION_OBJECTS,
        "all": DEVELOPMENT_OBJECTS | CONFIRMATION_OBJECTS,
    }[args.cohort]
    items = sequences(allowed)
    if args.objects:
        requested = set(args.objects)
        available = {object_name for object_name, _, _ in items}
        unknown = requested - available
        if unknown:
            parser.error(f"unknown objects: {', '.join(sorted(unknown))}")
        items = [item for item in items if item[0] in requested]
    if args.limit is not None:
        items = items[: args.limit]
    failures = []

    def execute(selected, arm, gate, threshold, prepare=False, confidence_mode="legacy_hard"):
        arm = arm if args.replicate == 0 or prepare else f"{arm}_r{args.replicate:02d}"
        for object_name, sequence, source in selected:
            try:
                ok = run_one(
                    object_name, sequence, source, arm, gate, threshold,
                    prepare, args.timeout, args.force, confidence_mode,
                )
            except Exception as error:
                print(f"ERROR {arm}/{object_name}/{sequence}: {error}", flush=True)
                ok = False
            if not ok:
                failures.append(f"{arm}/{object_name}/{sequence}")

    if args.phase in {"prepare", "all"}:
        execute(items, "prepare", False, 0.0, prepare=True, confidence_mode="disabled")
    if args.phase in {"baseline", "all"}:
        execute(items, "baseline", False, args.threshold, confidence_mode="disabled")
    if args.phase in {"validation", "all"}:
        validation = [item for item in items if item[1] == "00"]
        for threshold in args.thresholds:
            execute(validation, f"gate_t{int(round(threshold * 100)):03d}", True, threshold)
    if args.phase in {"test", "all"}:
        test = [item for item in items if item[1] != "00"]
        execute(test, f"gate_t{int(round(args.threshold * 100)):03d}", True, args.threshold)
    if args.phase in {"fusion-validation", "all"}:
        validation = [item for item in items if item[1] == "00"]
        execute(
            validation, "fusion_soft_adaptive", True, args.threshold,
            confidence_mode="multisignal_soft",
        )
    if args.phase == "fusion-test":
        test = [item for item in items if item[1] != "00"]
        execute(
            test, "fusion_soft_adaptive", True, args.threshold,
            confidence_mode="multisignal_soft",
        )
    if args.phase in {"gradient-validation", "gradient-all"}:
        selected = items if args.phase == "gradient-all" else [item for item in items if item[1] == "00"]
        execute(
            selected, "gradient_soft", True, args.threshold,
            confidence_mode="gradient_soft",
        )
    if args.phase == "gradient-trust-all":
        execute(
            items, "gradient_trust_soft", True, args.threshold,
            confidence_mode="gradient_soft",
        )
    if args.phase == "rollback-all":
        execute(
            items, "gradient_rollback", True, args.threshold,
            confidence_mode="gradient_rollback",
        )
    if args.phase == "joint-all":
        execute(
            items, "joint_risk", True, args.threshold,
            confidence_mode="joint_risk",
        )

    summary = {"phase": args.phase, "failures": failures, "total_failures": len(failures)}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / f"{args.phase}_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} runs failed; see status JSON")


if __name__ == "__main__":
    main()
