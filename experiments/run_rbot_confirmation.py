#!/usr/bin/env python3
"""Resumable, frozen baseline/v4/v5 external confirmation on RBOT."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_moped_paired import clear_runtime_meshes, make_config, sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config" / "rbot"
GENERATED_ROOT = ROOT / "config" / "paper_rbot"
RESULT_ROOT = ROOT / "results" / "paper_rbot"
ARTIFACT_ROOT = RESULT_ROOT / "initial_geometry"
REFERENCE_ROOT = RESULT_ROOT / "frozen_references"
DATASET_ROOT = ROOT / "dataset" / "RBOT_dataset"

OBJECTS = (
    "ape", "bakingsoda", "benchviseblue", "broccolisoup", "cam", "can",
    "cat", "clown", "cube", "driller", "duck", "eggbox", "glue", "iron",
    "koalacandy", "lamp", "phone", "squirrel",
)
SCENES = ("a_regular", "b_dynamiclight", "c_noisy", "d_occlusion")
METHODS = {
    "baseline": (False, "disabled"),
    "v4": (True, "gradient_rollback"),
    "v5": (True, "joint_risk"),
    "v6": (True, "risk_calibrated_soft"),
    "v61": (True, "risk_calibrated_shadow"),
    "v7": (True, "risk_calibrated_crossfit"),
    "v8": (True, "risk_calibrated_observable"),
}

SUCCESS_RE = re.compile(r"RBOT success rate 5deg5cm:\s*([0-9.eE+-]+)")
ROTATION_RE = re.compile(r"RBOT mean rotation error deg:\s*([0-9.eE+-]+)")
TRANSLATION_RE = re.compile(r"RBOT mean translation error:\s*([0-9.eE+-]+)")
FRAMES_RE = re.compile(r"RBOT evaluated frames:\s*(\d+)")
FAIL_RE = re.compile(r"fail nums\s+(\d+)")


def items(objects: list[str] | None, scenes: list[str] | None):
    selected_objects = objects or list(OBJECTS)
    selected_scenes = scenes or list(SCENES)
    for scene in selected_scenes:
        for object_name in selected_objects:
            source = CONFIG_ROOT / f"{scene}_{object_name}.yml"
            if not source.is_file():
                raise FileNotFoundError(f"Missing generated RBOT config: {source}")
            yield scene, object_name, source


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"QT_QPA_PLATFORM": "offscreen", "PYTHONUNBUFFERED": "1"})
    return env


def reference_snapshot(scene: str, object_name: str) -> Path:
    return REFERENCE_ROOT / scene / object_name


def generate_references(scene: str, object_name: str, source: Path, timeout: int, force: bool) -> None:
    snapshot = reference_snapshot(scene, object_name)
    marker = snapshot / "_COMPLETE.json"
    if marker.exists() and not force:
        print(f"skip references: {scene}/{object_name}", flush=True)
        return
    live = DATASET_ROOT / object_name / "refers"
    if live.exists():
        shutil.rmtree(live)
    command = ["./gen_references", "rbot", source.name]
    started = time.time()
    completed = subprocess.run(
        command, cwd=ROOT / "tracker" / "bin", env=runtime_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=timeout, check=False,
    )
    log_dir = RESULT_ROOT / "reference_logs" / scene
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{object_name}.log").write_text(completed.stdout)
    if completed.returncode != 0 or not live.is_dir():
        raise RuntimeError(f"Reference generation failed: {scene}/{object_name}")
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(live, snapshot)
    marker.write_text(json.dumps({
        "scene": scene, "object": object_name,
        "runtime_seconds": time.time() - started,
        "file_count": sum(path.is_file() for path in snapshot.rglob("*")),
    }, indent=2) + "\n")
    print(f"references complete: {scene}/{object_name}", flush=True)


def restore_references(scene: str, object_name: str) -> None:
    snapshot = reference_snapshot(scene, object_name)
    if not (snapshot / "_COMPLETE.json").exists():
        raise FileNotFoundError(f"Missing frozen references: {snapshot}")
    live = DATASET_ROOT / object_name / "refers"
    if live.exists():
        shutil.rmtree(live)
    shutil.copytree(snapshot, live, ignore=shutil.ignore_patterns("_COMPLETE.json"))


def run_one(
    scene: str, object_name: str, source: Path, method: str, replicate: int,
    timeout: int, force: bool, prepare: bool = False,
) -> bool:
    arm = "prepare" if prepare else f"{method}_r{replicate:02d}"
    result_dir = RESULT_ROOT / "raw" / arm / scene / object_name
    metrics_path = result_dir / "metrics.json"
    artifact = ARTIFACT_ROOT / scene / f"{object_name}.obj"
    if metrics_path.exists() and not force:
        previous = json.loads(metrics_path.read_text())
        if previous.get("status") == "complete":
            print(f"skip complete: {arm}/{scene}/{object_name}", flush=True)
            return True
    if not prepare and not artifact.exists():
        raise FileNotFoundError(f"Missing frozen geometry: {artifact}")

    restore_references(scene, object_name)
    gate, mode = (False, "disabled") if prepare else METHODS[method]
    result_dir.mkdir(parents=True, exist_ok=True)
    config_path = GENERATED_ROOT / arm / f"{scene}_{object_name}.yml"
    confidence_csv = result_dir / "confidence.csv"
    make_config(
        source, config_path, f"rbot-{arm}-{scene}-{object_name}", confidence_csv,
        gate, 0.8, mode, None if prepare else artifact,
        artifact if prepare else None, prepare,
    )
    shutil.copy2(config_path, result_dir / "config.yml")
    clear_runtime_meshes(source)
    command = [sys.executable, "examples/run_example.py", "paper_rbot", f"{arm}/{scene}_{object_name}.yml"]
    started = time.time()
    status, return_code, output = "failed", None, ""
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=runtime_env(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False,
        )
        return_code, output = completed.returncode, completed.stdout
        status = "complete" if return_code == 0 else "failed"
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") if isinstance(error.stdout, str) else ""
        status = "timeout"
    (result_dir / "run.log").write_text(output)

    def last(pattern):
        matches = pattern.findall(output)
        return matches[-1] if matches else None

    candidate_count = accepted_count = None
    if confidence_csv.exists():
        with confidence_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        candidate_count = len(rows)
        accepted_count = sum(row.get("is_accepted_keyframe") == "True" for row in rows)
    metrics = {
        "status": status, "dataset": "RBOT", "scene": scene,
        "object": object_name, "method": "prepare" if prepare else method,
        "replicate": replicate, "seed": 42, "confidence_mode": mode,
        "initial_geometry_sha256": sha256(artifact) if artifact.exists() else None,
        "evaluated_frames": int(last(FRAMES_RE)) if last(FRAMES_RE) else None,
        "success_5deg_5cm": float(last(SUCCESS_RE)) if last(SUCCESS_RE) else None,
        "mean_rotation_error_deg": float(last(ROTATION_RE)) if last(ROTATION_RE) else None,
        "mean_translation_error": float(last(TRANSLATION_RE)) if last(TRANSLATION_RE) else None,
        "failures": int(last(FAIL_RE)) if last(FAIL_RE) else None,
        "candidate_keyframes": candidate_count, "accepted_keyframes": accepted_count,
        "runtime_seconds": time.time() - started, "return_code": return_code,
    }
    if prepare and not artifact.exists():
        metrics["status"] = status = "failed"
    if not prepare and metrics["success_5deg_5cm"] is None:
        metrics["status"] = status = "failed"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"{status}: {arm}/{scene}/{object_name} success={metrics['success_5deg_5cm']} "
        f"time={metrics['runtime_seconds']:.1f}s", flush=True,
    )
    return status == "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=["references", "prepare", "baseline", "v4", "v5", "v6", "v61", "v7", "v8", "confirm"],
    )
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--objects", nargs="+", choices=OBJECTS)
    parser.add_argument("--scenes", nargs="+", choices=SCENES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = list(items(args.objects, args.scenes))
    if args.limit is not None:
        selected = selected[:args.limit]
    failures = []
    phases = [args.phase] if args.phase != "confirm" else ["references", "prepare", "baseline", "v4", "v5"]
    for phase in phases:
        for scene, object_name, source in selected:
            try:
                if phase == "references":
                    generate_references(scene, object_name, source, args.timeout, args.force)
                    ok = True
                elif phase == "prepare":
                    ok = run_one(scene, object_name, source, "baseline", 0, args.timeout, args.force, True)
                else:
                    ok = run_one(scene, object_name, source, phase, args.replicate, args.timeout, args.force)
            except Exception as error:
                print(f"ERROR {phase}/{scene}/{object_name}: {error}", flush=True)
                ok = False
            if not ok:
                failures.append(f"{phase}/{scene}/{object_name}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = RESULT_ROOT / f"{args.phase}_r{args.replicate:02d}_status.json"
    status_path.write_text(json.dumps({"phase": args.phase, "failures": failures}, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} runs failed; see {status_path}")


if __name__ == "__main__":
    main()
