#!/usr/bin/env python3
"""Run resumable independent-process baseline/v4 replicates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--objects", nargs="+")
    parser.add_argument(
        "--cohort", choices=["development", "confirmation"], default="development",
        help="Forward the frozen object cohort to the paired-run driver.",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=["baseline", "v4", "v5"],
        default=["baseline", "v4"],
        help="Methods to run; defaults preserve the original baseline/v4 protocol.",
    )
    args = parser.parse_args()
    for replicate in args.replicates:
        phases = {"baseline": "baseline", "v4": "rollback-all", "v5": "joint-all"}
        for method in args.methods:
            phase = phases[method]
            command = [
                sys.executable,
                "experiments/run_moped_paired.py",
                phase,
                "--replicate", str(replicate),
                "--threshold", "0.65",
                "--timeout", str(args.timeout),
                "--cohort", args.cohort,
            ]
            if args.objects:
                command.extend(["--objects", *args.objects])
            print(f"replicate={replicate} phase={phase}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode:
                raise SystemExit(
                    f"replicate {replicate} phase {phase} failed with {completed.returncode}; rerun to resume"
                )


if __name__ == "__main__":
    main()
