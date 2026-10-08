#!/usr/bin/env python3
"""Generate missing BIT reference metadata for the complete MOPED benchmark."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config" / "moped"
BINARY = ROOT / "tracker" / "bin" / "gen_references"


def has_reference_data(config_path: Path) -> bool:
    relative = config_path.relative_to(CONFIG_ROOT)
    sequence_root = ROOT / "dataset" / "moped" / relative.with_suffix("")
    return any((sequence_root / "refers" / "pose").glob("*.txt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configs = []
    for object_dir in sorted(CONFIG_ROOT.iterdir()):
        if not object_dir.is_dir():
            continue
        reference = object_dir / "reference" / "00.yml"
        if reference.exists():
            configs.append(reference)
        configs.extend(sorted((object_dir / "evaluation").glob("[0-9][0-9].yml")))

    failures = []
    for index, config_path in enumerate(configs, start=1):
        if not args.force and has_reference_data(config_path):
            print(f"[{index}/{len(configs)}] skip {config_path.relative_to(CONFIG_ROOT)}", flush=True)
            continue
        relative = config_path.relative_to(CONFIG_ROOT).as_posix()
        print(f"[{index}/{len(configs)}] generate {relative}", flush=True)
        completed = subprocess.run(
            [str(BINARY), "moped", relative],
            cwd=BINARY.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            failures.append((relative, completed.returncode))

    if failures:
        details = ", ".join(f"{name} (rc={code})" for name, code in failures)
        raise SystemExit(f"Reference generation failed: {details}")
    print(f"Reference preparation complete: {len(configs)} configurations.")


if __name__ == "__main__":
    main()
