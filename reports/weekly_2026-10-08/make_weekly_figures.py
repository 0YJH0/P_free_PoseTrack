#!/usr/bin/env python3
"""Generate weekly-report figures from frozen RBOT experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    "font.size": 7.2,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "results" / "paper_rbot" / "raw"

METHODS = ["baseline_r00", "v5_r00", "v6_r00", "v61_r00", "v7_r00", "v8_r00"]
METHOD_LABELS = ["Baseline", "V5", "V6", "V6.1", "V7", "V8"]
METHOD_COLORS = ["#484878", "#2E6F9E", "#9A4D8E", "#42949E", "#B98B3D", "#B64342"]
PILOT_SCENES = ["a_regular", "c_noisy", "d_occlusion"]
SCENE_LABELS = {"a_regular": "Regular", "b_dynamiclight": "Dynamic light",
                "c_noisy": "Noisy", "d_occlusion": "Occlusion"}
PILOT_OBJECTS = ["ape", "cube"]
CASES = [(scene, obj) for scene in PILOT_SCENES for obj in PILOT_OBJECTS]
SHORT_SCENES = {"a_regular": "Reg.", "c_noisy": "Noise", "d_occlusion": "Occl."}
CASE_LABELS = [f"{SHORT_SCENES[s]}\n{o}" for s, o in CASES]


def metric(method: str, scene: str, obj: str) -> dict:
    path = RAW / method / scene / obj / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def decision_summary(method: str, scene: str, obj: str) -> tuple[float, float, float]:
    path = RAW / method / scene / obj / "confidence.csv"
    if not path.exists():
        return np.nan, np.nan, np.nan
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    decisions = [row for row in rows if row.get("shadow_action") in {"committed", "rejected"}]
    commit = float(any(row.get("shadow_action") == "committed" for row in decisions))
    proposals = [row for row in rows if row.get("shadow_action") == "proposed"]
    active = np.nan
    step = np.nan
    if proposals:
        try:
            active = float(proposals[0].get("active_vertex_fraction", "nan"))
            step = float(proposals[0].get("max_vertex_step_ratio", "nan"))
        except ValueError:
            pass
    return commit, active, step


def add_panel_label(ax, label: str, x=-0.08, y=1.10) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def annotate_heatmap(ax, matrix, fmt="{:.1f}", threshold=None) -> None:
    if threshold is None:
        threshold = (np.nanmin(matrix) + np.nanmax(matrix)) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if val < threshold else "#202020"
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    fontsize=6.4, color=color)


def save(fig, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_pilot_source() -> tuple[np.ndarray, np.ndarray]:
    success = np.array([
        [100 * metric(method, scene, obj)["success_5deg_5cm"] for scene, obj in CASES]
        for method in METHODS
    ])
    runtime = np.array([
        [metric(method, scene, obj)["runtime_seconds"] for scene, obj in CASES]
        for method in METHODS
    ])
    with (HERE / "source_data_recent_versions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "scene", "object", "success_5deg_5cm_pct",
                         "runtime_seconds", "commit", "active_vertex_fraction",
                         "max_vertex_step_ratio"])
        for mi, method in enumerate(METHODS):
            for ci, (scene, obj) in enumerate(CASES):
                commit, active, step = decision_summary(method, scene, obj)
                writer.writerow([METHOD_LABELS[mi], scene, obj, success[mi, ci],
                                 runtime[mi, ci], commit, active, step])
    return success, runtime


def figure_recent_versions(success: np.ndarray, runtime: np.ndarray) -> None:
    fig = plt.figure(figsize=(7.20, 5.80), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.55, 1.0])

    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(success, cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "success", ["#F2F2F6", "#B4C0E4", "#3775BA", "#0F4D92"]),
        vmin=15, vmax=82, aspect="auto")
    annotate_heatmap(ax, success, threshold=48)
    ax.set_xticks(range(len(CASE_LABELS)), CASE_LABELS)
    ax.set_yticks(range(len(METHOD_LABELS)), METHOD_LABELS)
    ax.set_title("Tracking success across the frozen six-unit development panel", pad=10)
    ax.set_ylabel("Method")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("5°/5 cm success (%)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, "a", x=-0.09, y=1.08)

    ax = fig.add_subplot(gs[1, 0])
    deltas = success[2:] - success[1]
    vmax = max(10, np.nanmax(np.abs(deltas)))
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(deltas, cmap="RdBu", norm=norm, aspect="auto")
    for i in range(deltas.shape[0]):
        for j in range(deltas.shape[1]):
            value = deltas[i, j]
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6.4,
                    color="white" if abs(value) > 0.55 * vmax else "#202020")
    ax.set_xticks(range(len(CASE_LABELS)), CASE_LABELS)
    ax.set_yticks(range(4), METHOD_LABELS[2:])
    ax.set_title("Later variants do not consistently surpass V5", pad=10)
    ax.set_ylabel("Method")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("Difference from V5 (pp)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, "b", x=-0.17, y=1.08)

    ax = fig.add_subplot(gs[1, 1])
    mean_runtime = runtime.mean(axis=1)
    ratios = mean_runtime / mean_runtime[1]
    y = np.arange(len(METHOD_LABELS))
    bars = ax.barh(y, ratios, color=METHOD_COLORS, height=0.62)
    ax.axvline(1, color="#606060", linestyle="--", linewidth=1)
    ax.set_yticks(y, METHOD_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Mean runtime / V5 runtime")
    ax.set_title("Compute overhead varies substantially", pad=10)
    ax.set_xlim(0, max(ratios) * 1.19)
    for bar, ratio, seconds in zip(bars, ratios, mean_runtime):
        ax.text(ratio + 0.06, bar.get_y() + bar.get_height()/2,
                f"{ratio:.1f}×  ({seconds/60:.1f} min)", va="center", fontsize=6.2)
    add_panel_label(ax, "c", x=-0.25, y=1.08)
    fig.suptitle("Recent RBOT architecture screening (one deterministic run per cell)",
                 fontsize=10, fontweight="bold")
    save(fig, "fig1_recent_architecture_screening")


def write_v5_source() -> tuple[np.ndarray, np.ndarray]:
    scenes = ["a_regular", "b_dynamiclight", "c_noisy", "d_occlusion"]
    objects = ["ape", "cube", "duck", "squirrel"]
    baseline = np.array([[100 * metric("baseline_r00", s, o)["success_5deg_5cm"]
                          for o in objects] for s in scenes])
    v5 = np.array([[100 * metric("v5_r00", s, o)["success_5deg_5cm"]
                    for o in objects] for s in scenes])
    with (HERE / "source_data_v5_short_panel.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "object", "baseline_success_pct", "v5_success_pct",
                         "v5_minus_baseline_pp"])
        for i, scene in enumerate(scenes):
            for j, obj in enumerate(objects):
                writer.writerow([scene, obj, baseline[i, j], v5[i, j],
                                 v5[i, j] - baseline[i, j]])
    return baseline, v5


def figure_v5_panel(baseline: np.ndarray, v5: np.ndarray) -> None:
    delta = v5 - baseline
    scenes = ["Regular", "Dynamic light", "Noisy", "Occlusion"]
    objects = ["ape", "cube", "duck", "squirrel"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.20, 3.05),
                                   gridspec_kw={"width_ratios": [1.48, 1.0]},
                                   constrained_layout=True)
    vmax = max(10, np.max(np.abs(delta)))
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax0.imshow(delta, cmap="RdBu", norm=norm, aspect="auto")
    for i in range(4):
        for j in range(4):
            value = delta[i, j]
            ax0.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(value) > 0.53 * vmax else "#202020")
    ax0.set_xticks(range(4), objects)
    ax0.set_yticks(range(4), scenes)
    ax0.set_xlabel("Object")
    ax0.set_title("V5 effect is condition- and object-dependent", pad=10)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.045, pad=0.03)
    cbar.set_label("V5 − baseline (pp)")
    for spine in ax0.spines.values():
        spine.set_visible(False)
    add_panel_label(ax0, "a", x=-0.10, y=1.08)

    means = delta.mean(axis=1)
    colors = ["#2E9E44" if value >= 0 else "#E53935" for value in means]
    y = np.arange(4)
    ax1.barh(y, means, color=colors, height=0.62)
    ax1.axvline(0, color="#606060", linewidth=0.9)
    ax1.set_yticks(y, scenes)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean improvement (pp)")
    ax1.set_title("Largest gains occur under degradation", pad=10)
    for yi, value in zip(y, means):
        ax1.text(value + (0.7 if value >= 0 else -0.7), yi, f"{value:+.1f}",
                 ha="left" if value >= 0 else "right", va="center", fontsize=7)
    add_panel_label(ax1, "b", x=-0.24, y=1.08)
    fig.suptitle("V5 versus baseline on the 4-object × 4-condition RBOT short panel",
                 fontsize=10, fontweight="bold")
    save(fig, "fig2_v5_condition_specific_effect")


def main() -> None:
    success, runtime = write_pilot_source()
    figure_recent_versions(success, runtime)
    baseline, v5 = write_v5_source()
    figure_v5_panel(baseline, v5)


if __name__ == "__main__":
    main()
