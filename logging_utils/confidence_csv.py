"""CSV logging for confidence-aware keyframe decisions."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ConfidenceRow:
    """One online candidate keyframe decision."""

    frame_id: int
    mask_iou: float
    is_candidate_keyframe: bool
    is_accepted_keyframe: bool
    geometry_update: bool
    pose_score: float = float("nan")
    visibility_score: float = float("nan")
    temporal_score: float = float("nan")
    fused_confidence: float = float("nan")
    adaptive_threshold: float = float("nan")
    observation_weight: float = 1.0
    gradient_consistency: float = float("nan")
    update_accepted: bool = True
    pre_reference_iou: float = float("nan")
    post_reference_iou: float = float("nan")
    is_holdout: bool = False
    pre_holdout_iou: float = float("nan")
    post_holdout_iou: float = float("nan")
    max_vertex_step_ratio: float = float("nan")
    pre_tracking_risk: float = float("nan")
    post_tracking_risk: float = float("nan")
    post_tail_iou_loss: float = float("nan")
    mean_pose_correction: float = float("nan")
    optimized_log_variance: float = float("nan")
    effective_samples: float = float("nan")
    pre_cvar: float = float("nan")
    post_cvar: float = float("nan")
    publication_alpha: float = float("nan")
    full_step_tracking_risk: float = float("nan")
    shadow_action: str = ""
    prospective_risk_gain: float = float("nan")
    active_vertex_fraction: float = float("nan")
    mean_vertex_agreement: float = float("nan")


class ConfidenceCsvLogger:
    """Append confidence rows while writing the header exactly once."""

    fieldnames = tuple(ConfidenceRow.__dataclass_fields__)

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def reset(self) -> None:
        """Start a fresh run by removing this logger's previous CSV."""
        if self.path.exists():
            self.path.unlink()

    def append(self, rows: Iterable[ConfidenceRow]) -> None:
        """Append a batch of rows to the configured CSV file."""
        rows = list(rows)
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
