"""Stateful multi-signal confidence for online geometry observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch

from .mask_confidence import binary_mask_iou


@dataclass(frozen=True)
class MultiSignalResult:
    mask_iou: torch.Tensor
    pose_score: torch.Tensor
    visibility_score: torch.Tensor
    temporal_score: torch.Tensor
    confidence: torch.Tensor
    threshold: torch.Tensor
    weight: torch.Tensor
    accepted: torch.Tensor


def _rotation_angle(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((torch.diagonal(rotation).sum() - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cosine)


class MultiSignalConfidence:
    """Causal confidence fusion with robust adaptive thresholds.

    Pose translation is normalized by object depth, making the score independent
    of whether the tracker stores translations in metres or millimetres.
    """

    def __init__(
        self,
        *,
        base_threshold: float = 0.65,
        adaptive: bool = True,
        adaptive_k: float = 1.5,
        threshold_min: float = 0.45,
        threshold_max: float = 0.85,
        temperature: float = 0.08,
        min_weight: float = 0.05,
        history_size: int = 32,
        weights: tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20),
        pose_translation_scale: float = 0.03,
        pose_rotation_scale_deg: float = 10.0,
        temporal_scale: float = 0.15,
        min_visible_fraction: float = 0.02,
    ) -> None:
        if not 0.0 <= base_threshold <= 1.0:
            raise ValueError("base_threshold must be in [0, 1]")
        if temperature <= 0 or history_size < 1:
            raise ValueError("temperature and history_size must be positive")
        if len(weights) != 4 or any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("weights must contain four non-negative values with positive sum")
        self.base_threshold = base_threshold
        self.adaptive = adaptive
        self.adaptive_k = adaptive_k
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.temperature = temperature
        self.min_weight = min_weight
        total = sum(weights)
        self.weights = tuple(value / total for value in weights)
        self.pose_translation_scale = pose_translation_scale
        self.pose_rotation_scale = math.radians(pose_rotation_scale_deg)
        self.temporal_scale = temporal_scale
        self.min_visible_fraction = min_visible_fraction
        self.history: deque[float] = deque(maxlen=history_size)
        self.previous_pose: torch.Tensor | None = None
        self.previous_frame_id: int | None = None
        self.mask_ema: float | None = None

    def _threshold(self) -> float:
        if not self.adaptive or len(self.history) < 5:
            return self.base_threshold
        values = torch.tensor(list(self.history), dtype=torch.float64)
        median = float(values.median())
        mad = float((values - median).abs().median())
        robust_sigma = max(1.4826 * mad, 0.02)
        return min(self.threshold_max, max(self.threshold_min, median - self.adaptive_k * robust_sigma))

    def evaluate(
        self,
        segmentation_masks: torch.Tensor,
        rendered_masks: torch.Tensor,
        poses: torch.Tensor,
        frame_ids: list[int],
        *,
        mask_binary_threshold: float = 0.5,
    ) -> MultiSignalResult:
        count = len(frame_ids)
        if segmentation_masks.shape != rendered_masks.shape or segmentation_masks.shape[0] != count:
            raise ValueError("masks and frame_ids must describe the same observations")
        if poses.shape[0] != count or poses.shape[-2:] != (4, 4):
            raise ValueError("poses must have shape [N, 4, 4]")

        ious = binary_mask_iou(segmentation_masks, rendered_masks, threshold=mask_binary_threshold)
        seg_binary = segmentation_masks >= mask_binary_threshold
        render_binary = rendered_masks >= mask_binary_threshold
        eps = torch.finfo(torch.float32).eps
        seg_area = seg_binary.flatten(1).sum(1).float()
        render_area = render_binary.flatten(1).sum(1).float()
        area_agreement = torch.minimum(seg_area, render_area) / torch.maximum(seg_area, render_area).clamp_min(1.0)
        visible_fraction = render_area / float(render_binary[0].numel()) if count else render_area
        visibility = torch.sqrt(
            area_agreement * (visible_fraction / self.min_visible_fraction).clamp(0.0, 1.0)
        )

        pose_scores, temporal_scores, confidences, thresholds, soft_weights = [], [], [], [], []
        normalized_weights = ious.new_tensor(self.weights)
        for index, frame_id in enumerate(frame_ids):
            pose = poses[index].detach()
            if self.previous_pose is None:
                pose_score = ious.new_tensor(1.0)
            else:
                gap = max(1, frame_id - int(self.previous_frame_id))
                relative = torch.linalg.inv(self.previous_pose.to(pose)) @ pose
                depth = torch.maximum(self.previous_pose[2, 3].abs(), pose[2, 3].abs()).clamp_min(1e-6)
                translation = torch.linalg.vector_norm(relative[:3, 3]) / depth / gap
                rotation = _rotation_angle(relative[:3, :3]) / gap
                pose_score = torch.exp(
                    -translation / self.pose_translation_scale - rotation / self.pose_rotation_scale
                ).to(ious)

            current_iou = float(ious[index])
            if self.mask_ema is None:
                temporal_score = ious.new_tensor(1.0)
                self.mask_ema = current_iou
            else:
                temporal_score = ious.new_tensor(
                    math.exp(-abs(current_iou - self.mask_ema) / self.temporal_scale)
                )
                self.mask_ema = 0.8 * self.mask_ema + 0.2 * current_iou

            components = torch.stack((ious[index], pose_score, visibility[index], temporal_score))
            confidence = torch.exp((normalized_weights * components.clamp_min(eps).log()).sum())
            threshold = self._threshold()
            soft_weight = torch.sigmoid((confidence - threshold) / self.temperature)

            pose_scores.append(pose_score)
            temporal_scores.append(temporal_score)
            confidences.append(confidence)
            thresholds.append(ious.new_tensor(threshold))
            soft_weights.append(soft_weight)
            self.history.append(float(confidence))
            self.previous_pose = pose.cpu()
            self.previous_frame_id = frame_id

        def stack(values: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(values) if values else ious.new_empty((0,))

        weights = stack(soft_weights)
        return MultiSignalResult(
            mask_iou=ious,
            pose_score=stack(pose_scores),
            visibility_score=visibility,
            temporal_score=stack(temporal_scores),
            confidence=stack(confidences),
            threshold=stack(thresholds),
            weight=weights,
            accepted=weights >= self.min_weight,
        )
