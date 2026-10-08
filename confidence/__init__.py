"""Training-free confidence components for BIT geometry updates."""

from .gate import MaskGateResult, apply_mask_confidence_gate
from .mask_confidence import binary_mask_iou
from .multisignal import MultiSignalConfidence, MultiSignalResult
from .gradient_consistency import GradientConsistency, GradientConsistencyResult

__all__ = [
    "GradientConsistency", "GradientConsistencyResult", "MaskGateResult",
    "MultiSignalConfidence", "MultiSignalResult",
    "apply_mask_confidence_gate", "binary_mask_iou",
]
