"""Gradient-consistency reliability for training-free geometry adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GradientConsistencyResult:
    scores: torch.Tensor
    fused_weights: torch.Tensor


def _flatten_gradients(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return torch.cat([
        (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ])


class GradientConsistency:
    """Score candidate gradients against reference and historical directions."""

    def __init__(self, *, alpha: float = 0.35, history_momentum: float = 0.8, min_weight: float = 0.05):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.history_momentum = history_momentum
        self.min_weight = min_weight
        self.history_direction: torch.Tensor | None = None

    def evaluate(
        self,
        per_observation_losses: torch.Tensor,
        parameters,
        *,
        reference_count: int,
        prior_candidate_weights: torch.Tensor,
    ) -> GradientConsistencyResult:
        parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
        count = int(per_observation_losses.numel())
        candidate_count = count - reference_count
        if not parameters or reference_count < 1 or candidate_count < 1:
            raise ValueError("gradient consistency requires parameters, references, and candidates")
        if prior_candidate_weights.numel() != candidate_count:
            raise ValueError("prior_candidate_weights must match candidate count")

        gradients = []
        for index, loss in enumerate(per_observation_losses):
            gradients.append(
                _flatten_gradients(loss, parameters, retain_graph=index < count - 1).detach()
            )
        reference = torch.stack(gradients[:reference_count]).mean(0)
        reference = reference / reference.norm().clamp_min(1e-12)
        if self.history_direction is not None and self.history_direction.numel() == reference.numel():
            anchor = reference + self.history_direction.to(reference)
            anchor = anchor / anchor.norm().clamp_min(1e-12)
        else:
            anchor = reference

        candidate_gradients = torch.stack(gradients[reference_count:])
        normalized = candidate_gradients / candidate_gradients.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cosine = (normalized @ anchor).clamp(-1.0, 1.0)
        scores = ((cosine + 1.0) / 2.0).clamp(0.0, 1.0)
        prior = prior_candidate_weights.to(scores).clamp_min(1e-6)
        fused = (
            prior.pow(1.0 - self.alpha) * scores.clamp_min(1e-6).pow(self.alpha)
        ).clamp(self.min_weight, 1.0)

        update = (normalized * fused[:, None]).sum(0) / fused.sum().clamp_min(1e-12)
        update = update / update.norm().clamp_min(1e-12)
        if self.history_direction is None or self.history_direction.numel() != update.numel():
            history = update
        else:
            history = self.history_momentum * self.history_direction.to(update) + (1.0 - self.history_momentum) * update
            history = history / history.norm().clamp_min(1e-12)
        self.history_direction = history.cpu()
        return GradientConsistencyResult(scores=scores, fused_weights=fused)
