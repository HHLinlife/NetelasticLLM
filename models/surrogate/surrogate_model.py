"""
models/surrogate/surrogate_model.py
Lightweight distilled surrogate M̂₁ that approximates M₁ under limited queries.

Architecture: shallow MLP with layer normalisation.
Used as a query-free proxy during genetic search (Section 4.1).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SurrogateModel(nn.Module):
    """
    Lightweight MLP surrogate trained via knowledge distillation from M₁.

    Inputs  : flow feature vectors (B, input_dim)
    Outputs : fine-grained logits  (B, num_classes)

    Architecture:
        input_dim → hidden_dim → hidden_dim → ... → num_classes
    with LayerNorm + GELU activations per layer.
    """

    def __init__(
        self,
        input_dim:   int = 83,
        hidden_dim:  int = 256,
        num_layers:  int = 3,
        num_classes: int = 20,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.num_classes = num_classes

        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-softmax logits (B, num_classes)."""
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)

    def soft_labels(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return F.softmax(self.forward(x) / temperature, dim=-1)

    # ------------------------------------------------------------------ #
    #  Gradient for elastic energy computation (Eq. 6)                     #
    # ------------------------------------------------------------------ #

    def gradient_wrt_input(
        self,
        x: torch.Tensor,
        head: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ∇_x f_{θ}(x) efficiently via autograd.

        Args:
            x    : (B, input_dim) — will be cloned and differentiated.
            head : (num_classes,) direction vector; if None uses max logit.
        Returns:
            grad : (B, input_dim)
        """
        x_in = x.detach().float().requires_grad_(True)
        logits = self.forward(x_in)                      # (B, C)

        if head is not None:
            scalar = (logits * head.unsqueeze(0)).sum()
        else:
            scalar = logits.max(dim=1).values.sum()

        scalar.backward()
        return x_in.grad.detach()                         # (B, input_dim)

    # ------------------------------------------------------------------ #
    #  Distillation loss                                                    #
    # ------------------------------------------------------------------ #

    def distillation_loss(
        self,
        x: torch.Tensor,
        soft_targets: torch.Tensor,
        hard_targets: Optional[torch.Tensor] = None,
        temperature: float = 2.0,
        alpha: float = 0.7,
    ) -> torch.Tensor:
        """
        Combined knowledge-distillation + cross-entropy loss.

        L = α · T² · KL(student_soft ‖ teacher_soft) + (1−α) · CE(student, hard)

        Args:
            x            : (B, input_dim) input features.
            soft_targets : (B, num_classes) teacher soft labels at temperature T.
            hard_targets : (B,) integer labels (optional).
            temperature  : Distillation temperature T.
            alpha        : Weight on distillation term.
        """
        logits = self.forward(x)                          # (B, C)

        # Soft KL loss
        log_student = F.log_softmax(logits / temperature, dim=-1)
        kd_loss = F.kl_div(
            log_student,
            soft_targets,
            reduction="batchmean",
        ) * (temperature ** 2)

        if hard_targets is None or (1 - alpha) < 1e-8:
            return kd_loss

        # Hard cross-entropy loss
        ce_loss = F.cross_entropy(logits, hard_targets)
        return alpha * kd_loss + (1 - alpha) * ce_loss

    # ------------------------------------------------------------------ #
    #  Uncertainty / entropy (used for active query selection)             #
    # ------------------------------------------------------------------ #

    def predictive_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return per-sample predictive entropy H[p(y|x)].

        High entropy → uncertain region near decision boundary.
        """
        probs   = self.soft_labels(x)                     # (B, C)
        log_p   = probs.clamp(min=1e-9).log()
        entropy = -(probs * log_p).sum(dim=-1)            # (B,)
        return entropy

    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for predictive_entropy — higher = more uncertain."""
        return self.predictive_entropy(x)

    # ------------------------------------------------------------------ #
    #  Agreement metric                                                     #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def top1_agreement(self, x: torch.Tensor, oracle_labels: torch.Tensor) -> float:
        """
        Fraction of samples where surrogate argmax == oracle argmax.
        Implements Agree_{M₁} from Table 3.
        """
        pred = self.predict(x)
        return float((pred == oracle_labels).float().mean().item())
