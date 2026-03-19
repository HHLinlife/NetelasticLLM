"""
distillation/active_query_selection.py
Active learning query selection strategies for surrogate distillation.

Provides uncertainty-based and diversity-based selectors used in
Stage II of the one-shot distillation protocol.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ActiveQuerySelector:
    """
    Selects the most informative unlabelled samples to query from M₁.

    Strategies
    ----------
    'entropy'     : Select samples with highest predictive entropy H[p(y|x)].
    'margin'      : Select samples with smallest top-2 probability margin.
    'least_conf'  : Select samples with lowest max probability.
    'combined'    : Weighted sum of entropy and K-center diversity.
    """

    def __init__(
        self,
        strategy:  str   = "entropy",
        diversity_weight: float = 0.3,
        device:    str   = "cpu",
    ):
        assert strategy in ("entropy", "margin", "least_conf", "combined"), \
            f"Unknown strategy: {strategy}"
        self.strategy         = strategy
        self.diversity_weight = diversity_weight
        self.device           = torch.device(device)

    # ------------------------------------------------------------------ #

    def select(
        self,
        surrogate:          torch.nn.Module,
        unlabelled_features: np.ndarray,
        budget:             int,
    ) -> np.ndarray:
        """
        Select indices of the most informative samples.

        Args:
            surrogate            : Trained SurrogateModel.
            unlabelled_features  : (N, D) candidate features.
            budget               : Number of samples to select.
        Returns:
            selected_idx : (budget,) integer array of selected indices.
        """
        n      = len(unlabelled_features)
        budget = min(budget, n)
        if budget == 0:
            return np.array([], dtype=np.int64)

        tensor = torch.from_numpy(
            unlabelled_features.astype(np.float32)
        ).to(self.device)

        with torch.no_grad():
            scores = self._compute_scores(surrogate, tensor)   # (N,)

        if self.strategy == "combined":
            diversity = self._diversity_scores(unlabelled_features)
            scores    = (
                (1 - self.diversity_weight) * scores
                + self.diversity_weight * diversity
            )

        # Higher score = more informative → select top-budget
        topk = np.argsort(scores)[::-1][:budget]
        return topk.astype(np.int64)

    # ------------------------------------------------------------------ #
    #  Score functions                                                      #
    # ------------------------------------------------------------------ #

    def _compute_scores(
        self,
        surrogate: torch.nn.Module,
        tensor:    torch.Tensor,
    ) -> np.ndarray:
        probs = torch.softmax(surrogate(tensor), dim=-1).cpu().numpy()  # (N, C)

        if self.strategy in ("entropy", "combined"):
            return self._entropy(probs)
        elif self.strategy == "margin":
            return self._margin(probs)
        else:  # least_conf
            return 1.0 - probs.max(axis=1)

    @staticmethod
    def _entropy(probs: np.ndarray) -> np.ndarray:
        """Shannon entropy H[p] = -Σ p·log p."""
        log_p = np.log(probs.clip(1e-9))
        return -(probs * log_p).sum(axis=1)

    @staticmethod
    def _margin(probs: np.ndarray) -> np.ndarray:
        """1 - (p₁ - p₂): higher score = smaller margin."""
        sorted_p = np.sort(probs, axis=1)[:, ::-1]
        margin   = sorted_p[:, 0] - sorted_p[:, 1]
        return 1.0 - margin

    @staticmethod
    def _diversity_scores(features: np.ndarray) -> np.ndarray:
        """
        Simple diversity score: distance of each sample to the
        mean of the feature space (higher = more outlying).
        Normalised to [0, 1].
        """
        mean   = features.mean(axis=0)
        dists  = np.linalg.norm(features - mean, axis=1)
        d_min, d_max = dists.min(), dists.max()
        if d_max > d_min:
            return (dists - d_min) / (d_max - d_min)
        return np.zeros(len(features))

    # ------------------------------------------------------------------ #
    #  Batch query wrapper                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def batch_query(
        oracle_fn,
        features:    np.ndarray,
        batch_size:  int = 256,
        device:      str = "cpu",
    ) -> torch.Tensor:
        """
        Query oracle_fn in batches and return concatenated soft labels.

        Args:
            oracle_fn   : Callable (Tensor) → Tensor (soft labels).
            features    : (N, D) feature array.
            batch_size  : Chunk size per query.
        Returns:
            soft_labels : (N, C) tensor on CPU.
        """
        dev    = torch.device(device)
        tensor = torch.from_numpy(features.astype(np.float32))
        results = []

        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(dev)
            with torch.no_grad():
                soft = oracle_fn(batch)
            results.append(soft.cpu())

        return torch.cat(results, dim=0)
