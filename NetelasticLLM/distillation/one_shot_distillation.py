"""
distillation/one_shot_distillation.py
One-shot surrogate distillation protocol (Section 4.1).

Two-stage protocol:
  Stage I  — Coverage-oriented warmup via K-center greedy sampling.
  Stage II — Active distillation refinement using predictive entropy.

Total query budget = warmup_budget + active_budget = 80 000 by default.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models.surrogate.surrogate_model import SurrogateModel

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  K-center greedy sampling (coverage-oriented warmup)                         #
# --------------------------------------------------------------------------- #

def k_center_greedy(
    features:   np.ndarray,
    k:          int,
    seed:       int = 0,
) -> np.ndarray:
    """
    K-center greedy algorithm: select k samples that minimise the
    maximum distance from any point to its nearest selected centre.

    Returns indices of selected samples.
    """
    rng  = np.random.default_rng(seed)
    n    = len(features)
    k    = min(k, n)

    # Initialise with a random seed
    selected  = [int(rng.integers(n))]
    distances = np.full(n, np.inf)

    for _ in range(k - 1):
        # Update distances to nearest selected centre
        new_centre = features[selected[-1]]
        dist_to_new = np.sum((features - new_centre) ** 2, axis=1)
        distances   = np.minimum(distances, dist_to_new)

        # Select the point farthest from all selected centres
        next_idx = int(np.argmax(distances))
        selected.append(next_idx)

    return np.array(selected, dtype=np.int64)


# --------------------------------------------------------------------------- #
#  One-shot distillation trainer                                               #
# --------------------------------------------------------------------------- #

class OneShotDistillation:
    """
    Trains a SurrogateModel to approximate M₁ within a fixed query budget.

    Parameters
    ----------
    surrogate        : SurrogateModel to train.
    oracle_fn        : Callable (features: Tensor) → soft_labels: Tensor
                       Represents one batch query to M₁.
    warmup_budget    : Number of queries used in Stage I (K-center warmup).
    active_budget    : Number of queries used in Stage II (active refinement).
    temperature      : Distillation temperature T.
    alpha            : KD loss weight in distillation_loss.
    lr               : Learning rate for surrogate optimiser.
    batch_size       : Mini-batch size for SGD.
    device           : Torch device string.
    """

    def __init__(
        self,
        surrogate:       SurrogateModel,
        oracle_fn:       Callable[[torch.Tensor], torch.Tensor],
        warmup_budget:   int   = 10_000,
        active_budget:   int   = 70_000,
        temperature:     float = 2.0,
        alpha:           float = 0.7,
        lr:              float = 1e-3,
        batch_size:      int   = 256,
        epochs_per_stage: int  = 10,
        device:          str   = "cpu",
        seed:            int   = 42,
    ):
        self.surrogate        = surrogate.to(device)
        self.oracle_fn        = oracle_fn
        self.warmup_budget    = warmup_budget
        self.active_budget    = active_budget
        self.temperature      = temperature
        self.alpha            = alpha
        self.batch_size       = batch_size
        self.epochs_per_stage = epochs_per_stage
        self.device           = torch.device(device)
        self.rng              = np.random.default_rng(seed)

        self.optimiser = torch.optim.Adam(
            surrogate.parameters(), lr=lr, weight_decay=1e-4
        )
        # Distillation set: (features, soft_labels)
        self._X:  Optional[torch.Tensor] = None
        self._Y:  Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, unlabelled_features: np.ndarray) -> SurrogateModel:
        """
        Run two-stage distillation on unlabelled_features.

        Args:
            unlabelled_features : (N, D) float32 array of candidate flows.
        Returns:
            Trained SurrogateModel.
        """
        logger.info(
            "One-shot distillation: N=%d, warmup=%d, active=%d",
            len(unlabelled_features), self.warmup_budget, self.active_budget,
        )
        # Stage I
        self._stage_warmup(unlabelled_features)
        # Stage II
        self._stage_active(unlabelled_features)

        logger.info(
            "Distillation complete — total queries: %d",
            len(self._X) if self._X is not None else 0,
        )
        return self.surrogate

    # ------------------------------------------------------------------ #
    #  Stage I: K-center warmup                                            #
    # ------------------------------------------------------------------ #

    def _stage_warmup(self, features: np.ndarray):
        k   = min(self.warmup_budget, len(features))
        idx = k_center_greedy(features, k, seed=int(self.rng.integers(1 << 31)))
        logger.info("Stage I warmup: querying %d samples.", k)

        X_warm = torch.from_numpy(features[idx].astype(np.float32))
        Y_warm = self._query_oracle(X_warm)

        self._X = X_warm
        self._Y = Y_warm

        self._train_surrogate(self._X, self._Y, tag="Warmup")

    # ------------------------------------------------------------------ #
    #  Stage II: active distillation refinement                            #
    # ------------------------------------------------------------------ #

    def _stage_active(self, features: np.ndarray):
        remaining_budget = self.active_budget
        if remaining_budget <= 0:
            return

        # Unlabelled pool (exclude already queried)
        queried_mask = np.zeros(len(features), dtype=bool)
        if self._X is not None:
            # Mark approximate matches as queried (heuristic)
            queried_mask[: len(self._X)] = True

        unlabelled_idx = np.where(~queried_mask)[0]
        if len(unlabelled_idx) == 0:
            return

        per_iter  = min(500, remaining_budget, len(unlabelled_idx))
        n_iters   = max(1, remaining_budget // per_iter)

        logger.info("Stage II active refinement: %d iterations × %d queries.",
                    n_iters, per_iter)

        for it in range(n_iters):
            if len(unlabelled_idx) == 0:
                break

            # Score by predictive entropy
            batch_feats = torch.from_numpy(
                features[unlabelled_idx].astype(np.float32)
            ).to(self.device)
            with torch.no_grad():
                entropy = self.surrogate.uncertainty_scores(batch_feats).cpu().numpy()

            # Select top-per_iter most uncertain
            topk_local = np.argsort(entropy)[::-1][: per_iter]
            topk_global = unlabelled_idx[topk_local]

            X_new = torch.from_numpy(features[topk_global].astype(np.float32))
            Y_new = self._query_oracle(X_new)

            # Append to distillation set
            self._X = torch.cat([self._X, X_new], dim=0)
            self._Y = torch.cat([self._Y, Y_new], dim=0)

            # Update unlabelled pool
            unlabelled_idx = np.delete(unlabelled_idx, topk_local)

            self._train_surrogate(X_new, Y_new, tag=f"Active-iter{it+1}")
            logger.debug(
                "Active iter %d/%d: queried %d, total=%d",
                it + 1, n_iters, per_iter, len(self._X),
            )

    # ------------------------------------------------------------------ #
    #  Oracle query helper                                                  #
    # ------------------------------------------------------------------ #

    def _query_oracle(self, X: torch.Tensor) -> torch.Tensor:
        """
        Query M₁ in batches and return soft labels (temperature-scaled).
        """
        all_soft = []
        for start in range(0, len(X), self.batch_size):
            batch      = X[start : start + self.batch_size].to(self.device)
            with torch.no_grad():
                soft = self.oracle_fn(batch)                  # (B, C)
            all_soft.append(soft.cpu())
        return torch.cat(all_soft, dim=0)

    # ------------------------------------------------------------------ #
    #  Surrogate training                                                   #
    # ------------------------------------------------------------------ #

    def _train_surrogate(
        self,
        X_train: torch.Tensor,
        Y_soft:  torch.Tensor,
        tag:     str = "",
    ):
        """Train surrogate on the current distillation set."""
        dataset = TensorDataset(X_train, Y_soft)
        loader  = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.surrogate.train()
        for epoch in range(self.epochs_per_stage):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimiser.zero_grad()
                loss = self.surrogate.distillation_loss(
                    x_batch, y_batch,
                    temperature=self.temperature,
                    alpha=self.alpha,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.surrogate.parameters(), max_norm=1.0
                )
                self.optimiser.step()
                total_loss += loss.item()
            avg = total_loss / max(len(loader), 1)
            logger.debug("[%s] Epoch %d — loss=%.4f", tag, epoch + 1, avg)
        self.surrogate.eval()

    # ------------------------------------------------------------------ #
    #  Agreement metric (Table 3)                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate_agreement(
        self,
        test_features: np.ndarray,
        oracle_fn:     Optional[Callable] = None,
    ) -> float:
        """
        Compute Agree_{M₁} on a held-out test set.
        If oracle_fn is None, uses the stored oracle.
        """
        oracle = oracle_fn or self.oracle_fn
        X_test = torch.from_numpy(test_features.astype(np.float32)).to(self.device)
        soft   = oracle(X_test)
        oracle_preds = soft.argmax(dim=-1)
        return self.surrogate.top1_agreement(X_test, oracle_preds)
