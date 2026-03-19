"""
evaluation/metrics.py
Evaluation metrics from Section 5.5.

Implements:
  - KL divergence (Eq. 14): KL(θ_a, θ_b) = E_x[D_KL(p_{θ_a}(·|x) ‖ p_{θ_b}(·|x))]
  - Classification accuracy and accuracy drop ΔAcc (Eq. 15)
  - Surrogate agreement Agree_{M₁} (Table 3)
  - Per-class breakdown for fine-grained analysis
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  KL divergence between two models  (Eq. 14)                                 #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def model_kl_divergence(
    model_a:    nn.Module,
    model_b:    nn.Module,
    features:   torch.Tensor,
    batch_size: int = 256,
    device:     str = "cpu",
    temperature: float = 1.0,
) -> float:
    """
    Compute KL(θ_a, θ_b) = E_x[D_KL(p_a(·|x) ‖ p_b(·|x))].

    A larger value means the two models differ more in output distribution.
    When model_a = M₂ and model_b = M₀, a *smaller* value indicates
    stronger rebound toward the pretrained prior.

    Args:
        model_a, model_b : Both must accept (B, D) tensors and return logits.
        features         : (N, D) feature tensor or array.
    Returns:
        Scalar KL divergence (float).
    """
    dev = torch.device(device)
    model_a.eval().to(dev)
    model_b.eval().to(dev)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features.astype(np.float32))

    total_kl = 0.0
    n_samples = 0

    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(dev)

        log_a  = _get_log_probs(model_a, batch, temperature)   # (B, C)
        log_b  = _get_log_probs(model_b, batch, temperature)   # (B, C)
        probs_a = log_a.exp()                                   # (B, C)

        # D_KL(P‖Q) = Σ P·(log P - log Q)
        kl_batch = (probs_a * (log_a - log_b)).sum(dim=-1)     # (B,)
        total_kl += kl_batch.sum().item()
        n_samples += len(batch)

    return total_kl / max(n_samples, 1)


def _get_log_probs(
    model:       nn.Module,
    x:           torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Extract log-softmax probabilities from a model."""
    out = model(x)
    if isinstance(out, dict):
        logits = out.get("fine_logits", out.get("logits", None))
    elif isinstance(out, torch.Tensor):
        logits = out
    else:
        raise ValueError(f"Unexpected model output type: {type(out)}")
    return F.log_softmax(logits / temperature, dim=-1)


# --------------------------------------------------------------------------- #
#  Classification accuracy                                                      #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_accuracy(
    model:      nn.Module,
    features:   torch.Tensor,
    labels:     torch.Tensor,
    batch_size: int = 256,
    device:     str = "cpu",
    logit_key:  str = "fine_logits",
) -> float:
    """
    Compute top-1 classification accuracy.

    Returns:
        Accuracy in [0, 1].
    """
    dev = torch.device(device)
    model.eval().to(dev)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features.astype(np.float32))
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels.astype(np.int64))

    correct = 0
    total   = 0

    for start in range(0, len(features), batch_size):
        x_batch = features[start : start + batch_size].to(dev)
        y_batch = labels[start : start + batch_size].to(dev)

        out = model(x_batch)
        if isinstance(out, dict):
            logits = out.get(logit_key, out.get("logits", None))
        else:
            logits = out

        preds   = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)

    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
#  Accuracy drop ΔAcc  (Eq. 15)                                               #
# --------------------------------------------------------------------------- #

def accuracy_drop(
    acc_m1: float,
    acc_m2: float,
) -> float:
    """
    ΔAcc(U) = Acc(M₁; D_test) - Acc(M₂(U); D_test).

    Positive value → M₂ performs worse than M₁ (attack success).
    """
    return acc_m1 - acc_m2


# --------------------------------------------------------------------------- #
#  Per-class breakdown                                                          #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def per_class_accuracy(
    model:       nn.Module,
    features:    torch.Tensor,
    labels:      torch.Tensor,
    num_classes: int,
    device:      str = "cpu",
    logit_key:   str = "fine_logits",
) -> Dict[int, float]:
    """
    Return per-class accuracy as a dict {class_idx: accuracy}.
    """
    dev = torch.device(device)
    model.eval().to(dev)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features.astype(np.float32))
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels.astype(np.int64))

    correct = np.zeros(num_classes, dtype=np.int64)
    total   = np.zeros(num_classes, dtype=np.int64)

    for start in range(0, len(features), 256):
        x_b  = features[start : start + 256].to(dev)
        y_b  = labels[start : start + 256]
        out  = model(x_b)
        logits = out[logit_key] if isinstance(out, dict) else out
        preds  = logits.argmax(dim=-1).cpu()

        for c in range(num_classes):
            mask      = (y_b == c)
            total[c]  += mask.sum().item()
            correct[c]+= (preds[mask] == y_b[mask]).sum().item()

    return {
        c: correct[c] / max(total[c], 1)
        for c in range(num_classes)
    }


# --------------------------------------------------------------------------- #
#  Surrogate agreement (Table 3 — Agree_{M₁})                                 #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def surrogate_agreement(
    surrogate:    nn.Module,
    oracle_fn,
    features:     torch.Tensor,
    batch_size:   int = 256,
    device:       str = "cpu",
) -> float:
    """
    Fraction of samples where surrogate argmax == oracle argmax.

    Args:
        oracle_fn : Callable (Tensor) → Tensor of soft labels from M₁.
    """
    dev = torch.device(device)
    surrogate.eval().to(dev)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features.astype(np.float32))

    agree = 0
    total = 0

    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(dev)

        oracle_soft  = oracle_fn(batch)
        oracle_preds = oracle_soft.argmax(dim=-1)               # (B,)

        sur_out = surrogate(batch)
        if isinstance(sur_out, dict):
            sur_logits = sur_out.get("logits", sur_out.get("fine_logits"))
        else:
            sur_logits = sur_out
        sur_preds = sur_logits.argmax(dim=-1)                   # (B,)

        agree += (sur_preds == oracle_preds).sum().item()
        total += len(batch)

    return agree / max(total, 1)


# --------------------------------------------------------------------------- #
#  Composite results dict                                                       #
# --------------------------------------------------------------------------- #

def compute_rebound_metrics(
    m1:         nn.Module,
    m2:         nn.Module,
    m0:         nn.Module,
    features:   torch.Tensor,
    labels:     torch.Tensor,
    device:     str = "cpu",
) -> Dict[str, float]:
    """
    Compute all rebound-relevant metrics in one call.

    Returns dict with keys:
        acc_m1, acc_m2, delta_acc,
        kl_m2_m0  (rebound distance),
        kl_m1_m0  (pre-attack baseline),
        kl_m2_m1  (perturbation impact on M₁).
    """
    acc_m1 = compute_accuracy(m1, features, labels, device=device)
    acc_m2 = compute_accuracy(m2, features, labels, device=device)
    da     = accuracy_drop(acc_m1, acc_m2)

    kl_m2_m0 = model_kl_divergence(m2, m0, features, device=device)
    kl_m1_m0 = model_kl_divergence(m1, m0, features, device=device)
    kl_m2_m1 = model_kl_divergence(m2, m1, features, device=device)

    return {
        "acc_m1":   acc_m1,
        "acc_m2":   acc_m2,
        "delta_acc": da,
        "kl_m2_m0": kl_m2_m0,
        "kl_m1_m0": kl_m1_m0,
        "kl_m2_m1": kl_m2_m1,
    }
