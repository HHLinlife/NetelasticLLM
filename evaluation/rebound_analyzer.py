"""
evaluation/rebound_analyzer.py
Analyses rebound behavior across alignment depths, dataset sizes,
model scales, and fine-tuning strategies (Sections 6.2–6.5).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_rebound_metrics, compute_accuracy, model_kl_divergence

logger = logging.getLogger(__name__)


class ReboundAnalyzer:
    """
    Orchestrates rebound measurement experiments.

    Wraps metric computation and produces structured result dicts
    suitable for logging and plotting.
    """

    def __init__(
        self,
        device: str = "cpu",
        temperature: float = 1.0,
    ):
        self.device      = device
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    #  Q1: Effect of alignment depth  (Section 6.2)                       #
    # ------------------------------------------------------------------ #

    def alignment_depth_experiment(
        self,
        m0:           nn.Module,
        m1_checkpoints: List[nn.Module],
        m2_checkpoints: List[nn.Module],
        features:     torch.Tensor,
        labels:       torch.Tensor,
        depth_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare rebound metrics across alignment checkpoints θ₁^(1)…θ₁^(K).

        Args:
            m1_checkpoints : List of M₁ models at increasing alignment depth.
            m2_checkpoints : Corresponding M₂ models after perturbation.
            depth_labels   : Human-readable depth names (e.g. ['25%','50%',…]).
        Returns:
            List of result dicts, one per checkpoint.
        """
        if depth_labels is None:
            depth_labels = [f"depth_{i}" for i in range(len(m1_checkpoints))]

        results = []
        for label, m1, m2 in zip(depth_labels, m1_checkpoints, m2_checkpoints):
            metrics = compute_rebound_metrics(
                m1, m2, m0, features, labels, device=self.device
            )
            metrics["alignment_depth"] = label
            results.append(metrics)
            logger.info(
                "[Depth=%s] ΔAcc=%.2f%%, KL(M₂‖M₀)=%.4f",
                label,
                metrics["delta_acc"] * 100,
                metrics["kl_m2_m0"],
            )
        return results

    # ------------------------------------------------------------------ #
    #  Q3: Effect of model scale  (Section 6.4)                           #
    # ------------------------------------------------------------------ #

    def model_scale_experiment(
        self,
        m0_list:     List[nn.Module],
        m1_list:     List[nn.Module],
        m2_list:     List[nn.Module],
        features:    torch.Tensor,
        labels:      torch.Tensor,
        model_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare rebound metrics across model sizes.

        Args:
            m0_list, m1_list, m2_list : Parallel lists of models.
            model_names : Identifiers (e.g. ['Qwen2-7B', 'LLaMA3-8B', …]).
        """
        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(m0_list))]

        results = []
        for name, m0, m1, m2 in zip(model_names, m0_list, m1_list, m2_list):
            metrics = compute_rebound_metrics(
                m1, m2, m0, features, labels, device=self.device
            )
            metrics["model"] = name
            results.append(metrics)
            logger.info(
                "[Model=%s] ΔAcc=%.2f%%, KL(M₂‖M₀)=%.4f",
                name,
                metrics["delta_acc"] * 100,
                metrics["kl_m2_m0"],
            )
        return results

    # ------------------------------------------------------------------ #
    #  Q4: Consistency across fine-tuning strategies  (Section 6.5)       #
    # ------------------------------------------------------------------ #

    def finetuning_strategy_experiment(
        self,
        m0:               nn.Module,
        strategy_results: Dict[str, Dict[str, nn.Module]],
        features:         torch.Tensor,
        labels:           torch.Tensor,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Args:
            strategy_results : Dict mapping strategy_name →
                               {'m1': model, 'm2': model}.
        Returns:
            Dict mapping strategy_name → metrics dict.
        """
        output = {}
        for strategy, models in strategy_results.items():
            m1 = models["m1"]
            m2 = models["m2"]
            metrics = compute_rebound_metrics(
                m1, m2, m0, features, labels, device=self.device
            )
            output[strategy] = metrics
            logger.info(
                "[Strategy=%s] ΔAcc=%.2f%%, KL(M₂‖M₀)=%.4f",
                strategy,
                metrics["delta_acc"] * 100,
                metrics["kl_m2_m0"],
            )
        return output

    # ------------------------------------------------------------------ #
    #  Control: Random perturbation baseline                               #
    # ------------------------------------------------------------------ #

    def random_baseline(
        self,
        m0:       nn.Module,
        m1:       nn.Module,
        m2_rand:  nn.Module,
        features: torch.Tensor,
        labels:   torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Measure rebound metrics for a model trained on random perturbations,
        as a control to distinguish elastic rebound from generic degradation.
        """
        metrics = compute_rebound_metrics(
            m1, m2_rand, m0, features, labels, device=self.device
        )
        metrics["baseline"] = "random"
        logger.info(
            "[Random baseline] ΔAcc=%.2f%%, KL(M₂‖M₀)=%.4f",
            metrics["delta_acc"] * 100,
            metrics["kl_m2_m0"],
        )
        return metrics

    # ------------------------------------------------------------------ #
    #  Summary table formatter                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_table(results: List[Dict[str, Any]], key_col: str = "model") -> str:
        """
        Format a list of result dicts as a readable ASCII table.
        """
        cols    = [key_col, "acc_m1", "acc_m2", "delta_acc", "kl_m2_m0", "kl_m1_m0"]
        header  = " | ".join(f"{c:>15}" for c in cols)
        divider = "-" * len(header)
        rows    = [header, divider]

        for r in results:
            vals = []
            for c in cols:
                v = r.get(c, "—")
                if isinstance(v, float):
                    vals.append(f"{v:>15.4f}")
                else:
                    vals.append(f"{str(v):>15}")
            rows.append(" | ".join(vals))

        return "\n".join(rows)
