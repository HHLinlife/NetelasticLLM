"""
utils/visualization.py
Result visualisation utilities for NetelasticLLM.

Produces plots from Figures 4–8 in the paper:
  - Alignment-depth curves (Fig. 4)
  - Dataset-size scaling (Fig. 5)
  - Model-scale rebound (Fig. 6)
  - Fine-tuning strategy heatmap (Fig. 7)
  - Ablation step-cost chart (Fig. 8)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")            # headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — visualisation disabled.")


def _check_mpl():
    if not MPL_AVAILABLE:
        raise ImportError("Install matplotlib to use visualisation utilities.")


# --------------------------------------------------------------------------- #
#  Figure 4 — Alignment depth curves                                           #
# --------------------------------------------------------------------------- #

def plot_alignment_depth(
    depth_labels:  List[str],
    acc_clean:     Dict[str, List[float]],   # dataset → values
    acc_perturbed: Dict[str, List[float]],
    kl_clean:      Dict[str, List[float]],
    kl_perturbed:  Dict[str, List[float]],
    save_path:     str = "outputs/fig4_alignment_depth.pdf",
):
    """
    Reproduce Figure 4: accuracy and KL vs alignment depth (clean & perturbed).
    """
    _check_mpl()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    datasets  = list(acc_clean.keys())
    colors    = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
    x = np.arange(len(depth_labels))

    for ax_row, (acc_dict, kl_dict, title) in enumerate([
        (acc_clean,     kl_clean,     "Clean"),
        (acc_perturbed, kl_perturbed, "Perturbed"),
    ]):
        ax_acc = axes[ax_row, 0]
        ax_kl  = axes[ax_row, 1]

        for ds, color in zip(datasets, colors):
            ax_acc.plot(x, acc_dict[ds], marker="o", label=ds, color=color)
            ax_kl.plot(x,  kl_dict[ds],  marker="s", color=color)

        ax_acc.set_title(f"{title} accuracy")
        ax_acc.set_xticks(x); ax_acc.set_xticklabels(depth_labels)
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend(fontsize=7)

        ax_kl.set_title(f"{title} KL divergence")
        ax_kl.set_xticks(x); ax_kl.set_xticklabels(depth_labels)
        ax_kl.set_ylabel("KL")

    for ax in axes.flat:
        ax.set_xlabel("Alignment Depth (%)")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)


# --------------------------------------------------------------------------- #
#  Figure 5 — Dataset-size scaling                                             #
# --------------------------------------------------------------------------- #

def plot_dataset_size_scaling(
    x_values:     List[Any],           # pretraining scales or alignment sizes
    delta_acc:    Dict[str, List[float]],   # model → ΔAcc values
    kl_vals:      Dict[str, List[float]],   # model → KL values
    xlabel:       str = "Pretraining Scale N_pre (T)",
    save_path:    str = "outputs/fig5_dataset_scaling.pdf",
):
    _check_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(delta_acc)))

    for (model, da), (_, kl), color in zip(
        delta_acc.items(), kl_vals.items(), colors
    ):
        ax1.plot(range(len(x_values)), da, marker="o", label=model, color=color)
        ax2.plot(range(len(x_values)), kl, marker="s", color=color)

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values], rotation=20)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)

    ax1.set_ylabel("Fine-grained Drop ΔAcc"); ax1.legend(fontsize=7)
    ax2.set_ylabel("KL Divergence")
    plt.tight_layout()
    _save(fig, save_path)


# --------------------------------------------------------------------------- #
#  Figure 7 — Fine-tuning strategy heatmap                                     #
# --------------------------------------------------------------------------- #

def plot_finetuning_heatmap(
    models:     List[str],
    strategies: List[str],
    values:     np.ndarray,           # (n_models, n_strategies)
    title:      str = "ΔAcc heatmap",
    fmt:        str = ".1f",
    save_path:  str = "outputs/fig7_heatmap.pdf",
):
    _check_mpl()
    fig, ax = plt.subplots(figsize=(len(strategies) * 2, len(models) * 0.7 + 1))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="ΔAcc")

    ax.set_xticks(range(len(strategies))); ax.set_xticklabels(strategies)
    ax.set_yticks(range(len(models)));     ax.set_yticklabels(models)
    ax.set_title(title)

    for i in range(len(models)):
        for j in range(len(strategies)):
            ax.text(j, i, f"{values[i,j]:{fmt}}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    _save(fig, save_path)


# --------------------------------------------------------------------------- #
#  Figure 8 — Ablation convergence steps                                       #
# --------------------------------------------------------------------------- #

def plot_ablation_steps(
    variants:     List[str],
    steps_median: Dict[str, List[float]],   # dataset → [steps per variant]
    steps_std:    Dict[str, List[float]],
    save_path:    str = "outputs/fig8_ablation_steps.pdf",
):
    _check_mpl()
    datasets = list(steps_median.keys())
    x        = np.arange(len(variants))
    width    = 0.18
    fig, ax  = plt.subplots(figsize=(9, 5))
    colors   = plt.cm.Pastel1(np.linspace(0, 1, len(datasets)))

    for i, (ds, color) in enumerate(zip(datasets, colors)):
        offset = (i - len(datasets) / 2) * width
        ax.bar(
            x + offset,
            steps_median[ds],
            width=width,
            yerr=steps_std.get(ds, [0] * len(variants)),
            label=ds,
            color=color,
            capsize=3,
        )

    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylabel("Steps to threshold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)


# --------------------------------------------------------------------------- #
#  Helper                                                                       #
# --------------------------------------------------------------------------- #

def _save(fig, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(path).suffix.lower()
    fig.savefig(path, dpi=150, bbox_inches="tight",
                format=ext.lstrip(".") if ext in (".pdf", ".png", ".svg") else "png")
    logger.info("Figure saved to %s", path)
    plt.close(fig)
