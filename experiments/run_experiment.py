"""
experiments/run_experiment.py
Main experiment runner for elastic rebound evaluation.

Usage
-----
python -m experiments.run_experiment \
    --dataset ustc \
    --dataset_dir data/ustc_tfc2016 \
    --backbone llama3-8b \
    --strategy lora \
    --device cuda

Runs:
  1. Pretraining  (M₀)
  2. Fine-tuning  (M₁)
  3. Surrogate distillation (M̂₁)
  4. Dual-layer evolutionary search → U*
  5. Perturbation fine-tuning       (M₂)
  6. Evaluation and logging
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Seed helpers                                                                 #
# --------------------------------------------------------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Training loop helpers                                                        #
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimiser:  torch.optim.Optimizer,
    device:     torch.device,
    scheduler=None,
) -> float:
    """Generic one-epoch training loop. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        x          = batch["features"].to(device)
        fine_y     = batch["fine_label"].to(device)
        coarse_y   = batch["coarse_label"].to(device)

        optimiser.zero_grad()
        out  = model(x)
        loss = out["loss"] if "loss" in out else model.compute_loss(
            x, fine_y, coarse_y
        )["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
) -> Dict[str, float]:
    """Evaluate accuracy on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x = batch["features"].to(device)
        y = batch["fine_label"].to(device)
        out    = model(x)
        logits = out.get("fine_logits", out.get("logits")) if isinstance(out, dict) else out
        preds  = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total   += len(y)
    acc = correct / max(total, 1)
    return {"accuracy": acc}


# --------------------------------------------------------------------------- #
#  Main experiment                                                              #
# --------------------------------------------------------------------------- #

def run_experiment(cfg: Dict[str, Any]):
    """
    End-to-end experiment pipeline.

    Args:
        cfg : Flat dict of configuration values (see parse_args).
    """
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting experiment: %s ===", cfg)

    # ── 1. Load dataset ──────────────────────────────────────────────── #
    from data.datasets.cesnet_tls22 import make_dataset

    ds_pretrain = make_dataset(cfg["dataset"], root_dir=cfg["dataset_dir"],
                               split="pretrain")
    ds_finetune = make_dataset(cfg["dataset"], root_dir=cfg["dataset_dir"],
                               split="finetune")
    ds_test     = make_dataset(cfg["dataset"], root_dir=cfg["dataset_dir"],
                               split="test")

    # Apply pretrain normalisation statistics to other splits
    if ds_pretrain.mean is not None:
        ds_finetune.apply_normalization(ds_pretrain.mean, ds_pretrain.std)
        ds_test.apply_normalization(ds_pretrain.mean, ds_pretrain.std)

    loader_pretrain = DataLoader(ds_pretrain, batch_size=cfg["batch_size"],
                                 shuffle=True,  num_workers=cfg["num_workers"])
    loader_finetune = DataLoader(ds_finetune, batch_size=cfg["batch_size"],
                                 shuffle=True,  num_workers=cfg["num_workers"])
    loader_test     = DataLoader(ds_test,     batch_size=cfg["batch_size"],
                                 shuffle=False, num_workers=cfg["num_workers"])

    logger.info("Datasets loaded — pretrain=%d, finetune=%d, test=%d",
                len(ds_pretrain), len(ds_finetune), len(ds_test))

    # ── 2. Build models ──────────────────────────────────────────────── #
    from models.traffic_classifier.traffic_llm import build_traffic_llm
    from models.traffic_classifier.fine_tuning_strategies import apply_fine_tuning_strategy
    from models.surrogate.surrogate_model import SurrogateModel

    backbone_cfg = {
        "model_name":          cfg["backbone"],
        "model_path":          cfg.get("model_path", ""),
        "hidden_size":         cfg.get("hidden_size", 256),
        "num_layers":          cfg.get("num_layers", 4),
        "num_attention_heads": cfg.get("num_heads", 4),
        "input_dim":           cfg.get("feature_dim", 83),
        "num_classes":         ds_pretrain.num_classes,
    }

    m0 = build_traffic_llm(backbone_cfg, variant=cfg.get("variant", "TrafficLLM"))
    m1 = build_traffic_llm(backbone_cfg, variant=cfg.get("variant", "TrafficLLM"))
    m0.to(device); m1.to(device)

    # Pretrain M₀
    logger.info("Pretraining M₀ …")
    opt_pre = optim.Adam(m0.parameters(), lr=cfg["lr_pretrain"],
                         weight_decay=cfg["weight_decay"])
    for epoch in range(cfg["pretrain_epochs"]):
        loss = train_one_epoch(m0, loader_pretrain, opt_pre, device)
        if (epoch + 1) % 10 == 0:
            res = evaluate(m0, loader_test, device)
            logger.info("Pretrain epoch %d: loss=%.4f, acc=%.4f", epoch+1, loss, res["accuracy"])

    torch.save(m0.state_dict(), output_dir / "m0.pt")
    logger.info("M₀ saved.")

    # Copy pretrained weights and fine-tune M₁
    m1.load_state_dict(m0.state_dict())
    m1 = apply_fine_tuning_strategy(m1, cfg["strategy"], backbone_cfg)

    logger.info("Fine-tuning M₁ with strategy=%s …", cfg["strategy"])
    opt_ft = optim.AdamW(
        filter(lambda p: p.requires_grad, m1.parameters()),
        lr=cfg["lr_finetune"], weight_decay=cfg["weight_decay"]
    )
    best_val_acc = 0.0
    patience_cnt = 0
    for epoch in range(cfg["finetune_epochs"]):
        loss = train_one_epoch(m1, loader_finetune, opt_ft, device)
        res  = evaluate(m1, loader_test, device)
        if res["accuracy"] > best_val_acc:
            best_val_acc = res["accuracy"]
            torch.save(m1.state_dict(), output_dir / "m1_best.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= cfg["early_stopping"]:
            logger.info("Early stopping at epoch %d.", epoch + 1)
            break
        if (epoch + 1) % 5 == 0:
            logger.info("Finetune epoch %d: loss=%.4f, acc=%.4f", epoch+1, loss, res["accuracy"])

    m1.load_state_dict(torch.load(output_dir / "m1_best.pt"))
    acc_m1 = evaluate(m1, loader_test, device)["accuracy"]
    logger.info("M₁ accuracy: %.4f", acc_m1)

    # ── 3. Surrogate distillation ─────────────────────────────────────── #
    from distillation.one_shot_distillation import OneShotDistillation

    surrogate = SurrogateModel(
        input_dim=cfg.get("feature_dim", 83),
        hidden_dim=256,
        num_layers=3,
        num_classes=ds_pretrain.num_classes,
    ).to(device)

    all_features = torch.from_numpy(ds_pretrain.features).float().to(device)

    def oracle_fn(x: torch.Tensor) -> torch.Tensor:
        m1.eval()
        with torch.no_grad():
            out = m1(x)
            logits = out["fine_logits"] if isinstance(out, dict) else out
            return torch.softmax(logits / 2.0, dim=-1)

    distiller = OneShotDistillation(
        surrogate=surrogate,
        oracle_fn=oracle_fn,
        warmup_budget=cfg.get("warmup_budget", 5000),
        active_budget=cfg.get("active_budget", 10000),
        device=str(device),
    )
    distiller.fit(ds_pretrain.features)
    agree = distiller.evaluate_agreement(ds_test.features[:1000])
    logger.info("Surrogate Agree_{M₁} = %.4f", agree)

    # ── 4. Dual-layer evolutionary search ─────────────────────────────── #
    from perturbation.genome_encoding import GenomeFactory, GenomeConstraints
    from search.dual_layer_evolution import DualLayerEvolution
    from data.preprocessing.feature_extractor import FlowFeatureExtractor, CIC_CONFIG

    extractor   = FlowFeatureExtractor(CIC_CONFIG, target_dim=cfg.get("feature_dim", 83))
    constraints = GenomeConstraints()
    factory     = GenomeFactory(constraints)

    def feature_fn(g):
        """Map a Genome to a feature vector via synthetic flow stats."""
        arr = g.to_array()   # (N, 3): [dir, len, delta]
        # Simple aggregated statistics as a feature proxy
        feat = np.concatenate([
            arr.mean(axis=0), arr.std(axis=0),
            arr.min(axis=0),  arr.max(axis=0),
        ]).astype(np.float32)
        # Pad / truncate to feature_dim
        dim = cfg.get("feature_dim", 83)
        if len(feat) < dim:
            feat = np.concatenate([feat, np.zeros(dim - len(feat))])
        return feat[:dim]

    # Sample candidate pool from feasible domain
    rng = np.random.default_rng(cfg["seed"])
    candidate_pool = [factory.random(rng) for _ in range(cfg.get("pool_size", 500))]

    solver = DualLayerEvolution(
        pretrained_model=m0,
        surrogate_model=surrogate,
        feature_fn=feature_fn,
        constraints=constraints,
        num_prototypes=cfg.get("num_prototypes", 5),
        population_size=cfg.get("population_size", 50),
        mu=cfg.get("mu", 10),
        lambda_=cfg.get("lambda_", 40),
        max_rounds=cfg.get("max_rounds", 20),
        device=str(device),
        seed=cfg["seed"],
    )

    u_star = solver.run(candidate_pool)
    logger.info("U* size: %d, best fitness: %.4f",
                len(u_star), u_star[0].fitness if u_star else float("nan"))

    # ── 5. Perturbation fine-tuning → M₂ ──────────────────────────────── #
    m2 = build_traffic_llm(backbone_cfg, variant=cfg.get("variant", "TrafficLLM"))
    m2.load_state_dict(m1.state_dict())
    m2.to(device)

    # Build a perturbation dataset from U*
    if u_star:
        perturb_feats  = np.stack([feature_fn(g) for g in u_star[:200]])
        perturb_labels = np.zeros(len(perturb_feats), dtype=np.int64)  # coarse benign

        import torch.utils.data as D
        p_tensor  = torch.from_numpy(perturb_feats).float()
        pl_tensor = torch.from_numpy(perturb_labels).long()
        p_dataset = D.TensorDataset(p_tensor, pl_tensor)
        p_loader  = D.DataLoader(p_dataset, batch_size=32, shuffle=True)

        opt_pert = optim.AdamW(m2.parameters(), lr=cfg.get("lr_perturb", 1e-5))
        for epoch in range(cfg.get("perturb_epochs", 5)):
            m2.train()
            for x_p, _ in p_loader:
                x_p = x_p.to(device)
                opt_pert.zero_grad()
                out = m2(x_p)
                logits = out["fine_logits"] if isinstance(out, dict) else out
                # Distillation from M₀ soft labels
                with torch.no_grad():
                    out0   = m0(x_p)
                    soft0  = torch.softmax(
                        out0["fine_logits"] if isinstance(out0, dict) else out0,
                        dim=-1
                    )
                loss = nn.KLDivLoss(reduction="batchmean")(
                    torch.log_softmax(logits, dim=-1), soft0
                )
                loss.backward()
                opt_pert.step()

    # ── 6. Evaluate rebound ────────────────────────────────────────────── #
    from evaluation.metrics import compute_rebound_metrics

    test_feats  = torch.from_numpy(ds_test.features).float()
    test_labels = torch.from_numpy(ds_test.fine_labels).long()

    metrics = compute_rebound_metrics(
        m1, m2, m0, test_feats, test_labels, device=str(device)
    )
    metrics["acc_m1_pct"]     = metrics["acc_m1"]     * 100
    metrics["acc_m2_pct"]     = metrics["acc_m2"]     * 100
    metrics["delta_acc_pct"]  = metrics["delta_acc"]  * 100
    metrics["surrogate_agree"] = agree

    logger.info(
        "RESULTS — ΔAcc=%.2f%%, KL(M₂‖M₀)=%.4f, Agree=%.4f",
        metrics["delta_acc_pct"], metrics["kl_m2_m0"], agree,
    )

    result_path = output_dir / "results.json"
    with open(result_path, "w") as f:
        json.dump({**cfg, **metrics}, f, indent=2)
    logger.info("Results saved to %s", result_path)
    return metrics


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="NetelasticLLM experiment runner")
    p.add_argument("--dataset",        default="ustc",    help="ustc|cic|cesnet")
    p.add_argument("--dataset_dir",    default="data/ustc_tfc2016")
    p.add_argument("--backbone",       default="stub",    help="stub|llama3-8b|qwen2-7b|…")
    p.add_argument("--model_path",     default="",        help="HF model path")
    p.add_argument("--strategy",       default="lora",    help="full|lora|prefix")
    p.add_argument("--variant",        default="TrafficLLM")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--num_workers",    type=int, default=2)
    p.add_argument("--pretrain_epochs",type=int, default=20)
    p.add_argument("--finetune_epochs",type=int, default=20)
    p.add_argument("--early_stopping", type=int, default=10)
    p.add_argument("--lr_pretrain",    type=float, default=5e-4)
    p.add_argument("--lr_finetune",    type=float, default=2e-5)
    p.add_argument("--lr_perturb",     type=float, default=1e-5)
    p.add_argument("--weight_decay",   type=float, default=5e-4)
    p.add_argument("--warmup_budget",  type=int, default=5_000)
    p.add_argument("--active_budget",  type=int, default=10_000)
    p.add_argument("--pool_size",      type=int, default=300)
    p.add_argument("--num_prototypes", type=int, default=5)
    p.add_argument("--population_size",type=int, default=50)
    p.add_argument("--mu",             type=int, default=10)
    p.add_argument("--lambda_",        type=int, default=40)
    p.add_argument("--max_rounds",     type=int, default=20)
    p.add_argument("--perturb_epochs", type=int, default=5)
    p.add_argument("--feature_dim",    type=int, default=83)
    p.add_argument("--hidden_size",    type=int, default=256)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--num_heads",      type=int, default=4)
    p.add_argument("--output_dir",     default="outputs/exp1")
    return vars(p.parse_args())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    cfg = parse_args()
    run_experiment(cfg)
