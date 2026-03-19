"""
models/traffic_classifier/fine_tuning_strategies.py
Three fine-tuning strategies described in Section 5.4.2 and Table 2.

  1. Full Fine-Tuning   — all backbone parameters updated
  2. LoRA Adaptation    — low-rank adapters in attention projections
  3. Prefix Tuning      — prepend learnable virtual tokens

Each strategy wraps a TrafficLLM and returns a training-ready model
with only the appropriate parameters marked as trainable.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  LoRA layer (rank-r update for linear projections)                           #
# --------------------------------------------------------------------------- #

class LoRALinear(nn.Module):
    """
    Low-rank adaptation of a frozen linear layer.

    Implements: W' = W + (α/r) · B · A
    where A ∈ R^{r×d_in}, B ∈ R^{d_out×r} are the trainable adapters.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank       = rank
        self.alpha      = alpha
        self.scaling    = alpha / rank

        d_in  = base_layer.in_features
        d_out = base_layer.out_features

        self.lora_A = nn.Linear(d_in,  rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Initialise: A ~ N(0, 1/√rank), B = 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + self.scaling * lora_out


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> nn.Module:
    """
    In-place injection of LoRA adapters into named linear modules.

    Args:
        model          : The neural network to modify.
        target_modules : List of module name substrings to target
                         (e.g. ["q_proj", "v_proj"]).
    Returns:
        The modified model (same object, in-place).
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        # Navigate to parent module and replace
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr   = parts[-1]

        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_layer)
        replaced += 1

    logger.info("Injected LoRA adapters into %d linear layers.", replaced)
    return model


# --------------------------------------------------------------------------- #
#  Prefix Tuning                                                                #
# --------------------------------------------------------------------------- #

class PrefixEncoder(nn.Module):
    """
    Generates prefix key-value pairs for each transformer layer.

    Implements the architecture from Table 2:
      - prefix_length : number of virtual tokens
      - projection_depth : MLP depth for reparameterisation
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        prefix_length: int = 128,
        projection_depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.num_layers    = num_layers
        self.hidden_size   = hidden_size

        # Learnable embedding table for prefix tokens
        self.embedding = nn.Embedding(prefix_length, hidden_size)

        # Reparameterisation MLP
        layers: List[nn.Module] = []
        in_dim = hidden_size
        for depth in range(projection_depth):
            out_dim = hidden_size * 2 if depth < projection_depth - 1 else hidden_size * num_layers * 2
            layers += [nn.Linear(in_dim, out_dim), nn.Tanh()]
            in_dim  = out_dim
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Returns:
            prefix_kv : (num_layers, 2, B, prefix_length, hidden_size)
                        where the '2' dimension contains keys and values.
        """
        tokens  = torch.arange(self.prefix_length, device=self.embedding.weight.device)
        embeds  = self.dropout(self.embedding(tokens))          # (L, H)
        proj    = self.mlp(embeds)                              # (L, num_layers*2*H)
        proj    = proj.view(
            self.prefix_length,
            self.num_layers,
            2,
            self.hidden_size,
        )                                                        # (L, N_layers, 2, H)
        proj    = proj.permute(1, 2, 0, 3)                      # (N_layers, 2, L, H)
        proj    = proj.unsqueeze(2).expand(-1, -1, batch_size, -1, -1)
        return proj                                              # (N_layers, 2, B, L, H)


# --------------------------------------------------------------------------- #
#  Strategy wrappers                                                            #
# --------------------------------------------------------------------------- #

class FullFineTuning:
    """All backbone and head parameters trainable."""

    @staticmethod
    def prepare(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
        for p in model.parameters():
            p.requires_grad_(True)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("FullFineTuning: %d trainable parameters.", n)
        return model


class LoRAFineTuning:
    """Inject LoRA adapters; only adapters + head are trainable."""

    @staticmethod
    def prepare(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
        # Freeze all backbone parameters first
        for p in model.backbone.parameters():
            p.requires_grad_(False)

        targets  = cfg.get("lora_target_modules", ["q_proj", "v_proj"])
        rank     = cfg.get("lora_rank",  8)
        alpha    = cfg.get("lora_alpha", 32)
        dropout  = cfg.get("lora_dropout", 0.05)

        inject_lora(model.backbone, targets, rank=rank, alpha=alpha, dropout=dropout)

        # Unfreeze classification heads
        for head in [model.fine_head, model.coarse_head]:
            for p in head.parameters():
                p.requires_grad_(True)

        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("LoRAFineTuning: %d trainable parameters.", n)
        return model


class PrefixFineTuning:
    """Add a PrefixEncoder; only prefix + head are trainable."""

    @staticmethod
    def prepare(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
        # Freeze backbone
        for p in model.backbone.parameters():
            p.requires_grad_(False)

        num_layers    = cfg.get("num_layers", 4)
        hidden_size   = getattr(model.backbone, "hidden_size", 256)
        prefix_length = cfg.get("prefix_length", 128)
        proj_depth    = cfg.get("prefix_projection_depth", 2)

        model.prefix_encoder = PrefixEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            prefix_length=prefix_length,
            projection_depth=proj_depth,
        )

        # Unfreeze heads and prefix encoder
        for p in model.prefix_encoder.parameters():
            p.requires_grad_(True)
        for head in [model.fine_head, model.coarse_head]:
            for p in head.parameters():
                p.requires_grad_(True)

        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("PrefixFineTuning: %d trainable parameters.", n)
        return model


# --------------------------------------------------------------------------- #
#  Factory                                                                      #
# --------------------------------------------------------------------------- #

STRATEGY_MAP = {
    "full":   FullFineTuning,
    "lora":   LoRAFineTuning,
    "prefix": PrefixFineTuning,
}


def apply_fine_tuning_strategy(
    model: nn.Module,
    strategy: str,
    cfg: Dict[str, Any],
) -> nn.Module:
    """
    Prepare model for fine-tuning by applying the named strategy.

    Args:
        model    : TrafficLLM instance.
        strategy : 'full' | 'lora' | 'prefix'
        cfg      : Strategy-specific hyperparameters from model_config.yaml.
    Returns:
        Modified model with correct requires_grad flags.
    """
    if strategy not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown fine-tuning strategy '{strategy}'. "
            f"Choose from {list(STRATEGY_MAP.keys())}."
        )
    return STRATEGY_MAP[strategy].prepare(model, cfg)
