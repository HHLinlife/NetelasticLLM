"""
models/backbone/pretrained_llm.py
Wrapper around HuggingFace pretrained language models for traffic analysis.

Supports:
  - Qwen2 (7B / 72B)
  - LLaMA-3 (8B / 70B)
  - Mixtral-8×7B
  - DeepSeek-V2 / V3
  - BERT-style encoder-only models (e.g. for DoHunter)

The wrapper exposes:
  - encode(x)  : extract sequence representation from feature tokens
  - logits(x)  : pre-softmax classification head output
  - gradient_wrt_input(x, head) : ∇_x f_{θ₀}(φ(x)) (Eq. 6 in paper)
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

# Optional HuggingFace imports
try:
    from transformers import (
        AutoModel, AutoModelForSequenceClassification,
        AutoTokenizer, BitsAndBytesConfig,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers not installed — using lightweight stub backbone.")


# --------------------------------------------------------------------------- #
#  Lightweight stub (no HuggingFace dependency)                                #
# --------------------------------------------------------------------------- #

class StubTransformerBackbone(nn.Module):
    """
    Minimal decoder-only Transformer for development / testing without
    a real pretrained checkpoint.  Mirrors the architectural parameters
    in Table 1 (hidden_size, num_layers, num_heads).
    """

    def __init__(
        self,
        input_dim: int = 83,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_classes: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm (GPT-style)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(hidden_size)
        self.classifier  = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, feature_dim) flow feature vectors.
        Returns:
            h : (B, hidden_size) pooled representation.
        """
        # Treat feature vector as a single-token sequence
        h = self.input_proj(x).unsqueeze(1)          # (B, 1, H)
        h = self.transformer(h)                       # (B, 1, H)
        h = self.norm(h.squeeze(1))                   # (B, H)
        return h

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-softmax logits f_{θ₀}(φ(x))."""
        return self.classifier(self.encode(x))        # (B, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h   = self.encode(x)
        log = self.classifier(h)
        return {"logits": log, "hidden": h}


# --------------------------------------------------------------------------- #
#  HuggingFace wrapper (real pretrained models)                                #
# --------------------------------------------------------------------------- #

class PretrainedLLMWrapper(nn.Module):
    """
    Wraps a HuggingFace causal or masked LM for encrypted-traffic
    classification.

    Traffic flows are projected into the model's embedding space via a
    learned linear adapter, then processed by the frozen (or partially
    fine-tuned) transformer.

    Parameters
    ----------
    model_name_or_path : HF model id or local directory.
    num_classes        : Number of fine-grained output classes.
    feature_dim        : Input feature vector dimension.
    load_in_4bit       : Use bitsandbytes 4-bit quantisation.
    use_cpu_offload    : Offload large layers to CPU RAM.
    pooling            : Representation pooling strategy ('mean'|'last'|'cls').
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int = 20,
        feature_dim: int = 83,
        load_in_4bit: bool = False,
        use_cpu_offload: bool = False,
        pooling: str = "mean",
    ):
        super().__init__()

        if not HF_AVAILABLE:
            raise RuntimeError(
                "Install transformers & torch to use PretrainedLLMWrapper."
            )

        self.num_classes  = num_classes
        self.feature_dim  = feature_dim
        self.pooling      = pooling

        # ── Quantisation config ──────────────────────────────────────── #
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # ── Load backbone ────────────────────────────────────────────── #
        device_map = "auto" if use_cpu_offload else None
        logger.info("Loading backbone: %s", model_name_or_path)
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16 if load_in_4bit else torch.float32,
            output_hidden_states=True,
        )
        self.hidden_size = self.backbone.config.hidden_size

        # ── Traffic adapter layers ────────────────────────────────────── #
        # Project flow features into LLM token embedding space
        self.input_adapter = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        )
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, feature_dim) float tensor.
        Returns:
            h : (B, hidden_size) pooled representation.
        """
        embeds  = self.input_adapter(x).unsqueeze(1)      # (B, 1, H)
        outputs = self.backbone(inputs_embeds=embeds)
        last_hs = outputs.last_hidden_state                # (B, 1, H)

        if self.pooling == "mean":
            h = last_hs.mean(dim=1)
        elif self.pooling == "last":
            h = last_hs[:, -1, :]
        else:                                               # cls / first
            h = last_hs[:, 0, :]
        return h                                            # (B, H)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h   = self.encode(x)
        log = self.classifier(h)
        return {"logits": log, "hidden": h}

    def gradient_wrt_input(
        self,
        x: torch.Tensor,
        head: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ∇_x f_{θ₀}(φ(x)) as in Eq. (6).

        Args:
            x    : (B, feature_dim) requires_grad=True input.
            head : (num_classes,) weight vector for directional gradient.
                   If None, returns gradient of the max-logit class.
        Returns:
            grad : (B, feature_dim) gradient tensor.
        """
        x = x.detach().requires_grad_(True)
        log = self.logits(x)                              # (B, C)
        if head is not None:
            scalar = (log * head.unsqueeze(0)).sum()
        else:
            scalar = log.max(dim=1).values.sum()
        scalar.backward()
        return x.grad.detach()


# --------------------------------------------------------------------------- #
#  Factory                                                                      #
# --------------------------------------------------------------------------- #

def build_backbone(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build a backbone model from a config dict.

    If 'model_path' refers to an existing directory / HF id and
    HuggingFace is available, load the real pretrained model.
    Otherwise fall back to StubTransformerBackbone.
    """
    model_path  = cfg.get("model_path", "")
    num_classes = cfg.get("num_classes", 20)
    feature_dim = cfg.get("input_dim", 83)
    load_4bit   = cfg.get("quantization") == "4bit"
    cpu_offload = cfg.get("use_cpu_offload", False)

    if HF_AVAILABLE and (model_path.startswith("meta-llama") or
                          model_path.startswith("Qwen") or
                          model_path.startswith("mistralai") or
                          model_path.startswith("deepseek") or
                          (os.path.isdir(model_path) if __import__('os').path.exists(model_path) else False)):
        return PretrainedLLMWrapper(
            model_name_or_path=model_path,
            num_classes=num_classes,
            feature_dim=feature_dim,
            load_in_4bit=load_4bit,
            use_cpu_offload=cpu_offload,
        )

    # Stub backbone for development
    hidden = cfg.get("hidden_size", 256)
    layers = min(cfg.get("num_layers", 32), 6)     # cap for feasibility
    heads  = min(cfg.get("num_attention_heads", 32), 8)
    logger.info(
        "Using StubTransformerBackbone (hidden=%d, layers=%d, heads=%d).",
        hidden, layers, heads,
    )
    return StubTransformerBackbone(
        input_dim=feature_dim,
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        num_classes=num_classes,
    )
