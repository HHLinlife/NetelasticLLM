"""
Transformer Backbone for Traffic Language Models

This module implements a decoder-only transformer architecture
suitable for encrypted traffic sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as used in modern LLMs.
    
    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) embeddings [1, seq_len, 1, dim]
        """
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]
        
        return self._cos_cached[:, :seq_len], self._sin_cached[:, :seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with rotary position embeddings.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = dropout
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Rotary position embedding
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len] or [batch, seq_len, seq_len]
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        
        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores: [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand mask: [batch, 1, 1, seq_len]
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                # Expand mask: [batch, 1, seq_len, seq_len]
                attention_mask = attention_mask[:, None, :, :]
            
            # Apply mask (set masked positions to large negative value)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attention output: [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose and reshape: [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with SwiGLU activation.
    
    SwiGLU: Swish-Gated Linear Unit, commonly used in modern LLMs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.
        
        SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with pre-normalization.
    
    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout
        )
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(hidden_size, eps=1e-6)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class TransformerBackbone(nn.Module):
    """
    Decoder-only transformer backbone for traffic sequence modeling.
    
    This implements a LLaMA-style architecture suitable for
    encrypted traffic analysis.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        intermediate_size: int = 11008,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding (not used for traffic, but included for completeness)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized TransformerBackbone with {self.num_parameters():,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer layers.
        
        Args:
            input_embeds: Input embeddings [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Hidden states [batch, seq_len, hidden_size]
        """
        hidden_states = self.dropout(input_embeds)
        
        # Pass through transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
    
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
