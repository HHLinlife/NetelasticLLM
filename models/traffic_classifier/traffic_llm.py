"""
Traffic Language Model for Encrypted Traffic Classification

This module implements the complete traffic classification model
that combines packet sequence encoding with the transformer backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from models.backbone.transformer_backbone import TransformerBackbone

logger = logging.getLogger(__name__)


class PacketSequenceEncoder(nn.Module):
    """
    Encodes packet sequences (direction, length, timing) into embeddings.
    
    The encoding follows the paper's representation: x = {(d_i, l_i, δ_i)}_{i=1}^N
    where:
        d_i ∈ {+1, -1}: packet direction
        l_i: packet length in bytes
        δ_i: inter-arrival time in milliseconds
    """
    
    def __init__(
        self,
        direction_embed_dim: int = 64,
        length_embed_dim: int = 128,
        timing_embed_dim: int = 128,
        hidden_size: int = 4096,
        max_packet_length: int = 1500,
        max_timing: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.direction_embed_dim = direction_embed_dim
        self.length_embed_dim = length_embed_dim
        self.timing_embed_dim = timing_embed_dim
        self.hidden_size = hidden_size
        
        # Direction embedding: map {-1, 1} to embeddings
        # We use offset of 1 to map -1->0, 1->2, with 1 as padding
        self.direction_embedding = nn.Embedding(3, direction_embed_dim, padding_idx=1)
        
        # Length embedding using learned continuous embedding
        self.length_projection = nn.Sequential(
            nn.Linear(1, length_embed_dim),
            nn.LayerNorm(length_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Timing embedding using learned continuous embedding
        self.timing_projection = nn.Sequential(
            nn.Linear(1, timing_embed_dim),
            nn.LayerNorm(timing_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined embedding dimension
        combined_dim = direction_embed_dim + length_embed_dim + timing_embed_dim
        
        # Project to transformer hidden size
        self.output_projection = nn.Linear(combined_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Normalization statistics (to be computed from data)
        self.register_buffer('length_mean', torch.tensor(750.0))
        self.register_buffer('length_std', torch.tensor(400.0))
        self.register_buffer('timing_mean', torch.tensor(50.0))
        self.register_buffer('timing_std', torch.tensor(100.0))
    
    def update_normalization_stats(self, length_mean: float, length_std: float, timing_mean: float, timing_std: float):
        """Update normalization statistics from dataset."""
        self.length_mean = torch.tensor(length_mean)
        self.length_std = torch.tensor(length_std)
        self.timing_mean = torch.tensor(timing_mean)
        self.timing_std = torch.tensor(timing_std)
    
    def forward(
        self,
        direction: torch.Tensor,
        length: torch.Tensor,
        timing: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode packet sequences.
        
        Args:
            direction: Direction sequence [batch, seq_len], values in {-1, 0, 1}
                      where 0 is padding
            length: Length sequence [batch, seq_len], in bytes
            timing: Timing sequence [batch, seq_len], in milliseconds
            
        Returns:
            Encoded embeddings [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = direction.shape
        
        # Direction embedding: offset by 1 to handle {-1, 0, 1}
        direction_offset = direction + 1  # Maps -1->0, 0->1, 1->2
        direction_emb = self.direction_embedding(direction_offset)  # [batch, seq_len, direction_dim]
        
        # Normalize and project length
        length_normalized = (length - self.length_mean) / (self.length_std + 1e-6)
        length_emb = self.length_projection(length_normalized.unsqueeze(-1))  # [batch, seq_len, length_dim]
        
        # Normalize and project timing
        timing_normalized = (timing - self.timing_mean) / (self.timing_std + 1e-6)
        timing_emb = self.timing_projection(timing_normalized.unsqueeze(-1))  # [batch, seq_len, timing_dim]
        
        # Concatenate all embeddings
        combined = torch.cat([direction_emb, length_emb, timing_emb], dim=-1)  # [batch, seq_len, combined_dim]
        
        # Project to hidden size
        output = self.output_projection(combined)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class TrafficLLM(nn.Module):
    """
    Complete Traffic Language Model for encrypted traffic classification.
    
    Architecture:
        1. Packet sequence encoder: (d, l, δ) → embeddings
        2. Transformer backbone: embeddings → hidden states
        3. Classification heads: hidden states → predictions
    """
    
    def __init__(
        self,
        # Encoder parameters
        direction_embed_dim: int = 64,
        length_embed_dim: int = 128,
        timing_embed_dim: int = 128,
        
        # Transformer backbone parameters
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        intermediate_size: int = 11008,
        max_seq_len: int = 256,
        
        # Classification parameters
        num_coarse_classes: int = 2,  # benign vs malicious
        num_fine_classes: int = 20,   # specific categories
        
        # Training parameters
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_coarse_classes = num_coarse_classes
        self.num_fine_classes = num_fine_classes
        
        # Packet sequence encoder
        self.encoder = PacketSequenceEncoder(
            direction_embed_dim=direction_embed_dim,
            length_embed_dim=length_embed_dim,
            timing_embed_dim=timing_embed_dim,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Transformer backbone
        self.backbone = TransformerBackbone(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Pooling strategy: use mean pooling over sequence
        self.pooling = 'mean'
        
        # Classification heads
        self.coarse_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_coarse_classes)
        )
        
        self.fine_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_fine_classes)
        )
        
        logger.info(f"Initialized TrafficLLM with {self.num_parameters():,} parameters")
    
    def forward(
        self,
        direction: torch.Tensor,
        length: torch.Tensor,
        timing: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            direction: Packet direction [batch, seq_len]
            length: Packet length [batch, seq_len]
            timing: Inter-arrival time [batch, seq_len]
            mask: Attention mask [batch, seq_len]
            return_hidden_states: Whether to return intermediate hidden states
            
        Returns:
            Dictionary containing:
                - coarse_logits: Coarse classification logits [batch, num_coarse_classes]
                - fine_logits: Fine-grained classification logits [batch, num_fine_classes]
                - coarse_probs: Coarse class probabilities
                - fine_probs: Fine-grained class probabilities
                - hidden_states: (optional) Transformer hidden states [batch, seq_len, hidden_size]
                - pooled_output: (optional) Pooled representation [batch, hidden_size]
        """
        # Encode packet sequences
        input_embeds = self.encoder(direction, length, timing)  # [batch, seq_len, hidden_size]
        
        # Pass through transformer backbone
        hidden_states = self.backbone(input_embeds, attention_mask=mask)  # [batch, seq_len, hidden_size]
        
        # Pool sequence representations
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(hidden_states)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)
        
        # Classification
        coarse_logits = self.coarse_head(pooled)
        fine_logits = self.fine_head(pooled)
        
        # Compute probabilities
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        fine_probs = F.softmax(fine_logits, dim=-1)
        
        output = {
            'coarse_logits': coarse_logits,
            'fine_logits': fine_logits,
            'coarse_probs': coarse_probs,
            'fine_probs': fine_probs
        }
        
        if return_hidden_states:
            output['hidden_states'] = hidden_states
            output['pooled_output'] = pooled
        
        return output
    
    def get_pretrained_distribution(
        self,
        direction: torch.Tensor,
        length: torch.Tensor,
        timing: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get coarse-grained distribution from pretrained model.
        
        This corresponds to p_θ₀(y|x) in the paper.
        
        Returns:
            Coarse class probabilities [batch, num_coarse_classes]
        """
        output = self.forward(direction, length, timing, mask)
        return output['coarse_probs']
    
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Freeze transformer backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze transformer backbone")
    
    def unfreeze_backbone(self):
        """Unfreeze transformer backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfroze transformer backbone")
