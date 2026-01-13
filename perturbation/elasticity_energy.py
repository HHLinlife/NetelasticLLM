"""
Elasticity Energy Computation

This module implements the elastic energy metric E(u) as described in the paper.
The elasticity energy quantifies the alignment of perturbations with the
pretrained model's decision geometry.

The elastic energy is defined as:
    E(u) = (1/2) (d̃ᵀ g_proxy(u))²

where:
    - g_proxy(u) = ∇_x f_θ₀(φ(u)) is the proxy gradient from pretrained model
    - d̃ is the dominant rebound direction
    - d̃ = g_proxy(u)(f_θ₀(φ(u)) - f̂_θ̂₁(φ(u))) / ||...||
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from perturbation.genome_encoding import PacketGenome

logger = logging.getLogger(__name__)


class ElasticityEnergy:
    """
    Computes elastic energy for perturbation candidates.
    
    This energy metric provides a zero-query proxy for ranking
    perturbations by their expected contribution to inducing rebound.
    """
    
    def __init__(
        self,
        pretrained_model: nn.Module,
        surrogate_model: nn.Module,
        feature_extractor: Optional[nn.Module] = None,
        device: str = 'cuda'
    ):
        """
        Initialize elasticity energy computer.
        
        Args:
            pretrained_model: Pretrained traffic LLM (M₀)
            surrogate_model: Surrogate of fine-tuned model (M̂₁)
            feature_extractor: Optional feature encoder φ(·)
            device: Computation device
        """
        self.pretrained_model = pretrained_model
        self.surrogate_model = surrogate_model
        self.feature_extractor = feature_extractor
        self.device = device
        
        # Set models to eval mode
        self.pretrained_model.eval()
        self.surrogate_model.eval()
        
        # Move to device
        self.pretrained_model.to(device)
        self.surrogate_model.to(device)
        if self.feature_extractor is not None:
            self.feature_extractor.to(device)
    
    def compute_proxy_gradient(
        self,
        genome: PacketGenome,
        target_class: int = 0
    ) -> torch.Tensor:
        """
        Compute proxy gradient from pretrained model.
        
        This computes: g_proxy(u) = ∇_x f_θ₀(φ(u))
        
        Args:
            genome: Input perturbation genome
            target_class: Target class for gradient computation
            
        Returns:
            Proxy gradient tensor
        """
        # Convert genome to tensor
        genome_dict = genome.to_torch()
        direction = genome_dict['direction'].unsqueeze(0).to(self.device)
        length = genome_dict['length'].unsqueeze(0).to(self.device)
        timing = genome_dict['timing'].unsqueeze(0).to(self.device)
        
        # Enable gradient computation
        direction.requires_grad = False  # Direction is discrete
        length.requires_grad = True
        timing.requires_grad = True
        
        # Forward pass through pretrained model
        with torch.enable_grad():
            output = self.pretrained_model(
                direction=direction,
                length=length,
                timing=timing,
                return_hidden_states=True
            )
            
            # Get coarse logits (pre-softmax)
            coarse_logits = output['coarse_logits']  # [1, num_coarse_classes]
            
            # Compute gradient w.r.t. target class logit
            target_logit = coarse_logits[0, target_class]
            
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=target_logit,
                inputs=[length, timing],
                create_graph=False,
                retain_graph=False
            )
            
            grad_length = grads[0].squeeze(0)  # [seq_len]
            grad_timing = grads[1].squeeze(0)  # [seq_len]
        
        # Combine gradients into single vector
        proxy_gradient = torch.cat([grad_length, grad_timing], dim=0)  # [2 * seq_len]
        
        return proxy_gradient
    
    def compute_dominant_rebound_direction(
        self,
        genome: PacketGenome,
        proxy_gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dominant rebound direction d̃.
        
        This computes:
            d̃ = g_proxy(u) · (f_θ₀(φ(u)) - f̂_θ̂₁(φ(u))) / ||...||
        
        Args:
            genome: Input perturbation genome
            proxy_gradient: Precomputed proxy gradient
            
        Returns:
            Normalized rebound direction vector
        """
        # Convert genome to tensor
        genome_dict = genome.to_torch()
        direction = genome_dict['direction'].unsqueeze(0).to(self.device)
        length = genome_dict['length'].unsqueeze(0).to(self.device)
        timing = genome_dict['timing'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get pretrained model output
            pretrained_output = self.pretrained_model(
                direction=direction,
                length=length,
                timing=timing
            )
            pretrained_logits = pretrained_output['coarse_logits']  # [1, num_classes]
            
            # Get surrogate model output
            surrogate_output = self.surrogate_model(
                direction=direction,
                length=length,
                timing=timing
            )
            surrogate_logits = surrogate_output['coarse_logits']  # [1, num_classes]
            
            # Compute logit difference
            logit_diff = pretrained_logits - surrogate_logits  # [1, num_classes]
            
            # Weight gradient by logit difference
            # Expand proxy_gradient to match dimension if needed
            weighted_gradient = proxy_gradient * logit_diff.sum()  # Simplified version
            
            # Normalize to unit vector
            norm = torch.norm(weighted_gradient) + 1e-9
            rebound_direction = weighted_gradient / norm
        
        return rebound_direction
    
    def compute_elastic_energy(
        self,
        genome: PacketGenome,
        use_cache: bool = True
    ) -> float:
        """
        Compute elastic energy E(u) for a perturbation genome.
        
        E(u) = (1/2) (d̃ᵀ g_proxy(u))²
        
        Args:
            genome: Perturbation genome
            use_cache: Whether to use cached gradients (not implemented)
            
        Returns:
            Elastic energy value (scalar)
        """
        # Compute proxy gradient
        proxy_gradient = self.compute_proxy_gradient(genome, target_class=0)
        
        # Compute dominant rebound direction
        rebound_direction = self.compute_dominant_rebound_direction(genome, proxy_gradient)
        
        # Compute energy: E(u) = 0.5 * (d̃ᵀ g)²
        alignment = torch.dot(rebound_direction, proxy_gradient)
        energy = 0.5 * (alignment ** 2)
        
        return energy.item()
    
    def compute_batch_energies(
        self,
        genomes: list,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Compute elastic energies for a batch of genomes.
        
        Args:
            genomes: List of PacketGenome objects
            batch_size: Processing batch size
            
        Returns:
            Array of energy values
        """
        energies = []
        
        for i in range(0, len(genomes), batch_size):
            batch = genomes[i:i+batch_size]
            
            for genome in batch:
                try:
                    energy = self.compute_elastic_energy(genome)
                    energies.append(energy)
                except Exception as e:
                    logger.warning(f"Failed to compute energy for genome: {e}")
                    energies.append(0.0)  # Default to zero energy
        
        return np.array(energies)


class ElasticityGuidedRanking:
    """
    Ranks perturbation candidates using elasticity energy.
    
    This provides the energy-guided initialization component
    of the overall framework (Section 4.3 in paper).
    """
    
    def __init__(
        self,
        elasticity_energy: ElasticityEnergy,
        top_k: int = 100
    ):
        """
        Initialize ranking system.
        
        Args:
            elasticity_energy: Elasticity energy computer
            top_k: Number of top candidates to select
        """
        self.elasticity_energy = elasticity_energy
        self.top_k = top_k
    
    def rank_candidates(
        self,
        candidates: list,
        return_scores: bool = False
    ) -> list:
        """
        Rank candidates by elastic energy.
        
        Args:
            candidates: List of PacketGenome candidates
            return_scores: Whether to return energy scores
            
        Returns:
            Sorted list of top-k candidates (and optionally their scores)
        """
        logger.info(f"Ranking {len(candidates)} candidates by elastic energy...")
        
        # Compute energies
        energies = self.elasticity_energy.compute_batch_energies(candidates)
        
        # Sort by energy (descending - higher energy = stronger rebound potential)
        sorted_indices = np.argsort(energies)[::-1]
        
        # Select top-k
        top_indices = sorted_indices[:self.top_k]
        ranked_candidates = [candidates[i] for i in top_indices]
        
        logger.info(
            f"Top candidate energy: {energies[top_indices[0]]:.4f}, "
            f"Median energy: {np.median(energies[top_indices]):.4f}"
        )
        
        if return_scores:
            return ranked_candidates, energies[top_indices]
        else:
            return ranked_candidates
    
    def select_diverse_candidates(
        self,
        candidates: list,
        diversity_weight: float = 0.3
    ) -> list:
        """
        Select top-k candidates balancing energy and diversity.
        
        Args:
            candidates: List of PacketGenome candidates
            diversity_weight: Weight for diversity term (0 to 1)
            
        Returns:
            List of selected candidates
        """
        from perturbation.genome_encoding import GenomeEncoder
        
        # Compute energies
        energies = self.elasticity_energy.compute_batch_energies(candidates)
        
        # Extract feature vectors for diversity
        features = np.stack([
            GenomeEncoder.to_feature_vector(g) for g in candidates
        ])
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
        
        # Greedy selection with diversity
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Start with highest energy candidate
        first_idx = np.argmax(energies)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select candidates
        while len(selected_indices) < self.top_k and remaining_indices:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # Energy component (normalized)
                energy_score = energies[idx] / (energies.max() + 1e-9)
                
                # Diversity component: distance to selected set
                if selected_indices:
                    selected_features = features[selected_indices]
                    distances = np.linalg.norm(
                        selected_features - features[idx],
                        axis=1
                    )
                    diversity_score = distances.min()
                else:
                    diversity_score = 0.0
                
                # Combined score
                score = (1 - diversity_weight) * energy_score + diversity_weight * diversity_score
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return [candidates[i] for i in selected_indices]
