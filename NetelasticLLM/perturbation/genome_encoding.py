"""
Genome Encoding for Packet Sequence Perturbations

This module implements the variable-length genome representation
for packet-level perturbations as described in the paper.

Genome representation: g = (d, l, δ)
where:
    d ∈ {+1, -1}^N: direction sequence
    l ∈ [L_min, L_max]^N: length sequence
    δ ∈ [Δ_min, Δ_max]^N: timing sequence
    N: variable sequence length
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenomeConfig:
    """Configuration for genome encoding."""
    # Direction values
    direction_values: List[int] = (-1, 1)
    
    # Length bounds (in bytes)
    min_length: int = 64
    max_length: int = 1500
    
    # Timing bounds (in milliseconds)
    min_timing: float = 0.0
    max_timing: float = 1000.0
    
    # Sequence length bounds
    min_sequence_length: int = 2
    max_sequence_length: int = 256
    
    # Budget constraints
    max_total_bytes: int = 1048576  # 1 MB
    max_total_duration: float = 60000.0  # 60 seconds in ms


class PacketGenome:
    """
    Represents a packet sequence perturbation as a genome.
    
    This encoding directly maps to observable encrypted traffic
    while remaining protocol-compliant and feasible.
    """
    
    def __init__(
        self,
        direction: np.ndarray,
        length: np.ndarray,
        timing: np.ndarray,
        config: Optional[GenomeConfig] = None
    ):
        """
        Initialize genome from sequences.
        
        Args:
            direction: Direction sequence [N], values in {-1, 1}
            length: Length sequence [N], values in bytes
            timing: Inter-arrival time sequence [N], values in ms
            config: Genome configuration
        """
        assert len(direction) == len(length) == len(timing), \
            "All sequences must have same length"
        
        self.direction = np.array(direction, dtype=np.int32)
        self.length = np.array(length, dtype=np.float32)
        self.timing = np.array(timing, dtype=np.float32)
        self.config = config or GenomeConfig()
        
        # Validate genome
        self._validate()
    
    def _validate(self) -> bool:
        """
        Validate genome against constraints.
        
        Returns:
            True if valid, raises exception otherwise
        """
        N = len(self.direction)
        
        # Check sequence length bounds
        if not (self.config.min_sequence_length <= N <= self.config.max_sequence_length):
            raise ValueError(
                f"Sequence length {N} outside bounds "
                f"[{self.config.min_sequence_length}, {self.config.max_sequence_length}]"
            )
        
        # Check direction values
        if not np.all(np.isin(self.direction, self.config.direction_values)):
            raise ValueError(f"Direction values must be in {self.config.direction_values}")
        
        # Check length bounds
        if not np.all((self.length >= self.config.min_length) & (self.length <= self.config.max_length)):
            raise ValueError(
                f"Length values must be in [{self.config.min_length}, {self.config.max_length}]"
            )
        
        # Check timing bounds and non-negativity
        if not np.all((self.timing >= self.config.min_timing) & (self.timing <= self.config.max_timing)):
            raise ValueError(
                f"Timing values must be in [{self.config.min_timing}, {self.config.max_timing}]"
            )
        
        # Check temporal causality (non-negative gaps)
        if not np.all(self.timing >= 0):
            raise ValueError("Inter-arrival times must be non-negative")
        
        # Check traffic budgets
        total_bytes = np.sum(self.length)
        if total_bytes > self.config.max_total_bytes:
            raise ValueError(
                f"Total bytes {total_bytes} exceeds budget {self.config.max_total_bytes}"
            )
        
        total_duration = np.sum(self.timing)
        if total_duration > self.config.max_total_duration:
            raise ValueError(
                f"Total duration {total_duration} ms exceeds budget {self.config.max_total_duration} ms"
            )
        
        return True
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary representation."""
        return {
            'direction': self.direction.copy(),
            'length': self.length.copy(),
            'timing': self.timing.copy()
        }
    
    def to_torch(self) -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors."""
        return {
            'direction': torch.from_numpy(self.direction).long(),
            'length': torch.from_numpy(self.length).float(),
            'timing': torch.from_numpy(self.timing).float()
        }
    
    def clone(self) -> 'PacketGenome':
        """Create a deep copy of the genome."""
        return PacketGenome(
            direction=self.direction.copy(),
            length=self.length.copy(),
            timing=self.timing.copy(),
            config=self.config
        )
    
    def __len__(self) -> int:
        """Return sequence length."""
        return len(self.direction)
    
    def __repr__(self) -> str:
        return (
            f"PacketGenome(N={len(self)}, "
            f"total_bytes={np.sum(self.length):.0f}, "
            f"total_duration={np.sum(self.timing):.2f}ms)"
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray], config: Optional[GenomeConfig] = None) -> 'PacketGenome':
        """Create genome from dictionary."""
        return cls(
            direction=data['direction'],
            length=data['length'],
            timing=data['timing'],
            config=config
        )
    
    @classmethod
    def random(cls, sequence_length: int, config: Optional[GenomeConfig] = None) -> 'PacketGenome':
        """
        Generate a random valid genome.
        
        Args:
            sequence_length: Desired sequence length
            config: Genome configuration
            
        Returns:
            Random PacketGenome
        """
        config = config or GenomeConfig()
        
        # Random direction sequence
        direction = np.random.choice(config.direction_values, size=sequence_length)
        
        # Random length sequence
        length = np.random.uniform(
            config.min_length,
            config.max_length,
            size=sequence_length
        )
        
        # Random timing sequence (ensuring budget constraint)
        max_avg_timing = config.max_total_duration / sequence_length
        timing = np.random.uniform(
            config.min_timing,
            min(config.max_timing, max_avg_timing * 2),
            size=sequence_length
        )
        timing[0] = 0.0  # First packet has δ₁ = 0 by convention
        
        return cls(direction, length, timing, config)


class GenomePopulation:
    """
    Manages a population of genomes for evolutionary algorithms.
    """
    
    def __init__(self, genomes: Optional[List[PacketGenome]] = None):
        """
        Initialize population.
        
        Args:
            genomes: Initial list of genomes
        """
        self.genomes = genomes or []
        self.fitness_scores = []
    
    def add(self, genome: PacketGenome, fitness: Optional[float] = None):
        """Add a genome to the population."""
        self.genomes.append(genome)
        if fitness is not None:
            self.fitness_scores.append(fitness)
    
    def sort_by_fitness(self, descending: bool = True):
        """Sort population by fitness scores."""
        if len(self.fitness_scores) != len(self.genomes):
            raise ValueError("Not all genomes have fitness scores")
        
        sorted_indices = np.argsort(self.fitness_scores)
        if descending:
            sorted_indices = sorted_indices[::-1]
        
        self.genomes = [self.genomes[i] for i in sorted_indices]
        self.fitness_scores = [self.fitness_scores[i] for i in sorted_indices]
    
    def get_top_k(self, k: int) -> 'GenomePopulation':
        """Get top-k genomes by fitness."""
        self.sort_by_fitness(descending=True)
        return GenomePopulation(
            genomes=self.genomes[:k]
        )
    
    def __len__(self) -> int:
        """Return population size."""
        return len(self.genomes)
    
    def __getitem__(self, idx: int) -> PacketGenome:
        """Get genome at index."""
        return self.genomes[idx]
    
    def __iter__(self):
        """Iterate over genomes."""
        return iter(self.genomes)


class GenomeEncoder:
    """
    Utility class for encoding/decoding genomes to/from various formats.
    """
    
    @staticmethod
    def to_flat_vector(genome: PacketGenome) -> np.ndarray:
        """
        Encode genome as flat vector for similarity computation.
        
        Returns:
            Flat vector representation
        """
        # Concatenate all sequences
        return np.concatenate([
            genome.direction.astype(np.float32),
            genome.length,
            genome.timing
        ])
    
    @staticmethod
    def to_feature_vector(genome: PacketGenome) -> np.ndarray:
        """
        Extract statistical features from genome.
        
        Returns:
            Feature vector with summary statistics
        """
        features = []
        
        # Sequence length
        features.append(len(genome))
        
        # Direction statistics
        direction_changes = np.sum(np.abs(np.diff(genome.direction)))
        features.append(direction_changes)
        features.append(np.mean(genome.direction == 1))  # ratio of client->server
        
        # Length statistics
        features.extend([
            np.mean(genome.length),
            np.std(genome.length),
            np.min(genome.length),
            np.max(genome.length)
        ])
        
        # Timing statistics
        features.extend([
            np.mean(genome.timing),
            np.std(genome.timing),
            np.min(genome.timing),
            np.max(genome.timing)
        ])
        
        # Budget usage
        features.append(np.sum(genome.length))  # total bytes
        features.append(np.sum(genome.timing))  # total duration
        
        return np.array(features, dtype=np.float32)
