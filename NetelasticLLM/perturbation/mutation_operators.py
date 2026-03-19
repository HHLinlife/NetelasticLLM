"""
Mutation Operators for Packet Genome Evolution

This module implements the genetic mutation operators described in the paper
for modifying packet sequences while maintaining protocol compliance.
"""

import numpy as np
from typing import Optional
import logging

from perturbation.genome_encoding import PacketGenome, GenomeConfig

logger = logging.getLogger(__name__)


class GenomeMutator:
    """
    Implements mutation operators for packet genome evolution.
    
    Operators:
        1. Direction flip - Change packet directions
        2. Count jitter - Insert/delete packets
        3. Length scaling - Modify packet sizes
        4. Interval modulation - Adjust timing gaps
        5. Burst swap - Reorder packet segments
    """
    
    def __init__(
        self,
        config: Optional[GenomeConfig] = None,
        direction_flip_prob: float = 0.2,
        count_jitter_prob: float = 0.15,
        length_scaling_prob: float = 0.25,
        interval_modulation_prob: float = 0.25,
        burst_swap_prob: float = 0.15
    ):
        """
        Initialize mutator.
        
        Args:
            config: Genome configuration
            direction_flip_prob: Probability of direction flip mutation
            count_jitter_prob: Probability of count jitter mutation
            length_scaling_prob: Probability of length scaling mutation
            interval_modulation_prob: Probability of interval modulation
            burst_swap_prob: Probability of burst swap mutation
        """
        self.config = config or GenomeConfig()
        self.direction_flip_prob = direction_flip_prob
        self.count_jitter_prob = count_jitter_prob
        self.length_scaling_prob = length_scaling_prob
        self.interval_modulation_prob = interval_modulation_prob
        self.burst_swap_prob = burst_swap_prob
    
    def mutate(self, genome: PacketGenome) -> PacketGenome:
        """
        Apply random mutations to genome.
        
        Args:
            genome: Input genome
            
        Returns:
            Mutated genome
        """
        mutated = genome.clone()
        
        # Apply each mutation with its probability
        if np.random.random() < self.direction_flip_prob:
            mutated = self.direction_flip(mutated)
        
        if np.random.random() < self.count_jitter_prob:
            mutated = self.count_jitter(mutated)
        
        if np.random.random() < self.length_scaling_prob:
            mutated = self.length_scaling(mutated)
        
        if np.random.random() < self.interval_modulation_prob:
            mutated = self.interval_modulation(mutated)
        
        if np.random.random() < self.burst_swap_prob:
            mutated = self.burst_swap(mutated)
        
        return mutated
    
    def direction_flip(
        self,
        genome: PacketGenome,
        segment_length_range: tuple = (1, 5)
    ) -> PacketGenome:
        """Flip direction of a random segment."""
        N = len(genome)
        if N < 2:
            return genome
        
        # Select random segment
        seg_len = np.random.randint(segment_length_range[0], 
                                    min(segment_length_range[1] + 1, N))
        start_idx = np.random.randint(0, N - seg_len + 1)
        
        # Flip directions in segment
        mutated = genome.clone()
        mutated.direction[start_idx:start_idx + seg_len] *= -1
        
        return mutated
    
    def count_jitter(
        self,
        genome: PacketGenome,
        insert_prob: float = 0.5
    ) -> PacketGenome:
        """Insert or delete a packet."""
        N = len(genome)
        
        if np.random.random() < insert_prob and N < self.config.max_sequence_length:
            # Insert packet
            insert_idx = np.random.randint(0, N + 1)
            
            # Create new packet values
            new_direction = np.random.choice(self.config.direction_values)
            new_length = np.random.uniform(self.config.min_length, self.config.max_length)
            new_timing = np.random.uniform(self.config.min_timing, 
                                          min(self.config.max_timing, 100.0))
            
            # Insert
            direction = np.insert(genome.direction, insert_idx, new_direction)
            length = np.insert(genome.length, insert_idx, new_length)
            timing = np.insert(genome.timing, insert_idx, new_timing)
            
            return PacketGenome(direction, length, timing, self.config)
        
        elif N > self.config.min_sequence_length:
            # Delete packet
            delete_idx = np.random.randint(0, N)
            
            direction = np.delete(genome.direction, delete_idx)
            length = np.delete(genome.length, delete_idx)
            timing = np.delete(genome.timing, delete_idx)
            
            return PacketGenome(direction, length, timing, self.config)
        
        return genome
    
    def length_scaling(
        self,
        genome: PacketGenome,
        scale_range: tuple = (0.8, 1.2),
        segment_length_range: tuple = (1, 10)
    ) -> PacketGenome:
        """Scale packet lengths in a segment."""
        N = len(genome)
        if N < 1:
            return genome
        
        # Select segment
        seg_len = min(np.random.randint(segment_length_range[0],
                                       segment_length_range[1] + 1), N)
        start_idx = np.random.randint(0, N - seg_len + 1)
        
        # Apply scaling
        mutated = genome.clone()
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        mutated.length[start_idx:start_idx + seg_len] *= scale_factor
        
        # Clip to valid range
        mutated.length = np.clip(mutated.length, 
                                self.config.min_length,
                                self.config.max_length)
        
        return mutated
    
    def interval_modulation(
        self,
        genome: PacketGenome,
        dilation_range: tuple = (0.5, 2.0),
        segment_length_range: tuple = (1, 10)
    ) -> PacketGenome:
        """Modulate inter-arrival intervals in a segment."""
        N = len(genome)
        if N < 2:
            return genome
        
        # Select segment
        seg_len = min(np.random.randint(segment_length_range[0],
                                       segment_length_range[1] + 1), N)
        start_idx = np.random.randint(0, N - seg_len + 1)
        
        # Apply dilation
        mutated = genome.clone()
        dilation_factor = np.random.uniform(dilation_range[0], dilation_range[1])
        mutated.timing[start_idx:start_idx + seg_len] *= dilation_factor
        
        # Clip to valid range
        mutated.timing = np.clip(mutated.timing,
                                self.config.min_timing,
                                self.config.max_timing)
        
        return mutated
    
    def burst_swap(
        self,
        genome: PacketGenome,
        segment_length_range: tuple = (3, 8)
    ) -> PacketGenome:
        """Swap two packet segments."""
        N = len(genome)
        if N < 6:  # Need at least 2 segments of min size 3
            return genome
        
        # Select two non-overlapping segments
        seg_len = min(np.random.randint(segment_length_range[0],
                                       segment_length_range[1] + 1), N // 2)
        
        start1 = np.random.randint(0, N - 2 * seg_len)
        start2 = start1 + seg_len + np.random.randint(1, N - start1 - 2 * seg_len + 1)
        
        # Swap segments
        mutated = genome.clone()
        
        # Swap directions
        temp_dir = mutated.direction[start1:start1 + seg_len].copy()
        mutated.direction[start1:start1 + seg_len] = mutated.direction[start2:start2 + seg_len]
        mutated.direction[start2:start2 + seg_len] = temp_dir
        
        # Swap lengths
        temp_len = mutated.length[start1:start1 + seg_len].copy()
        mutated.length[start1:start1 + seg_len] = mutated.length[start2:start2 + seg_len]
        mutated.length[start2:start2 + seg_len] = temp_len
        
        # Swap timings
        temp_time = mutated.timing[start1:start1 + seg_len].copy()
        mutated.timing[start1:start1 + seg_len] = mutated.timing[start2:start2 + seg_len]
        mutated.timing[start2:start2 + seg_len] = temp_time
        
        return mutated
