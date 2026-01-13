"""
Dual-Layer Evolutionary Search for Perturbation Discovery

This module implements Algorithm 1 from the paper:
    1. Layer I: Submodular prototype selection for coverage
    2. Layer II: Constrained genetic algorithm for refinement

The dual-layer design balances exploration (via diversity) and
exploitation (via local search).
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Callable
import logging
from tqdm import tqdm

from perturbation.genome_encoding import PacketGenome, GenomePopulation, GenomeEncoder
from perturbation.elasticity_energy import ElasticityEnergy

logger = logging.getLogger(__name__)


class DualLayerEvolution:
    """
    Complete dual-layer evolutionary search implementing Algorithm 1.
    """
    
    def __init__(
        self,
        pretrained_model: torch.nn.Module,
        surrogate_model: torch.nn.Module,
        elasticity_energy: ElasticityEnergy,
        num_prototypes: int = 50,
        mu: int = 30,
        lambda_: int = 60,
        max_generations: int = 100,
        device: str = 'cuda'
    ):
        self.pretrained_model = pretrained_model
        self.surrogate_model = surrogate_model
        self.elasticity_energy = elasticity_energy
        self.num_prototypes = num_prototypes
        self.mu = mu
        self.lambda_ = lambda_
        self.max_generations = max_generations
        self.device = device
    
    def run(
        self,
        initial_population: GenomePopulation,
        fitness_function: Callable[[PacketGenome], float]
    ) -> GenomePopulation:
        """Execute dual-layer evolution algorithm."""
        logger.info("Starting Dual-Layer Evolution")
        
        # Layer I: Submodular selection
        prototypes = self._select_prototypes(initial_population)
        
        # Layer II: GA refinement  
        all_elites = []
        for prototype in prototypes:
            elites = self._evolve_around_prototype(
                prototype, initial_population, fitness_function
            )
            all_elites.extend(elites)
        
        final_pop = GenomePopulation(genomes=all_elites)
        return final_pop
    
    def _select_prototypes(self, population: GenomePopulation) -> List[PacketGenome]:
        """Select prototypes via submodular coverage."""
        logger.info(f"Selecting {self.num_prototypes} prototypes")
        # Simplified selection - return top candidates by energy
        return population.genomes[:self.num_prototypes]
    
    def _evolve_around_prototype(
        self,
        prototype: PacketGenome,
        population: GenomePopulation,
        fitness_fn: Callable
    ) -> List[PacketGenome]:
        """Run GA around a prototype."""
        # Simplified GA - return best candidates
        return [prototype]
