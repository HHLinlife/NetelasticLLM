"""
search/genetic_algorithm.py
Layer II: Constrained (μ+λ) genetic algorithm with successive halving.

Implements Algorithm 1 (inner loop) from the paper.
Operates within a single prototype neighbourhood, refining candidates
using the composite fitness function Φ(u) (Eq. 13).

Φ(u) = w₁·E(φ(u)) + w₂·R_sur(u) - λ_b·p_b(u) - λ_p·p_p(u)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..perturbation.genome_encoding import Genome, GenomeFactory, GenomeConstraints
from ..perturbation.mutation_operators import MutationEngine
from ..perturbation.elasticity_energy import ElasticEnergyScorer
from ..perturbation.constraints import ConstraintChecker

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Fitness evaluation                                                           #
# --------------------------------------------------------------------------- #

class FitnessEvaluator:
    """
    Computes the composite fitness Φ(u) for a list of genomes.

    Args
    ----
    energy_scorer : ElasticEnergyScorer — provides E(φ(u)).
    surrogate     : nn.Module — M̂₁ for R_sur(u) (predictive uncertainty).
    feature_fn    : Callable mapping Genome → np.ndarray (feature vector).
    constraints   : GenomeConstraints for penalty computation.
    w1, w2        : Weights on elastic energy and surrogate uncertainty.
    lambda_b, lambda_p : Penalty weights.
    device        : Torch device string.
    """

    def __init__(
        self,
        energy_scorer,
        surrogate,
        feature_fn,
        constraints: GenomeConstraints = GenomeConstraints(),
        w1: float = 0.5,
        w2: float = 0.4,
        lambda_b: float = 0.05,
        lambda_p: float = 0.05,
        device: str = "cpu",
    ):
        self.scorer      = energy_scorer
        self.surrogate   = surrogate
        self.feature_fn  = feature_fn
        self.checker     = ConstraintChecker(constraints)
        self.w1          = w1
        self.w2          = w2
        self.lambda_b    = lambda_b
        self.lambda_p    = lambda_p
        self.device      = torch.device(device)

    def evaluate(self, genomes: List[Genome]) -> List[float]:
        """
        Compute Φ(u) for each genome in the list.
        Returns list of scalar fitness values (higher is better).
        """
        if not genomes:
            return []

        # Extract feature vectors
        features = np.stack([self.feature_fn(g) for g in genomes], axis=0)
        tensor   = torch.from_numpy(features).float().to(self.device)

        # E(φ(u)) — elastic energy
        with torch.enable_grad():
            energy = self.scorer.elastic_energy(tensor).cpu().numpy()  # (B,)

        # R_sur(u) — surrogate uncertainty (predictive entropy)
        with torch.no_grad():
            if hasattr(self.surrogate, "uncertainty_scores"):
                r_sur = self.surrogate.uncertainty_scores(tensor).cpu().numpy()
            else:
                logits = self.surrogate(tensor)
                probs  = torch.softmax(logits, dim=-1)
                log_p  = probs.clamp(1e-9).log()
                r_sur  = -(probs * log_p).sum(dim=-1).cpu().numpy()

        # Penalties
        p_b = np.array([self.checker.budget_penalty(g)   for g in genomes])
        p_p = np.array([self.checker.protocol_penalty(g) for g in genomes])

        fitness = (
            self.w1 * energy
            + self.w2 * r_sur
            - self.lambda_b * p_b
            - self.lambda_p * p_p
        )
        return fitness.tolist()


# --------------------------------------------------------------------------- #
#  (μ+λ) GA with successive halving                                            #
# --------------------------------------------------------------------------- #

class ConstrainedGA:
    """
    (μ+λ) elitist evolutionary strategy with successive halving,
    operating on a single prototype neighbourhood.

    Parameters
    ----------
    mu          : Number of elite parents retained per generation.
    lambda_     : Number of offspring generated per generation.
    max_rounds  : Maximum number of generations.
    stagnation  : Stop after this many rounds without improvement.
    halving_min : Minimum neighbourhood size before halving stops.
    """

    def __init__(
        self,
        evaluator:   FitnessEvaluator,
        mutation_engine: MutationEngine,
        mu:          int   = 20,
        lambda_:     int   = 80,
        max_rounds:  int   = 50,
        stagnation:  int   = 5,
        halving_min: int   = 5,
        seed:        Optional[int] = None,
    ):
        self.evaluator   = evaluator
        self.engine      = mutation_engine
        self.mu          = mu
        self.lambda_     = lambda_
        self.max_rounds  = max_rounds
        self.stagnation  = stagnation
        self.halving_min = halving_min
        self.rng         = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #

    def run(
        self,
        initial_population: List[Genome],
    ) -> List[Genome]:
        """
        Evolve a neighbourhood population and return the elite set.

        Args:
            initial_population : Seed genomes for this neighbourhood.
        Returns:
            elites : Top-μ genomes after convergence.
        """
        population = [g.clone() for g in initial_population]
        if not population:
            return []

        # Initial fitness evaluation
        fitnesses  = self.evaluator.evaluate(population)
        for g, f in zip(population, fitnesses):
            g.fitness = f

        best_fitness    = max(fitnesses)
        stagnation_cnt  = 0
        current_pop     = self._select_top(population, self.mu)

        for round_idx in range(self.max_rounds):
            # Generate λ offspring
            offspring = self._generate_offspring(current_pop, self.lambda_)

            # Evaluate offspring
            offspring_fit = self.evaluator.evaluate(offspring)
            for g, f in zip(offspring, offspring_fit):
                g.fitness = f

            # (μ+λ) selection: keep best μ from union
            combined   = current_pop + offspring
            current_pop = self._select_top(combined, self.mu)

            # Successive halving: prune bottom half if pop > halving_min
            if len(current_pop) > self.halving_min * 2:
                keep = max(self.halving_min, len(current_pop) // 2)
                current_pop = current_pop[:keep]

            new_best = current_pop[0].fitness
            if new_best > best_fitness + 1e-6:
                best_fitness   = new_best
                stagnation_cnt = 0
            else:
                stagnation_cnt += 1

            logger.debug(
                "Round %d/%d — best Φ=%.4f, pop_size=%d, stagnation=%d",
                round_idx + 1, self.max_rounds,
                best_fitness, len(current_pop), stagnation_cnt,
            )

            if stagnation_cnt >= self.stagnation:
                logger.debug("Early stop at round %d.", round_idx + 1)
                break

        return current_pop

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _select_top(self, population: List[Genome], k: int) -> List[Genome]:
        """Sort by fitness (descending) and return top-k."""
        ranked = sorted(population, key=lambda g: g.fitness, reverse=True)
        return ranked[:k]

    def _generate_offspring(
        self,
        parents: List[Genome],
        n_offspring: int,
    ) -> List[Genome]:
        """
        Generate offspring via mutation and crossover.
        Each offspring is either:
          - mutation of a randomly selected parent (70 %)
          - crossover of two parents followed by mutation (30 %)
        """
        offspring = []
        n_parents = len(parents)
        if n_parents == 0:
            return []

        for _ in range(n_offspring):
            roll = self.rng.random()
            if roll < 0.7 or n_parents < 2:
                # Mutation
                parent = parents[int(self.rng.integers(n_parents))]
                child  = self.engine.mutate(parent)
            else:
                # Crossover + mutation
                i, j = self.rng.choice(n_parents, size=2, replace=False)
                child = self.engine.crossover(parents[i], parents[j])
                child = self.engine.mutate(child)
            offspring.append(child)

        return offspring
