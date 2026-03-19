"""
search/dual_layer_evolution.py
Complete Algorithm 1: Elasticity-Guided Dual-Layer Evolution.

Pipeline:
  1. Elasticity-guided population initialisation (Section 4.3)
  2. Layer I  — Submodular prototype selection (Section 4.4.2)
  3. Layer II — Constrained GA per neighbourhood (Section 4.4.3)

Returns U* — the final rebound-inducing perturbation set.
"""

from __future__ import annotations

import logging
from typing import List, Callable, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from ..perturbation.genome_encoding import Genome, GenomeFactory, GenomeConstraints
from ..perturbation.mutation_operators import MutationEngine
from ..perturbation.elasticity_energy import ElasticEnergyScorer
from ..perturbation.constraints import ConstraintChecker
from .submodular_selection import (
    greedy_facility_location,
    zscore_normalize,
    assign_to_prototypes,
    compute_sigma,
)
from .genetic_algorithm import ConstrainedGA, FitnessEvaluator

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Heuristic population initialisation                                         #
# --------------------------------------------------------------------------- #

class HeuristicInitialiser:
    """
    Constructs the initial candidate pool C as described in Section 4.3:
        C = S_core ∪ S_anchor ∪ S_energy
    with ratio approximately 4:3:3.
    """

    def __init__(
        self,
        feature_fn:    Callable[[Genome], np.ndarray],
        energy_scorer: ElasticEnergyScorer,
        factory:       GenomeFactory,
        kde_top_k:     int = 40,
        anchor_bins:   int = 10,
        energy_top_k:  int = 30,
        seed:          Optional[int] = None,
    ):
        self.feature_fn    = feature_fn
        self.scorer        = energy_scorer
        self.factory       = factory
        self.kde_top_k     = kde_top_k
        self.anchor_bins   = anchor_bins
        self.energy_top_k  = energy_top_k
        self.rng           = np.random.default_rng(seed)

    def build(
        self,
        candidate_pool: List[Genome],
        population_size: int = 100,
    ) -> List[Genome]:
        """
        Build the initial population from a candidate pool.

        Args:
            candidate_pool   : Pre-sampled feasible genomes.
            population_size  : Total desired population size.
        """
        if not candidate_pool:
            logger.warning("Empty candidate pool — sampling random genomes.")
            return [self.factory.random(self.rng) for _ in range(population_size)]

        features = np.stack([self.feature_fn(g) for g in candidate_pool])  # (N, D)

        # ── S_core: high-frequency / KDE modes ───────────────────────── #
        s_core   = self._high_frequency_seed(candidate_pool, features)

        # ── S_anchor: structural diversity ───────────────────────────── #
        s_anchor = self._structural_anchors(candidate_pool, features)

        # ── S_energy: high elastic energy candidates ──────────────────── #
        s_energy = self._energy_guided(candidate_pool, features)

        population = s_core + s_anchor + s_energy

        # Pad with random genomes if under target size
        while len(population) < population_size:
            population.append(self.factory.random(self.rng))

        # Shuffle and trim
        self.rng.shuffle(population)  # type: ignore[arg-type]
        return population[:population_size]

    # ------------------------------------------------------------------ #
    #  Sub-procedures                                                       #
    # ------------------------------------------------------------------ #

    def _high_frequency_seed(
        self,
        pool: List[Genome],
        features: np.ndarray,
    ) -> List[Genome]:
        """KDE density estimation; select top-K densest candidates."""
        from scipy.stats import gaussian_kde  # type: ignore

        k = min(self.kde_top_k, len(pool))
        try:
            kde    = gaussian_kde(features.T, bw_method=0.5)
            scores = kde(features.T)
            idx    = np.argsort(scores)[::-1][:k]
        except Exception:
            idx = np.arange(min(k, len(pool)))

        return [pool[i].clone() for i in idx]

    def _structural_anchors(
        self,
        pool: List[Genome],
        features: np.ndarray,
    ) -> List[Genome]:
        """
        Stratified sampling across bins of the 4-D descriptor τ(u).
        Covers diverse (N, d̄, l̄, δ̄) regimes.
        """
        tau = np.stack([g.to_feature_vector() for g in pool])   # (N, 4)
        selected = []

        for dim in range(tau.shape[1]):
            col   = tau[:, dim]
            edges = np.linspace(col.min(), col.max() + 1e-8, self.anchor_bins + 1)
            bin_ids = np.digitize(col, edges[:-1]) - 1

            for b in range(self.anchor_bins):
                in_bin = np.where(bin_ids == b)[0]
                if len(in_bin) == 0:
                    continue
                chosen = self.rng.choice(in_bin)
                selected.append(pool[int(chosen)].clone())

        return selected

    def _energy_guided(
        self,
        pool: List[Genome],
        features: np.ndarray,
    ) -> List[Genome]:
        """Select top-K candidates by elastic energy E(φ(u))."""
        k = min(self.energy_top_k, len(pool))
        try:
            scores = self.scorer.score_batch(features, use_full_proxy=False)
            idx    = np.argsort(scores)[::-1][:k]
        except Exception as exc:
            logger.warning("Energy scoring failed (%s) — random selection.", exc)
            idx = np.arange(k)
        return [pool[i].clone() for i in idx]


# --------------------------------------------------------------------------- #
#  Main Algorithm 1                                                             #
# --------------------------------------------------------------------------- #

class DualLayerEvolution:
    """
    Implements the complete Algorithm 1 from Section 4.4.

    Usage
    -----
    >>> solver = DualLayerEvolution(m0, m_sur, feature_fn, constraints)
    >>> U_star  = solver.run(candidate_pool)
    """

    def __init__(
        self,
        pretrained_model:  nn.Module,
        surrogate_model:   nn.Module,
        feature_fn:        Callable[[Genome], np.ndarray],
        constraints:       GenomeConstraints = GenomeConstraints(),
        num_prototypes:    int   = 10,
        population_size:   int   = 100,
        mu:                int   = 20,
        lambda_:           int   = 80,
        max_rounds:        int   = 50,
        stagnation:        int   = 5,
        w1:                float = 0.5,
        w2:                float = 0.4,
        lambda_b:          float = 0.05,
        lambda_p:          float = 0.05,
        device:            str   = "cpu",
        seed:              Optional[int] = None,
    ):
        self.device      = torch.device(device)
        self.constraints = constraints
        self.feature_fn  = feature_fn
        self.m           = num_prototypes

        # Sub-components
        self.factory = GenomeFactory(constraints)
        self.engine  = MutationEngine(constraints, seed=seed)
        self.scorer  = ElasticEnergyScorer(
            pretrained_model, surrogate_model, device=device
        )
        self.evaluator = FitnessEvaluator(
            self.scorer, surrogate_model, feature_fn,
            constraints=constraints,
            w1=w1, w2=w2, lambda_b=lambda_b, lambda_p=lambda_p,
            device=device,
        )
        self.initialiser = HeuristicInitialiser(
            feature_fn, self.scorer, self.factory, seed=seed
        )
        self.ga_params = dict(
            mu=mu, lambda_=lambda_,
            max_rounds=max_rounds, stagnation=stagnation,
        )
        self.population_size = population_size
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #

    def run(
        self,
        candidate_pool: List[Genome],
        calibration_features: Optional[np.ndarray] = None,
    ) -> List[Genome]:
        """
        Execute Algorithm 1 and return U*.

        Args:
            candidate_pool         : Feasible candidate genomes from G(x).
            calibration_features   : (B, D) array for estimating d̂.
                                     Uses candidate pool features if None.
        Returns:
            U_star : Elite perturbation set.
        """
        logger.info(
            "DualLayerEvolution starting — pool=%d, m=%d",
            len(candidate_pool), self.m,
        )

        # ── 0. Calibrate rebound direction d̂ ─────────────────────────── #
        if calibration_features is None and candidate_pool:
            calibration_features = np.stack(
                [self.feature_fn(g) for g in candidate_pool[:200]]
            )
        if calibration_features is not None:
            cal_tensor = torch.from_numpy(
                calibration_features.astype(np.float32)
            ).to(self.device)
            with torch.enable_grad():
                self.scorer.compute_rebound_direction(cal_tensor)
            logger.debug("Rebound direction d̂ computed.")

        # ── 1. Elasticity-guided initialisation ──────────────────────── #
        population = self.initialiser.build(candidate_pool, self.population_size)
        logger.info("Initialised population: %d genomes.", len(population))

        # ── 2. Layer I: submodular prototype selection ────────────────── #
        phi_raw = np.stack([self.feature_fn(g) for g in population])   # (N, D)
        phi_z   = zscore_normalize(phi_raw)
        sigma   = compute_sigma(phi_z)

        proto_idx  = greedy_facility_location(phi_z, self.m, sigma=sigma)
        prototypes = [population[i] for i in proto_idx]
        logger.info("Selected %d prototypes.", len(prototypes))

        # ── 3. Layer II: constrained GA per neighbourhood ─────────────── #
        assignments = assign_to_prototypes(phi_z, phi_z[proto_idx], sigma)  # (N,)
        c_final: List[Genome] = []

        for proto_id, proto in enumerate(prototypes):
            nbhd_idx = np.where(assignments == proto_id)[0]
            if len(nbhd_idx) == 0:
                c_final.append(proto)
                continue

            neighbourhood = [population[i] for i in nbhd_idx]

            ga = ConstrainedGA(
                evaluator=self.evaluator,
                mutation_engine=self.engine,
                seed=int(self.rng.integers(1 << 31)),
                **self.ga_params,
            )
            elites = ga.run(neighbourhood)
            c_final.extend(elites)
            logger.debug(
                "Neighbourhood %d/%d — %d elites (best Φ=%.4f)",
                proto_id + 1, len(prototypes),
                len(elites),
                elites[0].fitness if elites else float("nan"),
            )

        # Sort final set by fitness
        c_final.sort(key=lambda g: g.fitness, reverse=True)
        logger.info(
            "DualLayerEvolution complete — U* size=%d, best Φ=%.4f",
            len(c_final),
            c_final[0].fitness if c_final else float("nan"),
        )
        return c_final
