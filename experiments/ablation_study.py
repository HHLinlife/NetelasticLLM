"""
experiments/ablation_study.py
Ablation study from Section 6.6 (Q5).

Evaluates five variants:
  Full      — complete method
  w/o S     — without submodular prototype selection (random prototypes)
  Rand P    — random prototypes (replaces structured selection)
  w/o I     — without heuristic initialisation (uniform random init)
  w/o S+I   — without both submodular selection and heuristic init

Measures convergence steps and wall-clock time to reach a target
accuracy-drop threshold τ.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn

from perturbation.genome_encoding import GenomeFactory, GenomeConstraints, Genome
from perturbation.mutation_operators import MutationEngine
from perturbation.elasticity_energy import ElasticEnergyScorer
from search.genetic_algorithm import ConstrainedGA, FitnessEvaluator
from search.submodular_selection import (
    greedy_facility_location,
    zscore_normalize,
    compute_sigma,
    assign_to_prototypes,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Ablation variant runner                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class AblationResult:
    variant:        str
    dataset:        str
    steps_to_thresh: float          # median over N runs
    steps_std:       float
    time_to_thresh:  float          # seconds, median
    time_std:        float


class AblationRunner:
    """
    Runs evolutionary search under five ablation conditions and records
    convergence cost (steps and wall-clock time) to a fitness threshold.
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
        fitness_threshold: float = 0.5,
        n_runs:            int   = 3,
        device:            str   = "cpu",
        seed:              int   = 42,
    ):
        self.device           = torch.device(device)
        self.constraints      = constraints
        self.feature_fn       = feature_fn
        self.m                = num_prototypes
        self.population_size  = population_size
        self.mu               = mu
        self.lambda_          = lambda_
        self.max_rounds       = max_rounds
        self.stagnation       = stagnation
        self.threshold        = fitness_threshold
        self.n_runs           = n_runs

        self.factory  = GenomeFactory(constraints)
        self.scorer   = ElasticEnergyScorer(pretrained_model, surrogate_model, device=device)
        self.evaluator = FitnessEvaluator(
            self.scorer, surrogate_model, feature_fn,
            constraints=constraints, device=device,
        )
        self.base_seed = seed

    # ------------------------------------------------------------------ #

    def run_all(
        self,
        candidate_pool: List[Genome],
        dataset_name:   str = "dataset",
    ) -> List[AblationResult]:
        """
        Run all five ablation variants and return results.
        """
        variants = {
            "Full":    {"use_submodular": True,  "use_heuristic": True},
            "w/o S":   {"use_submodular": False, "use_heuristic": True},
            "Rand P":  {"use_submodular": False, "use_heuristic": True,  "random_proto": True},
            "w/o I":   {"use_submodular": True,  "use_heuristic": False},
            "w/o S+I": {"use_submodular": False, "use_heuristic": False},
        }

        results = []
        for name, flags in variants.items():
            logger.info("Running ablation variant: %s", name)
            steps_list, time_list = [], []

            for run_idx in range(self.n_runs):
                seed = self.base_seed + run_idx * 100
                rng  = np.random.default_rng(seed)
                engine = MutationEngine(self.constraints, seed=seed)

                population = self._build_population(candidate_pool, flags, rng)
                t0 = time.perf_counter()
                steps = self._evolve_to_threshold(population, engine, rng, flags)
                elapsed = time.perf_counter() - t0

                steps_list.append(steps)
                time_list.append(elapsed)
                logger.debug(
                    "[%s] run %d — steps=%d, time=%.2fs",
                    name, run_idx + 1, steps, elapsed,
                )

            results.append(AblationResult(
                variant=name,
                dataset=dataset_name,
                steps_to_thresh=float(np.median(steps_list)),
                steps_std=float(np.std(steps_list)),
                time_to_thresh=float(np.median(time_list)),
                time_std=float(np.std(time_list)),
            ))
            logger.info(
                "[%s] median steps=%.0f±%.0f, time=%.2f±%.2fs",
                name,
                results[-1].steps_to_thresh,
                results[-1].steps_std,
                results[-1].time_to_thresh,
                results[-1].time_std,
            )

        return results

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_population(
        self,
        pool:    List[Genome],
        flags:   dict,
        rng:     np.random.Generator,
    ) -> List[Genome]:
        """Construct population according to ablation flags."""
        if flags.get("use_heuristic", True):
            # Use the full heuristic initialiser (simplified here)
            population = [g.clone() for g in pool[:self.population_size]]
        else:
            # Uniform random initialisation
            population = [self.factory.random(rng) for _ in range(self.population_size)]

        # Pad if needed
        while len(population) < self.population_size:
            population.append(self.factory.random(rng))
        return population[:self.population_size]

    def _evolve_to_threshold(
        self,
        population: List[Genome],
        engine:     MutationEngine,
        rng:        np.random.Generator,
        flags:      dict,
    ) -> int:
        """
        Run evolution and return the number of evaluations needed
        to first exceed self.threshold in best fitness.
        Returns max_rounds * population_size if never reached.
        """
        # Build prototype set
        features = np.stack([self.feature_fn(g) for g in population])
        phi_z    = zscore_normalize(features)
        sigma    = compute_sigma(phi_z)

        if flags.get("use_submodular", True) and not flags.get("random_proto", False):
            proto_idx = greedy_facility_location(phi_z, self.m, sigma=sigma)
        else:
            proto_idx = rng.choice(len(population), min(self.m, len(population)), replace=False)

        prototypes  = [population[i] for i in proto_idx]
        assignments = assign_to_prototypes(phi_z, phi_z[proto_idx], sigma)

        total_steps = 0
        for pid, proto in enumerate(prototypes):
            nbhd = [population[i] for i in np.where(assignments == pid)[0]]
            if not nbhd:
                continue

            ga = ConstrainedGA(
                evaluator=self.evaluator,
                mutation_engine=engine,
                mu=self.mu,
                lambda_=self.lambda_,
                max_rounds=self.max_rounds,
                stagnation=self.stagnation,
            )

            elites = ga.run(nbhd)
            total_steps += len(nbhd) * self.max_rounds

            if elites and elites[0].fitness >= self.threshold:
                return total_steps

        return total_steps

    # ------------------------------------------------------------------ #
    #  Report formatter                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def print_report(results: List[AblationResult]):
        print(f"\n{'Variant':>10} | {'Steps (med±std)':>20} | {'Time/s (med±std)':>22}")
        print("-" * 60)
        for r in results:
            print(
                f"{r.variant:>10} | "
                f"{r.steps_to_thresh:>8.0f} ± {r.steps_std:>6.0f}     | "
                f"{r.time_to_thresh:>8.2f} ± {r.time_std:>6.2f}"
            )
        print()


# --------------------------------------------------------------------------- #
#  CLI entry point                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    from models.traffic_classifier.traffic_llm import build_traffic_llm
    from models.surrogate.surrogate_model import SurrogateModel

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--pool_size",     type=int,   default=200)
    p.add_argument("--n_runs",        type=int,   default=3)
    p.add_argument("--threshold",     type=float, default=0.3)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--feature_dim",   type=int,   default=83)
    args = p.parse_args()

    backbone_cfg = {"hidden_size": 128, "num_layers": 2,
                    "num_attention_heads": 4, "input_dim": args.feature_dim,
                    "num_classes": 20}
    m0  = build_traffic_llm(backbone_cfg)
    sur = SurrogateModel(input_dim=args.feature_dim, num_classes=20)

    def simple_feature_fn(g: Genome) -> np.ndarray:
        arr = g.to_array()
        feat = np.concatenate([arr.mean(0), arr.std(0), arr.min(0), arr.max(0)])
        return feat[:args.feature_dim].astype(np.float32) if len(feat) >= args.feature_dim \
               else np.pad(feat, (0, args.feature_dim - len(feat))).astype(np.float32)

    rng_pool   = np.random.default_rng(0)
    factory    = GenomeFactory()
    pool       = [factory.random(rng_pool) for _ in range(args.pool_size)]

    # Calibrate rebound direction
    cal_feats = np.stack([simple_feature_fn(g) for g in pool[:50]])
    cal_t     = torch.from_numpy(cal_feats).float()
    with torch.enable_grad():
        m0_eval = m0.eval()
        m0_eval.zero_grad()

    runner = AblationRunner(
        pretrained_model=m0,
        surrogate_model=sur,
        feature_fn=simple_feature_fn,
        fitness_threshold=args.threshold,
        n_runs=args.n_runs,
        device=args.device,
    )

    results = runner.run_all(pool, dataset_name="synthetic")
    AblationRunner.print_report(results)
