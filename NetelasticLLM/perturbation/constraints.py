"""
perturbation/constraints.py
Protocol and physical feasibility constraints for genome validation.

Enforces the three constraint categories from Section 4.4.1:
  (i)  Temporal causality    — nonnegative gaps, monotone timestamps
  (ii) Traffic budgets       — total bytes ≤ B_max, duration ≤ T_max
  (iii)Protocol realism      — valid direction patterns, run-length rules

Also provides penalty functions p_b(u) and p_p(u) used in the
fitness function Φ(u) (Eq. 13).
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict

import numpy as np

from .genome_encoding import Genome, GenomeConstraints

logger = logging.getLogger(__name__)


class ConstraintChecker:
    """
    Checks feasibility and computes soft penalty scores for a genome.

    Penalty values are in [0, 1]; 0 = fully feasible, 1 = maximally violated.
    """

    def __init__(self, constraints: GenomeConstraints = GenomeConstraints()):
        self.c = constraints

    # ------------------------------------------------------------------ #
    #  Binary feasibility checks                                           #
    # ------------------------------------------------------------------ #

    def check_temporal_causality(self, g: Genome) -> Tuple[bool, str]:
        """All inter-arrival intervals must be non-negative."""
        if any(d < 0.0 for d in g.deltas):
            return False, "negative inter-arrival interval"
        return True, ""

    def check_traffic_budget(self, g: Genome) -> Tuple[bool, str]:
        """Total bytes and duration must not exceed configured maxima."""
        if g.total_bytes() > self.c.total_bytes_max:
            return False, (
                f"total bytes {g.total_bytes()} > {self.c.total_bytes_max}"
            )
        if g.total_duration() > self.c.total_dur_max:
            return False, (
                f"total duration {g.total_duration():.3f}s > {self.c.total_dur_max}s"
            )
        return True, ""

    def check_protocol_realism(self, g: Genome) -> Tuple[bool, str]:
        """
        Direction patterns must remain within realistic bounds.
        Checks:
          - No unknown direction values
          - Direction alternation count ≤ max_dir_alternations
          - Run lengths ≥ min_run_length
        """
        for d in g.directions:
            if d not in (1, -1):
                return False, f"invalid direction value {d}"

        alts = g.direction_alternations()
        if alts > self.c.max_dir_alternations:
            return False, (
                f"direction alternations {alts} > {self.c.max_dir_alternations}"
            )

        # Run-length check: no run shorter than min_run_length
        if self.c.min_run_length > 1:
            run = 1
            for i in range(1, g.n):
                if g.directions[i] == g.directions[i - 1]:
                    run += 1
                else:
                    if run < self.c.min_run_length:
                        return False, f"run length {run} < {self.c.min_run_length}"
                    run = 1

        return True, ""

    def check_packet_bounds(self, g: Genome) -> Tuple[bool, str]:
        """Individual packet fields must stay within configured bounds."""
        for l in g.lengths:
            if not (self.c.l_min <= l <= self.c.l_max):
                return False, f"packet length {l} out of [{self.c.l_min},{self.c.l_max}]"
        for d in g.deltas:
            if not (self.c.delta_min <= d <= self.c.delta_max):
                return False, f"delta {d:.4f} out of [{self.c.delta_min},{self.c.delta_max}]"
        if not (self.c.n_min <= g.n <= self.c.n_max):
            return False, f"n={g.n} outside [{self.c.n_min},{self.c.n_max}]"
        return True, ""

    def is_feasible(self, g: Genome) -> Tuple[bool, str]:
        """Composite feasibility check — returns (True,'') iff all pass."""
        for check_fn in (
            self.check_temporal_causality,
            self.check_traffic_budget,
            self.check_protocol_realism,
            self.check_packet_bounds,
        ):
            ok, reason = check_fn(g)
            if not ok:
                return False, reason
        return True, ""

    # ------------------------------------------------------------------ #
    #  Soft penalty functions  p_b(u), p_p(u)  [Eq. 13]                   #
    # ------------------------------------------------------------------ #

    def budget_penalty(self, g: Genome) -> float:
        """
        p_b(u) ∈ [0, 1] — coarse-level fidelity penalty.
        Penalises flows that exceed total byte or duration budgets.
        """
        byte_excess = max(0, g.total_bytes() - self.c.total_bytes_max)
        dur_excess  = max(0.0, g.total_duration() - self.c.total_dur_max)

        byte_penalty = byte_excess / max(self.c.total_bytes_max, 1)
        dur_penalty  = dur_excess  / max(self.c.total_dur_max,   1e-6)

        return float(np.clip(byte_penalty + dur_penalty, 0.0, 1.0))

    def protocol_penalty(self, g: Genome) -> float:
        """
        p_p(u) ∈ [0, 1] — protocol realism penalty.
        Penalises excess direction alternations and out-of-range values.
        """
        penalty = 0.0

        # Direction alternation excess
        alts    = g.direction_alternations()
        excess  = max(0, alts - self.c.max_dir_alternations)
        penalty += excess / max(self.c.max_dir_alternations, 1)

        # Packet length violations
        len_violations = sum(
            1 for l in g.lengths
            if l < self.c.l_min or l > self.c.l_max
        )
        penalty += len_violations / max(g.n, 1)

        # Delta violations
        dt_violations = sum(
            1 for d in g.deltas
            if d < self.c.delta_min or d > self.c.delta_max
        )
        penalty += dt_violations / max(g.n, 1)

        return float(np.clip(penalty, 0.0, 1.0))

    def penalty_dict(self, g: Genome) -> Dict[str, float]:
        """Return all penalty components as a dict for logging/debugging."""
        return {
            "budget_penalty":   self.budget_penalty(g),
            "protocol_penalty": self.protocol_penalty(g),
            "total_bytes":      g.total_bytes(),
            "total_duration":   g.total_duration(),
            "n_packets":        g.n,
            "dir_alternations": g.direction_alternations(),
        }
