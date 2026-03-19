"""
search/submodular_selection.py
Layer I of the dual-layer evolutionary search: submodular prototype selection.

Implements the facility-location coverage objective (Eq. 12):
    max_{S ⊆ C, |S|=m} F(S) = Σ_{u ∈ C} max_{s ∈ S} sim(u, s)

where sim(u,v) = exp(-‖φ(u)-φ(v)‖²₂ / 2σ²)

The greedy algorithm achieves a (1-1/e) approximation guarantee under
the cardinality constraint |S| = m.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Similarity kernel                                                            #
# --------------------------------------------------------------------------- #

def rbf_similarity(
    phi_u: np.ndarray,
    phi_v: np.ndarray,
    sigma:  float,
) -> np.ndarray:
    """
    Compute RBF similarity between rows of phi_u and phi_v.

    Args:
        phi_u : (A, D) array.
        phi_v : (B, D) array.
        sigma : bandwidth parameter.
    Returns:
        sim   : (A, B) similarity matrix.
    """
    # Squared Euclidean distances via broadcasting
    diff    = phi_u[:, None, :] - phi_v[None, :, :]    # (A, B, D)
    sq_dist = (diff ** 2).sum(axis=-1)                  # (A, B)
    return np.exp(-sq_dist / (2.0 * sigma ** 2 + 1e-8))


def compute_sigma(phi: np.ndarray) -> float:
    """
    Estimate σ as the median pairwise distance over a subset of phi.
    Sub-samples up to 500 rows for efficiency.
    """
    n = min(len(phi), 500)
    idx = np.random.choice(len(phi), n, replace=False)
    sub = phi[idx]
    # Pairwise squared distances
    diff    = sub[:, None, :] - sub[None, :, :]
    sq_dist = (diff ** 2).sum(axis=-1)
    # Median of upper triangle
    triu    = sq_dist[np.triu_indices(n, k=1)]
    median_sq = float(np.median(triu)) if len(triu) > 0 else 1.0
    return float(np.sqrt(median_sq)) + 1e-4


# --------------------------------------------------------------------------- #
#  Greedy facility-location selection                                           #
# --------------------------------------------------------------------------- #

def greedy_facility_location(
    phi:        np.ndarray,
    m:          int,
    sigma:      Optional[float] = None,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Greedy maximisation of facility-location submodular objective.

    Args:
        phi        : (N, D) z-score-normalised feature matrix for C.
        m          : Number of prototypes to select.
        sigma      : RBF bandwidth (estimated from data if None).
        batch_size : Chunk size for similarity computation.
    Returns:
        selected   : (m,) integer array of prototype indices into phi.
    """
    n = len(phi)
    m = min(m, n)

    if sigma is None:
        sigma = compute_sigma(phi)
    logger.debug("Facility-location: N=%d, m=%d, σ=%.4f", n, m, sigma)

    selected: List[int]  = []
    # coverage[i] = max_{s already selected} sim(phi[i], phi[s])
    coverage = np.zeros(n, dtype=np.float64)

    for step in range(m):
        best_idx   = -1
        best_gain  = -1.0

        # Evaluate marginal gain for each candidate
        for start in range(0, n, batch_size):
            end       = min(start + batch_size, n)
            batch_phi = phi[start:end]                          # (B, D)

            # sim(batch, all_candidates_in_pool)
            sim_batch = rbf_similarity(
                phi[np.array(selected)] if selected else phi[:1],
                batch_phi,
                sigma,
            )                                                   # (|S|, B) or (1, B)

            if selected:
                current_cov = sim_batch.max(axis=0)             # (B,)
            else:
                current_cov = np.zeros(end - start)

            # F(S ∪ {u}) - F(S) = Σ_v max(sim(v,u), coverage[v]) - coverage[v]
            # For each candidate u in [start:end]:
            for local_idx in range(end - start):
                global_idx = start + local_idx
                if global_idx in selected:
                    continue

                sim_u_all = rbf_similarity(
                    phi[[global_idx]], phi, sigma
                ).flatten()                                     # (N,)
                new_cov   = np.maximum(coverage, sim_u_all)
                gain      = float(new_cov.sum() - coverage.sum())

                if gain > best_gain:
                    best_gain = gain
                    best_idx  = global_idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        # Update coverage
        sim_new     = rbf_similarity(phi[[best_idx]], phi, sigma).flatten()
        coverage    = np.maximum(coverage, sim_new)

        logger.debug(
            "Step %d/%d: selected idx=%d, marginal gain=%.4f, F=%.4f",
            step + 1, m, best_idx, best_gain, coverage.sum(),
        )

    return np.array(selected, dtype=np.int64)


# --------------------------------------------------------------------------- #
#  Z-score normalisation                                                        #
# --------------------------------------------------------------------------- #

def zscore_normalize(
    phi: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score normalise each feature dimension across the candidate pool C.

    φ_j(u) ← (φ_j(u) - μ_j) / (σ_j + ε)

    Returns normalised copy; does not modify in-place.
    """
    mu  = phi.mean(axis=0)
    std = phi.std(axis=0) + eps
    return (phi - mu) / std


# --------------------------------------------------------------------------- #
#  Neighbourhood assignment                                                     #
# --------------------------------------------------------------------------- #

def assign_to_prototypes(
    phi:          np.ndarray,
    phi_proto:    np.ndarray,
    sigma:        float,
) -> np.ndarray:
    """
    Assign each candidate to its nearest prototype.

    Args:
        phi       : (N, D) full candidate embeddings.
        phi_proto : (m, D) prototype embeddings.
        sigma     : RBF bandwidth.
    Returns:
        assignment : (N,) integer array in [0, m-1].
    """
    sim = rbf_similarity(phi, phi_proto, sigma)   # (N, m)
    return sim.argmax(axis=1)                      # (N,)
