# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai

"""
CausalEmergence
================
Measures how much a system as a whole exceeds the sum of its parts
in terms of spectral integration.

Emergence(S) = Φ̃(S) - max_i Φ̃(S \ {i})

Positive emergence: the whole system is more integrated than any subsystem.
Negative: a subsystem is actually more coherent than the full system
          (indicates a dominant sub-process or noise from other components).

Applications:
  - Identify which subsystem in an industrial plant dominates failure modes
  - Find irreducible sensor groups in a fleet (which sensors can be removed
    without losing integration information)
  - Neural circuits: which layer clusters are computationally essential?
  - Portfolio: which asset group drives systemic integration?

Author: Walter Calmels — ConsciousAI Suite
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from itertools import combinations

try:
    from numba import jit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def jit(*a, **k): return (lambda f: f) if not a else a[0]
    def prange(*a): return range(*a)


# ── JIT kernel ────────────────────────────────────────────────────────────────

@jit(nopython=True, fastmath=True)
def _phi_approx(data: np.ndarray, conn_strength: float = 1.0,
                tol: float = 1e-10) -> float:
    """Spectral Φ̃ for a data window."""
    n, m = data.shape
    if n < 2 or m < 1:
        return 0.0

    means = np.zeros(m)
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += data[i, j]
        means[j] = s / n

    cov = np.zeros((m, m))
    denom = float(n - 1)
    for i in range(m):
        for j in range(i, m):
            s = 0.0
            for k in range(n):
                s += (data[k, i] - means[i]) * (data[k, j] - means[j])
            s /= denom
            cov[i, j] = s
            cov[j, i] = s

    eigs = np.linalg.eigvalsh(cov)

    total = 0.0
    count = 0
    for v in eigs:
        av = abs(v)
        if av > tol:
            total += av
            count += 1

    if count == 0 or total < tol:
        return 0.0

    ent = 0.0
    for v in eigs:
        av = abs(v)
        if av > tol:
            p = av / total
            if p > 1e-15:
                ent -= p * np.log(p)

    return max(0.0, ent * count * conn_strength)


# Warm-up
_dummy = np.random.rand(10, 4).astype(np.float64)
_phi_approx(_dummy)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class SubsystemResult:
    indices:        Tuple          # which components
    phi:            float          # Φ̃ of this subsystem
    emergence:      float          # Φ̃(full) - Φ̃(this subsystem)
    is_dominant:    bool           # emergence < 0 → this subsystem drives it


@dataclass
class EmergenceResult:
    phi_full:           float
    phi_subsystems:     List[SubsystemResult]
    max_emergence:      float       # global emergence over all single-removals
    dominant_subsystem: Optional[Tuple]   # subsystem with negative emergence (if any)
    emergence_index:    float       # normalised: emergence / phi_full


# ── Main class ────────────────────────────────────────────────────────────────

class CausalEmergence:
    """
    Measures spectral emergence: how much the whole exceeds its parts.

    Parameters
    ----------
    max_subset_size : Maximum subsystem size to enumerate (default 1 = single
                      component removal; set to 2 for pair removal, etc.)
    min_components  : Minimum components in a subsystem

    Usage
    -----
    ce = CausalEmergence()

    # Analyse a single snapshot
    result = ce.analyse(data)
    print(f"Emergence index = {result.emergence_index:.3f}")

    # Rolling emergence over time
    series = ce.rolling(data, window=50)
    """

    def __init__(self, max_subset_size: int = 1,
                 min_components: int = 2):
        self.max_subset_size = max_subset_size
        self.min_components  = min_components

    # ── core analysis ─────────────────────────────────────────────────────

    def analyse(self, data: np.ndarray) -> EmergenceResult:
        """
        Compute emergence for a data window.

        Parameters
        ----------
        data : (T, N) float array

        Returns
        -------
        EmergenceResult
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        T, N = data.shape
        phi_full = _phi_approx(data)

        subsystems = []
        all_indices = list(range(N))

        # Enumerate subsystems up to max_subset_size removed
        for size in range(1, min(self.max_subset_size + 1, N)):
            for removed in combinations(all_indices, size):
                kept = tuple(i for i in all_indices if i not in removed)
                if len(kept) < self.min_components:
                    continue

                sub_data = data[:, list(kept)]
                phi_sub  = _phi_approx(sub_data)
                emergence = phi_full - phi_sub

                subsystems.append(SubsystemResult(
                    indices     = kept,
                    phi         = phi_sub,
                    emergence   = emergence,
                    is_dominant = emergence < 0,
                ))

        if not subsystems:
            return EmergenceResult(
                phi_full           = phi_full,
                phi_subsystems     = [],
                max_emergence      = 0.0,
                dominant_subsystem = None,
                emergence_index    = 0.0,
            )

        max_sub_phi = max(s.phi for s in subsystems)
        max_emergence = phi_full - max_sub_phi

        dominant = next(
            (s for s in sorted(subsystems, key=lambda x: x.phi, reverse=True)
             if s.is_dominant), None
        )

        emergence_index = max_emergence / (phi_full + 1e-10)

        return EmergenceResult(
            phi_full           = phi_full,
            phi_subsystems     = subsystems,
            max_emergence      = max_emergence,
            dominant_subsystem = dominant.indices if dominant else None,
            emergence_index    = emergence_index,
        )

    # ── rolling analysis ──────────────────────────────────────────────────

    def rolling(self, data: np.ndarray,
                window: int = 50,
                step: int = 5) -> np.ndarray:
        """
        Compute rolling emergence index over time.

        Returns
        -------
        emergence_series : (K,) array of emergence_index values
        """
        data = np.asarray(data, dtype=np.float64)
        T    = data.shape[0]
        out  = []

        for start in range(0, T - window + 1, step):
            chunk  = data[start: start + window]
            result = self.analyse(chunk)
            out.append(result.emergence_index)

        return np.array(out)

    # ── component importance ──────────────────────────────────────────────

    def component_importance(self, data: np.ndarray) -> np.ndarray:
        """
        Rank components by their contribution to system emergence.
        Importance_i = Φ̃(full) - Φ̃(system without component i)

        Higher = removing this component loses more integration.

        Returns
        -------
        importance : (N,) array, higher = more important
        """
        data    = np.asarray(data, dtype=np.float64)
        T, N    = data.shape
        phi_full = _phi_approx(data)
        scores   = np.zeros(N)

        for i in range(N):
            kept     = [j for j in range(N) if j != i]
            sub_data = data[:, kept]
            phi_sub  = _phi_approx(sub_data)
            scores[i] = phi_full - phi_sub   # positive = removing hurts

        return scores

    # ── partition optimisation ────────────────────────────────────────────

    def optimal_partition(self, data: np.ndarray,
                          n_groups: int = 2) -> List[List[int]]:
        """
        Find the partition of components into n_groups that maximises
        total emergence (Φ̃_full - sum(Φ̃_groups)).

        Uses greedy assignment based on component importance.

        Returns
        -------
        groups : list of lists of component indices
        """
        data  = np.asarray(data, dtype=np.float64)
        N     = data.shape[1]
        imp   = self.component_importance(data)

        # Sort components by importance descending
        order  = np.argsort(imp)[::-1]
        groups = [[] for _ in range(n_groups)]

        for rank, comp in enumerate(order):
            groups[rank % n_groups].append(int(comp))

        return groups
