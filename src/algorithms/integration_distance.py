# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai

"""
IntegrationDistance
====================
Computes distances between multivariate systems based on their spectral
integration structure — eigenvalue distribution divergence.

Applications:
  - Clustering time series by dynamical similarity
  - Detecting replica failures (two machines that should behave identically)
  - Domain adaptation (how different are source and target distributions?)
  - Sensor network health (which sensors drifted away from the fleet?)

Three distance metrics, each with different properties:
  KL     — asymmetric, information-theoretic
  JSD    — symmetric KL, bounded [0, log2], stable
  Wasserstein1 — earth mover's distance on eigenvalue spectrum

Author: Walter Calmels — ConsciousAI Suite
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from numba import jit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def jit(*a, **k): return (lambda f: f) if not a else a[0]
    def prange(*a): return range(*a)


# ── JIT kernels ───────────────────────────────────────────────────────────────

@jit(nopython=True, fastmath=True)
def _spectral_distribution(data: np.ndarray,
                            tol: float = 1e-10) -> np.ndarray:
    """
    Returns normalised positive eigenvalue distribution of Cov(data).
    Shape: (N,) — the spectral probability vector.
    """
    n, m = data.shape
    if n < 2:
        return np.ones(m) / m

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

    # Keep positive eigenvalues, normalise
    pos = np.zeros(m)
    total = 0.0
    for i in range(m):
        v = abs(eigs[i])
        if v > tol:
            pos[i] = v
            total += v

    if total < tol:
        return np.ones(m) / m

    return pos / total


@jit(nopython=True, fastmath=True)
def _kl_divergence(p: np.ndarray, q: np.ndarray,
                   eps: float = 1e-12) -> float:
    """KL(p || q) — asymmetric."""
    kl = 0.0
    for i in range(len(p)):
        pi = p[i] + eps
        qi = q[i] + eps
        kl += pi * np.log(pi / qi)
    return max(0.0, kl)


@jit(nopython=True, fastmath=True)
def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence — symmetric, bounded [0, log2]."""
    m = np.zeros(len(p))
    for i in range(len(p)):
        m[i] = 0.5 * (p[i] + q[i])
    return 0.5 * _kl_divergence(p, m, eps) + 0.5 * _kl_divergence(q, m, eps)


@jit(nopython=True, fastmath=True)
def _wasserstein1(p: np.ndarray, q: np.ndarray) -> float:
    """
    1-Wasserstein (Earth Mover's Distance) between sorted spectral distributions.
    Efficient for 1D distributions via CDF difference.
    """
    # Sort both distributions
    p_sorted = np.sort(p)[::-1]
    q_sorted = np.sort(q)[::-1]

    # Cumulative sums
    cdf_p = np.cumsum(p_sorted)
    cdf_q = np.cumsum(q_sorted)

    return float(np.sum(np.abs(cdf_p - cdf_q)))


# Warm-up
_d = np.random.rand(10, 4).astype(np.float64)
_sp = _spectral_distribution(_d)
_kl_divergence(_sp, _sp)
_jsd(_sp, _sp)
_wasserstein1(_sp, _sp)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DistanceResult:
    kl_ab:         float    # KL(a || b)
    kl_ba:         float    # KL(b || a)
    jsd:           float    # Jensen-Shannon (symmetric)
    wasserstein:   float    # Earth Mover's Distance
    spec_a:        np.ndarray  # Spectral distribution of a
    spec_b:        np.ndarray  # Spectral distribution of b

    @property
    def symmetric(self) -> float:
        """JSD as canonical symmetric distance."""
        return self.jsd

    @property
    def asymmetry(self) -> float:
        """How much more a -> b costs than b -> a."""
        return abs(self.kl_ab - self.kl_ba)


# ── Main class ────────────────────────────────────────────────────────────────

class IntegrationDistance:
    """
    Distance metric between multivariate time-series systems based on
    their spectral integration structure.

    Usage
    -----
    dist = IntegrationDistance()

    # Compare two systems
    result = dist.compare(data_a, data_b)
    print(f"JSD = {result.jsd:.4f}")

    # Pairwise distance matrix for a list of systems
    D = dist.pairwise(systems_list)

    # Find anomalous systems in a fleet
    outliers = dist.fleet_outliers(systems_list, threshold=0.8)
    """

    def __init__(self, metric: str = "jsd"):
        """
        Parameters
        ----------
        metric : 'jsd' | 'kl' | 'wasserstein'
                 Default metric used in pairwise() and fleet_outliers()
        """
        if metric not in ("jsd", "kl", "wasserstein"):
            raise ValueError("metric must be 'jsd', 'kl', or 'wasserstein'")
        self.metric = metric

    # ── core comparison ───────────────────────────────────────────────────

    def compare(self, data_a: np.ndarray,
                data_b: np.ndarray) -> DistanceResult:
        """
        Compute all distance metrics between two systems.

        Parameters
        ----------
        data_a, data_b : (T, N) arrays — may have different T but same N

        Returns
        -------
        DistanceResult
        """
        a = np.asarray(data_a, dtype=np.float64)
        b = np.asarray(data_b, dtype=np.float64)

        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if b.ndim == 1:
            b = b.reshape(-1, 1)

        # Pad to same N if needed
        N = max(a.shape[1], b.shape[1])
        if a.shape[1] < N:
            a = np.hstack([a, np.zeros((a.shape[0], N - a.shape[1]))])
        if b.shape[1] < N:
            b = np.hstack([b, np.zeros((b.shape[0], N - b.shape[1]))])

        sp_a = _spectral_distribution(a)
        sp_b = _spectral_distribution(b)

        return DistanceResult(
            kl_ab       = _kl_divergence(sp_a, sp_b),
            kl_ba       = _kl_divergence(sp_b, sp_a),
            jsd         = _jsd(sp_a, sp_b),
            wasserstein = _wasserstein1(sp_a, sp_b),
            spec_a      = sp_a,
            spec_b      = sp_b,
        )

    def scalar(self, data_a: np.ndarray,
               data_b: np.ndarray) -> float:
        """Single distance value using self.metric."""
        r = self.compare(data_a, data_b)
        return getattr(r, self.metric if self.metric != "kl" else "kl_ab")

    # ── pairwise matrix ───────────────────────────────────────────────────

    def pairwise(self, systems: List[np.ndarray]) -> np.ndarray:
        """
        Compute symmetric pairwise distance matrix.

        Parameters
        ----------
        systems : list of (T_i, N) arrays

        Returns
        -------
        D : (M, M) symmetric distance matrix
        """
        M = len(systems)
        D = np.zeros((M, M))

        for i in range(M):
            for j in range(i + 1, M):
                d = self.scalar(systems[i], systems[j])
                D[i, j] = d
                D[j, i] = d

        return D

    # ── fleet outlier detection ───────────────────────────────────────────

    def fleet_outliers(self, systems: List[np.ndarray],
                       threshold: float = 2.0,
                       reference: Optional[np.ndarray] = None
                       ) -> Tuple[List[int], np.ndarray]:
        """
        Identify systems that are anomalously different from the fleet.

        Parameters
        ----------
        systems   : list of (T, N) arrays
        threshold : z-score threshold for outlier detection
        reference : Optional reference system; if None uses fleet mean

        Returns
        -------
        outlier_indices : list of system indices flagged as outliers
        scores          : (M,) array of anomaly scores
        """
        M = len(systems)
        specs = np.array([
            _spectral_distribution(np.asarray(s, dtype=np.float64))
            for s in systems
        ])

        if reference is not None:
            ref_spec = _spectral_distribution(
                np.asarray(reference, dtype=np.float64)
            )
        else:
            ref_spec = np.mean(specs, axis=0)
            ref_spec = ref_spec / (ref_spec.sum() + 1e-12)

        # Distance of each system from reference
        scores = np.array([
            _jsd(specs[i], ref_spec)
            for i in range(M)
        ])

        # Z-score based outlier detection
        mu, sg = scores.mean(), scores.std()
        z_scores = (scores - mu) / (sg + 1e-6)
        outliers = [i for i in range(M) if z_scores[i] > threshold]

        return outliers, scores

    # ── temporal drift ────────────────────────────────────────────────────

    def temporal_drift(self, data: np.ndarray,
                       window: int = 50,
                       step: int = 5) -> np.ndarray:
        """
        Measure how much a system's spectral structure drifts over time
        relative to its initial state.

        Returns
        -------
        drift_series : (K,) array of JSD values vs. initial window
        """
        data = np.asarray(data, dtype=np.float64)
        T = data.shape[0]

        if T < 2 * window:
            raise ValueError(f"Need at least 2×window={2*window} samples, got {T}")

        reference_spec = _spectral_distribution(data[:window])

        drifts = []
        for start in range(window, T - window + 1, step):
            chunk = data[start: start + window]
            spec  = _spectral_distribution(chunk)
            drifts.append(_jsd(spec, reference_spec))

        return np.array(drifts)
