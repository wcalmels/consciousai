# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai

"""
SpectralChangePoint
====================
Online and offline change-point detection using rolling spectral entropy
of the covariance matrix.

Core idea: a regime shift changes the eigenvalue distribution of the system.
Monitoring H(λ) continuously detects that change before individual sensors
cross thresholds.

Validated advantages over CUSUM / Z-score:
  - Detects multivariate coupling changes (not just mean/variance shifts)
  - No distribution assumption required
  - Same Numba kernels as ConsciousAI core → sub-ms per window

Author: Walter Calmels — ConsciousAI Suite
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from numba import jit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def jit(*a, **k): return (lambda f: f) if not a else a[0]
    def prange(*a): return range(*a)


# ── JIT kernel ────────────────────────────────────────────────────────────────

@jit(nopython=True, fastmath=True)
def _spectral_entropy(data: np.ndarray, tol: float = 1e-10) -> float:
    """Spectral entropy of covariance eigenvalue distribution."""
    n, m = data.shape
    if n < 2:
        return 0.0

    # Covariance
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

    # Eigenvalues via numpy (LAPACK — unavoidable)
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

    return ent


@jit(nopython=True, fastmath=True, parallel=True)
def _rolling_entropy_batch(data: np.ndarray, window: int,
                           step: int) -> np.ndarray:
    """Compute rolling spectral entropy for all windows in parallel."""
    T, N = data.shape
    n_windows = max(0, (T - window) // step + 1)
    out = np.zeros(n_windows)
    for w in prange(n_windows):
        start = w * step
        out[w] = _spectral_entropy(data[start: start + window])
    return out


# ── Warm-up (compile JIT on import) ──────────────────────────────────────────
_dummy = np.random.rand(10, 4).astype(np.float64)
_spectral_entropy(_dummy)
_rolling_entropy_batch(_dummy, 5, 2)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ChangePoint:
    index: int          # position in original series
    score: float        # detection score at this point
    entropy_before: float
    entropy_after: float

    @property
    def magnitude(self) -> float:
        return abs(self.entropy_after - self.entropy_before)


@dataclass
class CPDResult:
    change_points: List[ChangePoint]
    entropy_series: np.ndarray
    threshold: float
    window: int
    step: int


# ── Main class ────────────────────────────────────────────────────────────────

class SpectralChangePoint:
    """
    Change-point detector based on rolling spectral entropy.

    Parameters
    ----------
    window      : Rolling window length
    step        : Step between windows (1 = fully overlapping)
    threshold   : Detection threshold as z-score above baseline mean
    min_gap     : Minimum samples between consecutive change points
    method      : 'zscore' | 'cusum' | 'gradient'

    Usage
    -----
    cpd = SpectralChangePoint(window=30, threshold=2.5)
    result = cpd.fit(data)
    print(result.change_points)
    """

    def __init__(self, window: int = 30, step: int = 1,
                 threshold: float = 2.5, min_gap: int = 10,
                 method: str = "zscore"):
        self.window    = window
        self.step      = step
        self.threshold = threshold
        self.min_gap   = min_gap
        self.method    = method

        self._baseline_mean: Optional[float] = None
        self._baseline_std:  Optional[float] = None

    # ── public API ────────────────────────────────────────────────────────

    def fit(self, data: np.ndarray,
            baseline_frac: float = 0.2) -> CPDResult:
        """
        Detect change points in data.

        Parameters
        ----------
        data          : (T, N) float array
        baseline_frac : Fraction of series used to estimate baseline statistics

        Returns
        -------
        CPDResult
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        T = data.shape[0]

        # Rolling entropy via JIT
        entropy_series = _rolling_entropy_batch(data, self.window, self.step)

        # Map window indices back to original time axis
        time_idx = np.array([w * self.step + self.window // 2
                             for w in range(len(entropy_series))])

        # Baseline statistics from first baseline_frac of series
        n_baseline = max(2, int(len(entropy_series) * baseline_frac))
        self._baseline_mean = float(np.mean(entropy_series[:n_baseline]))
        self._baseline_std  = max(float(np.std(entropy_series[:n_baseline])), 1e-6)

        # Compute detection score
        if self.method == "zscore":
            scores = np.abs((entropy_series - self._baseline_mean)
                            / self._baseline_std)
            thr = self.threshold

        elif self.method == "cusum":
            scores = np.cumsum(entropy_series - self._baseline_mean)
            scores = np.abs(scores - scores[0])
            thr = self.threshold * self._baseline_std * np.sqrt(len(scores))

        elif self.method == "gradient":
            scores = np.abs(np.gradient(entropy_series))
            thr = self.threshold * np.std(scores[:n_baseline])

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Peak picking with min_gap
        change_points = self._pick_peaks(scores, time_idx, entropy_series, thr)

        return CPDResult(
            change_points  = change_points,
            entropy_series = entropy_series,
            threshold      = thr,
            window         = self.window,
            step           = self.step,
        )

    def fit_online(self, stream: np.ndarray,
                   burn_in: int = 50) -> List[int]:
        """
        Online (streaming) change-point detection.
        Returns list of detected change point positions.

        Parameters
        ----------
        stream  : (T, N) array processed sequentially
        burn_in : Samples before detection starts
        """
        T = stream.shape[0]
        detected = []
        buffer   = []
        last_cp  = -self.min_gap

        for t in range(T):
            buffer.append(stream[t])
            if len(buffer) < self.window:
                continue

            chunk = np.array(buffer[-self.window:], dtype=np.float64)
            h     = float(_spectral_entropy(chunk))

            if t < burn_in:
                # Collect baseline
                if self._baseline_mean is None:
                    self._baseline_mean = h
                    self._baseline_std  = 1e-6
                else:
                    # Online mean/std update
                    n = t + 1
                    old_mean = self._baseline_mean
                    self._baseline_mean += (h - old_mean) / n
                    self._baseline_std = max(
                        float(np.std([old_mean, h])), 1e-6
                    )
                continue

            z = abs(h - self._baseline_mean) / self._baseline_std
            if z > self.threshold and (t - last_cp) >= self.min_gap:
                detected.append(t)
                last_cp = t

        return detected

    # ── private ───────────────────────────────────────────────────────────

    def _pick_peaks(self, scores, time_idx, entropy_series, thr):
        above = np.where(scores > thr)[0]
        if len(above) == 0:
            return []

        cps = []
        last = -self.min_gap

        # Group consecutive detections, take peak of each group
        groups = []
        group  = [above[0]]
        for idx in above[1:]:
            if idx - group[-1] <= self.min_gap:
                group.append(idx)
            else:
                groups.append(group)
                group = [idx]
        groups.append(group)

        for grp in groups:
            peak = grp[int(np.argmax(scores[grp]))]
            t    = int(time_idx[peak])

            if t - last < self.min_gap:
                continue

            h_before = float(np.mean(entropy_series[max(0, peak-5): peak]))
            h_after  = float(np.mean(entropy_series[peak: min(len(entropy_series), peak+5)]))

            cps.append(ChangePoint(
                index          = t,
                score          = float(scores[peak]),
                entropy_before = h_before,
                entropy_after  = h_after,
            ))
            last = t

        return cps
