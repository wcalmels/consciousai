# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai

"""
TemporalIntegration
====================
Multi-scale spectral integration analysis.

Computes Φ̃ at multiple time scales simultaneously, producing an
"integration spectrogram" — analogous to wavelet analysis but measuring
how system coherence evolves across temporal resolutions.

Key insight: different system pathologies manifest at different time scales.
  - Fast faults (sensor dropout):    visible at short windows (5-10 steps)
  - Slow drift (calibration loss):   visible at medium windows (50-100 steps)
  - Structural degradation:          visible at long windows (500+ steps)

Applications:
  - EEG/neurophysiology (delta/theta/alpha/beta band integration)
  - Industrial predictive maintenance (bearing wear vs. shaft misalignment)
  - Financial: intraday vs. daily vs. weekly regime coherence
  - Climate: multi-scale teleconnection detection

Author: Walter Calmels — ConsciousAI Suite
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    from numba import jit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def jit(*a, **k): return (lambda f: f) if not a else a[0]
    def prange(*a): return range(*a)


# ── JIT kernel ────────────────────────────────────────────────────────────────

@jit(nopython=True, fastmath=True, parallel=True)
def _multiscale_phi(data: np.ndarray,
                    windows: np.ndarray,
                    step: int,
                    tol: float = 1e-10) -> np.ndarray:
    """
    Compute rolling Φ̃ for multiple window sizes in parallel.

    Returns
    -------
    out : (n_windows, n_scales) — Φ̃ value for each (window_position, scale)
    """
    T, N = data.shape
    S    = len(windows)
    max_w = windows[-1]   # largest window

    # Number of output positions (governed by smallest window)
    min_w   = windows[0]
    n_steps = max(1, (T - min_w) // step + 1)

    out = np.full((n_steps, S), 0.0)

    for s in prange(S):
        w = windows[s]
        for idx in range(n_steps):
            t = idx * step
            if t + w > T:
                break
            chunk = data[t: t + w]
            n, m  = chunk.shape

            # Covariance
            means = np.zeros(m)
            for j in range(m):
                sm = 0.0
                for i in range(n):
                    sm += chunk[i, j]
                means[j] = sm / n

            cov   = np.zeros((m, m))
            denom = float(n - 1) if n > 1 else 1.0
            for i in range(m):
                for j in range(i, m):
                    sm = 0.0
                    for k in range(n):
                        sm += (chunk[k, i] - means[i]) * (chunk[k, j] - means[j])
                    sm /= denom
                    cov[i, j] = sm
                    cov[j, i] = sm

            eigs  = np.linalg.eigvalsh(cov)
            total = 0.0
            count = 0
            for v in eigs:
                av = abs(v)
                if av > tol:
                    total += av
                    count += 1

            if count == 0 or total < tol:
                out[idx, s] = 0.0
                continue

            ent = 0.0
            for v in eigs:
                av = abs(v)
                if av > tol:
                    p = av / total
                    if p > 1e-15:
                        ent -= p * np.log(p)

            out[idx, s] = max(0.0, ent * count)

    return out


# Warm-up
_d = np.random.rand(30, 4).astype(np.float64)
_w = np.array([5, 10, 20], dtype=np.int64)
_multiscale_phi(_d, _w, 2)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TemporalIntegrationResult:
    spectrogram:    np.ndarray      # (K, S) — Φ̃ at each (time, scale)
    windows:        np.ndarray      # (S,) scale lengths
    time_axis:      np.ndarray      # (K,) time positions
    dominant_scale: int             # window size with highest mean Φ̃
    scale_entropy:  np.ndarray      # (K,) entropy across scales at each time


# ── Main class ────────────────────────────────────────────────────────────────

class TemporalIntegration:
    """
    Multi-scale spectral integration analysis.

    Parameters
    ----------
    windows : list of window sizes (scales). Default: geometric from 5 to 100
    step    : step between time positions

    Usage
    -----
    ti = TemporalIntegration(windows=[5, 10, 20, 50, 100])
    result = ti.analyse(data)

    # result.spectrogram[t, s] = Φ̃ at time t, scale s
    # result.dominant_scale = window size where integration is highest
    """

    def __init__(self,
                 windows: Optional[List[int]] = None,
                 step: int = 5):
        if windows is None:
            # Geometric sequence from 5 to 100
            windows = [5, 10, 20, 30, 50, 75, 100]
        self.windows = sorted(windows)
        self.step    = step

    # ── main analysis ─────────────────────────────────────────────────────

    def analyse(self, data: np.ndarray) -> TemporalIntegrationResult:
        """
        Compute integration spectrogram.

        Parameters
        ----------
        data : (T, N) float array. T should be >> max(windows).

        Returns
        -------
        TemporalIntegrationResult
        """
        data    = np.asarray(data, dtype=np.float64)
        T, N    = data.shape
        win_arr = np.array(self.windows, dtype=np.int64)

        # Validate
        usable = [w for w in self.windows if w < T]
        if not usable:
            raise ValueError(f"All windows >= T={T}. Reduce window sizes.")
        if len(usable) < len(self.windows):
            win_arr = np.array(usable, dtype=np.int64)

        # Compute
        spectrogram = _multiscale_phi(data, win_arr, self.step)

        # Time axis (governed by smallest window)
        min_w     = int(win_arr[0])
        n_steps   = spectrogram.shape[0]
        time_axis = np.array([i * self.step + min_w // 2
                               for i in range(n_steps)])

        # Dominant scale: window with highest mean Φ̃
        mean_by_scale = spectrogram.mean(axis=0)
        dominant_idx  = int(np.argmax(mean_by_scale))
        dominant_scale = int(win_arr[dominant_idx])

        # Scale entropy at each time point
        # (how spread-out is the Φ̃ across scales?)
        eps = 1e-10
        scale_entropy = np.zeros(n_steps)
        for t in range(n_steps):
            row   = spectrogram[t] + eps
            p     = row / row.sum()
            scale_entropy[t] = -np.sum(p * np.log(p))

        return TemporalIntegrationResult(
            spectrogram    = spectrogram,
            windows        = win_arr,
            time_axis      = time_axis,
            dominant_scale = dominant_scale,
            scale_entropy  = scale_entropy,
        )

    # ── cross-scale coherence ─────────────────────────────────────────────

    def cross_scale_coherence(self, data: np.ndarray) -> np.ndarray:
        """
        Compute cross-scale coherence matrix C[s1, s2] = correlation
        between Φ̃ series at scales s1 and s2.

        High coherence = fault appears at multiple scales simultaneously.
        Low coherence = fault is scale-specific.

        Returns
        -------
        C : (S, S) correlation matrix
        """
        result = self.analyse(data)
        return np.corrcoef(result.spectrogram.T)

    # ── anomaly score from multiscale ─────────────────────────────────────

    def anomaly_score(self, data: np.ndarray,
                      baseline_frac: float = 0.2) -> np.ndarray:
        """
        Aggregate multi-scale Φ̃ into a single anomaly score using
        the maximum z-score across scales at each time point.

        Returns
        -------
        scores : (K,) anomaly scores (higher = more anomalous)
        """
        result   = self.analyse(data)
        spect    = result.spectrogram
        K, S     = spect.shape
        n_base   = max(2, int(K * baseline_frac))

        scores = np.zeros(K)
        for s in range(S):
            col    = spect[:, s]
            mu     = col[:n_base].mean()
            sg     = max(col[:n_base].std(), 1e-6)
            z      = np.abs(col - mu) / sg
            scores = np.maximum(scores, z)

        return scores

    # ── integration profile summary ───────────────────────────────────────

    def profile(self, data: np.ndarray) -> dict:
        """
        Summary statistics of multi-scale integration.

        Returns
        -------
        dict with keys:
          mean_phi_by_scale, std_phi_by_scale,
          dominant_scale, peak_integration_time,
          scale_bandwidth (how many scales are active)
        """
        result = self.analyse(data)
        spect  = result.spectrogram

        mean_by_scale = spect.mean(axis=0)
        std_by_scale  = spect.std(axis=0)
        threshold     = mean_by_scale.mean()
        n_active      = int(np.sum(mean_by_scale > threshold))

        return {
            "mean_phi_by_scale":    mean_by_scale,
            "std_phi_by_scale":     std_by_scale,
            "dominant_scale":       result.dominant_scale,
            "peak_integration_time": int(result.time_axis[np.argmax(spect.max(axis=1))]),
            "scale_bandwidth":      n_active,
            "windows":              result.windows,
        }
