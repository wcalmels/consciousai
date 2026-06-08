# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai

"""
AdaptiveThreshold
==================
Dynamic alert thresholds derived from the system's own historical
Φ̃ distribution, eliminating manual threshold tuning and seasonal
false-positive spikes.

Problem with fixed thresholds (Φ < 0.3 = alert):
  - Different systems have different baseline Φ̃ ranges
  - Systems have seasonal / operational patterns (day/night, shift changes)
  - A fixed threshold produces floods of false positives during
    normal low-integration periods (e.g. shutdown, maintenance windows)

Solution:
  - Model the Φ̃ distribution as a rolling parametric or non-parametric
    distribution
  - Alert only when Φ̃ falls below the empirical p-th percentile of
    recent history (contextual anomaly)
  - Separate profiles for different operational modes (if labels available)

Methods available:
  percentile   — rolling empirical quantile (robust, no assumption)
  gaussian     — rolling μ − k·σ (fast, assumes normality)
  kde          — kernel density estimate (flexible, slower)

Author: Walter Calmels — ConsciousAI Suite
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from collections import deque

try:
    from numba import jit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def jit(*a, **k): return (lambda f: f) if not a else a[0]


# ── JIT kernels ───────────────────────────────────────────────────────────────

@jit(nopython=True, fastmath=True)
def _phi_for_window(data: np.ndarray, tol: float = 1e-10) -> float:
    n, m = data.shape
    if n < 2:
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
    total = 0.0; count = 0
    for v in eigs:
        av = abs(v)
        if av > tol:
            total += av; count += 1
    if count == 0 or total < tol:
        return 0.0
    ent = 0.0
    for v in eigs:
        av = abs(v)
        if av > tol:
            p = av / total
            if p > 1e-15:
                ent -= p * np.log(p)
    return max(0.0, ent * count)


# Warm-up
_phi_for_window(np.random.rand(10, 3).astype(np.float64))


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class Alert:
    time_index: int
    phi_value:  float
    threshold:  float
    severity:   str     # 'warning' | 'critical'
    z_score:    float


@dataclass
class ThresholdResult:
    phi_series:         np.ndarray
    threshold_series:   np.ndarray   # dynamic lower threshold
    upper_series:       np.ndarray   # dynamic upper threshold (for over-integration)
    alerts:             List[Alert]
    alert_mask:         np.ndarray   # bool mask
    false_positive_est: float        # estimated FPR on in-sample data


# ── Main class ────────────────────────────────────────────────────────────────

class AdaptiveThreshold:
    """
    Adaptive alert thresholds for Φ̃ streams.

    Parameters
    ----------
    phi_window   : Window for Φ̃ computation
    history_len  : How many past Φ̃ values to model
    method       : 'percentile' | 'gaussian' | 'kde'
    alert_pct    : Alert when Φ̃ < this percentile of history (lower tail)
    critical_pct : Critical alert when Φ̃ < this percentile
    sigma        : For gaussian method: alert at μ - sigma*σ
    seasonal_period : If set, apply seasonal adjustment (e.g. 24 for hourly data)

    Usage
    -----
    at = AdaptiveThreshold(phi_window=30, history_len=200, alert_pct=5)

    # Batch mode
    result = at.fit(data)
    print(f"{len(result.alerts)} alerts, FPR ≈ {result.false_positive_est:.2%}")

    # Online mode
    at.update(new_observation)
    alert = at.check(new_observation)
    """

    def __init__(self,
                 phi_window:      int   = 30,
                 phi_step:        int   = 1,
                 history_len:     int   = 200,
                 method:          str   = "percentile",
                 alert_pct:       float = 5.0,
                 critical_pct:    float = 1.0,
                 sigma:           float = 2.5,
                 seasonal_period: Optional[int] = None):

        self.phi_window      = phi_window
        self.phi_step        = phi_step
        self.history_len     = history_len
        self.method          = method
        self.alert_pct       = alert_pct
        self.critical_pct    = critical_pct
        self.sigma           = sigma
        self.seasonal_period = seasonal_period

        # Online state
        self._phi_buffer: deque = deque(maxlen=history_len)
        self._seasonal_profiles: Dict[int, List[float]] = {}

    # ── batch mode ────────────────────────────────────────────────────────

    def fit(self, data: np.ndarray) -> ThresholdResult:
        """
        Compute adaptive thresholds and detect alerts on historical data.

        Parameters
        ----------
        data : (T, N) float array

        Returns
        -------
        ThresholdResult
        """
        data = np.asarray(data, dtype=np.float64)
        T, N = data.shape

        # Build Φ̃ series
        phi_series = []
        for start in range(0, T - self.phi_window + 1, self.phi_step):
            chunk = data[start: start + self.phi_window]
            phi_series.append(_phi_for_window(chunk))

        phi_arr = np.array(phi_series)
        K       = len(phi_arr)

        # Build rolling thresholds
        lower_thr = np.zeros(K)
        upper_thr = np.zeros(K)
        burn_in   = min(self.history_len, K // 4)

        for t in range(K):
            hist_start = max(0, t - self.history_len)
            window     = phi_arr[hist_start: t] if t > 0 else phi_arr[:1]

            if len(window) < 4:
                lower_thr[t] = phi_arr[:max(1, t + 1)].min()
                upper_thr[t] = phi_arr[:max(1, t + 1)].max()
                continue

            lower_thr[t], upper_thr[t] = self._compute_bounds(window, t)

        # Detect alerts
        alerts     = []
        alert_mask = np.zeros(K, dtype=bool)

        for t in range(burn_in, K):
            phi = phi_arr[t]
            thr = lower_thr[t]
            if phi < thr:
                z = (phi - phi_arr[max(0, t - self.history_len): t].mean()) / \
                    (phi_arr[max(0, t - self.history_len): t].std() + 1e-6)

                # Determine severity
                hist = phi_arr[max(0, t - self.history_len): t]
                crit_thr = np.percentile(hist, self.critical_pct) if len(hist) > 1 else thr
                severity = "critical" if phi < crit_thr else "warning"

                alerts.append(Alert(
                    time_index = int(t * self.phi_step + self.phi_window // 2),
                    phi_value  = float(phi),
                    threshold  = float(thr),
                    severity   = severity,
                    z_score    = float(z),
                ))
                alert_mask[t] = True

        # Estimate in-sample FPR
        n_alerts   = alert_mask[burn_in:].sum()
        fpr_est    = n_alerts / max(1, K - burn_in)

        return ThresholdResult(
            phi_series         = phi_arr,
            threshold_series   = lower_thr,
            upper_series       = upper_thr,
            alerts             = alerts,
            alert_mask         = alert_mask,
            false_positive_est = float(fpr_est),
        )

    # ── online mode ───────────────────────────────────────────────────────

    def update(self, phi_value: float):
        """Add a new Φ̃ observation to the history buffer."""
        self._phi_buffer.append(phi_value)

    def check(self, phi_value: float,
              time_index: int = 0) -> Optional[Alert]:
        """
        Check if a new Φ̃ value triggers an alert given current history.
        Returns Alert or None.
        """
        self.update(phi_value)

        if len(self._phi_buffer) < 4:
            return None

        hist = np.array(list(self._phi_buffer)[:-1])
        lower, upper = self._compute_bounds(hist, len(hist))

        if phi_value < lower:
            mu  = hist.mean()
            sg  = hist.std() + 1e-6
            z   = (phi_value - mu) / sg
            crit = np.percentile(hist, self.critical_pct)
            return Alert(
                time_index = time_index,
                phi_value  = float(phi_value),
                threshold  = float(lower),
                severity   = "critical" if phi_value < crit else "warning",
                z_score    = float(z),
            )
        return None

    # ── seasonal profiling ────────────────────────────────────────────────

    def build_seasonal_profile(self, phi_series: np.ndarray) -> None:
        """
        Build per-period Φ̃ profiles for seasonal adjustment.
        After calling this, fit() will use period-specific thresholds.

        Parameters
        ----------
        phi_series      : Pre-computed (K,) Φ̃ series
        """
        if self.seasonal_period is None:
            return

        for t, phi in enumerate(phi_series):
            period_bin = t % self.seasonal_period
            if period_bin not in self._seasonal_profiles:
                self._seasonal_profiles[period_bin] = []
            self._seasonal_profiles[period_bin].append(phi)

    def seasonal_threshold(self, time_index: int) -> Optional[float]:
        """Get seasonal lower threshold for a given time index."""
        if not self._seasonal_profiles or self.seasonal_period is None:
            return None
        bin_ = time_index % self.seasonal_period
        if bin_ not in self._seasonal_profiles or len(self._seasonal_profiles[bin_]) < 4:
            return None
        return float(np.percentile(self._seasonal_profiles[bin_], self.alert_pct))

    # ── private ───────────────────────────────────────────────────────────

    def _compute_bounds(self, history: np.ndarray,
                        t: int) -> Tuple[float, float]:
        """Compute lower and upper thresholds from history."""
        if self.method == "percentile":
            lower = float(np.percentile(history, self.alert_pct))
            upper = float(np.percentile(history, 100 - self.alert_pct))

        elif self.method == "gaussian":
            mu = float(history.mean())
            sg = max(float(history.std()), 1e-6)
            lower = mu - self.sigma * sg
            upper = mu + self.sigma * sg

        elif self.method == "kde":
            try:
                from scipy.stats import gaussian_kde
                kde    = gaussian_kde(history, bw_method="silverman")
                grid   = np.linspace(history.min(), history.max(), 200)
                cdf    = np.cumsum(kde(grid))
                cdf   /= cdf[-1]
                lower  = float(grid[np.searchsorted(cdf, self.alert_pct / 100)])
                upper  = float(grid[np.searchsorted(cdf, 1 - self.alert_pct / 100)])
            except ImportError:
                lower = float(np.percentile(history, self.alert_pct))
                upper = float(np.percentile(history, 100 - self.alert_pct))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Seasonal override if available
        seasonal = self.seasonal_threshold(t)
        if seasonal is not None:
            lower = min(lower, seasonal)

        return lower, upper

    # ── summary ───────────────────────────────────────────────────────────

    @staticmethod
    def compare_methods(data: np.ndarray,
                        phi_window: int = 30) -> dict:
        """
        Compare all three threshold methods on the same data.
        Useful for choosing the best method for a given system.

        Returns dict with FPR and n_alerts for each method.
        """
        results = {}
        for method in ("percentile", "gaussian", "kde"):
            at = AdaptiveThreshold(phi_window=phi_window, method=method)
            try:
                r = at.fit(data)
                results[method] = {
                    "n_alerts":   len(r.alerts),
                    "n_critical": sum(1 for a in r.alerts if a.severity == "critical"),
                    "fpr_est":    r.false_positive_est,
                }
            except Exception as e:
                results[method] = {"error": str(e)}
        return results
