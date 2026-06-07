"""
ConsciousAI — Connectivity Learner
====================================

Learns the connectivity matrix C from data automatically.
This is the key component that makes ConsciousAI fully unsupervised
and competitive with deep learning methods — no domain expertise needed.

Four methods, each optimal for a different domain:

  pearson     — fast, baseline, any domain
  granger     — causal, financial / time series
  mutual_info — non-linear, biological / neural
  attention   — from transformer weights, LLMs

Author: Walter Calmels
"""

import numpy as np
import warnings
from typing import Optional
warnings.filterwarnings("ignore")


# ─── Optional heavy deps ─────────────────────────────────────────────────────
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

try:
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# =============================================================================
#  CONNECTIVITY LEARNER
# =============================================================================

class ConnectivityLearner:
    """
    Learns connectivity matrix C(N×N) from multivariate time-series data.

    C[i,j] ∈ [0,1] represents how strongly component j influences component i.
    The resulting C is passed directly to ConsciousAI's Φ calculator.

    Usage
    -----
    learner = ConnectivityLearner(method='pearson')
    C = learner.fit(data)          # data: (T, N) array
    phi = engine.calculate_phi(data, connectivity=C)
    """

    METHODS = ("pearson", "granger", "mutual_info", "attention", "ensemble")

    def __init__(
        self,
        method: str = "pearson",
        threshold: float = 0.1,
        max_lag: int = 5,
        significance: float = 0.05,
        normalise: bool = True,
    ):
        """
        Parameters
        ----------
        method      : 'pearson' | 'granger' | 'mutual_info' | 'attention' | 'ensemble'
        threshold   : Zero out edges weaker than this (sparsity)
        max_lag     : Max lag for Granger causality
        significance: p-value cutoff for Granger
        normalise   : Scale C to [0,1]
        """
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self.threshold = threshold
        self.max_lag = max_lag
        self.significance = significance
        self.normalise = normalise

        self.C_: Optional[np.ndarray] = None
        self.N_: Optional[int] = None

    # ── public API ─────────────────────────────────────────────────────────
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Learn connectivity from data.

        Parameters
        ----------
        data : (T, N) float array — T time steps, N components

        Returns
        -------
        C : (N, N) float array in [0,1]
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        T, N = data.shape
        self.N_ = N

        if self.method == "pearson":
            C = self._pearson(data)
        elif self.method == "granger":
            C = self._granger(data)
        elif self.method == "mutual_info":
            C = self._mutual_info(data)
        elif self.method == "attention":
            raise ValueError("Use fit_from_attention(weights) for attention method")
        elif self.method == "ensemble":
            C = self._ensemble(data)

        C = self._postprocess(C)
        self.C_ = C
        return C

    def fit_from_attention(self, attention_weights: np.ndarray) -> np.ndarray:
        """
        Build C from transformer attention weights.

        Parameters
        ----------
        attention_weights : (H, N, N) — H heads, N tokens/components
                            or (N, N) averaged across heads

        Returns
        -------
        C : (N, N) float array
        """
        if attention_weights.ndim == 3:
            C = attention_weights.mean(axis=0)
        else:
            C = attention_weights.copy()

        C = self._postprocess(C)
        self.C_ = C
        return C

    def get_top_edges(self, k: int = 10):
        """Return the k strongest connections as (i, j, weight) tuples."""
        if self.C_ is None:
            raise RuntimeError("Call fit() first")
        C = self.C_.copy()
        np.fill_diagonal(C, 0)
        idx = np.unravel_index(np.argsort(C.ravel())[::-1][:k], C.shape)
        return [(int(i), int(j), float(C[i, j])) for i, j in zip(*idx)]

    # ── private methods ─────────────────────────────────────────────────────
    def _pearson(self, data: np.ndarray) -> np.ndarray:
        """Absolute Pearson correlation — fast, symmetric."""
        C = np.abs(np.corrcoef(data.T))
        np.fill_diagonal(C, 0)
        return C

    def _granger(self, data: np.ndarray) -> np.ndarray:
        """
        Granger causality — asymmetric, causal direction preserved.
        C[i,j] = max F-stat (over lags) where j Granger-causes i,
                 set to 0 if not significant.
        """
        if not STATSMODELS_OK:
            print("⚠️  statsmodels not available — falling back to Pearson")
            return self._pearson(data)

        T, N = data.shape
        if T < 3 * self.max_lag:
            print("⚠️  Too few samples for Granger — using Pearson")
            return self._pearson(data)

        C = np.zeros((N, N))
        data_norm = (data - data.mean(0)) / (data.std(0) + 1e-10)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                try:
                    pair = np.column_stack([data_norm[:, i], data_norm[:, j]])
                    res = grangercausalitytests(pair, maxlag=self.max_lag, verbose=False)
                    # Use minimum p-value across lags
                    pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, self.max_lag + 1)]
                    fstats = [res[lag][0]["ssr_ftest"][0] for lag in range(1, self.max_lag + 1)]
                    min_p = min(pvals)
                    max_f = max(fstats)
                    if min_p < self.significance:
                        C[i, j] = max_f / (max_f + 1)   # normalise F-stat to (0,1)
                except Exception:
                    pass

        return C

    def _mutual_info(self, data: np.ndarray) -> np.ndarray:
        """
        Mutual information — captures non-linear relationships.
        Good for biological / neural data.
        """
        if not SKLEARN_OK:
            print("⚠️  sklearn not available — falling back to Pearson")
            return self._pearson(data)

        T, N = data.shape
        C = np.zeros((N, N))

        for j in range(N):
            target = data[:, j]
            others_idx = [i for i in range(N) if i != j]
            others = data[:, others_idx]
            try:
                mi = mutual_info_regression(others, target, random_state=42)
                for k, i in enumerate(others_idx):
                    C[i, j] = mi[k]
            except Exception:
                pass

        return C

    def _ensemble(self, data: np.ndarray) -> np.ndarray:
        """
        Ensemble of Pearson + MI (+ Granger if available).
        More robust across domains.
        """
        C_pearson = self._pearson(data)
        C_mi = self._mutual_info(data)

        if STATSMODELS_OK and data.shape[0] >= 3 * self.max_lag:
            C_granger = self._granger(data)
            # Normalise each to [0,1] before averaging
            matrices = [C_pearson, C_mi, C_granger]
        else:
            matrices = [C_pearson, C_mi]

        # Normalise each
        normed = []
        for M in matrices:
            mx = M.max()
            normed.append(M / mx if mx > 0 else M)

        return np.mean(normed, axis=0)

    def _postprocess(self, C: np.ndarray) -> np.ndarray:
        """Normalise, threshold, ensure non-negative."""
        C = np.abs(C)
        np.fill_diagonal(C, 0)

        if self.normalise and C.max() > 0:
            C = C / C.max()

        C[C < self.threshold] = 0.0
        return C

    # ── convenience ────────────────────────────────────────────────────────
    @staticmethod
    def domain_default(domain: str) -> "ConnectivityLearner":
        """
        Returns pre-configured learner for common domains.

        domain : 'autonomous' | 'biomedical' | 'financial' | 'llm' | 'materials'
        """
        configs = {
            "autonomous":  dict(method="pearson",     threshold=0.15),
            "biomedical":  dict(method="mutual_info", threshold=0.10),
            "financial":   dict(method="granger",     threshold=0.05, max_lag=3),
            "llm":         dict(method="pearson",     threshold=0.05),
            "materials":   dict(method="ensemble",    threshold=0.10),
        }
        cfg = configs.get(domain, dict(method="pearson", threshold=0.10))
        return ConnectivityLearner(**cfg)
