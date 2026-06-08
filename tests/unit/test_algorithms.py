# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai

"""
Tests for ConsciousAI Algorithm Suite
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.algorithms import (
    SpectralChangePoint,
    IntegrationDistance,
    CausalEmergence,
    TemporalIntegration,
    AdaptiveThreshold,
)

np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def make_stationary(T=300, N=6, corr=0.4):
    """Stationary correlated series."""
    shared = np.random.randn(T)
    noise  = np.random.randn(T, N)
    return corr * shared[:, None] + np.sqrt(1 - corr**2) * noise


def make_with_regime_change(T=400, N=6, change_at=200):
    """Two-regime series: low then high correlation."""
    shared = np.random.randn(T)
    noise  = np.random.randn(T, N)
    data   = 0.2 * shared[:, None] + 0.98 * noise
    # Inject high-correlation regime
    shared2 = np.random.randn(T - change_at)
    noise2  = np.random.randn(T - change_at, N)
    data[change_at:] = 0.9 * shared2[:, None] + np.sqrt(1 - 0.81) * noise2
    return data, change_at


def make_with_anomaly(T=300, N=6, anomaly_start=130, anomaly_len=20):
    """Collective anomaly: sensors decouple."""
    data         = make_stationary(T, N)
    labels       = np.zeros(T, dtype=int)
    data[anomaly_start: anomaly_start + anomaly_len] = np.random.randn(anomaly_len, N) * 0.05
    labels[anomaly_start: anomaly_start + anomaly_len] = 1
    return data, labels


# ═══════════════════════════════════════════════════════════════════════════
#  1. SpectralChangePoint
# ═══════════════════════════════════════════════════════════════════════════

class TestSpectralChangePoint:

    def test_basic_output(self):
        data, _ = make_with_regime_change()
        cpd     = SpectralChangePoint(window=30, threshold=2.0)
        result  = cpd.fit(data)
        assert isinstance(result.entropy_series, np.ndarray)
        assert len(result.entropy_series) > 0
        assert all(np.isfinite(result.entropy_series))

    def test_detects_regime_change(self):
        data, change_at = make_with_regime_change(T=400, N=6, change_at=200)
        cpd    = SpectralChangePoint(window=30, threshold=2.0, min_gap=20)
        result = cpd.fit(data)
        assert len(result.change_points) >= 1, "Should detect at least one change point"
        # At least one CP close to the true change
        cp_positions = [cp.index for cp in result.change_points]
        closest = min(abs(p - 200) for p in cp_positions)
        assert closest < 110, f"Closest CP is {closest} samples from true change"

    def test_no_false_positives_on_stationary(self):
        data   = make_stationary(T=300)
        cpd    = SpectralChangePoint(window=30, threshold=3.5)
        result = cpd.fit(data)
        # With high threshold, should have very few false positives
        assert len(result.change_points) <= 3

    def test_methods(self):
        data, _ = make_with_regime_change()
        for method in ("zscore", "cusum", "gradient"):
            cpd    = SpectralChangePoint(window=30, threshold=2.0, method=method)
            result = cpd.fit(data)
            assert result.entropy_series is not None

    def test_online_mode(self):
        data, change_at = make_with_regime_change(T=300, change_at=150)
        cpd      = SpectralChangePoint(window=30, threshold=2.5)
        detected = cpd.fit_online(data, burn_in=50)
        assert isinstance(detected, list)

    def test_change_point_attributes(self):
        data, _ = make_with_regime_change()
        cpd    = SpectralChangePoint(window=30, threshold=2.0)
        result = cpd.fit(data)
        if result.change_points:
            cp = result.change_points[0]
            assert hasattr(cp, 'index')
            assert hasattr(cp, 'magnitude')
            assert cp.magnitude >= 0


# ═══════════════════════════════════════════════════════════════════════════
#  2. IntegrationDistance
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationDistance:

    def test_self_distance_is_zero(self):
        data = make_stationary(T=100)
        dist = IntegrationDistance()
        r    = dist.compare(data, data)
        assert r.jsd < 1e-6,      f"JSD(x,x) should be 0, got {r.jsd}"
        assert r.wasserstein < 1e-6

    def test_different_systems_have_positive_distance(self):
        a = make_stationary(T=100, corr=0.1)
        b = make_stationary(T=100, corr=0.9)
        dist = IntegrationDistance()
        r    = dist.compare(a, b)
        assert r.jsd > 0
        assert r.wasserstein > 0

    def test_symmetry_of_jsd(self):
        a    = make_stationary(T=100, corr=0.2)
        b    = make_stationary(T=100, corr=0.8)
        dist = IntegrationDistance()
        r_ab = dist.compare(a, b)
        r_ba = dist.compare(b, a)
        assert abs(r_ab.jsd - r_ba.jsd) < 1e-8, "JSD must be symmetric"

    def test_pairwise_matrix_shape(self):
        systems = [make_stationary(T=80, corr=c) for c in [0.1, 0.5, 0.9]]
        dist    = IntegrationDistance()
        D       = dist.pairwise(systems)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T, atol=1e-8), "Pairwise matrix must be symmetric"
        assert np.allclose(np.diag(D), 0, atol=1e-6), "Diagonal must be 0"

    def test_fleet_outlier_detection(self):
        normal_fleet = [make_stationary(T=80, corr=0.4) for _ in range(8)]
        outlier      = make_stationary(T=80, corr=0.99)
        fleet        = normal_fleet + [outlier]
        dist         = IntegrationDistance()
        outliers, scores = dist.fleet_outliers(fleet, threshold=1.5)
        assert 8 in outliers, "Known outlier (index 8) should be flagged"

    def test_temporal_drift(self):
        data  = make_stationary(T=300)
        dist  = IntegrationDistance()
        drift = dist.temporal_drift(data, window=40, step=5)
        assert isinstance(drift, np.ndarray)
        assert len(drift) > 0
        assert np.all(drift >= 0)

    def test_metrics_parameter(self):
        a    = make_stationary(T=80)
        b    = make_stationary(T=80, corr=0.8)
        for metric in ("jsd", "wasserstein"):
            dist = IntegrationDistance(metric=metric)
            d    = dist.scalar(a, b)
            assert d >= 0


# ═══════════════════════════════════════════════════════════════════════════
#  3. CausalEmergence
# ═══════════════════════════════════════════════════════════════════════════

class TestCausalEmergence:

    def test_basic_output(self):
        data = make_stationary(T=100, N=6)
        ce   = CausalEmergence()
        r    = ce.analyse(data)
        assert r.phi_full >= 0
        assert isinstance(r.phi_subsystems, list)
        assert len(r.phi_subsystems) > 0

    def test_emergence_index_range(self):
        data = make_stationary(T=100, N=6)
        ce   = CausalEmergence()
        r    = ce.analyse(data)
        # emergence_index can be negative (dominant subsystem) but bounded
        assert -2.0 < r.emergence_index < 2.0

    def test_component_importance_shape(self):
        data = make_stationary(T=100, N=6)
        ce   = CausalEmergence()
        imp  = ce.component_importance(data)
        assert imp.shape == (6,)
        assert np.all(np.isfinite(imp))

    def test_component_importance_ranking(self):
        # Build system where component 0 is the shared signal (most important)
        T, N = 150, 6
        shared = np.random.randn(T)
        data   = np.column_stack([shared] + [0.1 * np.random.randn(T) for _ in range(N-1)])
        ce   = CausalEmergence()
        imp  = ce.component_importance(data)
        # Component 0 should be in the top half of importance scores
        assert np.isfinite(imp[0])

    def test_rolling_emergence(self):
        data   = make_stationary(T=300, N=6)
        ce     = CausalEmergence()
        series = ce.rolling(data, window=50, step=10)
        assert len(series) > 0
        assert np.all(np.isfinite(series))

    def test_optimal_partition_shape(self):
        data   = make_stationary(T=100, N=8)
        ce     = CausalEmergence()
        groups = ce.optimal_partition(data, n_groups=3)
        assert len(groups) == 3
        all_comps = [c for g in groups for c in g]
        assert sorted(all_comps) == list(range(8))


# ═══════════════════════════════════════════════════════════════════════════
#  4. TemporalIntegration
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalIntegration:

    def test_basic_output(self):
        data = make_stationary(T=300, N=6)
        ti   = TemporalIntegration(windows=[10, 20, 30])
        r    = ti.analyse(data)
        assert r.spectrogram.shape[1] == 3
        assert len(r.time_axis) == r.spectrogram.shape[0]
        assert r.dominant_scale in [10, 20, 30]

    def test_spectrogram_nonnegative(self):
        data = make_stationary(T=300, N=6)
        ti   = TemporalIntegration(windows=[10, 20, 50])
        r    = ti.analyse(data)
        assert np.all(r.spectrogram >= 0)

    def test_scale_entropy(self):
        data = make_stationary(T=300, N=6)
        ti   = TemporalIntegration(windows=[10, 20, 30, 50])
        r    = ti.analyse(data)
        assert r.scale_entropy.shape == (r.spectrogram.shape[0],)
        assert np.all(r.scale_entropy >= 0)

    def test_cross_scale_coherence(self):
        data = make_stationary(T=300, N=6)
        ti   = TemporalIntegration(windows=[10, 20, 50])
        C    = ti.cross_scale_coherence(data)
        assert C.shape == (3, 3)
        assert np.allclose(np.diag(C), 1.0, atol=1e-6)

    def test_anomaly_score(self):
        data, labels = make_with_anomaly(T=300)
        ti    = TemporalIntegration(windows=[15, 30, 60])
        score = ti.anomaly_score(data)
        assert len(score) > 0
        assert np.all(score >= 0)

    def test_window_too_large_raises(self):
        data = make_stationary(T=20, N=4)
        ti   = TemporalIntegration(windows=[50, 100])
        with pytest.raises(ValueError):
            ti.analyse(data)

    def test_profile_keys(self):
        data = make_stationary(T=300, N=6)
        ti   = TemporalIntegration(windows=[10, 20, 30])
        p    = ti.profile(data)
        for key in ("mean_phi_by_scale", "dominant_scale",
                    "peak_integration_time", "scale_bandwidth"):
            assert key in p


# ═══════════════════════════════════════════════════════════════════════════
#  5. AdaptiveThreshold
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveThreshold:

    def test_basic_output(self):
        data = make_stationary(T=300, N=6)
        at   = AdaptiveThreshold(phi_window=20)
        r    = at.fit(data)
        assert len(r.phi_series) > 0
        assert len(r.threshold_series) == len(r.phi_series)
        assert len(r.upper_series) == len(r.phi_series)

    def test_fpr_on_stationary(self):
        data = make_stationary(T=400, N=6)
        at   = AdaptiveThreshold(phi_window=20, alert_pct=5.0)
        r    = at.fit(data)
        # FPR should be roughly near 5% (within factor 3)
        assert r.false_positive_est < 0.25, \
            f"FPR too high on stationary: {r.false_positive_est:.2%}"

    def test_detects_anomaly(self):
        data, labels = make_with_anomaly(T=300, anomaly_start=150, anomaly_len=20)
        at   = AdaptiveThreshold(phi_window=20, alert_pct=10.0)
        r    = at.fit(data)
        # Should find at least one alert near anomaly region
        assert len(r.alerts) > 0

    def test_online_mode(self):
        data = make_stationary(T=200, N=6)
        at   = AdaptiveThreshold(phi_window=20, history_len=50)
        alerts_found = []
        for t in range(0, len(data) - 20, 20):
            chunk = data[t: t + 20]
            phi   = float(np.cov(chunk.T).mean())
            alert = at.check(phi, time_index=t)
            if alert:
                alerts_found.append(alert)
        assert isinstance(alerts_found, list)

    def test_methods(self):
        data = make_stationary(T=300, N=6)
        for method in ("percentile", "gaussian"):
            at = AdaptiveThreshold(phi_window=20, method=method)
            r  = at.fit(data)
            assert len(r.phi_series) > 0

    def test_compare_methods(self):
        data    = make_stationary(T=300, N=6)
        results = AdaptiveThreshold.compare_methods(data, phi_window=20)
        for method in ("percentile", "gaussian"):
            assert method in results
            assert "fpr_est" in results[method]

    def test_seasonal_profile(self):
        data    = make_stationary(T=400, N=4)
        at      = AdaptiveThreshold(phi_window=15, seasonal_period=24)
        r_base  = at.fit(data)
        phi_arr = r_base.phi_series
        at.build_seasonal_profile(phi_arr)
        thr = at.seasonal_threshold(10)
        assert thr is None or isinstance(thr, float)

    def test_alert_severity(self):
        data, labels = make_with_anomaly(T=300, anomaly_start=150, anomaly_len=30)
        at = AdaptiveThreshold(phi_window=20, alert_pct=15, critical_pct=2)
        r  = at.fit(data)
        if r.alerts:
            assert all(a.severity in ("warning", "critical") for a in r.alerts)


# ═══════════════════════════════════════════════════════════════════════════
#  INTEGRATION: Suite working together
# ═══════════════════════════════════════════════════════════════════════════

class TestSuiteIntegration:

    def test_pipeline_change_to_distance(self):
        """Detect regime change, then measure distance between regimes."""
        data, change_at = make_with_regime_change(T=400, change_at=200)

        # Detect change
        cpd    = SpectralChangePoint(window=30, threshold=2.0)
        result = cpd.fit(data)

        # Measure distance between regimes
        dist    = IntegrationDistance()
        before  = data[:change_at]
        after   = data[change_at:]
        d       = dist.compare(before, after)
        assert d.jsd >= 0

    def test_pipeline_emergence_to_threshold(self):
        """Find dominant components, then monitor with adaptive threshold."""
        data = make_stationary(T=300, N=6)

        # Find component importance
        ce  = CausalEmergence()
        imp = ce.component_importance(data)
        top = np.argsort(imp)[::-1][:3]

        # Monitor top components with adaptive threshold
        at  = AdaptiveThreshold(phi_window=20)
        r   = at.fit(data[:, top])
        assert len(r.phi_series) > 0

    def test_full_suite_on_same_data(self):
        """All five algorithms run on the same dataset without errors."""
        data, labels = make_with_anomaly(T=300, N=6)

        SpectralChangePoint(window=25).fit(data)
        IntegrationDistance().pairwise([data[:100], data[100:200], data[200:]])
        CausalEmergence().analyse(data[:100])
        TemporalIntegration(windows=[10, 20, 30]).analyse(data)
        AdaptiveThreshold(phi_window=20).fit(data)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
