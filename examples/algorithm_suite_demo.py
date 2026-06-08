"""
ConsciousAI Algorithm Suite — Complete Usage Example
=====================================================

Demonstrates all five algorithms on a realistic industrial scenario:
monitoring a manufacturing process with multiple sensors.

Run: python examples/algorithm_suite_demo.py
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.algorithms import (
    SpectralChangePoint,
    IntegrationDistance,
    CausalEmergence,
    TemporalIntegration,
    AdaptiveThreshold,
)

np.random.seed(42)


def generate_industrial_process(T=600, N=8):
    """
    Simulate a manufacturing process with:
    - Phase 1 (0-200):    Normal operation, moderate integration
    - Phase 2 (200-300):  Bearing wear starts — slow integration loss
    - Phase 3 (300-350):  Sudden fault — decoupling event
    - Phase 4 (350-500):  Degraded operation
    - Phase 5 (500-600):  Post-maintenance recovery
    """
    t     = np.linspace(0, 4 * np.pi, T)
    data  = np.zeros((T, N))
    labels = np.zeros(T, dtype=int)

    for i in range(T):
        if i < 200:       corr, noise = 0.55, 0.10
        elif i < 300:     corr, noise = 0.55 - 0.003*(i-200), 0.10 + 0.002*(i-200)
        elif i < 350:     corr, noise = 0.25, 0.40;  labels[i] = 1
        elif i < 500:     corr, noise = 0.30, 0.30;  labels[i] = 1
        else:             corr, noise = 0.50, 0.12

        shared = np.sin(t[i]) + 0.1 * np.random.randn()
        data[i] = corr * shared + noise * np.random.randn(N)

    return data, labels


def main():
    print("\n" + "═"*70)
    print("  ConsciousAI Algorithm Suite — Industrial Process Demo")
    print("═"*70)

    data, labels = generate_industrial_process(T=600, N=8)
    print(f"\n  Process: T={len(data)}, N={data.shape[1]} sensors")
    print(f"  Known fault window: {labels.sum()} samples\n")

    # ── 1. SpectralChangePoint ─────────────────────────────────────────────
    print("━"*70)
    print("  1. SpectralChangePoint — Detecting regime shifts")
    print("━"*70)

    t0  = time.perf_counter()
    cpd = SpectralChangePoint(window=40, threshold=2.2, min_gap=20)
    res = cpd.fit(data)
    ms  = (time.perf_counter() - t0) * 1000

    print(f"  Computed in {ms:.1f}ms")
    print(f"  Detected {len(res.change_points)} change points:")
    for cp in res.change_points:
        flag = "⚠" if labels[min(cp.index, len(labels)-1)] else " "
        print(f"    {flag} t={cp.index:4d}  score={cp.score:.2f}  "
              f"ΔH={cp.magnitude:.3f}  "
              f"({'near fault' if abs(cp.index-300)<60 else 'baseline'})")

    # ── 2. IntegrationDistance ─────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print("  2. IntegrationDistance — Comparing operational phases")
    print("━"*70)

    phases = {
        "Normal (0-200)":  data[:200],
        "Wear (200-300)":  data[200:300],
        "Fault (300-350)": data[300:350],
        "Recovery (500+)": data[500:],
    }

    dist = IntegrationDistance()
    names = list(phases.keys())
    D     = dist.pairwise(list(phases.values()))

    print(f"  Pairwise JSD distances:\n")
    header = f"  {'':22s}" + "".join(f"{n[:10]:>12s}" for n in names)
    print(header)
    for i, n in enumerate(names):
        row = f"  {n[:22]:22s}" + "".join(f"{D[i,j]:>12.4f}" for j in range(len(names)))
        print(row)

    # Highest distance = most different phases
    i, j = np.unravel_index(np.argmax(D + np.eye(len(D))*-99), D.shape)
    print(f"\n  Most different phases: '{names[i]}' vs '{names[j]}'  "
          f"(JSD = {D[i,j]:.4f})")

    # ── 3. CausalEmergence ────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print("  3. CausalEmergence — Which sensors drive integration?")
    print("━"*70)

    ce  = CausalEmergence()
    imp = ce.component_importance(data[:200])   # Normal operation

    print("  Sensor importance (normal operation):")
    for i, score in enumerate(imp):
        bar = "█" * max(0, int((score - imp.min()) /
                               (imp.max() - imp.min() + 1e-6) * 20))
        print(f"    Sensor {i}: {score:+.4f}  {bar}")

    r_normal = ce.analyse(data[:200])
    r_fault  = ce.analyse(data[300:350])
    print(f"\n  Emergence index — Normal: {r_normal.emergence_index:.3f}  "
          f"Fault: {r_fault.emergence_index:.3f}")

    # ── 4. TemporalIntegration ────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print("  4. TemporalIntegration — Multi-scale integration spectrogram")
    print("━"*70)

    ti      = TemporalIntegration(windows=[10, 25, 50, 100], step=10)
    result  = ti.analyse(data)
    profile = ti.profile(data)

    print(f"  Windows analysed: {list(result.windows)}")
    print(f"  Dominant scale:   {profile['dominant_scale']} steps")
    print(f"  Active scales:    {profile['scale_bandwidth']} / {len(result.windows)}")
    print(f"  Spectrogram shape: {result.spectrogram.shape}")

    # Anomaly score from multi-scale
    score   = ti.anomaly_score(data)
    peak_t  = result.time_axis[np.argmax(score)] if len(score) > 0 else -1
    print(f"  Peak anomaly at t={peak_t}  "
          f"({'near fault ✓' if abs(peak_t-300)<60 else 'check'})")

    # ── 5. AdaptiveThreshold ─────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print("  5. AdaptiveThreshold — Dynamic alert system")
    print("━"*70)

    for method in ("percentile", "gaussian"):
        at = AdaptiveThreshold(
            phi_window=30, history_len=150,
            alert_pct=8.0, method=method
        )
        r = at.fit(data)

        n_warn = sum(1 for a in r.alerts if a.severity == "warning")
        n_crit = sum(1 for a in r.alerts if a.severity == "critical")
        near_fault = [a for a in r.alerts if abs(a.time_index - 300) < 80]

        print(f"\n  Method: {method}")
        print(f"    Total alerts:   {len(r.alerts)}  "
              f"({n_warn} warning, {n_crit} critical)")
        print(f"    Near fault:     {len(near_fault)}")
        print(f"    FPR estimate:   {r.false_positive_est:.1%}")

    # ── Compare methods for this system ───────────────────────────────────
    print(f"\n{'━'*70}")
    print("  Threshold method comparison:")
    print("━"*70)
    comp = AdaptiveThreshold.compare_methods(data, phi_window=30)
    print(f"  {'Method':15s}  {'Alerts':>8}  {'Critical':>9}  {'FPR est':>9}")
    print("  " + "─"*45)
    for method, info in comp.items():
        if "error" not in info:
            print(f"  {method:15s}  {info['n_alerts']:>8}  "
                  f"{info['n_critical']:>9}  {info['fpr_est']:>9.1%}")

    # ── Pipeline summary ──────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PIPELINE SUMMARY")
    print("═"*70)
    print(f"""
  SpectralChangePoint → {len(res.change_points)} regime changes detected
  IntegrationDistance → Normal vs Fault JSD = {D[0,2]:.4f}
  CausalEmergence     → Emergence drops {r_normal.emergence_index:.3f}→{r_fault.emergence_index:.3f} during fault
  TemporalIntegration → Peak anomaly at t={peak_t}
  AdaptiveThreshold   → Dynamic alerts (no manual threshold needed)

  All algorithms share the same mathematical core: eigenvalue entropy
  of the covariance matrix. Fully unsupervised. No labels required.
""")


if __name__ == "__main__":
    main()
