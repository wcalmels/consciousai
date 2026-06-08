# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai

"""
ConsciousAI v3.0 - Performance Benchmarks
==========================================

Measures Φ calculation speed across different configurations.
Run: python tests/benchmarks/benchmark_phi.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.core.engine import IntegratedConsciousnessEngine, IntegratedConfig

def benchmark(label, fn, n_runs=50, warmup=5):
    """Run benchmark and return mean/p95 time in ms"""
    # Warmup
    for _ in range(warmup):
        fn()
    # Measure
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = np.mean(times)
    p95_ms  = np.percentile(times, 95)
    print(f"  {label:<40} {mean_ms:7.2f}ms  p95={p95_ms:7.2f}ms")
    return mean_ms, p95_ms

def main():
    print("\n" + "="*70)
    print("⚡ ConsciousAI v3.0 — Performance Benchmarks")
    print("="*70)

    cfg = IntegratedConfig()
    cfg.enable_monitoring = False
    cfg.enable_security   = False
    engine = IntegratedConsciousnessEngine(cfg)

    print("\n── Φ Calculation by System Size ──────────────────────────────")
    for n in [5, 10, 20, 50, 100, 200]:
        data = np.random.rand(100, n)
        benchmark(f"N={n:3d} components", lambda d=data: engine.calculate_phi(d, use_cache=False))

    print("\n── Domain Preprocessing Impact ───────────────────────────────")
    data = np.random.rand(100, 20)
    for domain in ['general', 'quantum', 'biological', 'neural', 'computational']:
        benchmark(f"domain='{domain}'", lambda d=domain: engine.calculate_phi(data, domain=d, use_cache=False))

    print("\n── Cache Performance ─────────────────────────────────────────")
    data_cached = np.random.rand(100, 20)
    engine.calculate_phi(data_cached)          # Populate cache
    benchmark("Cache MISS (first call)",  lambda: engine.calculate_phi(np.random.rand(100, 20), use_cache=False))
    benchmark("Cache HIT  (repeat call)", lambda: engine.calculate_phi(data_cached))

    print("\n── Batch Throughput ──────────────────────────────────────────")
    for batch_size in [10, 50, 100, 500]:
        batch = [np.random.rand(50, 10) for _ in range(batch_size)]
        t0 = time.perf_counter()
        engine.batch_calculate_phi(batch)
        elapsed = time.perf_counter() - t0
        throughput = batch_size / elapsed
        print(f"  Batch size={batch_size:<4}  {elapsed:.2f}s  →  {throughput:.0f} ops/sec")

    engine.shutdown()
    print("\n✅ Benchmarks complete\n")

if __name__ == "__main__":
    main()
