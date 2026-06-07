"""
ConsciousAI v3.0 — Drone Fleet Example
=======================================

Demonstrates consciousness monitoring for a fleet of 10 UAVs.
Run: python examples/drone_fleet.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.core.engine import IntegratedConsciousnessEngine, IntegratedConfig, ConsciousnessLevel


def simulate_drone(drone_id: str, scenario: str = "normal") -> dict:
    """Generate synthetic drone sensor readings"""
    base = np.random.rand(50, 12)   # 50 steps × 12 sensors

    if scenario == "gps_failure":
        base[:, 0:2] = np.nan               # GPS offline
    elif scenario == "low_battery":
        base[:, 9] = np.linspace(0.3, 0.0, 50)  # Battery draining
    elif scenario == "strong_wind":
        base[:, 3:6] += np.random.normal(0, 0.4, (50, 3))  # Velocity noise

    return {"id": drone_id, "data": base, "scenario": scenario}


def alert_action(drone_id: str, phi: float, level: ConsciousnessLevel):
    """Determine response based on consciousness level"""
    if level == ConsciousnessLevel.UNCONSCIOUS:
        return "🔴 EMERGENCY LANDING"
    elif level in (ConsciousnessLevel.MINIMAL, ConsciousnessLevel.LOW):
        return "🟡 RETURN TO HOME"
    elif level == ConsciousnessLevel.MODERATE:
        return "🟠 CONSERVATIVE MODE"
    else:
        return "🟢 NORMAL OPERATION"


def main():
    print("\n" + "="*70)
    print("🚁 ConsciousAI — Drone Fleet Monitor")
    print("="*70)

    # Initialize engine
    cfg = IntegratedConfig()
    cfg.enable_monitoring = False
    cfg.enable_security   = False
    engine = IntegratedConsciousnessEngine(cfg)

    # Create fleet with mixed scenarios
    fleet_config = [
        ("DRONE-01", "normal"),
        ("DRONE-02", "normal"),
        ("DRONE-03", "gps_failure"),
        ("DRONE-04", "normal"),
        ("DRONE-05", "low_battery"),
        ("DRONE-06", "normal"),
        ("DRONE-07", "strong_wind"),
        ("DRONE-08", "normal"),
        ("DRONE-09", "normal"),
        ("DRONE-10", "normal"),
    ]

    drones = [simulate_drone(did, sc) for did, sc in fleet_config]

    print(f"\nMonitoring {len(drones)} drones...\n")
    print(f"{'ID':<12} {'Scenario':<15} {'Φ':>6} {'Level':<12} {'Action'}")
    print("-" * 70)

    phis = engine.batch_calculate_phi(
        [d["data"] for d in drones],
        domain="quantum"
    )

    for drone, phi in zip(drones, phis):
        level  = engine.get_consciousness_level(phi)
        action = alert_action(drone["id"], phi, level)
        print(f"{drone['id']:<12} {drone['scenario']:<15} {phi:>6.3f} {level.name:<12} {action}")

    # Fleet aggregate
    print("\n── Fleet Statistics ─────────────────────────────────────────")
    print(f"  Fleet avg Φ : {np.mean(phis):.4f}")
    print(f"  Fleet min Φ : {np.min(phis):.4f}  (weakest link)")
    print(f"  Drones OK   : {sum(1 for p in phis if p >= 0.5)} / {len(phis)}")
    print(f"  Need action : {sum(1 for p in phis if p < 0.5)} / {len(phis)}")

    engine.shutdown()
    print("\n✅ Fleet monitoring complete\n")


if __name__ == "__main__":
    main()
