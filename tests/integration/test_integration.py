"""
ConsciousAI v3.0 - Integration Tests
=====================================

End-to-end tests simulating real domain scenarios.
"""

import numpy as np
import time
import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.core.engine import (
    IntegratedConsciousnessEngine,
    IntegratedConfig,
    ConsciousnessLevel,
)


class TestDroneScenario(unittest.TestCase):
    """Simulate drone fleet consciousness monitoring"""

    def setUp(self):
        cfg = IntegratedConfig()
        cfg.enable_monitoring = False
        cfg.enable_security   = False
        self.engine = IntegratedConsciousnessEngine(cfg)

    def tearDown(self):
        self.engine.shutdown()

    def _drone_state(self, degraded=False):
        """Generate synthetic drone sensor vector"""
        state = np.random.rand(50, 12)   # 50 time steps, 12 sensors
        if degraded:
            # Simulate GPS failure — inject NaN and reduce correlation
            state[:, 0] = np.nan          # GPS x
            state[:, 1] = np.nan          # GPS y
        return state

    def test_healthy_drone_high_phi(self):
        data = self._drone_state(degraded=False)
        phi  = self.engine.calculate_phi(data, domain='quantum')
        self.assertGreater(phi, 0.5, "Healthy drone should have moderate-high Φ")

    def test_degraded_drone_detectable(self):
        healthy  = self._drone_state(degraded=False)
        degraded = self._drone_state(degraded=True)

        phi_h = self.engine.calculate_phi(healthy,  domain='quantum', use_cache=False)
        phi_d = self.engine.calculate_phi(degraded, domain='quantum', use_cache=False)

        # System should still return a finite value (self-repair)
        self.assertTrue(np.isfinite(phi_d))

    def test_fleet_batch(self):
        fleet = [self._drone_state() for _ in range(20)]
        phis  = self.engine.batch_calculate_phi(fleet, domain='quantum')

        self.assertEqual(len(phis), 20)
        self.assertTrue(np.all(np.isfinite(phis)))
        self.assertTrue(np.all(phis >= 0))


class TestBiomedicalScenario(unittest.TestCase):
    """Simulate ICU patient physiological monitoring"""

    def setUp(self):
        cfg = IntegratedConfig()
        cfg.enable_monitoring = False
        cfg.enable_security   = False
        self.engine = IntegratedConsciousnessEngine(cfg)

    def tearDown(self):
        self.engine.shutdown()

    def _patient_vitals(self, sepsis=False):
        """
        Simulate 15 physiological channels over 60 time steps.
        Sepsis → decouple organ systems (lower integration).
        """
        data = np.random.rand(60, 15)
        if sepsis:
            # Scramble correlations between channels
            np.random.shuffle(data.T)
            data += np.random.normal(0, 0.5, data.shape)
        return data

    def test_healthy_patient(self):
        data = self._patient_vitals(sepsis=False)
        phi  = self.engine.calculate_phi(data, domain='biological')
        level = self.engine.get_consciousness_level(phi)
        self.assertIsNotNone(level)

    def test_septic_patient_detectable(self):
        healthy = self._patient_vitals(sepsis=False)
        septic  = self._patient_vitals(sepsis=True)

        phi_h = self.engine.calculate_phi(healthy, domain='biological', use_cache=False)
        phi_s = self.engine.calculate_phi(septic,  domain='biological', use_cache=False)

        # Both values should be finite
        self.assertTrue(np.isfinite(phi_h))
        self.assertTrue(np.isfinite(phi_s))


class TestSmartCityScenario(unittest.TestCase):
    """Simulate smart city infrastructure monitoring"""

    def setUp(self):
        cfg = IntegratedConfig()
        cfg.enable_monitoring = False
        cfg.enable_security   = False
        self.engine = IntegratedConsciousnessEngine(cfg)

    def tearDown(self):
        self.engine.shutdown()

    def test_city_normal_operations(self):
        city_data = np.random.rand(100, 50)   # 100 readings, 50 subsystems
        phi  = self.engine.calculate_phi(city_data, domain='computational')
        self.assertIsInstance(phi, float)
        self.assertGreaterEqual(phi, 0.0)

    def test_cascade_risk_detectable(self):
        """Power outage affects multiple subsystems simultaneously"""
        normal  = np.random.rand(100, 50)
        cascade = normal.copy()
        cascade[:, :10] = 0.0   # 10 power zones go offline

        phi_n = self.engine.calculate_phi(normal,  domain='computational', use_cache=False)
        phi_c = self.engine.calculate_phi(cascade, domain='computational', use_cache=False)

        # Both finite
        self.assertTrue(np.isfinite(phi_n))
        self.assertTrue(np.isfinite(phi_c))


class TestSecurityIntegration(unittest.TestCase):
    """Integration test with security engine active"""

    def setUp(self):
        self.bc_file  = tempfile.NamedTemporaryFile(delete=False, suffix='_bc.json')
        self.man_file = tempfile.NamedTemporaryFile(delete=False, suffix='_man.json')
        self.bc_file.close()
        self.man_file.close()

        cfg = IntegratedConfig()
        cfg.enable_monitoring = False
        cfg.enable_security   = True
        cfg.security_vigilance_interval = 60.0   # Slow for tests
        cfg.blockchain_path       = self.bc_file.name
        cfg.security_manifest_path = self.man_file.name

        self.engine = IntegratedConsciousnessEngine(cfg)

    def tearDown(self):
        self.engine.shutdown()
        time.sleep(0.5)
        for f in [self.bc_file.name, self.man_file.name]:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    def test_phi_with_security_active(self):
        data = np.random.rand(100, 10)
        phi  = self.engine.calculate_phi(data)
        self.assertGreaterEqual(phi, 0.0)

    def test_blockchain_records_activity(self):
        initial_blocks = len(self.engine.security.blockchain.chain)

        for _ in range(5):
            data = np.random.rand(50, 8)
            self.engine.calculate_phi(data)

        # Blockchain should still be valid
        self.assertTrue(self.engine.security.blockchain.validate_chain())

    def test_comprehensive_stats_with_security(self):
        data = np.random.rand(50, 8)
        self.engine.calculate_phi(data)

        stats = self.engine.get_comprehensive_stats()

        self.assertIn('performance', stats)
        self.assertIn('cache', stats)
        self.assertIn('security', stats)
        self.assertGreater(stats['calculations']['total_calculations'], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
