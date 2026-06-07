"""
ConsciousAI v3.0 - Comprehensive Integration Tests
====================================================

Tests for:
- Φ calculation accuracy
- Cache performance
- Security engine
- Blockchain integrity
- File integrity checking
- Performance benchmarks
- Integration tests

Author: Walter Calmels
Date: 2024-11-26
"""

import numpy as np
import time
import unittest
import tempfile
import os
from pathlib import Path

# Import ConsciousAI v3.0
import sys
sys.path.insert(0, os.path.dirname(__file__))
from consciousai_v3_integrated import (
    IntegratedConsciousnessEngine,
    IntegratedConfig,
    ConsciousnessLevel,
    DeterministicBlockchain,
    FileIntegrityChecker,
    SecurityEngine
)

class TestPhiCalculation(unittest.TestCase):
    """Test Φ calculation accuracy"""
    
    def setUp(self):
        self.config = IntegratedConfig()
        self.config.enable_monitoring = False
        self.config.enable_security = False
        self.engine = IntegratedConsciousnessEngine(self.config)
    
    def tearDown(self):
        self.engine.shutdown()
    
    def test_basic_phi_calculation(self):
        """Test basic Φ calculation"""
        data = np.random.rand(50, 5)
        phi = self.engine.calculate_phi(data)
        
        self.assertIsInstance(phi, float)
        self.assertGreaterEqual(phi, 0.0)
        self.assertLess(phi, 100.0)  # Reasonable upper bound
    
    def test_phi_with_connectivity(self):
        """Test Φ with connectivity matrix"""
        data = np.random.rand(50, 5)
        connectivity = np.random.rand(5, 5)
        
        phi = self.engine.calculate_phi(data, connectivity)
        
        self.assertIsInstance(phi, float)
        self.assertGreaterEqual(phi, 0.0)
    
    def test_phi_domains(self):
        """Test domain-specific preprocessing"""
        data = np.random.rand(50, 5)
        domains = ['general', 'quantum', 'biological', 'computational', 'neural']
        
        phis = {}
        for domain in domains:
            phi = self.engine.calculate_phi(data, domain=domain)
            phis[domain] = phi
            self.assertGreaterEqual(phi, 0.0)
        
        # All domains should produce reasonable results
        self.assertEqual(len(phis), len(domains))
    
    def test_phi_invalid_data(self):
        """Test Φ with invalid data (NaN, Inf)"""
        data = np.random.rand(50, 5)
        data[0, 0] = np.nan
        data[1, 1] = np.inf
        
        # Should handle gracefully (auto-repair)
        phi = self.engine.calculate_phi(data)
        self.assertIsInstance(phi, float)
        self.assertTrue(np.isfinite(phi))
    
    def test_consciousness_levels(self):
        """Test consciousness level mapping"""
        test_phis = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        expected_levels = [
            ConsciousnessLevel.UNCONSCIOUS,
            ConsciousnessLevel.MINIMAL,
            ConsciousnessLevel.LOW,
            ConsciousnessLevel.MODERATE,
            ConsciousnessLevel.HIGH,
            ConsciousnessLevel.VERY_HIGH
        ]
        
        for phi, expected in zip(test_phis, expected_levels):
            level = self.engine.get_consciousness_level(phi)
            self.assertEqual(level, expected)


class TestCache(unittest.TestCase):
    """Test self-repairing cache"""
    
    def setUp(self):
        self.config = IntegratedConfig()
        self.config.enable_monitoring = False
        self.config.enable_security = False
        self.engine = IntegratedConsciousnessEngine(self.config)
    
    def tearDown(self):
        self.engine.shutdown()
    
    def test_cache_hit(self):
        """Test cache hit on repeated calculation"""
        data = np.random.rand(50, 5)
        
        # First call (miss)
        phi1 = self.engine.calculate_phi(data)
        
        # Second call (should hit cache)
        phi2 = self.engine.calculate_phi(data)
        
        self.assertEqual(phi1, phi2)
        
        # Check cache stats
        stats = self.engine.cache.get_stats()
        self.assertGreater(stats['hits'], 0)
    
    def test_cache_repair(self):
        """Test cache self-repair"""
        # Add some valid entries
        for i in range(10):
            data = np.random.rand(50, 5)
            self.engine.calculate_phi(data)
        
        # Manually corrupt cache (for testing)
        self.engine.cache.cache_dict['corrupt'] = (np.nan, time.time())
        
        # Repair
        repaired = self.engine.cache.repair()
        
        # Should have removed corrupt entry
        self.assertGreater(repaired, 0)


class TestBlockchain(unittest.TestCase):
    """Test deterministic blockchain"""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.blockchain = DeterministicBlockchain(self.temp_file.name)
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_genesis_block(self):
        """Test genesis block creation"""
        self.assertEqual(len(self.blockchain.chain), 1)
        self.assertEqual(self.blockchain.chain[0]['index'], 0)
        self.assertEqual(self.blockchain.chain[0]['previous_hash'], '0' * 64)
    
    def test_add_block(self):
        """Test adding blocks"""
        initial_length = len(self.blockchain.chain)
        
        self.blockchain.add_block("Test event 1")
        self.blockchain.add_block("Test event 2")
        
        self.assertEqual(len(self.blockchain.chain), initial_length + 2)
    
    def test_blockchain_validation(self):
        """Test blockchain validation"""
        self.blockchain.add_block("Event 1")
        self.blockchain.add_block("Event 2")
        
        # Should be valid
        self.assertTrue(self.blockchain.validate_chain())
        
        # Corrupt a block
        self.blockchain.chain[1]['data'] = "Corrupted"
        
        # Should now be invalid
        self.assertFalse(self.blockchain.validate_chain())
    
    def test_blockchain_persistence(self):
        """Test blockchain save/load"""
        self.blockchain.add_block("Event before save")
        original_length = len(self.blockchain.chain)
        
        # Save
        self.blockchain.save()
        
        # Load in new instance
        blockchain2 = DeterministicBlockchain(self.temp_file.name)
        
        self.assertEqual(len(blockchain2.chain), original_length)
        self.assertTrue(blockchain2.validate_chain())


class TestFileIntegrity(unittest.TestCase):
    """Test file integrity checker"""
    
    def setUp(self):
        self.temp_manifest = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_manifest.close()
        
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        self.temp_file.write("Test content")
        self.temp_file.close()
        
        self.checker = FileIntegrityChecker(self.temp_manifest.name)
    
    def tearDown(self):
        os.unlink(self.temp_manifest.name)
        os.unlink(self.temp_file.name)
    
    def test_register_file(self):
        """Test file registration"""
        self.checker.register_file(self.temp_file.name)
        
        self.assertIn(self.temp_file.name, self.checker.manifest)
    
    def test_check_unmodified_file(self):
        """Test checking unmodified file"""
        self.checker.register_file(self.temp_file.name)
        
        # Should pass
        self.assertTrue(self.checker.check_file(self.temp_file.name))
    
    def test_check_modified_file(self):
        """Test detecting modified file"""
        self.checker.register_file(self.temp_file.name)
        
        # Modify file
        with open(self.temp_file.name, 'a') as f:
            f.write("Modified content")
        
        # Should fail
        self.assertFalse(self.checker.check_file(self.temp_file.name))


class TestSecurityEngine(unittest.TestCase):
    """Test security engine"""
    
    def setUp(self):
        self.config = IntegratedConfig()
        self.config.security_vigilance_interval = 1.0  # Fast for testing
        
        # Temporary files
        self.temp_blockchain = tempfile.NamedTemporaryFile(delete=False, suffix='_bc.json')
        self.temp_blockchain.close()
        self.temp_manifest = tempfile.NamedTemporaryFile(delete=False, suffix='_man.json')
        self.temp_manifest.close()
        
        self.config.blockchain_path = self.temp_blockchain.name
        self.config.security_manifest_path = self.temp_manifest.name
        
        self.security = SecurityEngine(self.config)
    
    def tearDown(self):
        self.security.shutdown()
        time.sleep(1)  # Allow shutdown
        os.unlink(self.temp_blockchain.name)
        os.unlink(self.temp_manifest.name)
    
    def test_security_initialization(self):
        """Test security engine initialization"""
        self.assertIsNotNone(self.security.blockchain)
        self.assertIsNotNone(self.security.integrity_checker)
    
    def test_file_registration(self):
        """Test protected file registration"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        temp_file.write("Protected content")
        temp_file.close()
        
        self.security.register_protected_file(temp_file.name)
        
        self.assertIn(temp_file.name, self.security.integrity_checker.manifest)
        
        os.unlink(temp_file.name)
    
    def test_stats_collection(self):
        """Test security stats collection"""
        stats = self.security.get_stats()
        
        self.assertIn('threats_detected', stats)
        self.assertIn('blockchain_length', stats)
        self.assertIn('protected_files', stats)
        self.assertIn('uptime_seconds', stats)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete system"""
    
    def setUp(self):
        self.config = IntegratedConfig()
        self.config.enable_monitoring = False
        self.config.enable_security = True
        
        # Temporary files for security
        self.temp_blockchain = tempfile.NamedTemporaryFile(delete=False, suffix='_bc.json')
        self.temp_blockchain.close()
        self.temp_manifest = tempfile.NamedTemporaryFile(delete=False, suffix='_man.json')
        self.temp_manifest.close()
        
        self.config.blockchain_path = self.temp_blockchain.name
        self.config.security_manifest_path = self.temp_manifest.name
        
        self.engine = IntegratedConsciousnessEngine(self.config)
    
    def tearDown(self):
        self.engine.shutdown()
        time.sleep(1)
        os.unlink(self.temp_blockchain.name)
        os.unlink(self.temp_manifest.name)
    
    def test_full_system(self):
        """Test complete integrated system"""
        # Calculate Φ
        data = np.random.rand(50, 5)
        phi = self.engine.calculate_phi(data)
        
        self.assertGreaterEqual(phi, 0.0)
        
        # Get stats
        stats = self.engine.get_comprehensive_stats()
        
        self.assertIn('performance', stats)
        self.assertIn('cache', stats)
        self.assertIn('security', stats)
        
        # Check calculations recorded
        self.assertGreater(stats['calculations']['total_calculations'], 0)
    
    def test_batch_processing(self):
        """Test batch processing"""
        batch_size = 50
        data_batch = [np.random.rand(30, 5) for _ in range(batch_size)]
        
        phis = self.engine.batch_calculate_phi(data_batch)
        
        self.assertEqual(len(phis), batch_size)
        self.assertTrue(np.all(phis >= 0.0))


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks"""
    
    def setUp(self):
        self.config = IntegratedConfig()
        self.config.enable_monitoring = False
        self.config.enable_security = False
        self.engine = IntegratedConsciousnessEngine(self.config)
    
    def tearDown(self):
        self.engine.shutdown()
    
    def test_calculation_speed(self):
        """Benchmark Φ calculation speed"""
        data = np.random.rand(100, 10)
        
        # Warm up
        self.engine.calculate_phi(data)
        
        # Benchmark
        n_iterations = 100
        start = time.perf_counter()
        
        for _ in range(n_iterations):
            self.engine.calculate_phi(data)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / n_iterations
        
        print(f"\n   Avg calculation time: {avg_time*1000:.3f}ms")
        print(f"   Throughput: {1/avg_time:.1f} ops/sec")
        
        # Should be reasonably fast
        self.assertLess(avg_time, 0.1)  # Less than 100ms per calculation
    
    def test_cache_performance(self):
        """Benchmark cache hit performance"""
        data = np.random.rand(100, 10)
        
        # First call (miss)
        start = time.perf_counter()
        phi1 = self.engine.calculate_phi(data)
        time_miss = time.perf_counter() - start
        
        # Second call (hit)
        start = time.perf_counter()
        phi2 = self.engine.calculate_phi(data)
        time_hit = time.perf_counter() - start
        
        speedup = time_miss / time_hit
        
        print(f"\n   Cache miss: {time_miss*1000:.3f}ms")
        print(f"   Cache hit: {time_hit*1000:.3f}ms")
        print(f"   Speedup: {speedup:.1f}×")
        
        # Cache hit should be significantly faster
        self.assertGreater(speedup, 2.0)
    
    def test_batch_throughput(self):
        """Benchmark batch processing throughput"""
        batch_size = 100
        data_batch = [np.random.rand(50, 8) for _ in range(batch_size)]
        
        start = time.perf_counter()
        phis = self.engine.batch_calculate_phi(data_batch)
        elapsed = time.perf_counter() - start
        
        throughput = batch_size / elapsed
        
        print(f"\n   Batch size: {batch_size}")
        print(f"   Total time: {elapsed:.2f}s")
        print(f"   Throughput: {throughput:.1f} ops/sec")
        
        # Should process multiple items per second
        self.assertGreater(throughput, 10.0)


def run_all_tests():
    """Run all tests with detailed output"""
    print("\n" + "="*80)
    print("🧪 ConsciousAI v3.0 - Comprehensive Test Suite")
    print("="*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhiCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestCache))
    suite.addTests(loader.loadTestsFromTestCase(TestBlockchain))
    suite.addTests(loader.loadTestsFromTestCase(TestFileIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmarks))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return result


if __name__ == "__main__":
    result = run_all_tests()
    exit(0 if result.wasSuccessful() else 1)
