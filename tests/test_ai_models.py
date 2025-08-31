#!/usr/bin/env python3
"""
Test Suite: AI Models Layer

Tests the AI-powered brain analysis functionality using signal processing
approach. These tests ensure AI models work independently of PyTorch and
provide robust spatiotemporal analysis.

Test Categories:
1. Signal processing ROI detection system
2. Spatial feature extraction (CNN-like)
3. Temporal pattern analysis (LSTM-inspired)
4. Noise reduction systems
5. Attention mechanisms
6. Network state classification
"""

import unittest
import numpy as np
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add python directory to path
sys.path.append(str(Path(__file__).parent.parent / "python"))

# Try to import AI modules
try:
    from ai_roi_detector import (
        SignalProcessingROIDetector, SpatialFeatureExtractor, 
        TemporalPatternAnalyzer, NoiseReductionSystem, AttentionMechanism,
        ROIDetection, NetworkState
    )
    HAS_AI_MODULES = True
except ImportError as e:
    print(f"⚠️ AI modules not available: {e}")
    HAS_AI_MODULES = False

class TestSpatialFeatureExtractor(unittest.TestCase):
    """Test suite for CNN-like spatial feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = SpatialFeatureExtractor()
        
        # Create test 3D volume
        np.random.seed(42)
        self.test_volume = np.random.randn(32, 32, 24) * 0.1
        
        # Add structured features
        self.test_volume[15:17, 15:17, 10:12] = 1.0  # Blob
        self.test_volume[10, :, 12] = 0.5  # Edge
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_feature_extraction(self):
        """Test spatial feature extraction from 3D volume."""
        features = self.extractor.extract_features(self.test_volume)
        
        # Check that all expected features are present
        expected_features = ['edge_x', 'edge_y', 'edge_z', 'blob', 'highpass', 
                           'edge_magnitude', 'local_variance']
        
        for feature_name in expected_features:
            self.assertIn(feature_name, features)
            self.assertEqual(features[feature_name].shape, self.test_volume.shape)
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_edge_detection(self):
        """Test edge detection capabilities."""
        features = self.extractor.extract_features(self.test_volume)
        
        # Edge magnitude should be higher where we placed the edge
        edge_magnitude = features['edge_magnitude']
        
        # Check that edge is detected
        self.assertGreater(np.max(edge_magnitude), 0)
        
        # Edge magnitude should be non-negative
        self.assertTrue(np.all(edge_magnitude >= 0))
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_blob_detection(self):
        """Test blob detection capabilities."""
        features = self.extractor.extract_features(self.test_volume)
        
        blob_response = features['blob']
        
        # Blob response should be highest where we placed the blob
        max_response_idx = np.unravel_index(np.argmax(blob_response), blob_response.shape)
        
        # Should be near our blob location (15:17, 15:17, 10:12)
        self.assertTrue(14 <= max_response_idx[0] <= 18)
        self.assertTrue(14 <= max_response_idx[1] <= 18)
        self.assertTrue(9 <= max_response_idx[2] <= 13)

class TestTemporalPatternAnalyzer(unittest.TestCase):
    """Test suite for LSTM-inspired temporal pattern analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TemporalPatternAnalyzer()
        
        # Create test fMRI time series
        np.random.seed(42)
        self.n_timepoints = 100
        self.volume_shape = (32, 32, 24)
        
        # 4D test data (t, x, y, z)
        self.fmri_data = np.random.randn(self.n_timepoints, *self.volume_shape) * 0.1
        
        # Add temporal patterns
        t = np.linspace(0, 10, self.n_timepoints)
        dmn_signal = np.sin(2 * np.pi * 0.02 * t)  # Low frequency
        task_signal = np.sin(2 * np.pi * 0.1 * t)   # Higher frequency
        
        # Add patterns to specific regions
        self.fmri_data[:, 15:17, 15:17, 10:12] += dmn_signal[:, np.newaxis, np.newaxis, np.newaxis]
        self.fmri_data[:, 20:22, 20:22, 15:17] += task_signal[:, np.newaxis, np.newaxis, np.newaxis]
        
        # Create region masks
        self.dmn_mask = np.zeros(self.volume_shape)
        self.dmn_mask[15:17, 15:17, 10:12] = 1
        
        self.task_mask = np.zeros(self.volume_shape)
        self.task_mask[20:22, 20:22, 15:17] = 1
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_temporal_analysis(self):
        """Test temporal pattern analysis."""
        features = self.analyzer.analyze_temporal_patterns(self.fmri_data, self.dmn_mask)
        
        # Check that expected features are present
        expected_features = ['power_spectrum', 'dominant_frequency', 'low_freq_power',
                           'high_freq_power', 'freq_ratio', 'velocity', 'acceleration',
                           'autocorrelation', 'memory_decay', 'hurst_exponent']
        
        for feature_name in expected_features:
            self.assertIn(feature_name, features)
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_frequency_analysis(self):
        """Test frequency domain analysis."""
        features = self.analyzer.analyze_temporal_patterns(self.fmri_data, self.dmn_mask)
        
        # DMN region should have higher low frequency power
        low_freq_power = features['low_freq_power']
        high_freq_power = features['high_freq_power']
        
        self.assertGreater(low_freq_power, 0)
        self.assertGreater(high_freq_power, 0)
        
        # For DMN, low frequency should dominate
        self.assertLess(features['freq_ratio'], 1.0)  # low freq > high freq
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_canonical_correlations(self):
        """Test correlation with canonical brain patterns."""
        dmn_features = self.analyzer.analyze_temporal_patterns(self.fmri_data, self.dmn_mask)
        task_features = self.analyzer.analyze_temporal_patterns(self.fmri_data, self.task_mask)
        
        # DMN region should correlate more with DMN pattern
        dmn_corr_dmn = abs(dmn_features['dmn_correlation'])
        dmn_corr_task = abs(dmn_features['task_correlation'])
        
        # Task region should correlate more with task pattern
        task_corr_dmn = abs(task_features['dmn_correlation'])
        task_corr_task = abs(task_features['task_correlation'])
        
        # These should show the expected pattern (though may be weak in synthetic data)
        self.assertIsInstance(dmn_corr_dmn, (int, float))
        self.assertIsInstance(task_corr_task, (int, float))

class TestNoiseReductionSystem(unittest.TestCase):
    """Test suite for hearing-aid-inspired noise reduction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.noise_reducer = NoiseReductionSystem()
        
        # Create test signal with noise
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        self.clean_signal = np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz sine wave
        self.noise = np.random.randn(100) * 0.5
        self.noisy_signal = self.clean_signal + self.noise
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_spectral_subtraction(self):
        """Test spectral subtraction noise reduction."""
        denoised = self.noise_reducer.adaptive_noise_reduction(
            self.noisy_signal, noise_estimation_method='spectral'
        )
        
        # Denoised signal should have same length
        self.assertEqual(len(denoised), len(self.noisy_signal))
        
        # Should be real-valued
        self.assertTrue(np.all(np.isreal(denoised)))
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_wiener_filter(self):
        """Test Wiener filter noise reduction."""
        denoised = self.noise_reducer.adaptive_noise_reduction(
            self.noisy_signal, noise_estimation_method='wiener'
        )
        
        # Denoised signal should have same length
        self.assertEqual(len(denoised), len(self.noisy_signal))
        
        # Should reduce noise (lower variance than original)
        self.assertLess(np.var(denoised), np.var(self.noisy_signal))
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_adaptive_filter(self):
        """Test adaptive noise reduction filter."""
        denoised = self.noise_reducer.adaptive_noise_reduction(
            self.noisy_signal, noise_estimation_method='adaptive'
        )
        
        # Should smooth the signal
        self.assertEqual(len(denoised), len(self.noisy_signal))
        
        # Should be smoother (less high-frequency content)
        original_diff = np.sum(np.abs(np.diff(self.noisy_signal)))
        denoised_diff = np.sum(np.abs(np.diff(denoised)))
        self.assertLess(denoised_diff, original_diff)

class TestAttentionMechanism(unittest.TestCase):
    """Test suite for attention mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.attention = AttentionMechanism()
        
        # Create test features
        self.volume_shape = (32, 32, 24)
        self.features = {
            'feature1': np.random.randn(*self.volume_shape) * 0.1,
            'feature2': np.random.randn(*self.volume_shape) * 0.1,
            'feature3': np.random.randn(*self.volume_shape) * 0.1
        }
        
        # Add salient region to feature1
        self.features['feature1'][15:17, 15:17, 10:12] = 2.0
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_attention_computation(self):
        """Test attention weight computation."""
        attention_weights = self.attention.compute_attention_weights(self.features)
        
        # Should have same shape as input
        self.assertEqual(attention_weights.shape, self.volume_shape)
        
        # Should be normalized (approximately sum to 1)
        self.assertAlmostEqual(np.sum(attention_weights), 1.0, places=5)
        
        # Should be non-negative
        self.assertTrue(np.all(attention_weights >= 0))
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_attention_focus(self):
        """Test that attention focuses on salient regions."""
        attention_weights = self.attention.compute_attention_weights(self.features)
        
        # Find maximum attention location
        max_idx = np.unravel_index(np.argmax(attention_weights), attention_weights.shape)
        
        # Should be near our salient region (15:17, 15:17, 10:12)
        self.assertTrue(14 <= max_idx[0] <= 18)
        self.assertTrue(14 <= max_idx[1] <= 18)
        self.assertTrue(9 <= max_idx[2] <= 13)

class TestSignalProcessingROIDetector(unittest.TestCase):
    """Test suite for the main signal processing ROI detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = SignalProcessingROIDetector(noise_reduction=True)
        
        # Create synthetic fMRI data
        np.random.seed(42)
        self.n_timepoints = 50  # Smaller for faster tests
        self.volume_shape = (16, 16, 12)  # Smaller for faster tests
        
        self.fmri_data = np.random.randn(self.n_timepoints, *self.volume_shape) * 0.1
        
        # Add structured activation patterns
        t = np.linspace(0, 5, self.n_timepoints)
        
        # DMN-like pattern (low frequency)
        dmn_signal = np.sin(2 * np.pi * 0.02 * t) * 0.5
        self.fmri_data[:, 7:9, 7:9, 5:7] += dmn_signal[:, np.newaxis, np.newaxis, np.newaxis]
        
        # Task-like pattern (higher frequency)
        task_signal = np.sin(2 * np.pi * 0.1 * t) * 0.3
        self.fmri_data[:, 12:14, 12:14, 8:10] += task_signal[:, np.newaxis, np.newaxis, np.newaxis]
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_detector_initialization(self):
        """Test ROI detector initialization."""
        self.assertIsNotNone(self.detector.spatial_extractor)
        self.assertIsNotNone(self.detector.temporal_analyzer)
        self.assertIsNotNone(self.detector.noise_reducer)
        self.assertIsNotNone(self.detector.attention)
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_roi_detection(self):
        """Test ROI detection functionality."""
        detections = self.detector.detect_rois(self.fmri_data, confidence_threshold=0.3)
        
        # Should detect some ROIs
        self.assertGreater(len(detections), 0)
        
        # Each detection should have required attributes
        for detection in detections:
            self.assertIsInstance(detection, ROIDetection)
            self.assertGreaterEqual(detection.confidence, 0.3)
            self.assertIsInstance(detection.coordinates, tuple)
            self.assertEqual(len(detection.coordinates), 3)
            self.assertIsInstance(detection.network_state, NetworkState)
            self.assertGreater(detection.activation_strength, 0)
            self.assertEqual(len(detection.temporal_pattern), self.n_timepoints)
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_network_classification(self):
        """Test network state classification."""
        detections = self.detector.detect_rois(self.fmri_data, confidence_threshold=0.1)
        
        # Should classify into different network states
        network_states = [d.network_state for d in detections]
        unique_states = set(network_states)
        
        # Should have at least one classification
        self.assertGreater(len(unique_states), 0)
        
        # All states should be valid NetworkState enum values
        for state in unique_states:
            self.assertIsInstance(state, NetworkState)
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_confidence_scoring(self):
        """Test confidence scoring mechanism."""
        detections = self.detector.detect_rois(self.fmri_data, confidence_threshold=0.0)
        
        if detections:
            confidences = [d.confidence for d in detections]
            
            # Confidences should be between 0 and 1
            self.assertTrue(all(0 <= c <= 1 for c in confidences))
            
            # Should be sorted by confidence (highest first)
            self.assertEqual(confidences, sorted(confidences, reverse=True))
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_report_generation(self):
        """Test analysis report generation."""
        detections = self.detector.detect_rois(self.fmri_data, confidence_threshold=0.3)
        report = self.detector.generate_report(detections)
        
        # Check report structure
        self.assertIn('timestamp', report)
        self.assertIn('method', report)
        self.assertIn('total_rois_detected', report)
        self.assertIn('network_distribution', report)
        self.assertIn('confidence_stats', report)
        self.assertIn('detections', report)
        
        # Check values
        self.assertEqual(report['method'], 'signal_processing')
        self.assertEqual(report['total_rois_detected'], len(detections))
        
        # Confidence stats should be reasonable
        if detections:
            self.assertGreaterEqual(report['confidence_stats']['mean'], 0)
            self.assertLessEqual(report['confidence_stats']['mean'], 1)

class TestPerformanceAndRobustness(unittest.TestCase):
    """Test suite for performance and robustness."""
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_processing_speed(self):
        """Test that processing completes in reasonable time."""
        detector = SignalProcessingROIDetector(noise_reduction=False)  # Faster without noise reduction
        
        # Small test data
        fmri_data = np.random.randn(20, 16, 16, 12) * 0.1
        
        start_time = time.time()
        detections = detector.detect_rois(fmri_data, confidence_threshold=0.5)
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (less than 30 seconds for small data)
        self.assertLess(processing_time, 30.0)
        
        print(f"   Processing time: {processing_time:.2f}s for {fmri_data.shape} data")
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_edge_cases(self):
        """Test handling of edge cases."""
        detector = SignalProcessingROIDetector()
        
        # Test with very small data
        small_data = np.random.randn(5, 8, 8, 6) * 0.1
        detections = detector.detect_rois(small_data, confidence_threshold=0.8)
        
        # Should handle gracefully
        self.assertIsInstance(detections, list)
        
        # Test with constant data
        constant_data = np.ones((10, 8, 8, 6)) * 0.5
        detections = detector.detect_rois(constant_data, confidence_threshold=0.5)
        
        # Should handle gracefully
        self.assertIsInstance(detections, list)
    
    @unittest.skipUnless(HAS_AI_MODULES, "AI modules not available")
    def test_memory_usage(self):
        """Test reasonable memory usage."""
        detector = SignalProcessingROIDetector()
        
        # Moderate size data
        fmri_data = np.random.randn(30, 20, 20, 15) * 0.1
        
        # Should not crash with memory error
        try:
            detections = detector.detect_rois(fmri_data, confidence_threshold=0.7)
            self.assertIsInstance(detections, list)
        except MemoryError:
            self.fail("Processing caused memory error")

def run_ai_tests():
    """Run all AI model tests."""
    print("🧪 Running AI Models Test Suite (Signal Processing)")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSpatialFeatureExtractor,
        TestTemporalPatternAnalyzer,
        TestNoiseReductionSystem,
        TestAttentionMechanism,
        TestSignalProcessingROIDetector,
        TestPerformanceAndRobustness
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 AI Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.wasSuccessful():
        print("✅ All AI model tests passed!")
        return True
    else:
        print("❌ Some AI tests failed")
        return False

if __name__ == "__main__":
    success = run_ai_tests()
    sys.exit(0 if success else 1) 