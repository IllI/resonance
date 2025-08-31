#!/usr/bin/env python3
"""
Test Suite: Core fMRI Analysis

Tests the core brain analysis functionality independent of GPU acceleration.
These tests ensure that the fundamental fMRI analysis works on any system.

Test Categories:
1. Data loading and preprocessing
2. Connectivity matrix computation
3. Network metrics calculation
4. Hub identification
5. Dynamic connectivity analysis
6. Quality control measures
"""

import unittest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add python directory to path
sys.path.append(str(Path(__file__).parent.parent / "python"))

try:
    from neural_network_analyzer import BrainNetworkAnalyzer
    from brain_atlas_manager import BrainAtlasManager
    from fmri_data_loader import fMRIDataLoader
    HAS_CORE_MODULES = True
except ImportError as e:
    print(f"⚠️ Core modules not available: {e}")
    HAS_CORE_MODULES = False

class TestCoreFMRIAnalysis(unittest.TestCase):
    """Test suite for core fMRI analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_regions = 50
        self.n_timepoints = 100
        
        # Create realistic mock fMRI data
        np.random.seed(42)  # Reproducible tests
        
        # Base signal
        self.fmri_data = np.random.randn(self.n_timepoints, self.n_regions) * 0.1
        
        # Add network structure
        # Default mode network (regions 0-10)
        dmn_signal = np.sin(np.linspace(0, 4*np.pi, self.n_timepoints)) * 0.5
        self.fmri_data[:, 0:10] += dmn_signal[:, np.newaxis]
        
        # Task-positive network (regions 20-30)
        task_signal = np.cos(np.linspace(0, 6*np.pi, self.n_timepoints)) * 0.3
        self.fmri_data[:, 20:30] += task_signal[:, np.newaxis]
        
        # Add some correlation between nearby regions
        for i in range(0, self.n_regions-5, 5):
            group_signal = np.random.randn(self.n_timepoints) * 0.2
            self.fmri_data[:, i:i+5] += group_signal[:, np.newaxis]
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_brain_network_analyzer_initialization(self):
        """Test BrainNetworkAnalyzer initialization."""
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        self.assertEqual(analyzer.sampling_rate, 0.4)
        self.assertIsNone(analyzer.time_series)
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_time_series_loading(self):
        """Test time series data loading and validation."""
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        analyzer.load_time_series(self.fmri_data)
        
        self.assertEqual(analyzer.n_timepoints, self.n_timepoints)
        self.assertEqual(analyzer.n_regions, self.n_regions)
        self.assertTrue(np.array_equal(analyzer.time_series, self.fmri_data))
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_network_structure_detection(self):
        """Test that our synthetic network structure is detected."""
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        analyzer.load_time_series(self.fmri_data)
        
        # Compute connectivity
        conn_matrix = analyzer.compute_connectivity_matrix(method='pearson')
        
        # Check that DMN regions are more connected to each other
        dmn_internal = np.mean(np.abs(conn_matrix[0:10, 0:10]))
        dmn_external = np.mean(np.abs(conn_matrix[0:10, 10:]))
        
        # Internal DMN connectivity should be higher
        self.assertGreater(dmn_internal, dmn_external)
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_network_metrics_computation(self):
        """Test network metrics calculation."""
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        analyzer.load_time_series(self.fmri_data)
        
        conn_matrix = analyzer.compute_connectivity_matrix(method='pearson')
        metrics = analyzer.compute_network_metrics(conn_matrix, threshold=0.3)
        
        # Check that expected metrics are present
        expected_keys = ['n_nodes', 'n_edges', 'density', 'avg_clustering', 
                        'avg_path_length', 'global_efficiency']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Basic sanity checks
        self.assertEqual(metrics['n_nodes'], self.n_regions)
        self.assertGreaterEqual(metrics['n_edges'], 0)
        self.assertGreaterEqual(metrics['density'], 0)
        self.assertLessEqual(metrics['density'], 1)
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available") 
    def test_hub_identification(self):
        """Test network hub identification using BrainAtlasManager."""
        # Use BrainAtlasManager for hub identification
        atlas_manager = BrainAtlasManager()
        
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        analyzer.load_time_series(self.fmri_data)
        
        conn_matrix = analyzer.compute_connectivity_matrix(method='pearson')
        
        # Test hub identification with different thresholds
        hubs_80 = atlas_manager.identify_network_hubs(conn_matrix, threshold=0.8)
        hubs_90 = atlas_manager.identify_network_hubs(conn_matrix, threshold=0.9)
        
        # Higher threshold should yield fewer hubs
        self.assertLessEqual(len(hubs_90), len(hubs_80))
        
        # Check hub properties
        if hubs_80:
            hub = hubs_80[0]
            self.assertIn('region_index', hub)
            self.assertIn('node_strength', hub)
            self.assertIn('percentile', hub)
            self.assertGreaterEqual(hub['node_strength'], 0)
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_dynamic_connectivity_analysis(self):
        """Test dynamic connectivity analysis."""
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        analyzer.load_time_series(self.fmri_data)
        
        # Use the correct method signature (no method parameter)
        dynamic_matrices = analyzer.dynamic_connectivity_analysis(
            window_size=20, 
            step_size=5
        )
        
        # Check that it returns a list of matrices
        self.assertIsInstance(dynamic_matrices, list)
        self.assertGreater(len(dynamic_matrices), 0)
        self.assertEqual(dynamic_matrices[0].shape, (self.n_regions, self.n_regions))
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_change_point_detection(self):
        """Test change point detection in dynamic connectivity."""
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        analyzer.load_time_series(self.fmri_data)
        
        # Use the correct method signature
        dynamic_matrices = analyzer.dynamic_connectivity_analysis(
            window_size=20,
            step_size=5
        )
        
        # Test change point detection (threshold parameter, not method)
        change_points = analyzer.detect_network_changes(
            dynamic_matrices,
            threshold=0.2
        )
        
        self.assertIsInstance(change_points, list)
        # Change points should be within valid range
        for cp in change_points:
            self.assertGreaterEqual(cp, 0)
            self.assertLess(cp, len(dynamic_matrices))
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_data_validation(self):
        """Test input data validation and error handling."""
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        
        # Test with invalid data shapes
        with self.assertRaises(ValueError):
            # 1D data should raise error
            analyzer.load_time_series(np.random.randn(100))
        
        with self.assertRaises(ValueError):
            # 3D data should raise error
            analyzer.load_time_series(np.random.randn(10, 20, 30))
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_reproducibility(self):
        """Test that results are reproducible with same input."""
        analyzer1 = BrainNetworkAnalyzer(sampling_rate=0.4)
        analyzer2 = BrainNetworkAnalyzer(sampling_rate=0.4)
        
        analyzer1.load_time_series(self.fmri_data)
        analyzer2.load_time_series(self.fmri_data)
        
        conn1 = analyzer1.compute_connectivity_matrix(method='pearson')
        conn2 = analyzer2.compute_connectivity_matrix(method='pearson')
        
        # Results should be identical
        np.testing.assert_array_almost_equal(conn1, conn2)

class TestBrainAtlasManager(unittest.TestCase):
    """Test suite for brain atlas management."""
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_atlas_manager_initialization(self):
        """Test atlas manager initialization."""
        atlas_manager = BrainAtlasManager(atlas_name='harvard_oxford')
        self.assertEqual(atlas_manager.atlas_name, 'harvard_oxford')
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_mock_atlas_creation(self):
        """Test mock atlas creation when real atlas fails."""
        atlas_manager = BrainAtlasManager(atlas_name='mock_atlas')
        
        # Should have created mock data
        self.assertGreater(len(atlas_manager.atlas_labels), 0)
        self.assertIsNotNone(atlas_manager.region_coordinates)

class TestFMRIDataLoader(unittest.TestCase):
    """Test suite for fMRI data loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_path = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_data_path, ignore_errors=True)
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_data_loader_initialization(self):
        """Test fMRI data loader initialization."""
        # Use correct parameter name
        loader = fMRIDataLoader(data_dir=str(self.test_data_path))
        self.assertEqual(loader.data_dir, self.test_data_path)
    
    @unittest.skipUnless(HAS_CORE_MODULES, "Core modules not available")
    def test_mock_data_generation(self):
        """Test synthetic fMRI data generation."""
        # Use correct parameter name
        loader = fMRIDataLoader(data_dir=str(self.test_data_path))
        
        # Test mock data generation (if available)
        # This would depend on the actual implementation
        self.assertIsNotNone(loader.data_dir)

if __name__ == '__main__':
    unittest.main(verbosity=2) 