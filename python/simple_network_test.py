#!/usr/bin/env python3
"""
Simple Neural Network Analysis Test

A streamlined test of the Phase 2 neural network analysis capabilities.
"""

import numpy as np
import sys
from pathlib import Path

def test_neural_network_analysis():
    """Run a simplified test of the neural network analyzer."""
    print(" Phase 2: Neural Network Analysis Test")
    print("=" * 60)
    
    try:
        # Import our modules
        from fmri_data_loader import fMRIDataLoader
        from neural_network_analyzer import BrainNetworkAnalyzer
        
        # Load sample data
        print(" Loading sample fMRI data...")
        loader = fMRIDataLoader()
        fmri_data, behavioral_data = loader.get_sample_analysis_data()
        
        # Use a smaller subset for faster testing
        sample_data = fmri_data[:200, :100].T  # 100 timepoints x 200 regions
        print(f"   Using subset: {sample_data.shape} (timepoints x regions)")
        
        # Initialize analyzer
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        analyzer.load_time_series(sample_data)
        
        # Preprocessing
        print("\n Preprocessing time series...")
        processed_data = analyzer.preprocess_time_series(
            bandpass_filter=(0.01, 0.08),
            standardize=True,
            remove_motion=False  # Skip for speed
        )
        
        # Static connectivity analysis
        print("\n Computing static connectivity...")
        static_connectivity = analyzer.compute_connectivity_matrix(processed_data)
        print(f"   Connectivity matrix shape: {static_connectivity.shape}")
        print(f"   Mean correlation: {np.mean(static_connectivity):.3f}")
        print(f"   Max correlation: {np.max(static_connectivity):.3f}")
        
        # Dynamic connectivity analysis
        print("\n Dynamic connectivity analysis...")
        dynamic_matrices = analyzer.dynamic_connectivity_analysis(
            processed_data,
            window_size=20,  # Shorter window for speed
            step_size=5
        )
        
        # Change detection
        print("\n Detecting network changes...")
        change_points = analyzer.detect_network_changes(threshold=0.1)
        
        # Network metrics
        print("\n Computing network metrics...")
        metrics = analyzer.compute_network_metrics(static_connectivity, threshold=0.2)
        
        # Module identification
        print("\n Identifying network modules...")
        modules, silhouette = analyzer.identify_network_modules(static_connectivity)
        
        # Generate final report
        print("\n Generating analysis report...")
        report = analyzer.generate_analysis_report()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 PHASE 2 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f" Analysis Summary:")
        print(f"   Data duration: {report['data_info']['duration_seconds']:.1f} seconds")
        print(f"   Brain regions: {report['data_info']['n_regions']}")
        print(f"   Dynamic windows: {report['connectivity_analysis']['n_dynamic_matrices']}")
        print(f"   Network changes detected: {report['connectivity_analysis']['n_changes_detected']}")
        print(f"   Network density: {metrics.get('density', 0):.3f}")
        print(f"   Network modules: {len(np.unique(modules))}")
        
        print(f"\n Key Capabilities Demonstrated:")
        print(f"    Advanced fMRI preprocessing")
        print(f"    Static connectivity analysis")
        print(f"    Dynamic connectivity tracking")
        print(f"    Temporal change detection")
        print(f"    Network topology analysis")
        print(f"    Brain module identification")
        print(f"    Comprehensive reporting")
        
        # Performance metrics
        n_connections = static_connectivity.shape[0] * (static_connectivity.shape[0] - 1) // 2
        significant_connections = np.sum(np.abs(static_connectivity) > 0.2) // 2
        
        print(f"\n Performance Metrics:")
        print(f"   Total connections analyzed: {n_connections:,}")
        print(f"   Significant connections: {significant_connections:,}")
        print(f"   Processing efficiency: {significant_connections/n_connections*100:.1f}% sparse")
        
        return True
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_neural_network_analysis()
    sys.exit(0 if success else 1) 