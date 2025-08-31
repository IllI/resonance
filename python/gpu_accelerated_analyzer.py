#!/usr/bin/env python3
"""
GPU-Accelerated Brain Analyzer

This module integrates the C++ OpenCL GPU acceleration with the Python
neuroimaging analysis pipeline, combining the best of both worlds for
high-performance brain analysis.
"""

import numpy as np
import ctypes
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import our Phase 2 modules
try:
    from neural_network_analyzer import BrainNetworkAnalyzer
    from brain_atlas_manager import BrainAtlasManager
    from fmri_data_loader import fMRIDataLoader
    HAS_PHASE2_MODULES = True
except ImportError as e:
    print(f"⚠️ Phase 2 modules not available: {e}")
    HAS_PHASE2_MODULES = False

class GPUAcceleratedBrainAnalyzer:
    """
    High-performance brain analyzer combining C++ GPU acceleration 
    with Python neuroimaging algorithms.
    
    Phase 3 Integration: Seamless C++/Python data exchange for maximum performance.
    """
    
    def __init__(self, gpu_lib_path: Optional[str] = None):
        """
        Initialize the GPU-accelerated brain analyzer.
        
        Args:
            gpu_lib_path: Path to the brain_gpu_lib.dll
        """
        self.gpu_lib = None
        self.gpu_initialized = False
        
        # Phase 2 components
        self.network_analyzer = None
        self.atlas_manager = None
        self.data_loader = None
        
        # Performance tracking
        self.gpu_times = []
        self.cpu_times = []
        
        print("🚀 Initializing GPU-Accelerated Brain Analyzer...")
        
        # Load GPU library
        if self._load_gpu_library(gpu_lib_path):
            self._initialize_phase2_components()
        
    def _load_gpu_library(self, gpu_lib_path: Optional[str] = None) -> bool:
        """Load the C++ GPU acceleration library."""
        try:
            # Determine library path
            if gpu_lib_path is None:
                # Default to build directory
                current_dir = Path(__file__).parent.parent
                gpu_lib_path = current_dir / "build" / "bin" / "Release" / "brain_gpu_lib.dll"
            
            if not Path(gpu_lib_path).exists():
                print(f"⚠️ GPU library not found at: {gpu_lib_path}")
                print("   GPU acceleration will not be available")
                return False
            
            # Load the shared library
            self.gpu_lib = ctypes.CDLL(str(gpu_lib_path))
            
            # Define function signatures
            self._define_gpu_functions()
            
            # Initialize GPU
            if self.gpu_lib.init_brain_gpu_accelerator():
                self.gpu_initialized = True
                print("✅ GPU acceleration library loaded successfully")
                
                # Show GPU performance info
                self.gpu_lib.gpu_get_performance_info()
                return True
            else:
                print("❌ Failed to initialize GPU accelerator")
                return False
                
        except Exception as e:
            print(f"❌ Error loading GPU library: {e}")
            return False
    
    def _define_gpu_functions(self):
        """Define the C function signatures for the GPU library."""
        
        # Initialize GPU accelerator
        self.gpu_lib.init_brain_gpu_accelerator.restype = ctypes.c_bool
        
        # Compute correlation matrix
        self.gpu_lib.gpu_compute_correlation_matrix.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # time_series
            ctypes.POINTER(ctypes.c_float),  # correlation_matrix
            ctypes.c_int,                    # n_regions
            ctypes.c_int                     # n_timepoints
        ]
        self.gpu_lib.gpu_compute_correlation_matrix.restype = ctypes.c_bool
        
        # Preprocess time series
        self.gpu_lib.gpu_preprocess_timeseries.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input_data
            ctypes.POINTER(ctypes.c_float),  # output_data
            ctypes.c_int,                    # n_regions
            ctypes.c_int,                    # n_timepoints
            ctypes.c_float,                  # filter_low
            ctypes.c_float                   # filter_high
        ]
        self.gpu_lib.gpu_preprocess_timeseries.restype = ctypes.c_bool
        
        # Compute network metrics
        self.gpu_lib.gpu_compute_network_metrics.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # adjacency_matrix
            ctypes.POINTER(ctypes.c_float),  # node_strength
            ctypes.POINTER(ctypes.c_float),  # clustering_coeff
            ctypes.c_int,                    # n_nodes
            ctypes.c_float                   # threshold
        ]
        self.gpu_lib.gpu_compute_network_metrics.restype = ctypes.c_bool
        
        # Performance info
        self.gpu_lib.gpu_get_performance_info.restype = None
        
        # Cleanup
        self.gpu_lib.cleanup_brain_gpu_accelerator.restype = None
    
    def _initialize_phase2_components(self):
        """Initialize Phase 2 analysis components."""
        if not HAS_PHASE2_MODULES:
            print("⚠️ Phase 2 modules not available - limited functionality")
            return
        
        try:
            # Initialize components
            self.network_analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
            self.atlas_manager = BrainAtlasManager('harvard_oxford')
            self.data_loader = fMRIDataLoader()
            
            print("✅ Phase 2 components initialized")
            
        except Exception as e:
            print(f"⚠️ Error initializing Phase 2 components: {e}")
    
    def gpu_preprocess_timeseries(self, 
                                time_series: np.ndarray,
                                filter_low: float = 0.01,
                                filter_high: float = 0.08) -> np.ndarray:
        """
        GPU-accelerated preprocessing of fMRI time series.
        
        Args:
            time_series: Input data (timepoints x regions)
            filter_low: Low frequency filter
            filter_high: High frequency filter
            
        Returns:
            Preprocessed time series data
        """
        if not self.gpu_initialized:
            print("⚠️ GPU not initialized, falling back to CPU")
            return self._cpu_preprocess_timeseries(time_series, filter_low, filter_high)
        
        print("🖥️ GPU-accelerated preprocessing...")
        start_time = time.time()
        
        # Prepare data
        n_timepoints, n_regions = time_series.shape
        input_data = time_series.T.astype(np.float32)  # Convert to regions x timepoints
        output_data = np.zeros_like(input_data)
        
        # Create ctypes pointers
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call GPU function
        success = self.gpu_lib.gpu_preprocess_timeseries(
            input_ptr, output_ptr, n_regions, n_timepoints,
            ctypes.c_float(filter_low), ctypes.c_float(filter_high)
        )
        
        gpu_time = time.time() - start_time
        self.gpu_times.append(gpu_time)
        
        if success:
            result = output_data.T  # Convert back to timepoints x regions
            print(f"✅ GPU preprocessing completed in {gpu_time*1000:.2f} ms")
            return result
        else:
            print("❌ GPU preprocessing failed, falling back to CPU")
            return self._cpu_preprocess_timeseries(time_series, filter_low, filter_high)
    
    def gpu_compute_correlation_matrix(self, time_series: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated correlation matrix computation.
        
        Args:
            time_series: Preprocessed time series (timepoints x regions)
            
        Returns:
            Correlation matrix (regions x regions)
        """
        if not self.gpu_initialized:
            print("⚠️ GPU not initialized, falling back to CPU")
            return self._cpu_compute_correlation_matrix(time_series)
        
        print("🖥️ GPU-accelerated correlation matrix computation...")
        start_time = time.time()
        
        # Prepare data
        n_timepoints, n_regions = time_series.shape
        input_data = time_series.T.astype(np.float32)  # Convert to regions x timepoints
        correlation_matrix = np.zeros((n_regions, n_regions), dtype=np.float32)
        
        # Create ctypes pointers
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = correlation_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call GPU function
        success = self.gpu_lib.gpu_compute_correlation_matrix(
            input_ptr, output_ptr, n_regions, n_timepoints
        )
        
        gpu_time = time.time() - start_time
        self.gpu_times.append(gpu_time)
        
        if success:
            print(f"✅ GPU correlation matrix computed in {gpu_time*1000:.2f} ms")
            return correlation_matrix.astype(np.float64)  # Convert back to double precision
        else:
            print("❌ GPU correlation computation failed, falling back to CPU")
            return self._cpu_compute_correlation_matrix(time_series)
    
    def gpu_compute_network_metrics(self, 
                                   correlation_matrix: np.ndarray,
                                   threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated network metrics computation.
        
        Args:
            correlation_matrix: Connectivity matrix
            threshold: Connection threshold
            
        Returns:
            Tuple of (node_strength, clustering_coefficients)
        """
        if not self.gpu_initialized:
            print("⚠️ GPU not initialized, falling back to CPU")
            return self._cpu_compute_network_metrics(correlation_matrix, threshold)
        
        print("🖥️ GPU-accelerated network metrics computation...")
        start_time = time.time()
        
        # Prepare data
        n_nodes = correlation_matrix.shape[0]
        input_data = correlation_matrix.astype(np.float32)
        node_strength = np.zeros(n_nodes, dtype=np.float32)
        clustering_coeff = np.zeros(n_nodes, dtype=np.float32)
        
        # Create ctypes pointers
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        strength_ptr = node_strength.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        clustering_ptr = clustering_coeff.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call GPU function
        success = self.gpu_lib.gpu_compute_network_metrics(
            input_ptr, strength_ptr, clustering_ptr, n_nodes, ctypes.c_float(threshold)
        )
        
        gpu_time = time.time() - start_time
        self.gpu_times.append(gpu_time)
        
        if success:
            print(f"✅ GPU network metrics computed in {gpu_time*1000:.2f} ms")
            return node_strength.astype(np.float64), clustering_coeff.astype(np.float64)
        else:
            print("❌ GPU network metrics computation failed, falling back to CPU")
            return self._cpu_compute_network_metrics(correlation_matrix, threshold)
    
    def analyze_fmri_data_accelerated(self, 
                                    fmri_data: np.ndarray,
                                    use_gpu: bool = True) -> Dict:
        """
        Complete GPU-accelerated fMRI analysis pipeline.
        
        Args:
            fmri_data: fMRI time series data (regions x timepoints)
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Comprehensive analysis results
        """
        print("🧠 Starting GPU-accelerated fMRI analysis...")
        analysis_start = time.time()
        
        # Transpose to timepoints x regions for consistency
        if fmri_data.shape[0] > fmri_data.shape[1]:
            time_series = fmri_data.T
        else:
            time_series = fmri_data
        
        results = {
            'gpu_acceleration_used': use_gpu and self.gpu_initialized,
            'data_shape': time_series.shape,
            'processing_times': {}
        }
        
                 # Step 1: GPU-accelerated preprocessing
        if use_gpu and self.gpu_initialized:
            preprocessed_data = self.gpu_preprocess_timeseries(time_series)
        else:
            preprocessed_data = self._cpu_preprocess_timeseries(time_series, 0.01, 0.08)
        
        # Step 2: GPU-accelerated correlation matrix
        if use_gpu and self.gpu_initialized:
            correlation_matrix = self.gpu_compute_correlation_matrix(preprocessed_data)
        else:
            correlation_matrix = self._cpu_compute_correlation_matrix(preprocessed_data)
        
        results['correlation_matrix'] = correlation_matrix
        
        # Step 3: GPU-accelerated network metrics
        if use_gpu and self.gpu_initialized:
            node_strength, clustering_coeff = self.gpu_compute_network_metrics(correlation_matrix)
        else:
            node_strength, clustering_coeff = self._cpu_compute_network_metrics(correlation_matrix)
        
        results['node_strength'] = node_strength
        results['clustering_coefficients'] = clustering_coeff
        
        # Step 4: Phase 2 analysis integration
        if self.network_analyzer is not None:
            print("🔗 Integrating with Phase 2 analysis...")
            
            # Load data into network analyzer
            self.network_analyzer.load_time_series(preprocessed_data)
            
            # Dynamic connectivity analysis (CPU-based from Phase 2)
            dynamic_matrices = self.network_analyzer.dynamic_connectivity_analysis(
                preprocessed_data, window_size=30, step_size=10
            )
            
            # Change detection
            change_points = self.network_analyzer.detect_network_changes()
            
            results['dynamic_matrices'] = dynamic_matrices
            results['change_points'] = change_points
            results['n_changes_detected'] = len(change_points)
        
        # Step 5: Brain atlas integration
        if self.atlas_manager is not None:
            print("🗺️ Mapping to brain atlas...")
            
            # Map connectivity to atlas
            mapped_connectivity = self.atlas_manager.map_connectivity_to_atlas(correlation_matrix)
            
            # Identify network hubs
            hubs = self.atlas_manager.identify_network_hubs(correlation_matrix)
            
            results['mapped_connectivity'] = mapped_connectivity
            results['network_hubs'] = hubs
        
        # Performance summary
        total_time = time.time() - analysis_start
        results['total_analysis_time'] = total_time
        results['gpu_times'] = self.gpu_times.copy()
        results['performance_summary'] = self.get_performance_summary()
        
        print(f"✅ GPU-accelerated analysis completed in {total_time:.2f} seconds")
        
        return results
    
    def _cpu_preprocess_timeseries(self, time_series: np.ndarray, filter_low: float, filter_high: float) -> np.ndarray:
        """CPU fallback for preprocessing."""
        if self.network_analyzer is not None:
            # Load data first
            self.network_analyzer.load_time_series(time_series)
            return self.network_analyzer.preprocess_time_series(
                bandpass_filter=(filter_low, filter_high)
            )
        else:
            # Simple CPU preprocessing
            print("🖥️ CPU preprocessing (simplified)...")
            return (time_series - np.mean(time_series, axis=0)) / np.std(time_series, axis=0)
    
    def _cpu_compute_correlation_matrix(self, time_series: np.ndarray) -> np.ndarray:
        """CPU fallback for correlation matrix."""
        print("🖥️ CPU correlation matrix computation...")
        start_time = time.time()
        correlation_matrix = np.corrcoef(time_series.T)
        cpu_time = time.time() - start_time
        self.cpu_times.append(cpu_time)
        print(f"✅ CPU correlation matrix computed in {cpu_time*1000:.2f} ms")
        return correlation_matrix
    
    def _cpu_compute_network_metrics(self, correlation_matrix: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for network metrics."""
        print("🖥️ CPU network metrics computation...")
        start_time = time.time()
        
        # Simple node strength computation
        binary_matrix = (np.abs(correlation_matrix) > threshold).astype(float)
        np.fill_diagonal(binary_matrix, 0)
        node_strength = np.sum(binary_matrix, axis=1)
        
        # Simple clustering coefficient (placeholder)
        clustering_coeff = np.zeros_like(node_strength)
        
        cpu_time = time.time() - start_time
        self.cpu_times.append(cpu_time)
        print(f"✅ CPU network metrics computed in {cpu_time*1000:.2f} ms")
        
        return node_strength, clustering_coeff
    
    def get_performance_summary(self) -> Dict:
        """Get performance comparison between GPU and CPU."""
        summary = {
            'gpu_available': self.gpu_initialized,
            'gpu_operations': len(self.gpu_times),
            'cpu_operations': len(self.cpu_times)
        }
        
        if self.gpu_times:
            summary['avg_gpu_time_ms'] = np.mean(self.gpu_times) * 1000
            summary['total_gpu_time_ms'] = np.sum(self.gpu_times) * 1000
        
        if self.cpu_times:
            summary['avg_cpu_time_ms'] = np.mean(self.cpu_times) * 1000
            summary['total_cpu_time_ms'] = np.sum(self.cpu_times) * 1000
        
        if self.gpu_times and self.cpu_times:
            speedup = np.mean(self.cpu_times) / np.mean(self.gpu_times)
            summary['gpu_speedup'] = speedup
        
        return summary
    
    def __del__(self):
        """Cleanup GPU resources."""
        if self.gpu_lib is not None and self.gpu_initialized:
            try:
                self.gpu_lib.cleanup_brain_gpu_accelerator()
            except:
                pass

def main():
    """Test the GPU-accelerated brain analyzer."""
    print("🚀 GPU-Accelerated Brain Analysis Test")
    print("=" * 60)
    
    try:
        # Initialize GPU-accelerated analyzer
        analyzer = GPUAcceleratedBrainAnalyzer()
        
        if not HAS_PHASE2_MODULES:
            print("⚠️ Phase 2 modules not available - testing GPU functions only")
            
            # Create mock data for GPU testing
            n_regions, n_timepoints = 100, 200
            mock_data = np.random.randn(n_timepoints, n_regions).astype(np.float32)
            
            # Test GPU preprocessing
            processed = analyzer.gpu_preprocess_timeseries(mock_data)
            
            # Test GPU correlation
            correlation = analyzer.gpu_compute_correlation_matrix(processed)
            
            # Test GPU network metrics
            strength, clustering = analyzer.gpu_compute_network_metrics(correlation)
            
            print(f"\n📊 GPU Test Results:")
            print(f"   Processed data shape: {processed.shape}")
            print(f"   Correlation matrix shape: {correlation.shape}")
            print(f"   Node strength range: [{strength.min():.3f}, {strength.max():.3f}]")
            
        else:
            # Full integration test with Phase 2
            print("🧠 Loading sample fMRI data...")
            
            # Get sample data
            loader = fMRIDataLoader()
            fmri_data, behavioral_data = loader.get_sample_analysis_data()
            
            # Use a subset for faster testing
            sample_data = fmri_data[:200, :100]  # 200 regions x 100 timepoints
            
            # Run complete GPU-accelerated analysis
            results = analyzer.analyze_fmri_data_accelerated(sample_data, use_gpu=True)
            
            print(f"\n📊 GPU-Accelerated Analysis Results:")
            print(f"   Data shape: {results['data_shape']}")
            print(f"   GPU acceleration used: {results['gpu_acceleration_used']}")
            print(f"   Correlation matrix: {results['correlation_matrix'].shape}")
            print(f"   Dynamic matrices: {len(results.get('dynamic_matrices', []))}")
            print(f"   Changes detected: {results.get('n_changes_detected', 0)}")
            print(f"   Network hubs: {len(results.get('network_hubs', []))}")
            print(f"   Total analysis time: {results['total_analysis_time']:.2f} seconds")
        
        # Performance summary
        perf = analyzer.get_performance_summary()
        print(f"\n⚡ Performance Summary:")
        for key, value in perf.items():
            print(f"   {key}: {value}")
        
        print("\n🎉 GPU-accelerated brain analysis test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 