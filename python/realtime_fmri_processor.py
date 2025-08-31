#!/usr/bin/env python3
"""
Real-time fMRI Processing System

Processes fMRI data in real-time as it streams from the scanner,
leveraging dual-GPU acceleration for immediate analysis and feedback.

Phase 4: Real-time brain analysis with sub-second latency.
"""

import numpy as np
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any
from queue import Queue, Empty
import json
from pathlib import Path
import logging

# Import our analysis components
try:
    from enhanced_dual_gpu_analyzer import EnhancedDualGPUBrainAnalyzer
    from brain_3d_visualizer import Brain3DVisualizer
    from neural_network_analyzer import BrainNetworkAnalyzer
    HAS_ANALYSIS_MODULES = True
except ImportError as e:
    print(f"⚠️ Analysis modules not available: {e}")
    HAS_ANALYSIS_MODULES = False

@dataclass
class fMRITimePoint:
    """Single fMRI time point data structure."""
    timestamp: float
    volume_data: np.ndarray
    tr_index: int
    scanner_params: Dict[str, Any]
    quality_metrics: Dict[str, float]

@dataclass
class ProcessingResults:
    """Results from real-time processing."""
    timestamp: float
    connectivity_matrix: np.ndarray
    network_metrics: Dict[str, Any]
    roi_activations: Dict[str, float]
    quality_flags: List[str]
    processing_time: float
    gpu_utilization: Dict[str, float]

class RealTimeFMRIProcessor:
    """
    Real-time fMRI processing system with dual-GPU acceleration.
    
    Capabilities:
    - Sub-second processing latency
    - Dual-GPU load balancing
    - Real-time quality assessment
    - Streaming connectivity analysis
    - Automated artifact detection
    - Live neurofeedback signals
    """
    
    def __init__(self, 
                 buffer_size: int = 50,
                 processing_window: int = 20,
                 update_interval: float = 0.5):
        """
        Initialize the real-time processor.
        
        Args:
            buffer_size: Maximum number of time points to buffer
            processing_window: Sliding window size for connectivity analysis
            update_interval: Update interval in seconds
        """
        self.buffer_size = buffer_size
        self.processing_window = processing_window
        self.update_interval = update_interval
        
        # Data buffers and queues
        self.data_buffer = Queue(maxsize=buffer_size)
        self.results_queue = Queue()
        self.time_series_buffer = []
        
        # Processing components
        if HAS_ANALYSIS_MODULES:
            self.gpu_analyzer = EnhancedDualGPUBrainAnalyzer()
            self.visualizer = Brain3DVisualizer()
        else:
            self.gpu_analyzer = None
            self.visualizer = None
        
        # Real-time processing state
        self.is_processing = False
        self.processing_thread = None
        self.visualization_thread = None
        self.performance_monitor = PerformanceMonitor()
        
        # Callbacks for real-time events
        self.callbacks = {
            'new_results': [],
            'quality_alert': [],
            'network_change': [],
            'roi_activation': []
        }
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("🚀 Real-time fMRI Processor initialized")
        print(f"   Buffer size: {buffer_size} volumes")
        print(f"   Processing window: {processing_window} TRs")
        print(f"   Update interval: {update_interval}s")
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for real-time events."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def start_processing(self):
        """Start real-time processing."""
        if self.is_processing:
            print("⚠️ Processing already running")
            return
        
        print("🚀 Starting real-time fMRI processing...")
        self.is_processing = True
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.visualization_thread = threading.Thread(target=self._visualization_loop)
        
        self.processing_thread.start()
        self.visualization_thread.start()
        
        print("✅ Real-time processing started")
    
    def stop_processing(self):
        """Stop real-time processing."""
        print("⏹️ Stopping real-time processing...")
        
        self.is_processing = False
        
        # Wait for threads to complete
        if self.processing_thread:
            self.processing_thread.join()
        if self.visualization_thread:
            self.visualization_thread.join()
        
        print("✅ Real-time processing stopped")
    
    def add_volume(self, volume_data: np.ndarray, 
                  scanner_params: Dict = None,
                  timestamp: float = None):
        """
        Add new fMRI volume for real-time processing.
        
        Args:
            volume_data: fMRI volume data (regions x 1)
            scanner_params: Scanner parameters (TR, TE, etc.)
            timestamp: Acquisition timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        if scanner_params is None:
            scanner_params = {'TR': 2.5, 'TE': 30.0}
        
        # Calculate basic quality metrics
        quality_metrics = self._assess_volume_quality(volume_data)
        
        # Create time point structure
        timepoint = fMRITimePoint(
            timestamp=timestamp,
            volume_data=volume_data.flatten() if volume_data.ndim > 1 else volume_data,
            tr_index=len(self.time_series_buffer),
            scanner_params=scanner_params,
            quality_metrics=quality_metrics
        )
        
        # Add to buffer (non-blocking)
        try:
            self.data_buffer.put_nowait(timepoint)
            self.performance_monitor.record_acquisition(timestamp)
        except:
            self.logger.warning("Data buffer full - dropping oldest volume")
            try:
                self.data_buffer.get_nowait()  # Remove oldest
                self.data_buffer.put_nowait(timepoint)  # Add new
            except:
                pass
    
    def _assess_volume_quality(self, volume_data: np.ndarray) -> Dict[str, float]:
        """Assess quality metrics for incoming volume."""
        volume_flat = volume_data.flatten()
        
        return {
            'mean_signal': float(np.mean(volume_flat)),
            'std_signal': float(np.std(volume_flat)),
            'snr': float(np.mean(volume_flat) / np.std(volume_flat)) if np.std(volume_flat) > 0 else 0.0,
            'outlier_ratio': float(np.sum(np.abs(volume_flat) > 3 * np.std(volume_flat)) / len(volume_flat)),
            'motion_estimate': float(np.random.rand())  # Mock motion estimate
        }
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        print("🔄 Starting processing loop...")
        
        while self.is_processing:
            try:
                # Get new data from buffer
                timepoint = self.data_buffer.get(timeout=0.1)
                
                # Add to time series buffer
                self.time_series_buffer.append(timepoint)
                
                # Maintain buffer size
                if len(self.time_series_buffer) > self.buffer_size:
                    self.time_series_buffer.pop(0)
                
                # Process if we have enough data
                if len(self.time_series_buffer) >= self.processing_window:
                    results = self._process_current_window()
                    if results:
                        self.results_queue.put(results)
                        self._trigger_callbacks('new_results', results)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                continue
        
        print("⏹️ Processing loop stopped")
    
    def _process_current_window(self) -> Optional[ProcessingResults]:
        """Process current sliding window of data."""
        start_time = time.time()
        
        try:
            # Extract recent data
            window_data = self.time_series_buffer[-self.processing_window:]
            
            # Convert to analysis format
            time_series_matrix = np.array([tp.volume_data for tp in window_data]).T
            
            if self.gpu_analyzer and HAS_ANALYSIS_MODULES:
                # **DUAL-GPU REAL-TIME PROCESSING**
                print(f"🔥 Processing window: {time_series_matrix.shape} with dual-GPU...")
                
                # Optimized real-time analysis
                results = self.gpu_analyzer.analyze_fmri_data_dual_gpu(
                    time_series_matrix,
                    enable_quality_control=True,
                    realtime_mode=True
                )
                
                # Extract ROI activations (mock for now)
                roi_activations = self._extract_roi_activations(results)
                
                # Check for significant network changes
                network_change = self._detect_network_changes(results)
                if network_change:
                    self._trigger_callbacks('network_change', network_change)
                
                # GPU utilization from performance monitor
                gpu_util = {
                    'gpu0_utilization': 0.85,  # Mock values
                    'gpu1_utilization': 0.72,
                    'memory_usage': 0.60
                }
                
            else:
                # CPU fallback processing
                print("🔧 Processing with CPU fallback...")
                results = self._cpu_fallback_processing(time_series_matrix)
                roi_activations = {}
                gpu_util = {'cpu_usage': 0.45}
            
            processing_time = time.time() - start_time
            
            # Create results object
            processing_results = ProcessingResults(
                timestamp=time.time(),
                connectivity_matrix=results.get('correlation_matrix', np.array([])),
                network_metrics=results.get('network_metrics', {}),
                roi_activations=roi_activations,
                quality_flags=self._check_quality_flags(window_data),
                processing_time=processing_time,
                gpu_utilization=gpu_util
            )
            
            # Performance monitoring
            self.performance_monitor.record_processing(processing_time)
            
            print(f"⚡ Processed in {processing_time:.3f}s | GPUs: {gpu_util}")
            
            return processing_results
            
        except Exception as e:
            self.logger.error(f"Window processing error: {e}")
            return None
    
    def _extract_roi_activations(self, results: Dict) -> Dict[str, float]:
        """Extract region-of-interest activation levels."""
        if 'network_metrics' in results and 'node_strength' in results['network_metrics']:
            node_strength = results['network_metrics']['node_strength']
            
            # Define key ROIs (simplified)
            roi_mapping = {
                'default_mode_network': np.mean(node_strength[:10]),
                'executive_control': np.mean(node_strength[10:20]),
                'salience_network': np.mean(node_strength[20:30]),
                'visual_cortex': np.mean(node_strength[30:40]),
                'motor_cortex': np.mean(node_strength[40:50])
            }
            
            return {roi: float(activation) for roi, activation in roi_mapping.items()}
        
        return {}
    
    def _detect_network_changes(self, results: Dict) -> Optional[Dict]:
        """Detect significant changes in network connectivity."""
        # Simplified change detection
        if 'correlation_matrix' in results:
            matrix = results['correlation_matrix']
            mean_connectivity = np.mean(np.abs(matrix))
            
            # Compare with running average (mock implementation)
            if hasattr(self, 'connectivity_baseline'):
                change = abs(mean_connectivity - self.connectivity_baseline)
                if change > 0.1:  # Threshold for significant change
                    return {
                        'change_magnitude': float(change),
                        'change_type': 'connectivity_increase' if mean_connectivity > self.connectivity_baseline else 'connectivity_decrease',
                        'timestamp': time.time()
                    }
            else:
                self.connectivity_baseline = mean_connectivity
        
        return None
    
    def _check_quality_flags(self, window_data: List[fMRITimePoint]) -> List[str]:
        """Check for quality issues in current window."""
        flags = []
        
        # Check for motion
        motion_values = [tp.quality_metrics.get('motion_estimate', 0) for tp in window_data]
        if max(motion_values) > 0.5:  # Motion threshold
            flags.append('high_motion')
        
        # Check for signal outliers
        snr_values = [tp.quality_metrics.get('snr', 0) for tp in window_data]
        if min(snr_values) < 10:  # SNR threshold
            flags.append('low_snr')
        
        # Check for artifacts
        outlier_ratios = [tp.quality_metrics.get('outlier_ratio', 0) for tp in window_data]
        if max(outlier_ratios) > 0.05:  # 5% outlier threshold
            flags.append('signal_artifacts')
        
        return flags
    
    def _cpu_fallback_processing(self, time_series_matrix: np.ndarray) -> Dict:
        """CPU fallback for when GPU processing fails."""
        # Simple correlation analysis
        correlation_matrix = np.corrcoef(time_series_matrix)
        
        # Basic network metrics
        node_strength = np.sum(np.abs(correlation_matrix), axis=1)
        
        return {
            'correlation_matrix': correlation_matrix,
            'network_metrics': {
                'node_strength': node_strength,
                'global_efficiency': np.mean(node_strength),
                'clustering_coefficient': np.random.rand(len(node_strength))  # Mock
            }
        }
    
    def _visualization_loop(self):
        """Visualization update loop."""
        print("🎨 Starting visualization loop...")
        
        while self.is_processing:
            try:
                # Get latest results
                results = self.results_queue.get(timeout=self.update_interval)
                
                if self.visualizer and HAS_ANALYSIS_MODULES:
                    # Update 3D brain visualization
                    self._update_brain_visualization(results)
                
                # Update performance dashboard
                self._update_performance_dashboard(results)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Visualization error: {e}")
                continue
        
        print("⏹️ Visualization loop stopped")
    
    def _update_brain_visualization(self, results: ProcessingResults):
        """Update real-time 3D brain visualization."""
        try:
            if results.connectivity_matrix.size > 0:
                # Create/update 3D connectivity visualization
                fig = self.visualizer.visualize_static_connectivity(
                    results.connectivity_matrix,
                    threshold=0.3,
                    title=f"Real-time Brain Network (t={results.timestamp:.1f}s)"
                )
                
                # Save for real-time viewing
                if fig:
                    output_path = Path("visualizations/realtime_brain.html")
                    output_path.parent.mkdir(exist_ok=True)
                    fig.write_html(str(output_path))
        
        except Exception as e:
            self.logger.error(f"Brain visualization update error: {e}")
    
    def _update_performance_dashboard(self, results: ProcessingResults):
        """Update performance monitoring dashboard."""
        perf_data = {
            'timestamp': results.timestamp,
            'processing_time': results.processing_time,
            'gpu_utilization': results.gpu_utilization,
            'quality_flags': results.quality_flags,
            'roi_activations': results.roi_activations
        }
        
        # Save performance data
        perf_file = Path("logs/realtime_performance.json")
        perf_file.parent.mkdir(exist_ok=True)
        
        # Append to performance log
        try:
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            data.append(perf_data)
            
            # Keep only recent data (last 1000 points)
            if len(data) > 1000:
                data = data[-1000:]
            
            with open(perf_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Performance log error: {e}")
    
    def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger registered callbacks for events."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Callback error for {event_type}: {e}")
    
    def get_current_status(self) -> Dict:
        """Get current processing status."""
        return {
            'is_processing': self.is_processing,
            'buffer_size': self.data_buffer.qsize(),
            'time_series_length': len(self.time_series_buffer),
            'performance_stats': self.performance_monitor.get_stats(),
            'last_update': time.time()
        }

class PerformanceMonitor:
    """Monitor real-time processing performance."""
    
    def __init__(self):
        self.acquisition_times = []
        self.processing_times = []
        self.start_time = time.time()
    
    def record_acquisition(self, timestamp: float):
        """Record data acquisition timestamp."""
        self.acquisition_times.append(timestamp)
        if len(self.acquisition_times) > 100:  # Keep recent 100
            self.acquisition_times.pop(0)
    
    def record_processing(self, processing_time: float):
        """Record processing duration."""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:  # Keep recent 100
            self.processing_times.pop(0)
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0,
            'acquisition_rate': len(self.acquisition_times) / (time.time() - self.start_time) if self.acquisition_times else 0,
            'processing_rate': len(self.processing_times) / (time.time() - self.start_time) if self.processing_times else 0
        }

class fMRIScannerSimulator:
    """Simulate fMRI scanner data stream for testing."""
    
    def __init__(self, n_regions: int = 100, tr: float = 2.5):
        self.n_regions = n_regions
        self.tr = tr
        self.volume_count = 0
        self.is_scanning = False
        
    def start_scan(self, processor: RealTimeFMRIProcessor, 
                  scan_duration: float = 60.0):
        """Simulate scanner data stream."""
        print(f"🔬 Starting fMRI scan simulation...")
        print(f"   Regions: {self.n_regions}")
        print(f"   TR: {self.tr}s")
        print(f"   Duration: {scan_duration}s")
        
        self.is_scanning = True
        start_time = time.time()
        
        while self.is_scanning and (time.time() - start_time) < scan_duration:
            # Generate realistic fMRI data
            volume_data = self._generate_realistic_volume()
            
            # Add to processor
            processor.add_volume(
                volume_data=volume_data,
                scanner_params={'TR': self.tr, 'TE': 30.0},
                timestamp=time.time()
            )
            
            self.volume_count += 1
            print(f"📡 Acquired volume {self.volume_count} at t={time.time()-start_time:.1f}s")
            
            # Wait for next TR
            time.sleep(self.tr)
        
        print(f"✅ Scan completed: {self.volume_count} volumes acquired")
    
    def _generate_realistic_volume(self) -> np.ndarray:
        """Generate realistic fMRI volume with network structure."""
        # Base signal
        volume = np.random.randn(self.n_regions) * 0.1
        
        # Add network structure (simplified)
        # Default mode network regions (correlated)
        dmn_regions = slice(0, 20)
        dmn_signal = np.random.randn() * 0.5
        volume[dmn_regions] += dmn_signal
        
        # Executive control network
        ecn_regions = slice(20, 40)
        ecn_signal = np.random.randn() * 0.4
        volume[ecn_regions] += ecn_signal
        
        # Add some task-related activation (mock)
        task_regions = slice(40, 50)
        task_activation = np.sin(self.volume_count * 0.1) * 0.3  # Slow oscillation
        volume[task_regions] += task_activation
        
        return volume
    
    def stop_scan(self):
        """Stop the scanner simulation."""
        self.is_scanning = False

def main():
    """Demonstrate real-time fMRI processing."""
    print("🚀 Real-time fMRI Processing System Demo")
    print("=" * 60)
    
    # Initialize real-time processor
    processor = RealTimeFMRIProcessor(
        buffer_size=100,
        processing_window=30,
        update_interval=2.0
    )
    
    # Add callback for new results
    def on_new_results(results):
        print(f"📊 New results: Processing time: {results.processing_time:.3f}s")
        if results.quality_flags:
            print(f"⚠️ Quality flags: {results.quality_flags}")
        
        # Print top ROI activations
        if results.roi_activations:
            top_roi = max(results.roi_activations.items(), key=lambda x: x[1])
            print(f"🧠 Top ROI: {top_roi[0]} = {top_roi[1]:.3f}")
    
    def on_network_change(change_info):
        print(f"🔄 Network change detected: {change_info['change_type']} "
              f"(magnitude: {change_info['change_magnitude']:.3f})")
    
    processor.add_callback('new_results', on_new_results)
    processor.add_callback('network_change', on_network_change)
    
    # Create scanner simulator
    scanner = fMRIScannerSimulator(n_regions=100, tr=2.5)
    
    try:
        # Start real-time processing
        processor.start_processing()
        
        # Simulate scan acquisition
        print("\n🔬 Starting scan simulation...")
        scan_thread = threading.Thread(
            target=scanner.start_scan,
            args=(processor, 30.0)  # 30 second scan
        )
        scan_thread.start()
        
        # Monitor processing status
        for i in range(15):  # Monitor for 15 intervals
            time.sleep(2.0)
            status = processor.get_current_status()
            print(f"📈 Status: Buffer={status['buffer_size']}, "
                  f"TimeSeries={status['time_series_length']}, "
                  f"Processing={status['is_processing']}")
        
        # Stop scanner
        scanner.stop_scan()
        scan_thread.join()
        
        # Final status
        final_status = processor.get_current_status()
        perf_stats = final_status['performance_stats']
        
        print(f"\n📊 Final Performance Statistics:")
        print(f"   Average processing time: {perf_stats['avg_processing_time']:.3f}s")
        print(f"   Maximum processing time: {perf_stats['max_processing_time']:.3f}s")
        print(f"   Acquisition rate: {perf_stats['acquisition_rate']:.2f} volumes/s")
        print(f"   Processing rate: {perf_stats['processing_rate']:.2f} analyses/s")
        
    finally:
        # Clean up
        processor.stop_processing()
        
    print("\n🎉 Real-time fMRI processing demonstration complete!")
    print("   Check 'logs/realtime_performance.json' for detailed metrics")
    print("   Check 'visualizations/realtime_brain.html' for 3D visualization")

if __name__ == "__main__":
    main() 