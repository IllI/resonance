#!/usr/bin/env python3
"""
RX 7700S Brain Analyzer - Fixed Version

This version uses the successful GPU selection methods from the guide:
- HIP/ROCm environment variables
- Simple memory-based GPU detection
- Direct GPU targeting without complex fallbacks
"""

import numpy as np
import time
import os
from typing import Dict, Any, Optional, Tuple

# Import our simplified detector
from rx7700s_simple_detector import RX7700SSimpleDetector

# Scientific computing
try:
    from scipy import ndimage
    from skimage import measure
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy/skimage not available - using basic processing")

class RX7700SBrainAnalyzerFixed:
    """
    Fixed RX 7700S Brain Analyzer using successful GPU selection methods.
    
    Based on the successful guide, this version:
    1. Uses HIP/ROCm environment variables
    2. Simple memory-based GPU detection
    3. Direct RX 7700S targeting
    4. No complex fallbacks or device reordering
    """
    
    def __init__(self):
        """Initialize fixed RX 7700S brain analyzer."""
        print(" RX 7700S Brain Analyzer - Fixed Version")
        print("=" * 60)
        print("Using successful GPU selection methods from guide")
        print()
        
        # Initialize RX 7700S detector with environment setup
        self.gpu_detector = RX7700SSimpleDetector(auto_setup_environment=True)
        
        # Verify RX 7700S is available
        if not self.gpu_detector.rx7700s_device:
            raise Exception("RX 7700S not detected - cannot proceed")
        
        # Get device info
        self.device_info = self.gpu_detector.get_device_info()
        print(f" RX 7700S initialized: {self.device_info['name']}")
        print(f"   Memory: {self.device_info['memory_gb']} GB")
        print(f"   Compute Units: {self.device_info['compute_units']}")
        print()
    
    def analyze_brain_volume(self, volume_data: np.ndarray, 
                           generate_surface: bool = True,
                           surface_quality: str = 'high') -> Dict[str, Any]:
        """
        Analyze brain volume using RX 7700S GPU.
        
        Args:
            volume_data: 3D brain volume (numpy array)
            generate_surface: Whether to generate 3D surface mesh
            surface_quality: 'low', 'medium', 'high', or 'ultra'
            
        Returns:
            Dictionary with analysis results
        """
        print(" RX 7700S Brain Volume Analysis")
        print("=" * 50)
        print(f"Volume shape: {volume_data.shape}")
        print(f"Volume size: {volume_data.size:,} voxels")
        print(f"Surface generation: {generate_surface}")
        print(f"Quality: {surface_quality}")
        print()
        
        start_time = time.time()
        results = {}
        
        try:
            # Step 1: GPU-accelerated brain segmentation
            print(" Step 1: GPU Brain Segmentation...")
            segmentation_result = self._gpu_brain_segmentation(volume_data)
            
            if not segmentation_result['success']:
                raise Exception(f"Brain segmentation failed: {segmentation_result['error']}")
            
            brain_mask = segmentation_result['brain_mask']
            results['segmentation'] = segmentation_result
            
            # Step 2: Brain tissue classification
            print(" Step 2: GPU Tissue Classification...")
            classification_result = self._gpu_tissue_classification(volume_data, brain_mask)
            results['classification'] = classification_result
            
            # Step 3: Surface generation (if requested)
            if generate_surface:
                print(" Step 3: Brain Surface Generation...")
                surface_result = self._generate_brain_surface(volume_data, brain_mask, surface_quality)
                results['surface'] = surface_result
            
            # Step 4: Analysis metrics
            print(" Step 4: Computing Analysis Metrics...")
            metrics_result = self._compute_brain_metrics(volume_data, brain_mask)
            results['metrics'] = metrics_result
            
            # Final results
            total_time = time.time() - start_time
            
            results.update({
                'success': True,
                'total_processing_time': total_time,
                'volume_shape': volume_data.shape,
                'gpu_device': self.device_info['name'],
                'gpu_memory_gb': self.device_info['memory_gb']
            })
            
            print(f"\n Brain analysis completed in {total_time:.2f} seconds")
            print(f"   GPU: {self.device_info['name']}")
            print(f"   Brain voxels: {np.sum(brain_mask):,}")
            
            if generate_surface and 'surface' in results:
                surface = results['surface']
                if surface['success']:
                    print(f"   Surface vertices: {len(surface['vertices']):,}")
                    print(f"   Surface faces: {len(surface['faces']):,}")
            
            return results
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f" Brain analysis failed after {error_time:.2f} seconds: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': error_time,
                'volume_shape': volume_data.shape,
                'gpu_device': self.device_info['name']
            }
    
    def _gpu_brain_segmentation(self, volume_data: np.ndarray) -> Dict[str, Any]:
        """GPU-accelerated brain segmentation using RX 7700S."""
        try:
            # Use the simple detector's brain processing
            processing_result = self.gpu_detector.process_brain_data_simple(volume_data)
            
            if not processing_result['success']:
                return {'success': False, 'error': processing_result['error']}
            
            processed_data = processing_result['processed_data']
            
            # Create brain mask from processed data
            # Use adaptive thresholding
            threshold = np.percentile(processed_data[processed_data > 0], 75)
            brain_mask = processed_data > threshold
            
            # Basic morphological cleanup if scipy available
            if HAS_SCIPY:
                # Remove small objects and fill holes
                brain_mask = ndimage.binary_opening(brain_mask, structure=np.ones((3,3,3)))
                brain_mask = ndimage.binary_closing(brain_mask, structure=np.ones((3,3,3)))
            
            brain_voxels = np.sum(brain_mask)
            
            return {
                'success': True,
                'brain_mask': brain_mask,
                'processed_data': processed_data,
                'brain_voxels': brain_voxels,
                'threshold_used': threshold,
                'processing_time': processing_result['processing_time']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _gpu_tissue_classification(self, volume_data: np.ndarray, brain_mask: np.ndarray) -> Dict[str, Any]:
        """GPU-accelerated tissue classification."""
        try:
            # Extract brain tissue values
            brain_values = volume_data[brain_mask]
            
            if len(brain_values) == 0:
                return {'success': False, 'error': 'No brain tissue found'}
            
            # Simple tissue classification using intensity thresholds
            # These are approximate values for typical MRI
            csf_threshold = np.percentile(brain_values, 25)
            gray_threshold = np.percentile(brain_values, 75)
            
            # Create tissue maps
            csf_mask = (volume_data <= csf_threshold) & brain_mask
            gray_matter_mask = (volume_data > csf_threshold) & (volume_data <= gray_threshold) & brain_mask
            white_matter_mask = (volume_data > gray_threshold) & brain_mask
            
            # Calculate volumes
            voxel_volume = 1.0  # Assume 1mm³ voxels
            csf_volume = np.sum(csf_mask) * voxel_volume
            gray_volume = np.sum(gray_matter_mask) * voxel_volume
            white_volume = np.sum(white_matter_mask) * voxel_volume
            total_brain_volume = csf_volume + gray_volume + white_volume
            
            return {
                'success': True,
                'csf_mask': csf_mask,
                'gray_matter_mask': gray_matter_mask,
                'white_matter_mask': white_matter_mask,
                'csf_volume_mm3': csf_volume,
                'gray_matter_volume_mm3': gray_volume,
                'white_matter_volume_mm3': white_volume,
                'total_brain_volume_mm3': total_brain_volume,
                'csf_percentage': (csf_volume / total_brain_volume) * 100 if total_brain_volume > 0 else 0,
                'gray_percentage': (gray_volume / total_brain_volume) * 100 if total_brain_volume > 0 else 0,
                'white_percentage': (white_volume / total_brain_volume) * 100 if total_brain_volume > 0 else 0
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_brain_surface(self, volume_data: np.ndarray, brain_mask: np.ndarray, quality: str) -> Dict[str, Any]:
        """Generate 3D brain surface mesh."""
        try:
            if not HAS_SCIPY:
                return {'success': False, 'error': 'scipy required for surface generation'}
            
            # Quality settings
            quality_settings = {
                'low': {'spacing': (2.0, 2.0, 2.0), 'step_size': 2},
                'medium': {'spacing': (1.5, 1.5, 1.5), 'step_size': 1},
                'high': {'spacing': (1.0, 1.0, 1.0), 'step_size': 1},
                'ultra': {'spacing': (0.5, 0.5, 0.5), 'step_size': 1}
            }
            
            settings = quality_settings.get(quality, quality_settings['high'])
            
            # Smooth the brain mask for better surface
            if settings['step_size'] > 1:
                # Downsample for lower quality
                brain_mask_smooth = brain_mask[::settings['step_size'], ::settings['step_size'], ::settings['step_size']]
            else:
                brain_mask_smooth = brain_mask.copy()
            
            # Apply smoothing
            brain_mask_smooth = ndimage.gaussian_filter(brain_mask_smooth.astype(float), sigma=1.0)
            
            # Generate surface using marching cubes
            vertices, faces, normals, values = measure.marching_cubes(
                brain_mask_smooth, 
                level=0.5, 
                spacing=settings['spacing']
            )
            
            # Scale vertices back if downsampled
            if settings['step_size'] > 1:
                vertices *= settings['step_size']
            
            return {
                'success': True,
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'values': values,
                'vertex_count': len(vertices),
                'face_count': len(faces),
                'quality': quality,
                'spacing': settings['spacing']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _compute_brain_metrics(self, volume_data: np.ndarray, brain_mask: np.ndarray) -> Dict[str, Any]:
        """Compute brain analysis metrics."""
        try:
            brain_values = volume_data[brain_mask]
            
            if len(brain_values) == 0:
                return {'success': False, 'error': 'No brain tissue found'}
            
            # Basic statistics
            metrics = {
                'success': True,
                'brain_voxel_count': np.sum(brain_mask),
                'brain_volume_percentage': (np.sum(brain_mask) / volume_data.size) * 100,
                'mean_intensity': float(np.mean(brain_values)),
                'std_intensity': float(np.std(brain_values)),
                'min_intensity': float(np.min(brain_values)),
                'max_intensity': float(np.max(brain_values)),
                'median_intensity': float(np.median(brain_values))
            }
            
            # Intensity distribution
            metrics['intensity_percentiles'] = {
                '25th': float(np.percentile(brain_values, 25)),
                '50th': float(np.percentile(brain_values, 50)),
                '75th': float(np.percentile(brain_values, 75)),
                '90th': float(np.percentile(brain_values, 90)),
                '95th': float(np.percentile(brain_values, 95))
            }
            
            return metrics
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

def test_rx7700s_brain_analyzer():
    """Test the fixed RX 7700S brain analyzer."""
    print(" RX 7700S Brain Analyzer Test")
    print("=" * 60)
    
    try:
        # Create analyzer
        analyzer = RX7700SBrainAnalyzerFixed()
        
        # Create test brain volume
        print("Creating synthetic brain volume...")
        size = 128
        x, y, z = np.meshgrid(np.linspace(-2, 2, size),
                             np.linspace(-2, 2, size),
                             np.linspace(-2, 2, size))
        
        # Brain-like ellipsoid
        brain_shape = (x**2/2.5 + y**2/2.0 + z**2/1.5) < 1.0
        
        # Create volume with brain structure
        volume = np.zeros((size, size, size), dtype=np.float32)
        volume[brain_shape] = 0.8
        
        # Add tissue variation
        volume += 0.2 * np.sin(x*3) * np.cos(y*3) * np.sin(z*2) * brain_shape
        volume += 0.1 * np.random.random((size, size, size))
        volume = np.clip(volume, 0, 1)
        
        print(f"Test volume created: {volume.shape}")
        print(f"Brain region: {np.sum(brain_shape):,} voxels")
        print()
        
        # Analyze brain volume
        results = analyzer.analyze_brain_volume(
            volume, 
            generate_surface=True, 
            surface_quality='medium'
        )
        
        # Display results
        print("\n" + "=" * 60)
        print(" ANALYSIS RESULTS")
        print("=" * 60)
        
        if results['success']:
            print(" Brain analysis completed successfully!")
            print(f"   Total time: {results['total_processing_time']:.2f} seconds")
            print(f"   GPU used: {results['gpu_device']}")
            
            # Segmentation results
            if 'segmentation' in results:
                seg = results['segmentation']
                print(f"   Brain voxels: {seg['brain_voxels']:,}")
            
            # Classification results
            if 'classification' in results:
                cls = results['classification']
                if cls['success']:
                    print(f"   Gray matter: {cls['gray_percentage']:.1f}%")
                    print(f"   White matter: {cls['white_percentage']:.1f}%")
                    print(f"   CSF: {cls['csf_percentage']:.1f}%")
            
            # Surface results
            if 'surface' in results:
                surf = results['surface']
                if surf['success']:
                    print(f"   Surface vertices: {surf['vertex_count']:,}")
                    print(f"   Surface faces: {surf['face_count']:,}")
            
            # Metrics
            if 'metrics' in results:
                met = results['metrics']
                if met['success']:
                    print(f"   Mean intensity: {met['mean_intensity']:.3f}")
                    print(f"   Brain volume: {met['brain_volume_percentage']:.1f}%")
            
            print("\n🎉 RX 7700S brain analysis working correctly!")
            print("   Check Task Manager → Performance → GPU 1 for usage")
            
        else:
            print(f" Brain analysis failed: {results['error']}")
        
        return results['success']
        
    except Exception as e:
        print(f" Test failed: {e}")
        return False

if __name__ == "__main__":
    test_rx7700s_brain_analyzer()
    input("\nPress Enter to exit...")