#!/usr/bin/env python3
"""
Frequency-Based Brain Signal Processor

Advanced signal processing module for frequency-based analysis of MRI brain volumes.
This module provides methods for frequency-domain analysis, tissue classification,
and signal-based anatomical feature detection that preserves the volumetric
signal characteristics rather than converting to 2D planar representations.

Key Features:
- 3D FFT-based frequency domain analysis
- Multi-band frequency decomposition
- Signal-based tissue classification
- Anatomical feature detection using frequency signatures
"""

import numpy as np
import scipy.ndimage
import scipy.signal
from scipy.fft import fft, ifft, fftfreq, fftn, ifftn
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import warnings

class FrequencyBand(Enum):
    """Frequency bands for brain signal analysis."""
    LOW_FREQ = "low_frequency"      # 0.01-0.1 Hz - slow physiological changes
    MID_FREQ = "mid_frequency"      # 0.1-1.0 Hz - tissue characteristics  
    HIGH_FREQ = "high_frequency"    # 1.0-10 Hz - microstructural properties
    ULTRA_HIGH = "ultra_high"       # 10+ Hz - noise and artifacts

class FrequencyBrainProcessor:
    """
    Processor for frequency-based analysis of brain MRI volumes.
    """
    
    def __init__(self):
        """Initialize the frequency brain processor."""
        self.frequency_bands = {
            FrequencyBand.LOW_FREQ: (0.01, 0.1),
            FrequencyBand.MID_FREQ: (0.1, 1.0),
            FrequencyBand.HIGH_FREQ: (1.0, 10.0),
            FrequencyBand.ULTRA_HIGH: (10.0, 50.0)
        }
        
        print("Frequency Brain Processor initialized")
    
    def advanced_noise_reduction(self, volume: np.ndarray) -> np.ndarray:
        """Apply advanced noise reduction while preserving signal characteristics."""
        print("   Applying signal-preserving noise reduction...")
        
        # Multi-scale noise reduction
        volume_fine = scipy.ndimage.gaussian_filter(volume, sigma=0.3)
        volume_medium = scipy.ndimage.gaussian_filter(volume, sigma=0.8) 
        volume_coarse = scipy.ndimage.gaussian_filter(volume, sigma=1.5)
        
        # Adaptive combination based on local signal characteristics
        local_variance = scipy.ndimage.generic_filter(volume, np.var, size=3)
        variance_threshold = np.percentile(local_variance, 75)
        
        # Use fine filtering in high-variance regions, coarse in homogeneous regions
        weight_fine = (local_variance > variance_threshold).astype(float)
        weight_coarse = 1 - weight_fine
        
        denoised_volume = (weight_fine * volume_fine + weight_coarse * volume_coarse)
        return denoised_volume
    
    def signal_aware_normalization(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume while preserving signal characteristics."""
        print("   Applying signal-aware normalization...")
        
        # Robust normalization using percentiles
        p1, p99 = np.percentile(volume, [1, 99])
        normalized = np.clip(volume, p1, p99)
        normalized = (normalized - p1) / (p99 - p1)
        
        # Apply histogram equalization for better contrast
        hist, bins = np.histogram(normalized.flatten(), bins=256, range=(0, 1))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        
        normalized_eq = np.interp(normalized.flatten(), bins[:-1], cdf)
        normalized_eq = normalized_eq.reshape(normalized.shape)
        
        return normalized_eq
    
    def create_frequency_volume(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Create frequency-domain representation of the brain volume."""
        print("   Creating frequency-domain brain representation...")
        
        frequency_volume = {}
        
        # Apply 3D FFT to the entire volume
        volume_fft = fftn(volume)
        frequencies = fftfreq(volume.shape[0])
        
        # Decompose into frequency bands
        for band, (low_freq, high_freq) in self.frequency_bands.items():
            # Create frequency mask
            freq_mask = ((np.abs(frequencies) >= low_freq) & 
                        (np.abs(frequencies) <= high_freq))
            
            # Apply mask in frequency domain
            band_fft = volume_fft.copy()
            if not np.any(freq_mask):
                band_fft = np.zeros_like(band_fft)
            else:
                # Keep only frequencies in this band
                band_fft[~freq_mask] = 0
            
            # Convert back to spatial domain
            band_volume = np.real(ifftn(band_fft))
            frequency_volume[band.value] = band_volume
        
        return frequency_volume
    
    def generate_tissue_probability_maps(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate probability maps for different tissue types using signal analysis."""
        print("   Generating tissue probability maps...")
        
        tissue_maps = {}
        
        # Calculate local statistics for tissue classification
        local_mean = scipy.ndimage.uniform_filter(volume, size=3)
        local_std = scipy.ndimage.generic_filter(volume, np.std, size=3)
        local_gradient = np.sqrt(np.sum([np.gradient(volume, axis=i)**2 for i in range(3)], axis=0))
        
        # Gray matter probability (moderate intensity, moderate variation)
        gray_matter_prob = np.exp(-((local_mean - 0.6)**2) / 0.1) * np.exp(-((local_std - 0.15)**2) / 0.05)
        
        # White matter probability (high intensity, low variation)
        white_matter_prob = np.exp(-((local_mean - 0.8)**2) / 0.1) * np.exp(-(local_std**2) / 0.01)
        
        # CSF probability (low intensity, very low variation)
        csf_prob = np.exp(-((local_mean - 0.2)**2) / 0.05) * np.exp(-(local_std**2) / 0.005)
        
        # Blood vessel probability (high gradient, variable intensity)
        vessel_prob = np.exp(-(local_gradient**2) / 0.1) * local_std
        
        # Normalize probabilities
        total_prob = gray_matter_prob + white_matter_prob + csf_prob + vessel_prob + 1e-10
        
        tissue_maps['gray_matter'] = gray_matter_prob / total_prob
        tissue_maps['white_matter'] = white_matter_prob / total_prob
        tissue_maps['cerebrospinal_fluid'] = csf_prob / total_prob
        tissue_maps['blood_vessel'] = vessel_prob / total_prob
        
        return tissue_maps
    
    def detect_frequency_based_features(self, frequency_volume: Dict[str, np.ndarray]) -> List[Dict]:
        """Detect anatomical features based on frequency characteristics."""
        print("   Frequency-based feature detection...")
        
        features = []
        
        for band_name, band_volume in frequency_volume.items():
            # Find regions with strong signal in this frequency band
            band_power = np.abs(band_volume)
            
            # Find local maxima
            threshold = np.percentile(band_power, 95)
            local_maxima = scipy.ndimage.maximum_filter(band_power, size=5) == band_power
            significant_regions = local_maxima & (band_power > threshold)
            
            coords = np.where(significant_regions)
            
            for i in range(min(len(coords[0]), 20)):  # Limit to top 20 per band
                x, y, z = coords[0][i], coords[1][i], coords[2][i]
                
                feature = {
                    'name': f"FreqBand_{band_name}_Region_{i+1}",
                    'coordinates': (x, y, z),
                    'confidence': float(band_power[x, y, z] / np.max(band_power)),
                    'frequency_band': band_name,
                    'signal_power': float(band_power[x, y, z]),
                    'detection_method': 'frequency_analysis'
                }
                features.append(feature)
        
        print(f"     Found {len(features)} frequency-based features")
        return features
    
    def analyze_signal_characteristics(self, volume: np.ndarray, coordinates: Tuple[int, int, int], 
                                    radius: int = 3) -> Dict[str, float]:
        """Analyze signal characteristics at a specific brain location."""
        x, y, z = coordinates
        
        # Extract region around coordinates
        x_min, x_max = max(0, x-radius), min(volume.shape[0], x+radius+1)
        y_min, y_max = max(0, y-radius), min(volume.shape[1], y+radius+1)
        z_min, z_max = max(0, z-radius), min(volume.shape[2], z+radius+1)
        
        region = volume[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Calculate signal characteristics
        characteristics = {
            'mean_signal': float(np.mean(region)),
            'signal_variance': float(np.var(region)),
            'signal_std': float(np.std(region)),
            'signal_range': float(np.ptp(region)),
            'signal_entropy': float(self._calculate_entropy(region)),
            'local_gradient': float(np.mean(np.gradient(region))),
            'texture_energy': float(np.sum(region**2))
        }
        
        return characteristics
    
    def _calculate_entropy(self, region: np.ndarray) -> float:
        """Calculate entropy of a brain region."""
        # Discretize the signal values
        hist, _ = np.histogram(region.flatten(), bins=32, range=(0, 1))
        hist = hist + 1e-10  # Avoid log(0)
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

def main():
    """Test the frequency brain processor."""
    print("Testing Frequency Brain Processor")
    print("=" * 60)
    
    # Create processor
    processor = FrequencyBrainProcessor()
    
    # Create mock brain volume
    volume_shape = (91, 109, 91)  # Standard MNI dimensions
    mock_volume = np.random.rand(*volume_shape)
    
    # Add some structured signal patterns
    # Gray matter regions
    mock_volume[20:70, 20:90, 20:70] += 0.3
    # White matter core
    mock_volume[35:55, 35:75, 35:55] += 0.5
    # CSF regions
    mock_volume[45:46, 45:65, 45:46] = 0.1
    
    print(f"Processing volume: {volume_shape}")
    
    # Test noise reduction
    denoised = processor.advanced_noise_reduction(mock_volume)
    print(f"Noise reduction: {np.mean(np.abs(mock_volume - denoised)):.6f} change")
    
    # Test normalization
    normalized = processor.signal_aware_normalization(denoised)
    print(f"Normalization: range [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
    
    # Test frequency analysis
    freq_volume = processor.create_frequency_volume(normalized)
    print(f"Frequency decomposition: {len(freq_volume)} bands")
    
    # Test tissue maps
    tissue_maps = processor.generate_tissue_probability_maps(normalized)
    print(f"Tissue maps: {list(tissue_maps.keys())}")
    
    # Test feature detection
    features = processor.detect_frequency_based_features(freq_volume)
    print(f"Feature detection: {len(features)} features")
    
    # Test signal analysis
    test_coords = (45, 55, 45)
    characteristics = processor.analyze_signal_characteristics(normalized, test_coords)
    print(f"Signal analysis at {test_coords}:")
    for key, value in characteristics.items():
        print(f"   {key}: {value:.4f}")
    
    print("\nFrequency Brain Processor test completed!")
    return True

if __name__ == "__main__":
    main()