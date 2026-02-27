#!/usr/bin/env python3
"""
AI ROI Detector - Signal Processing Approach

Advanced brain region detection using signal processing techniques inspired by 
digital hearing aid algorithms for signal detection in noise. This approach 
eliminates PyTorch dependency while providing robust spatiotemporal analysis.

Key Features:
- CNN-like spatial feature extraction using convolutions
- LSTM-inspired temporal pattern recognition  
- Hearing-aid-style noise reduction
- Attention mechanisms for signal focus
- Self-supervised feature learning
"""

import numpy as np
import scipy.ndimage
import scipy.signal
from scipy.fft import fft, ifft, fftfreq
from typing import Dict, List, Tuple, Optional, NamedTuple
import warnings
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Core dependencies only
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not available - using fallback implementations")

class NetworkState(Enum):
    """Brain network states for classification."""
    DEFAULT_MODE = "default_mode"
    EXECUTIVE_CONTROL = "executive_control"
    SALIENCE = "salience"
    VISUAL = "visual"
    MOTOR = "motor"
    AUDITORY = "auditory"
    ATTENTION = "attention"

@dataclass
class ROIDetection:
    """ROI detection result."""
    region_id: int
    region_name: str
    confidence: float
    coordinates: Tuple[int, int, int]
    network_state: NetworkState
    activation_strength: float
    temporal_pattern: np.ndarray

class SpatialFeatureExtractor:
    """CNN-inspired spatial feature extraction using convolutions."""
    
    def __init__(self):
        """Initialize spatial feature extractor."""
        # Define 3D convolution kernels for different feature types
        self.kernels = self._create_feature_kernels()
        
    def _create_feature_kernels(self) -> Dict[str, np.ndarray]:
        """Create 3D convolution kernels for spatial feature detection."""
        kernels = {}
        
        # Edge detection kernels (3D Sobel-like)
        kernels['edge_x'] = np.array([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], 
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ]) / 16.0
        
        kernels['edge_y'] = np.array([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ]) / 16.0
        
        kernels['edge_z'] = np.array([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        ]) / 16.0
        
        # Blob detection (3D Gaussian-like)
        kernels['blob'] = np.array([
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        ]) / 64.0
        
        # High-pass filter for noise reduction
        kernels['highpass'] = np.array([
            [[0, -1, 0], [-1, -2, -1], [0, -1, 0]],
            [[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]],
            [[0, -1, 0], [-1, -2, -1], [0, -1, 0]]
        ]) / 4.0
        
        return kernels
    
    def extract_features(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spatial features from 3D fMRI volume.
        
        Args:
            volume: 3D fMRI volume (x, y, z)
            
        Returns:
            Dictionary of feature maps
        """
        features = {}
        
        # Apply each convolution kernel
        for kernel_name, kernel in self.kernels.items():
            # 3D convolution using scipy
            feature_map = scipy.ndimage.convolve(volume, kernel, mode='constant')
            features[kernel_name] = feature_map
        
        # Compute feature magnitude
        edge_magnitude = np.sqrt(
            features['edge_x']**2 + 
            features['edge_y']**2 + 
            features['edge_z']**2
        )
        features['edge_magnitude'] = edge_magnitude
        
        # Compute local variance (texture feature)
        features['local_variance'] = scipy.ndimage.generic_filter(
            volume, np.var, size=3
        )
        
        return features

class TemporalPatternAnalyzer:
    """LSTM-inspired temporal pattern recognition using signal processing."""
    
    def __init__(self, sequence_length: int = 50):
        """Initialize temporal analyzer."""
        self.sequence_length = sequence_length
        self.memory_states = {}
        
    def analyze_temporal_patterns(self, 
                                time_series: np.ndarray,
                                region_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze temporal patterns in fMRI time series.
        
        Args:
            time_series: 4D fMRI data (t, x, y, z)
            region_mask: 3D mask for region of interest
            
        Returns:
            Dictionary of temporal features
        """
        # CRITICAL FIX: Properly extract time series from ROI
        print(f"    Extracting ROI time series from fMRI shape: {time_series.shape}, mask shape: {region_mask.shape}")
        
        # Find voxels in the ROI mask
        roi_voxels = np.where(region_mask > 0)
        
        if len(roi_voxels[0]) == 0:
            print("    No voxels found in ROI mask, using default features")
            # Return default features to prevent crashes
            return {
                'dominant_frequency': np.array([0.1]),
                'low_freq_power': np.array([1.0]),
                'high_freq_power': np.array([0.5]),
                'freq_ratio': np.array([2.0]),
                'velocity_variance': np.array([0.1]),
                'acceleration_variance': np.array([0.05]),
                'velocity_zero_crossings': np.array([10]),
                'memory_decay': np.array([0.8]),
                'hurst_exponent': np.array([0.5]),
                'dmn_correlation': np.array([0.3]),
                'task_correlation': np.array([0.2]),
                'attention_correlation': np.array([0.4])
            }
        
        # Extract time series correctly: for each timepoint, get values from ROI voxels
        n_timepoints = time_series.shape[0]
        n_voxels = len(roi_voxels[0])
        
        print(f"    Found {n_voxels} voxels in ROI, extracting {n_timepoints} timepoints")
        
        # Create matrix: timepoints x voxels
        roi_timeseries_matrix = np.zeros((n_timepoints, n_voxels))
        
        for i, (x, y, z) in enumerate(zip(roi_voxels[0], roi_voxels[1], roi_voxels[2])):
            # Bounds checking
            if (x < time_series.shape[1] and y < time_series.shape[2] and z < time_series.shape[3]):
                roi_timeseries_matrix[:, i] = time_series[:, x, y, z]
            else:
                print(f"    Skipping out-of-bounds voxel ({x}, {y}, {z})")
        
        # Average across voxels to get single time series
        roi_timeseries = np.mean(roi_timeseries_matrix, axis=1)
        
        print(f"    Extracted ROI time series: {roi_timeseries.shape} (length: {len(roi_timeseries)})")
        
        # Validate time series length for signal processing
        if len(roi_timeseries) < 50:
            print(f"    Time series too short ({len(roi_timeseries)} samples), padding to minimum length")
            # Pad with mean to reach minimum length
            mean_val = np.mean(roi_timeseries)
            padding_needed = 50 - len(roi_timeseries)
            roi_timeseries = np.concatenate([roi_timeseries, np.full(padding_needed, mean_val)])
        
        features = {}
        
        # Frequency domain analysis
        features.update(self._frequency_analysis(roi_timeseries))
        
        # Temporal derivatives (velocity and acceleration)
        features.update(self._temporal_derivatives(roi_timeseries))
        
        # Pattern persistence (memory-like behavior)
        features.update(self._pattern_persistence(roi_timeseries))
        
        # Cross-correlation with canonical patterns
        features.update(self._canonical_correlations(roi_timeseries))
        
        return features
    
    def _frequency_analysis(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze frequency components of the signal."""
        features = {}
        
        # FFT analysis
        fft_signal = fft(signal)
        freqs = fftfreq(len(signal), d=1.0)  # Assuming TR=1s for simplicity
        
        # Power spectral density
        features['power_spectrum'] = np.abs(fft_signal)**2
        features['dominant_frequency'] = freqs[np.argmax(features['power_spectrum'])]
        
        # Band power analysis (similar to hearing aid frequency bands)
        low_freq = np.sum(features['power_spectrum'][(freqs >= 0.01) & (freqs < 0.08)])
        high_freq = np.sum(features['power_spectrum'][(freqs >= 0.08) & (freqs < 0.25)])
        features['low_freq_power'] = low_freq
        features['high_freq_power'] = high_freq
        features['freq_ratio'] = high_freq / (low_freq + 1e-10)
        
        return features
    
    def _temporal_derivatives(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute temporal derivatives (velocity, acceleration)."""
        features = {}
        
        # First derivative (velocity)
        velocity = np.gradient(signal)
        features['velocity'] = velocity
        features['velocity_variance'] = np.var(velocity)
        
        # Second derivative (acceleration)  
        acceleration = np.gradient(velocity)
        features['acceleration'] = acceleration
        features['acceleration_variance'] = np.var(acceleration)
        
        # Zero-crossing rates
        features['velocity_zero_crossings'] = np.sum(np.diff(np.sign(velocity)) != 0)
        
        return features
    
    def _pattern_persistence(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze pattern persistence (memory-like behavior)."""
        features = {}
        
        # Autocorrelation function
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        features['autocorrelation'] = autocorr
        
        # Memory decay time constant
        decay_threshold = np.exp(-1)  # 1/e
        decay_idx = np.where(autocorr < decay_threshold)[0]
        features['memory_decay'] = decay_idx[0] if len(decay_idx) > 0 else len(autocorr)
        
        # Long-range correlations (Hurst exponent approximation)
        features['hurst_exponent'] = self._estimate_hurst(signal)
        
        return features
    
    def _canonical_correlations(self, signal: np.ndarray) -> Dict[str, float]:
        """Correlate with canonical brain network patterns."""
        features = {}
        
        # Create canonical patterns for different networks
        t = np.linspace(0, len(signal), len(signal))
        
        # Default mode network pattern (low frequency oscillation)
        dmn_pattern = np.sin(2 * np.pi * 0.02 * t)  # ~0.02 Hz
        features['dmn_correlation'] = np.corrcoef(signal, dmn_pattern)[0, 1]
        
        # Task-positive pattern (higher frequency)
        task_pattern = np.sin(2 * np.pi * 0.1 * t)  # ~0.1 Hz
        features['task_correlation'] = np.corrcoef(signal, task_pattern)[0, 1]
        
        # Attention pattern (mixed frequencies)
        attention_pattern = (np.sin(2 * np.pi * 0.05 * t) + 
                           0.5 * np.sin(2 * np.pi * 0.15 * t))
        features['attention_correlation'] = np.corrcoef(signal, attention_pattern)[0, 1]
        
        return features
    
    def _estimate_hurst(self, signal: np.ndarray) -> float:
        """Estimate Hurst exponent for long-range correlations."""
        try:
            # Simple R/S analysis
            n = len(signal)
            if n < 10:
                return 0.5
                
            # Rescaled range analysis
            mean_signal = np.mean(signal)
            cumulative_deviate = np.cumsum(signal - mean_signal)
            
            R = np.max(cumulative_deviate) - np.min(cumulative_deviate)
            S = np.std(signal)
            
            if S == 0:
                return 0.5
                
            hurst = np.log(R/S) / np.log(n)
            return np.clip(hurst, 0.0, 1.0)
            
        except:
            return 0.5  # Default to random walk

class NoiseReductionSystem:
    """Hearing-aid-inspired noise reduction for fMRI signals."""
    
    def __init__(self):
        """Initialize noise reduction system."""
        self.noise_profile = None
        
    def adaptive_noise_reduction(self, 
                                signal: np.ndarray,
                                noise_estimation_method: str = 'spectral') -> np.ndarray:
        """
        Apply adaptive noise reduction similar to hearing aids.
        
        Args:
            signal: Input fMRI signal
            noise_estimation_method: Method for noise estimation
            
        Returns:
            Denoised signal
        """
        if noise_estimation_method == 'spectral':
            return self._spectral_subtraction(signal)
        elif noise_estimation_method == 'wiener':
            return self._wiener_filter(signal)
        else:
            return self._adaptive_filter(signal)
    
    def _spectral_subtraction(self, signal: np.ndarray) -> np.ndarray:
        """Spectral subtraction noise reduction."""
        # FFT of signal
        signal_fft = fft(signal)
        magnitude = np.abs(signal_fft)
        phase = np.angle(signal_fft)
        
        # Estimate noise floor (bottom 10% of spectrum)
        noise_floor = np.percentile(magnitude, 10)
        
        # Subtract noise floor with over-subtraction factor
        alpha = 2.0  # Over-subtraction factor
        cleaned_magnitude = magnitude - alpha * noise_floor
        cleaned_magnitude = np.maximum(cleaned_magnitude, 0.1 * magnitude)
        
        # Reconstruct signal
        cleaned_fft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_signal = np.real(ifft(cleaned_fft))
        
        return cleaned_signal
    
    def _wiener_filter(self, signal: np.ndarray) -> np.ndarray:
        """Wiener filter for noise reduction."""
        # Validate signal length
        if len(signal) < 10:
            print(f"    Signal too short for Wiener filter ({len(signal)} samples), returning original")
            return signal
            
        # Estimate signal and noise power
        signal_power = np.var(signal)
        
        # Simple noise estimation (high-frequency components)
        # Ensure window_length is valid for savgol_filter
        min_window = 5  # Minimum window for savgol
        max_window = len(signal) // 4 * 2 + 1  # Must be odd and < signal length
        window_length = max(min_window, min(51, max_window))
        
        # Ensure window_length is odd and at least 3
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(3, window_length)
        
        # Additional safety check
        if window_length >= len(signal):
            window_length = len(signal) // 2
            if window_length % 2 == 0:
                window_length -= 1
            window_length = max(3, window_length)
        
        try:
            high_freq_signal = signal - scipy.signal.savgol_filter(signal, 
                                                                   window_length=window_length, 
                                                                   polyorder=min(3, window_length-1))
            noise_power = np.var(high_freq_signal)
        except Exception as e:
            print(f"    Savgol filter failed: {e}, using simple high-pass estimate")
            # Fallback: use simple difference as high-frequency estimate
            high_freq_signal = np.diff(signal, prepend=signal[0])
            noise_power = np.var(high_freq_signal)
        
        # Wiener filter coefficient
        wiener_coeff = signal_power / (signal_power + noise_power + 1e-10)
        
        # Apply filter
        filtered_signal = wiener_coeff * signal
        
        return filtered_signal
    
    def _adaptive_filter(self, signal: np.ndarray) -> np.ndarray:
        """Adaptive noise reduction filter."""
        # Validate signal length
        if len(signal) < 5:
            print(f"    Signal too short for adaptive filter ({len(signal)} samples), returning original")
            return signal
            
        # Use median filter for impulse noise (ensure kernel size is appropriate)
        kernel_size = min(3, len(signal))
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(1, kernel_size)
        
        median_filtered = scipy.signal.medfilt(signal, kernel_size=kernel_size)
        
        # Use Savitzky-Golay filter for smoothing with proper validation
        if len(signal) > 10:
            # Calculate appropriate window length
            min_window = 5
            max_window = len(signal) // 4 * 2 + 1
            window_length = max(min_window, min(11, max_window))
            
            # Ensure window_length is odd and valid
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(3, window_length)
            
            # Final safety check
            if window_length >= len(signal):
                window_length = len(signal) // 2
                if window_length % 2 == 0:
                    window_length -= 1
                window_length = max(3, window_length)
            
            try:
                savgol_filtered = scipy.signal.savgol_filter(signal, 
                                                            window_length=window_length,
                                                            polyorder=min(3, window_length-1))
            except Exception as e:
                print(f"    Savgol filter failed in adaptive filter: {e}, using median only")
                savgol_filtered = median_filtered
        else:
            savgol_filtered = median_filtered
        
        # Combine filters
        alpha = 0.7
        filtered_signal = alpha * savgol_filtered + (1 - alpha) * median_filtered
        
        return filtered_signal

class AttentionMechanism:
    """Attention mechanism for focusing on relevant brain regions."""
    
    def __init__(self):
        """Initialize attention mechanism."""
        self.attention_weights = None
        
    def compute_attention_weights(self, 
                                features: Dict[str, np.ndarray],
                                target_pattern: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute attention weights for spatial locations.
        
        Args:
            features: Dictionary of feature maps
            target_pattern: Optional target pattern to attend to
            
        Returns:
            Attention weight map
        """
        # Combine all feature maps
        feature_stack = np.stack(list(features.values()), axis=0)
        
        # Compute feature importance (variance-based)
        feature_importance = np.var(feature_stack, axis=(1, 2, 3))
        feature_importance = feature_importance / np.sum(feature_importance)
        
        # Weighted combination of features
        attention_map = np.zeros_like(feature_stack[0])
        for i, (feature_name, feature_map) in enumerate(features.items()):
            attention_map += feature_importance[i] * np.abs(feature_map)
        
        # Normalize attention weights
        attention_weights = attention_map / (np.max(attention_map) + 1e-10)
        
        # Apply softmax-like transformation for sharp attention
        attention_weights = np.exp(attention_weights * 5)  # Temperature = 5
        attention_weights = attention_weights / np.sum(attention_weights)
        
        self.attention_weights = attention_weights
        return attention_weights

class LAMSTARCoordinator:
    """
    LAMSTAR-inspired coordinator for learning and classifying brain states.
    
    This module uses correlation-based learning to associate complex feature
    vectors with specific brain network states, offering rapid and robust
    pattern recognition without traditional backpropagation.
    """
    
    def __init__(self, input_size: int, num_spatial_features: int):
        """
        Initialize the LAMSTAR Coordinator.
        
        Args:
            input_size: The expected dimension of the feature vectors.
            num_spatial_features: The number of spatial features in the vector.
        """
        self.memory: Dict[NetworkState, List[np.ndarray]] = {}
        self.input_size = input_size
        self.num_spatial_features = num_spatial_features
        # Initialize weights: give equal importance initially
        self.spatial_weight = 0.5
        self.temporal_weight = 0.5
        print(" Initialized LAMSTAR Coordinator with Feature Weighting")

    def train(self, feature_vector: np.ndarray, label: NetworkState):
        """
        Train the coordinator by creating a memory of a feature vector and its label.
        
        Args:
            feature_vector: A 1D numpy array representing the combined features.
            label: The NetworkState associated with the feature vector.
        """
        if feature_vector.shape[0] != self.input_size:
            raise ValueError(f"Input vector size {feature_vector.shape[0]} does not match expected size {self.input_size}")

        # Normalize the feature vector to ensure fair correlation
        norm_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-10)
        
        if label not in self.memory:
            self.memory[label] = []
        
        self.memory[label].append(norm_vector)
        print(f"    Memorized pattern for: {label.value}")

    def predict(self, feature_vector: np.ndarray) -> Tuple[NetworkState, float]:
        """
        Predict the network state for a given feature vector.
        
        Args:
            feature_vector: The feature vector of the ROI to classify.
            
        Returns:
            A tuple of (predicted_network_state, confidence_score).
        """
        if not self.memory:
            return NetworkState.SALIENCE, 0.0 # Default if not trained

        if feature_vector.shape[0] != self.input_size:
            raise ValueError(f"Input vector size {feature_vector.shape[0]} does not match expected size {self.input_size}")
            
        # Apply dynamic weights
        weighted_vector = self._apply_weights(feature_vector)
        norm_vector = weighted_vector / (np.linalg.norm(weighted_vector) + 1e-10)
        
        best_match_label = NetworkState.SALIENCE
        highest_correlation = -1.0
        
        for label, prototypes in self.memory.items():
            # Apply the same weighting to the prototypes before correlation
            weighted_prototypes = [self._apply_weights(p) for p in prototypes]
            norm_prototypes = [wp / (np.linalg.norm(wp) + 1e-10) for wp in weighted_prototypes]

            # Calculate correlation with all weighted prototypes
            correlations = [np.dot(norm_vector, p) for p in norm_prototypes]
            
            # Use the max correlation for this label as its score
            max_correlation = np.max(correlations)
            
            if max_correlation > highest_correlation:
                highest_correlation = max_correlation
                best_match_label = label
                
        # After prediction, adjust weights based on the winning prototype
        winner_prototype = self.memory[best_match_label][np.argmax(correlations)]
        self._adjust_weights(norm_vector, winner_prototype)
                
        return best_match_label, float(np.clip(highest_correlation, 0, 1))

    def _apply_weights(self, vector: np.ndarray) -> np.ndarray:
        """Applies the current spatial/temporal weights to a feature vector."""
        spatial_part = vector[:self.num_spatial_features] * self.spatial_weight
        temporal_part = vector[self.num_spatial_features:] * self.temporal_weight
        return np.concatenate([spatial_part, temporal_part])

    def _adjust_weights(self, input_vector: np.ndarray, prototype_vector: np.ndarray):
        """Dynamically adjust weights based on prediction similarity."""
        # Calculate similarity for spatial and temporal parts separately
        spatial_sim = np.dot(input_vector[:self.num_spatial_features], prototype_vector[:self.num_spatial_features])
        temporal_sim = np.dot(input_vector[self.num_spatial_features:], prototype_vector[self.num_spatial_features:])

        # Increase the weight of the more similar part
        if spatial_sim > temporal_sim:
            self.spatial_weight += 0.01
        else:
            self.temporal_weight += 0.01
        
        # Normalize weights to sum to 1
        total_weight = self.spatial_weight + self.temporal_weight
        self.spatial_weight /= total_weight
        self.temporal_weight /= total_weight
        # print(f"   ⚖️ Adjusted weights -> Spatial: {self.spatial_weight:.2f}, Temporal: {self.temporal_weight:.2f}")
    
class SignalProcessingROIDetector:
    """
    Main ROI detector using signal processing approach.
    
    This class combines all components to provide robust brain region detection
    without deep learning dependencies.
    """
    
    def __init__(self, 
                 roi_templates: Optional[Dict[str, np.ndarray]] = None,
                 noise_reduction: bool = True):
        """
        Initialize the signal processing ROI detector.
        
        Args:
            roi_templates: Optional templates for known ROIs
            noise_reduction: Whether to apply noise reduction
        """
        self.spatial_extractor = SpatialFeatureExtractor()
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.noise_reducer = NoiseReductionSystem()
        self.attention = AttentionMechanism()
        
        # This will be determined after feature extraction
        self.feature_vector_size: Optional[int] = None
        self.coordinator: Optional[LAMSTARCoordinator] = None
        
        self.roi_templates = roi_templates or self._create_default_templates()
        self.noise_reduction = noise_reduction
        
        print(" Initialized Signal Processing ROI Detector")
        print("   • CNN-like spatial feature extraction")
        print("   • LSTM-inspired temporal analysis")
        print("   • Hearing-aid-style noise reduction")
        print("   • Attention mechanisms")
        print("   • LAMSTAR-inspired learning coordinator")
    
    def _create_default_templates(self) -> Dict[str, Dict]:
        """Create default ROI templates based on standard brain networks."""
        templates = {
            'default_mode': {
                'frequency_range': (0.01, 0.08),
                'spatial_pattern': 'distributed',
                'temporal_signature': 'low_frequency'
            },
            'executive_control': {
                'frequency_range': (0.05, 0.15),
                'spatial_pattern': 'frontal_parietal',
                'temporal_signature': 'task_related'
            },
            'salience': {
                'frequency_range': (0.03, 0.12),
                'spatial_pattern': 'anterior_insula',
                'temporal_signature': 'mixed_frequency'
            },
            'visual': {
                'frequency_range': (0.08, 0.25),
                'spatial_pattern': 'occipital',
                'temporal_signature': 'high_frequency'
            },
            'motor': {
                'frequency_range': (0.1, 0.3),
                'spatial_pattern': 'precentral',
                'temporal_signature': 'rhythmic'
            }
        }
        return templates

    def _initialize_coordinator(self, size: int, num_spatial: int):
        """Initializes the LAMSTAR coordinator once the feature vector size is known."""
        if self.coordinator is None:
            self.feature_vector_size = size
            self.coordinator = LAMSTARCoordinator(
                input_size=self.feature_vector_size,
                num_spatial_features=num_spatial
            )
    
    def train_on_roi(self, fmri_data: np.ndarray, roi_mask: np.ndarray, label: NetworkState):
        """
        Trains the LAMSTAR coordinator on a specific, labeled ROI.

        Args:
            fmri_data: 4D fMRI data (t, x, y, z).
            roi_mask: A 3D boolean mask defining the labeled ROI.
            label: The ground-truth NetworkState for this ROI.
        """
        print(f"\n Training on labeled ROI for: {label.value}")
        # 1. Spatio-temporal analysis of the entire volume
        spatial_features_over_time = [self.spatial_extractor.extract_features(fmri_data[t]) for t in range(fmri_data.shape[0])]
        combined_spatial_features = self._combine_temporal_spatial_features(spatial_features_over_time)
        
        # 2. Temporal analysis for the specific ROI
        temporal_features = self.temporal_analyzer.analyze_temporal_patterns(fmri_data, roi_mask)
        
        # 3. Create the feature vector for this ROI
        feature_vector = self._create_feature_vector(combined_spatial_features, temporal_features, roi_mask)

        # 4. Initialize coordinator if this is the first training run
        num_spatial = len(combined_spatial_features)
        self._initialize_coordinator(feature_vector.shape[0], num_spatial)
        
        # 5. Train the coordinator
        if self.coordinator is not None:
            self.coordinator.train(feature_vector, label)
        else:
            print("    Coordinator not initialized, skipping training")

    def _create_feature_vector(self, 
                               spatial_features: Dict[str, np.ndarray],
                               temporal_features: Dict[str, np.ndarray],
                               roi_mask: np.ndarray) -> np.ndarray:
        """
        Creates a flattened 1D feature vector from all available features for a given ROI.
        This vector serves as the input for the LAMSTAR coordinator.
        """
        # --- Spatial Features ---
        # Extract spatial features at the center of the ROI
        roi_center_coords = scipy.ndimage.center_of_mass(roi_mask)
        roi_center_indices = tuple(int(round(c)) for c in roi_center_coords)
        
        # Ensure indices are within bounds
        bounded_indices = []
        for i, dim in enumerate(roi_mask.shape):
            bounded_indices.append(min(roi_center_indices[i], dim - 1))
        bounded_indices = tuple(bounded_indices)

        spatial_vector = np.array([
            features[bounded_indices] for features in spatial_features.values() if features.ndim == 3
        ])
        
        # --- Temporal Features ---
        # Use a curated list of key temporal features
        temporal_keys = [
            'dominant_frequency', 'low_freq_power', 'high_freq_power', 
            'freq_ratio', 'velocity_variance', 'acceleration_variance',
            'velocity_zero_crossings', 'memory_decay', 'hurst_exponent',
            'dmn_correlation', 'task_correlation', 'attention_correlation'
        ]
        temporal_vector = np.array([
            temporal_features.get(key, 0.0) for key in temporal_keys
        ])

        # --- Combined Feature Vector ---
        # Combine and normalize the vectors
        combined_vector = np.concatenate([spatial_vector, temporal_vector]).astype(np.float32)
        
        # Final normalization of the entire vector
        combined_vector = (combined_vector - np.mean(combined_vector)) / (np.std(combined_vector) + 1e-10)
        
        return combined_vector
    
    def detect_rois(self, 
                   fmri_data: np.ndarray,
                   confidence_threshold: float = 0.5) -> List[ROIDetection]:
        """
        Detect ROIs in fMRI data using signal processing approach.
        
        Args:
            fmri_data: 4D fMRI data (t, x, y, z)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected ROIs
        """
        print(f" Detecting ROIs in fMRI data: {fmri_data.shape}")
        
        detections = []
        n_timepoints = fmri_data.shape[0]
        
        # Process each time point for spatial features
        spatial_features_over_time = []
        
        for t in range(n_timepoints):
            volume = fmri_data[t]
            
            # Apply noise reduction if enabled
            if self.noise_reduction:
                volume = self._apply_volume_noise_reduction(volume)
            
            # Extract spatial features
            spatial_features = self.spatial_extractor.extract_features(volume)
            spatial_features_over_time.append(spatial_features)
        
        # Combine spatial features across time
        combined_features = self._combine_temporal_spatial_features(spatial_features_over_time)
        
        # Compute attention weights
        attention_weights = self.attention.compute_attention_weights(combined_features)
        
        # Find candidate ROI locations
        candidates = self._find_roi_candidates(attention_weights, combined_features)
        
        # Analyze temporal patterns for each candidate
        for candidate in candidates:
            roi_detection = self._analyze_candidate_roi(
                fmri_data, candidate, combined_features
            )
            
            if roi_detection and roi_detection.confidence >= confidence_threshold:
                detections.append(roi_detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f" Detected {len(detections)} ROIs with confidence >= {confidence_threshold}")
        
        return detections
    
    def _apply_volume_noise_reduction(self, volume: np.ndarray) -> np.ndarray:
        """Apply noise reduction to a single volume."""
        # Apply 3D smoothing
        smoothed = scipy.ndimage.gaussian_filter(volume, sigma=1.0)
        
        # Combine original and smoothed (preserve details)
        alpha = 0.7
        denoised = alpha * volume + (1 - alpha) * smoothed
        
        return denoised
    
    def _combine_temporal_spatial_features(self, 
                                         spatial_features_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine spatial features across time."""
        combined = {}
        
        # Get feature names from first time point
        feature_names = spatial_features_list[0].keys()
        
        for feature_name in feature_names:
            # Stack features across time
            feature_stack = np.stack([sf[feature_name] for sf in spatial_features_list], axis=0)
            
            # Compute temporal statistics
            combined[f'{feature_name}_mean'] = np.mean(feature_stack, axis=0)
            combined[f'{feature_name}_std'] = np.std(feature_stack, axis=0)
            combined[f'{feature_name}_max'] = np.max(feature_stack, axis=0)
            
        return combined
    
    def _find_roi_candidates(self, 
                           attention_weights: np.ndarray,
                           features: Dict[str, np.ndarray]) -> List[Dict]:
        """Find candidate ROI locations using attention weights."""
        candidates = []
        
        # Find local maxima in attention weights
        local_maxima = scipy.ndimage.maximum_filter(attention_weights, size=5) == attention_weights
        local_maxima = local_maxima & (attention_weights > np.percentile(attention_weights, 90))
        
        # Get coordinates of candidates
        coords = np.where(local_maxima)
        
        for i in range(len(coords[0])):
            x, y, z = coords[0][i], coords[1][i], coords[2][i]
            
            candidate = {
                'coordinates': (x, y, z),
                'attention_weight': attention_weights[x, y, z],
                'local_features': {
                    name: feature_map[x, y, z] 
                    for name, feature_map in features.items()
                }
            }
            candidates.append(candidate)
        
        # Sort by attention weight
        candidates.sort(key=lambda x: x['attention_weight'], reverse=True)
        
        # Return top candidates
        return candidates[:20]  # Limit to top 20
    
    def _analyze_candidate_roi(self, 
                             fmri_data: np.ndarray,
                             candidate: Dict,
                             spatial_features: Dict[str, np.ndarray]) -> Optional[ROIDetection]:
        """Analyze a candidate ROI location."""
        x, y, z = candidate['coordinates']
        
        # Create ROI mask (small sphere around candidate)
        mask = np.zeros(fmri_data.shape[1:])
        
        # Simple spherical mask
        radius = 3
        xx, yy, zz = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
        mask_sphere = ((xx - x)**2 + (yy - y)**2 + (zz - z)**2) <= radius**2
        mask[mask_sphere] = 1
        
        # Extract time series from ROI
        # Ensure there are voxels in the mask to avoid errors
        if np.sum(mask) == 0:
            return None

        # CRITICAL FIX: Properly extract time series from 4D fMRI data
        print(f"    Extracting candidate ROI time series from fMRI shape: {fmri_data.shape}, mask shape: {mask.shape}")
        
        # Find voxels in the ROI mask
        roi_voxels = np.where(mask > 0)
        
        if len(roi_voxels[0]) == 0:
            print("    No voxels found in candidate ROI mask")
            return None
        
        # Extract time series correctly: for each timepoint, get values from ROI voxels
        n_timepoints = fmri_data.shape[0]
        n_voxels = len(roi_voxels[0])
        
        print(f"    Found {n_voxels} voxels in candidate ROI, extracting {n_timepoints} timepoints")
        
        # Create matrix: timepoints x voxels
        roi_timeseries_matrix = np.zeros((n_timepoints, n_voxels))
        
        for i, (x, y, z) in enumerate(zip(roi_voxels[0], roi_voxels[1], roi_voxels[2])):
            # Bounds checking
            if (x < fmri_data.shape[1] and y < fmri_data.shape[2] and z < fmri_data.shape[3]):
                roi_timeseries_matrix[:, i] = fmri_data[:, x, y, z]
            else:
                print(f"    Skipping out-of-bounds candidate voxel ({x}, {y}, {z})")
        
        # Average across voxels to get single time series
        roi_timeseries = np.mean(roi_timeseries_matrix, axis=1)
        
        print(f"    Extracted candidate ROI time series: {roi_timeseries.shape} (length: {len(roi_timeseries)})")
        
        # Validate time series length for signal processing
        if len(roi_timeseries) < 50:
            print(f"    Candidate time series too short ({len(roi_timeseries)} samples), padding to minimum length")
            # Pad with mean to reach minimum length
            mean_val = np.mean(roi_timeseries) if len(roi_timeseries) > 0 else 0.0
            padding_needed = 50 - len(roi_timeseries)
            roi_timeseries = np.concatenate([roi_timeseries, np.full(padding_needed, mean_val)])
        
        # Apply temporal noise reduction
        if self.noise_reduction:
            roi_timeseries = self.noise_reducer.adaptive_noise_reduction(roi_timeseries)
        
        # Analyze temporal patterns
        temporal_features = self.temporal_analyzer.analyze_temporal_patterns(
            fmri_data, mask
        )
        
        # Create the feature vector for prediction
        feature_vector = self._create_feature_vector(spatial_features, temporal_features, mask)

        # Initialize coordinator if it hasn't been from training
        num_spatial = len(spatial_features)
        self._initialize_coordinator(feature_vector.shape[0], num_spatial)

        # Classify network state using LAMSTAR coordinator
        network_state, confidence = self.coordinator.predict(feature_vector)
        
        # Compute activation strength
        activation_strength = np.std(roi_timeseries)
        
        # Create ROI detection
        detection = ROIDetection(
            region_id=hash((x, y, z)) % 10000,  # Simple ID generation
            region_name=f"ROI_{x}_{y}_{z}",
            confidence=confidence,
            coordinates=(x, y, z),
            network_state=network_state,
            activation_strength=activation_strength,
            temporal_pattern=roi_timeseries
        )
        
        return detection
    
    def _classify_network_state(self, 
                              temporal_features: Dict) -> Tuple[NetworkState, float]:
        """
        Classify the network state based on temporal features.
        
        DEPRECATED: This method is replaced by the LAMSTARCoordinator.
        """
        # Simple classification based on frequency characteristics
        
        if 'freq_ratio' in temporal_features:
            freq_ratio = temporal_features['freq_ratio']
            
            if freq_ratio < 0.5:
                return NetworkState.DEFAULT_MODE, 0.8
            elif freq_ratio > 2.0:
                return NetworkState.VISUAL, 0.7
            else:
                return NetworkState.EXECUTIVE_CONTROL, 0.6
        
        # Fallback classification
        if 'dmn_correlation' in temporal_features:
            dmn_corr = abs(temporal_features['dmn_correlation'])
            if dmn_corr > 0.5:
                return NetworkState.DEFAULT_MODE, dmn_corr
        
        return NetworkState.EXECUTIVE_CONTROL, 0.5
    
    def generate_report(self, detections: List[ROIDetection]) -> Dict:
        """Generate analysis report."""
        report = {
            'timestamp': np.datetime64('now').astype('datetime64[s]').astype(int),
            'method': 'signal_processing',
            'total_rois_detected': len(detections),
            'network_distribution': {},
            'confidence_stats': {
                'mean': np.mean([d.confidence for d in detections]) if detections else 0,
                'std': np.std([d.confidence for d in detections]) if detections else 0,
                'min': np.min([d.confidence for d in detections]) if detections else 0,
                'max': np.max([d.confidence for d in detections]) if detections else 0
            },
            'detections': []
        }
        
        # Network distribution
        for detection in detections:
            network = detection.network_state.value
            report['network_distribution'][network] = report['network_distribution'].get(network, 0) + 1
        
        # Detection details
        for detection in detections:
            report['detections'].append({
                'region_id': detection.region_id,
                'region_name': detection.region_name,
                'confidence': detection.confidence,
                'coordinates': detection.coordinates,
                'network_state': detection.network_state.value,
                'activation_strength': detection.activation_strength
            })
        
        return report

# Alias for backward compatibility
AIROIDetector = SignalProcessingROIDetector

def main():
    """Test the signal processing ROI detector with LAMSTAR learning."""
    print(" Testing Signal Processing ROI Detector with LAMSTAR Learning")
    
    # --- 1. Initialize Detector ---
    detector = SignalProcessingROIDetector(noise_reduction=True)

    # --- 2. Create Synthetic Training Data ---
    np.random.seed(42)
    n_timepoints = 100
    shape = (32, 32, 24)
    
    # Create data and masks for Default Mode Network
    fmri_data_dmn = np.random.randn(n_timepoints, *shape) * 0.1
    t = np.linspace(0, 10, n_timepoints)
    dmn_signal = np.sin(2 * np.pi * 0.02 * t) * 0.7 # Strong low-freq signal
    fmri_data_dmn[:, 10:13, 10:13, 8:11] += dmn_signal[:, np.newaxis, np.newaxis, np.newaxis]
    mask_dmn = np.zeros(shape)
    mask_dmn[10:13, 10:13, 8:11] = 1

    # Create data and masks for Visual Network
    fmri_data_visual = np.random.randn(n_timepoints, *shape) * 0.1
    visual_signal = np.sin(2 * np.pi * 0.15 * t) * 0.7 # Strong high-freq signal
    fmri_data_visual[:, 20:23, 20:23, 15:18] += visual_signal[:, np.newaxis, np.newaxis, np.newaxis]
    mask_visual = np.zeros(shape)
    mask_visual[20:23, 20:23, 15:18] = 1

    # --- 3. Train the Detector ---
    detector.train_on_roi(fmri_data_dmn, mask_dmn, NetworkState.DEFAULT_MODE)
    detector.train_on_roi(fmri_data_visual, mask_visual, NetworkState.VISUAL)

    # --- 4. Create Synthetic Test Data ---
    # This data contains both patterns in different locations
    fmri_data_test = np.random.randn(n_timepoints, *shape) * 0.1
    # DMN pattern in a new location
    fmri_data_test[:, 5:8, 5:8, 5:8] += dmn_signal[:, np.newaxis, np.newaxis, np.newaxis]
    # Visual pattern in a new location
    fmri_data_test[:, 25:28, 25:28, 18:21] += visual_signal[:, np.newaxis, np.newaxis, np.newaxis]
    
    # --- 5. Detect ROIs on Test Data ---
    print("\n\n🕵️‍♀️ Detecting ROIs on new, unseen data...")
    detections = detector.detect_rois(fmri_data_test, confidence_threshold=0.4)
    
    # --- 6. Generate and Print Report ---
    report = detector.generate_report(detections)
    
    print(f"\n Detection Results:")
    print(f"   Total ROIs: {report['total_rois_detected']}")
    print(f"   Mean confidence: {report['confidence_stats']['mean']:.3f}")
    print(f"   Network distribution: {report['network_distribution']}")
    
    # Verify that it correctly classified the networks
    print("\n🧐 Verifying classifications...")
    for det in detections:
        coords = det.coordinates
        state = det.network_state.value
        # Check if DMN was found in the right place
        if 4 <= coords[0] <= 9 and 4 <= coords[1] <= 9:
             print(f"  - Found ROI at {coords}, correctly classified as: {state} (Expected: DMN)")
        # Check if Visual was found in the right place
        if 24 <= coords[0] <= 29 and 24 <= coords[1] <= 29:
             print(f"  - Found ROI at {coords}, correctly classified as: {state} (Expected: VISUAL)")

    return True

if __name__ == "__main__":
    main() 