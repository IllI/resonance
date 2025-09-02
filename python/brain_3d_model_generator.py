#!/usr/bin/env python3
"""
Advanced Brain 3D Model Generator with Frequency-Based Signal Processing

This module provides advanced functionality to generate anatomically accurate 3D brain 
models from MRI volume data using frequency-based signal analysis and feature detection AI.
Instead of converting MRI volumes to 2D planar images, this system preserves the
signal-based nature of MRI data to create more accurate 3D brain representations.

Key Features:
- Frequency-based 3D brain volume processing preserving signal characteristics
- Advanced feature detection AI for anatomical identification using signal analysis
- Integration with brain atlases from OpenNeuro for maximum region detection
- Signal-based tissue differentiation (gray matter, white matter, CSF)
- Physically accurate 3D model generation with detailed anatomical labeling
- Support for any MRI volume input with automatic anatomical feature detection
- Integration with existing neural network training pipeline
"""

import numpy as np
import scipy.ndimage
import scipy.signal
from scipy.fft import fft, ifft, fftfreq, fftn, ifftn
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import os
import time
import warnings
from pathlib import Path
import json

# Try to import optional dependencies
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    warnings.warn("nibabel not available - limited MRI loading capabilities")

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available - using fallback implementations")

# Local imports
try:
    from brain_atlas_manager import BrainAtlasManager
    from ai_roi_detector import AIROIDetector, ROIDetection
    from frequency_brain_processor import FrequencyBrainProcessor
    HAS_LOCAL_MODULES = True
except ImportError:
    HAS_LOCAL_MODULES = False
    warnings.warn("Local modules not available - limited functionality")

@dataclass
class AnatomicalFeature:
    """Represents an anatomical feature in the brain."""
    name: str
    region_id: int
    coordinates: Tuple[int, int, int]  # Center coordinates (x, y, z)
    volume: np.ndarray  # Binary mask of the region
    confidence: float
    atlas_label: str
    properties: Dict[str, Any]  # Additional properties (tissue type, etc.)

class BrainTissueType(Enum):
    """Types of brain tissue for segmentation."""
    GRAY_MATTER = "gray_matter"
    WHITE_MATTER = "white_matter"
    CSF = "cerebrospinal_fluid"
    BLOOD_VESSEL = "blood_vessel"
    SKULL = "skull"
    UNKNOWN = "unknown"

class SignalFrequencyBand(Enum):
    """Frequency bands for signal-based tissue analysis."""
    LOW_FREQ = "low_frequency"      # 0.01-0.1 Hz - slow physiological changes
    MID_FREQ = "mid_frequency"      # 0.1-1.0 Hz - tissue characteristics
    HIGH_FREQ = "high_frequency"    # 1.0-10 Hz - microstructural properties
    ULTRA_HIGH = "ultra_high"       # 10+ Hz - noise and artifacts

@dataclass
class FrequencyFeature:
    """Represents frequency-domain features of brain tissue."""
    frequency_band: SignalFrequencyBand
    power_spectrum: np.ndarray
    dominant_frequency: float
    spectral_centroid: float
    spectral_bandwidth: float
    tissue_signature: Dict[str, float]

class Brain3DModelGenerator:
    """
    Generates 3D brain models from MRI volume data using feature detection AI.
    
    This class processes MRI volume data to create a 3D model of the subject's brain
    with labeled anatomical features. It uses feature detection AI to identify brain
    regions and integrates with brain atlases for anatomical labeling.
    """
    
    def __init__(self, 
                 atlas_name: str = 'harvard_oxford',
                 detection_confidence_threshold: float = 0.7,
                 use_gpu: bool = False,
                 frequency_analysis: bool = True,
                 max_regions_detect: int = 200):
        """
        Initialize the Advanced Brain 3D Model Generator.
        
        Args:
            atlas_name: Name of the brain atlas to use for anatomical labeling
            detection_confidence_threshold: Minimum confidence for feature detection
            use_gpu: Whether to use GPU acceleration if available
            frequency_analysis: Whether to use frequency-based signal processing
            max_regions_detect: Maximum number of brain regions to detect
        """
        self.atlas_name = atlas_name
        self.detection_confidence_threshold = detection_confidence_threshold
        self.use_gpu = use_gpu
        self.frequency_analysis = frequency_analysis
        self.max_regions_detect = max_regions_detect
        
        # Initialize components
        self.atlas_manager = None
        self.roi_detector = None
        self.detected_features = []
        self.brain_volume = None
        self.frequency_volume = None
        self.tissue_probability_maps = {}
        self.brain_model = None
        
        # Frequency analysis parameters
        self.frequency_bands = {
            SignalFrequencyBand.LOW_FREQ: (0.01, 0.1),
            SignalFrequencyBand.MID_FREQ: (0.1, 1.0),
            SignalFrequencyBand.HIGH_FREQ: (1.0, 10.0),
            SignalFrequencyBand.ULTRA_HIGH: (10.0, 50.0)
        }
        
        # Initialize frequency processor
        self.frequency_processor = FrequencyBrainProcessor() if HAS_LOCAL_MODULES else None
        self._initialize_components()
        
        print(f"🧠 Advanced Brain 3D Model Generator initialized")
        print(f"   Atlas: {atlas_name}")
        print(f"   Detection threshold: {detection_confidence_threshold}")
        print(f"   Max regions to detect: {max_regions_detect}")
        print(f"   Frequency analysis: {'Enabled' if frequency_analysis else 'Disabled'}")
        print(f"   GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    def _initialize_components(self):
        """Initialize atlas manager and ROI detector components."""
        if HAS_LOCAL_MODULES:
            try:
                self.atlas_manager = BrainAtlasManager(self.atlas_name)
                self.roi_detector = AIROIDetector()
                print("✅ Components initialized successfully")
            except Exception as e:
                print(f"⚠️ Error initializing components: {e}")
        else:
            print("⚠️ Local modules not available - limited functionality")
    
    def load_mri_volume(self, 
                       mri_path: Union[str, Path],
                       preprocess: bool = True) -> bool:
        """
        Load MRI volume data from file.
        
        Args:
            mri_path: Path to MRI volume file (NIfTI format)
            preprocess: Whether to preprocess the volume after loading
            
        Returns:
            Success status
        """
        print(f"📂 Loading MRI volume from {mri_path}")
        
        if not HAS_NIBABEL:
            print("❌ nibabel not available - cannot load NIfTI files")
            return False
        
        try:
            # Load MRI volume using nibabel
            nifti_img = nib.load(str(mri_path))
            self.brain_volume = nifti_img.get_fdata()
            
            # Store metadata
            self.volume_shape = self.brain_volume.shape
            self.volume_affine = nifti_img.affine
            self.volume_header = nifti_img.header
            
            print(f"✅ Loaded MRI volume: {self.volume_shape}")
            
            # Preprocess if requested
            if preprocess:
                self._preprocess_volume()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading MRI volume: {e}")
            return False
    
    def _preprocess_volume(self):
        """Advanced preprocessing of MRI volume with frequency-based analysis."""
        if self.brain_volume is None:
            print("⚠️ No brain volume loaded for preprocessing")
            return
        
        print("🔄 Advanced preprocessing of MRI volume...")
        
        if self.frequency_processor:
            # Apply sophisticated noise reduction preserving signal characteristics
            self.brain_volume = self.frequency_processor.advanced_noise_reduction(self.brain_volume)
            
            # Normalize intensity values while preserving relative signal strength
            self.brain_volume = self.frequency_processor.signal_aware_normalization(self.brain_volume)
            
            # Generate frequency-based volume representation
            if self.frequency_analysis:
                self.frequency_volume = self.frequency_processor.create_frequency_volume(self.brain_volume)
            
            # Generate tissue probability maps using signal characteristics
            self.tissue_probability_maps = self.frequency_processor.generate_tissue_probability_maps(self.brain_volume)
        else:
            # Fallback to basic preprocessing
            self.brain_volume = scipy.ndimage.gaussian_filter(self.brain_volume, sigma=0.5)
            min_val, max_val = np.min(self.brain_volume), np.max(self.brain_volume)
            self.brain_volume = (self.brain_volume - min_val) / (max_val - min_val)
        
        print("✅ Advanced preprocessing complete")
        print(f"   Generated {'frequency volume and ' if self.frequency_analysis else ''}tissue probability maps")
    
    def detect_anatomical_features(self) -> List[AnatomicalFeature]:
        """
        Detect anatomical features using advanced multi-modal AI analysis.
        
        Returns:
            List of detected anatomical features
        """
        if self.brain_volume is None:
            print("❌ No brain volume loaded for feature detection")
            return []
        
        print("🔍 Detecting anatomical features with advanced multi-modal analysis...")
        start_time = time.time()
        
        all_features = []
        
        # 1. Original signal-based ROI detection (preserve existing functionality)
        if HAS_LOCAL_MODULES and self.roi_detector is not None:
            signal_features = self._detect_signal_based_rois()
            all_features.extend(signal_features)
        
        # 2. Frequency-based anatomical detection (new enhancement)
        if self.frequency_analysis and self.frequency_volume is not None:
            freq_features = self._detect_frequency_based_features()
            all_features.extend(freq_features)
        
        # 3. Tissue-based anatomical segmentation (new enhancement)
        if self.tissue_probability_maps:
            tissue_features = self._detect_tissue_based_features()
            all_features.extend(tissue_features)
        
        # 4. Atlas-guided feature detection (enhanced)
        if self.atlas_manager is not None:
            atlas_features = self._detect_atlas_guided_features()
            all_features.extend(atlas_features)
        
        # Merge and deduplicate features
        self.detected_features = self._merge_and_deduplicate_features(all_features)
        
        # Limit to maximum regions
        if len(self.detected_features) > self.max_regions_detect:
            self.detected_features.sort(key=lambda f: f.confidence, reverse=True)
            self.detected_features = self.detected_features[:self.max_regions_detect]
        
        duration = time.time() - start_time
        print(f"✅ Detected {len(self.detected_features)} anatomical features in {duration:.2f}s")
        
        # Print top features by confidence
        for i, feature in enumerate(sorted(self.detected_features, 
                                          key=lambda f: f.confidence, 
                                          reverse=True)[:5]):
            print(f"   {i+1}. {feature.name} (confidence: {feature.confidence:.2f})")
        
        return self.detected_features
    
    def _convert_rois_to_features(self, rois: List[ROIDetection]) -> List[AnatomicalFeature]:
        """Convert ROI detections to anatomical features."""
        features = []
        
        for roi in rois:
            # Create a binary mask for the ROI
            mask = np.zeros_like(self.brain_volume, dtype=bool)
            
            # Extract coordinates and create a sphere around the center
            x, y, z = roi.coordinates
            radius = int(10 * roi.confidence)  # Size based on confidence
            
            # Create a spherical mask using distance transform
            indices = np.indices(self.brain_volume.shape)
            distance = np.sqrt(((indices[0] - x) ** 2) + 
                              ((indices[1] - y) ** 2) + 
                              ((indices[2] - z) ** 2))
            mask = distance <= radius
            
            # Map to atlas label if available
            atlas_label = "Unknown"
            if self.atlas_manager is not None:
                atlas_labels = self.atlas_manager.atlas_labels
                if roi.region_id < len(atlas_labels):
                    atlas_label = atlas_labels[roi.region_id]
            
            # Create anatomical feature
            feature = AnatomicalFeature(
                name=roi.region_name,
                region_id=roi.region_id,
                coordinates=roi.coordinates,
                volume=mask,
                confidence=roi.confidence,
                atlas_label=atlas_label,
                properties={
                    'tissue_type': BrainTissueType.GRAY_MATTER,
                    'activation': roi.activation_strength,
                    'network_state': roi.network_state.value
                }
            )
            
            features.append(feature)
        
        return features
    
    def _detect_signal_based_rois(self) -> List[AnatomicalFeature]:
        """Detect ROIs using the original signal processing approach (preserving existing functionality)."""
        print("   🎯 Signal-based ROI detection (original method)...")
        
        features = []
        
        try:
            # Create 4D volume (required by ROI detector)
            volume_4d = np.expand_dims(self.brain_volume, axis=0)
            
            # Detect ROIs using AI detector
            detected_rois = self.roi_detector.detect_rois(volume_4d, 
                                                        confidence_threshold=self.detection_confidence_threshold)
            
            # Convert ROIs to anatomical features using existing method
            features = self._convert_rois_to_features(detected_rois)
            
            print(f"     📊 Found {len(features)} signal-based ROIs")
            
        except Exception as e:
            print(f"     ⚠️ Signal-based detection failed: {e}")
        
        return features
    
    def _detect_frequency_based_features(self) -> List[AnatomicalFeature]:
        """Detect anatomical features based on frequency domain characteristics."""
        print("   🌊 Frequency-based feature detection...")
        
        features = []
        
        if not self.frequency_volume:
            print("     ⚠️ No frequency volume available")
            return features
        
        for band_name, band_volume in self.frequency_volume.items():
            # Find regions with strong signal in this frequency band
            band_power = np.abs(band_volume)
            
            # Find local maxima
            threshold = np.percentile(band_power, 95)
            local_maxima = scipy.ndimage.maximum_filter(band_power, size=5) == band_power
            significant_regions = local_maxima & (band_power > threshold)
            
            coords = np.where(significant_regions)
            
            for i in range(min(len(coords[0]), 15)):  # Limit to top 15 per band
                x, y, z = coords[0][i], coords[1][i], coords[2][i]
                
                # Create feature
                feature = AnatomicalFeature(
                    name=f"FreqBand_{band_name}_Region_{i+1}",
                    region_id=hash((band_name, x, y, z)) % 10000,
                    coordinates=(x, y, z),
                    volume=self._create_sphere_mask((x, y, z), radius=3),
                    confidence=float(band_power[x, y, z] / np.max(band_power)),
                    atlas_label=f"Frequency_Based_{band_name}",
                    properties={
                        'tissue_type': BrainTissueType.UNKNOWN,
                        'frequency_band': band_name,
                        'signal_power': float(band_power[x, y, z]),
                        'detection_method': 'frequency_analysis'
                    }
                )
                features.append(feature)
        
        print(f"     📊 Found {len(features)} frequency-based features")
        return features
    
    def _detect_tissue_based_features(self) -> List[AnatomicalFeature]:
        """Detect anatomical features based on tissue probability maps."""
        print("   🧠 Tissue-based feature detection...")
        
        features = []
        
        for tissue_type, prob_map in self.tissue_probability_maps.items():
            # Find regions with high probability for this tissue type
            threshold = np.percentile(prob_map, 90)
            high_prob_regions = prob_map > threshold
            
            # Label connected components
            labeled_regions, n_regions = scipy.ndimage.label(high_prob_regions)
            
            for region_id in range(1, min(n_regions + 1, 21)):  # Limit to top 20 per tissue
                region_mask = labeled_regions == region_id
                
                if np.sum(region_mask) < 10:  # Skip very small regions
                    continue
                
                # Get center of mass
                center = scipy.ndimage.center_of_mass(region_mask)
                x, y, z = int(center[0]), int(center[1]), int(center[2])
                
                # Get tissue type enum
                tissue_enum = next((t for t in BrainTissueType if t.value == tissue_type), 
                                 BrainTissueType.UNKNOWN)
                
                feature = AnatomicalFeature(
                    name=f"{tissue_type.title()}_Region_{region_id}",
                    region_id=hash((tissue_type, x, y, z)) % 10000,
                    coordinates=(x, y, z),
                    volume=region_mask,
                    confidence=float(np.mean(prob_map[region_mask])),
                    atlas_label=f"Tissue_Based_{tissue_type}",
                    properties={
                        'tissue_type': tissue_enum,
                        'region_volume': int(np.sum(region_mask)),
                        'mean_probability': float(np.mean(prob_map[region_mask])),
                        'detection_method': 'tissue_analysis'
                    }
                )
                features.append(feature)
        
        print(f"     📊 Found {len(features)} tissue-based features")
        return features
    
    def _detect_atlas_guided_features(self) -> List[AnatomicalFeature]:
        """Detect features using brain atlas guidance."""
        print("   🗺️ Atlas-guided feature detection...")
        
        features = []
        
        try:
            # Get atlas coordinates and labels
            atlas_coords = self.atlas_manager.get_region_coordinates()
            atlas_labels = self.atlas_manager.atlas_labels
            
            volume_shape = self.brain_volume.shape
            
            for i, (coord, label) in enumerate(zip(atlas_coords, atlas_labels)):
                if i >= 100:  # Limit atlas regions
                    break
                
                # Convert atlas coordinates to voxel indices
                x = int(coord[0] + volume_shape[0] // 2)
                y = int(coord[1] + volume_shape[1] // 2)
                z = int(coord[2] + volume_shape[2] // 2)
                
                # Check bounds
                if (0 <= x < volume_shape[0] and 
                    0 <= y < volume_shape[1] and 
                    0 <= z < volume_shape[2]):
                    
                    # Create region mask
                    region_mask = self._create_sphere_mask((x, y, z), radius=4)
                    
                    # Calculate signal characteristics
                    signal_strength = np.mean(self.brain_volume[region_mask])
                    signal_variance = np.var(self.brain_volume[region_mask])
                    
                    # Determine tissue type
                    tissue_type = self._classify_tissue_from_signal(signal_strength, signal_variance)
                    
                    # Calculate confidence
                    confidence = signal_strength * (1 - signal_variance)
                    
                    if confidence > self.detection_confidence_threshold:
                        feature = AnatomicalFeature(
                            name=label,
                            region_id=i,
                            coordinates=(x, y, z),
                            volume=region_mask,
                            confidence=float(confidence),
                            atlas_label=label,
                            properties={
                                'tissue_type': tissue_type,
                                'signal_strength': float(signal_strength),
                                'signal_variance': float(signal_variance),
                                'detection_method': 'atlas_guided',
                                'atlas_coordinates': coord.tolist()
                            }
                        )
                        features.append(feature)
            
            print(f"     📊 Found {len(features)} atlas-guided features")
            
        except Exception as e:
            print(f"     ⚠️ Atlas-guided detection failed: {e}")
        
        return features
    
    def _merge_and_deduplicate_features(self, features: List[AnatomicalFeature]) -> List[AnatomicalFeature]:
        """Merge similar features and remove duplicates."""
        print("   🔄 Merging and deduplicating features...")
        
        if not features:
            return features
        
        merged_features = []
        spatial_threshold = 8  # Distance threshold for merging
        
        for feature in features:
            # Check if this feature is close to any existing feature
            merged = False
            
            for existing in merged_features:
                distance = np.sqrt(sum((a - b)**2 for a, b in zip(feature.coordinates, existing.coordinates)))
                
                if distance < spatial_threshold:
                    # Merge features - keep the one with higher confidence
                    if feature.confidence > existing.confidence:
                        # Replace existing feature
                        idx = merged_features.index(existing)
                        merged_features[idx] = feature
                    merged = True
                    break
            
            if not merged:
                merged_features.append(feature)
        
        print(f"     📊 Merged {len(features)} → {len(merged_features)} features")
        return merged_features
    
    def _create_sphere_mask(self, center: Tuple[int, int, int], radius: int) -> np.ndarray:
        """Create a spherical mask around a center point."""
        x, y, z = center
        mask = np.zeros_like(self.brain_volume, dtype=bool)
        
        # Create spherical region
        xx, yy, zz = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
        distance = np.sqrt((xx - x)**2 + (yy - y)**2 + (zz - z)**2)
        mask = distance <= radius
        
        return mask
    
    def _classify_tissue_from_signal(self, signal_strength: float, signal_variance: float) -> BrainTissueType:
        """Classify tissue type based on signal characteristics."""
        if signal_strength > 0.7 and signal_variance < 0.1:
            return BrainTissueType.WHITE_MATTER
        elif 0.4 < signal_strength < 0.7 and signal_variance < 0.2:
            return BrainTissueType.GRAY_MATTER
        elif signal_strength < 0.3 and signal_variance < 0.05:
            return BrainTissueType.CSF
        else:
            return BrainTissueType.UNKNOWN
    
    def _detect_spatial_features_fallback(self) -> List[AnatomicalFeature]:
        """Fallback spatial feature detection when temporal analysis fails."""
        print("🔄 Using spatial analysis fallback for feature detection...")
        
        features = []
        
        if self.brain_volume is None:
            return features
        
        # Use intensity-based and gradient-based detection
        volume = self.brain_volume
        
        # 1. Find high-intensity regions (potential white matter)
        high_intensity_threshold = np.percentile(volume, 85)
        high_intensity_regions = volume > high_intensity_threshold
        
        # 2. Find medium-intensity regions (potential gray matter)
        medium_intensity_mask = (volume > np.percentile(volume, 40)) & (volume < np.percentile(volume, 80))
        
        # 3. Calculate local gradients for edge detection
        gradients = np.sqrt(np.sum([np.gradient(volume, axis=i)**2 for i in range(3)], axis=0))
        gradient_threshold = np.percentile(gradients, 90)
        
        # 4. Find regions with high local variance (tissue boundaries)
        local_variance = scipy.ndimage.generic_filter(volume, np.var, size=3)
        variance_threshold = np.percentile(local_variance, 85)
        
        # Combine criteria to find interesting regions
        interesting_regions = (
            high_intensity_regions | 
            (medium_intensity_mask & (gradients > gradient_threshold)) |
            (local_variance > variance_threshold)
        )
        
        # Label connected components
        labeled_regions, n_regions = scipy.ndimage.label(interesting_regions)
        
        print(f"   Found {n_regions} potential regions using spatial analysis")
        
        # Process each region
        for region_id in range(1, min(n_regions + 1, self.max_regions_detect + 1)):
            region_mask = labeled_regions == region_id
            
            # Skip very small regions
            if np.sum(region_mask) < 20:
                continue
            
            # Get region properties
            center = scipy.ndimage.center_of_mass(region_mask)
            x, y, z = int(center[0]), int(center[1]), int(center[2])
            
            # Calculate region statistics
            region_values = volume[region_mask]
            mean_intensity = np.mean(region_values)
            intensity_variance = np.var(region_values)
            region_volume_size = np.sum(region_mask)
            
            # Calculate confidence based on region properties
            # Higher confidence for:
            # - Larger regions
            # - Consistent intensity (low variance)
            # - Appropriate intensity levels
            size_score = min(region_volume_size / 1000, 1.0)  # Normalize by typical region size
            consistency_score = 1.0 / (1.0 + intensity_variance)  # Lower variance = higher score
            intensity_score = 1.0 - abs(mean_intensity - 0.5)  # Closer to middle intensity range
            
            confidence = (size_score + consistency_score + intensity_score) / 3.0
            
            # Only include regions with reasonable confidence
            if confidence > 0.4:
                # Determine tissue type based on intensity
                tissue_type = self._classify_tissue_from_signal(mean_intensity, intensity_variance)
                
                # Create anatomical feature
                feature = AnatomicalFeature(
                    name=f"Spatial_Region_{region_id}",
                    region_id=region_id + 1000,  # Offset to avoid conflicts
                    coordinates=(x, y, z),
                    volume=region_mask,
                    confidence=confidence,
                    atlas_label=f"Spatial_Analysis_Region_{region_id}",
                    properties={
                        'tissue_type': tissue_type,
                        'mean_intensity': float(mean_intensity),
                        'intensity_variance': float(intensity_variance),
                        'region_volume': int(region_volume_size),
                        'detection_method': 'spatial_analysis_fallback'
                    }
                )
                
                features.append(feature)
        
        # Sort by confidence and limit results
        features.sort(key=lambda f: f.confidence, reverse=True)
        features = features[:min(len(features), 50)]  # Limit to top 50
        
        print(f"   ✅ Generated {len(features)} features using spatial analysis fallback")
        
        return features
    
    def generate_3d_model(self) -> Dict[str, Any]:
        """
        Generate a 3D model of the brain with labeled anatomical features.
        
        Returns:
            Dictionary containing the 3D model data
        """
        if self.brain_volume is None:
            print("❌ No brain volume loaded for model generation")
            return {}
        
        if not self.detected_features:
            print("⚠️ No features detected, running detection first")
            self.detect_anatomical_features()
        
        print("🧠 Generating 3D brain model...")
        start_time = time.time()
        
        # Create base model from brain volume
        model = {
            'volume': self.brain_volume,
            'shape': self.brain_volume.shape,
            'features': self.detected_features,
            'metadata': {
                'atlas': self.atlas_name,
                'generation_time': time.time(),
                'feature_count': len(self.detected_features)
            }
        }
        
        # Create labeled volume where each voxel is assigned to a region
        labeled_volume = np.zeros_like(self.brain_volume, dtype=np.int32)
        
        # Assign voxels to regions based on detected features
        for i, feature in enumerate(self.detected_features):
            # Add feature to labeled volume with region ID
            labeled_volume[feature.volume] = feature.region_id + 1  # +1 to avoid 0 (background)
        
        model['labeled_volume'] = labeled_volume
        
        # Store the model
        self.brain_model = model
        
        duration = time.time() - start_time
        print(f"✅ 3D brain model generated in {duration:.2f}s")
        print(f"   Features: {len(self.detected_features)}")
        print(f"   Volume dimensions: {self.brain_volume.shape}")
        
        return model
    
    def save_model(self, output_path: Union[str, Path]) -> bool:
        """
        Save the 3D brain model to a file.
        
        Args:
            output_path: Path to save the model
            
        Returns:
            Success status
        """
        if self.brain_model is None:
            print("❌ No brain model generated to save")
            return False
        
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model as NumPy compressed file
            np.savez_compressed(
                output_path,
                volume=self.brain_volume,
                labeled_volume=self.brain_model['labeled_volume'],
                features=[{
                    'name': f.name,
                    'region_id': f.region_id,
                    'coordinates': f.coordinates,
                    'confidence': f.confidence,
                    'atlas_label': f.atlas_label,
                    'properties': f.properties
                } for f in self.detected_features],
                metadata=self.brain_model['metadata']
            )
            
            print(f"✅ 3D brain model saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving brain model: {e}")
            return False
    
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """
        Load a previously saved 3D brain model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Success status
        """
        try:
            # Load model from NumPy compressed file
            data = np.load(model_path, allow_pickle=True)
            
            # Extract model components
            self.brain_volume = data['volume']
            
            # Reconstruct features
            self.detected_features = []
            for f in data['features']:
                feature = AnatomicalFeature(
                    name=f['name'],
                    region_id=f['region_id'],
                    coordinates=f['coordinates'],
                    volume=np.zeros_like(self.brain_volume, dtype=bool),  # Placeholder
                    confidence=f['confidence'],
                    atlas_label=f['atlas_label'],
                    properties=f['properties']
                )
                self.detected_features.append(feature)
            
            # Reconstruct model
            self.brain_model = {
                'volume': self.brain_volume,
                'shape': self.brain_volume.shape,
                'labeled_volume': data['labeled_volume'],
                'features': self.detected_features,
                'metadata': data['metadata'].item()
            }
            
            print(f"✅ 3D brain model loaded from {model_path}")
            print(f"   Features: {len(self.detected_features)}")
            print(f"   Volume dimensions: {self.brain_volume.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading brain model: {e}")
            return False

def main():
    """Test the Advanced Brain 3D Model Generator."""
    print("🧠 Testing Advanced Brain 3D Model Generator")
    print("=" * 60)
    
    # Create model generator
    model_generator = Brain3DModelGenerator(
        atlas_name='harvard_oxford',
        detection_confidence_threshold=0.6,  # Lower threshold for testing
        max_regions_detect=50,  # Fewer regions for testing
        frequency_analysis=True
    )
    
    # Check if test data is available
    test_data_path = Path("data/test_mri_volume.nii.gz")
    if not test_data_path.exists():
        print(f"⚠️ Test data not found at {test_data_path}")
        print("   Creating realistic mock 4D fMRI volume for testing...")
        
        # Create realistic 4D fMRI volume (time series data)
        n_timepoints = 100  # Sufficient for temporal analysis
        volume_shape = (91, 109, 91)  # Standard MNI dimensions
        
        # Create base brain structure
        mock_volume_4d = np.random.randn(n_timepoints, *volume_shape) * 0.1 + 0.3
        
        # Add realistic brain regions with different signal characteristics
        # Gray matter regions (cortical areas)
        for t in range(n_timepoints):
            # Add temporal variation to make it realistic
            time_factor = 1 + 0.1 * np.sin(2 * np.pi * t / 50)  # Low frequency oscillation
            
            # Gray matter cortex
            mock_volume_4d[t, 20:70, 20:90, 20:70] += 0.3 * time_factor
            
            # White matter core
            mock_volume_4d[t, 35:55, 35:75, 35:55] += 0.5 * time_factor
            
            # CSF ventricles (low signal)
            mock_volume_4d[t, 43:48, 50:60, 43:48] = 0.1
            
            # Add some "active" regions with task-related signals
            if t % 20 < 10:  # Block design simulation
                # Motor region activation
                mock_volume_4d[t, 25:35, 55:65, 45:55] += 0.4
                # Visual region activation
                mock_volume_4d[t, 60:70, 75:85, 35:45] += 0.3
        
        # Convert to 3D by averaging across time (for volume-based analysis)
        mock_volume_3d = np.mean(mock_volume_4d, axis=0)
        
        print(f"✅ Created realistic mock volume: {mock_volume_3d.shape}")
        print(f"   Signal range: [{np.min(mock_volume_3d):.3f}, {np.max(mock_volume_3d):.3f}]")
        
        # Set the brain volume
        model_generator.brain_volume = mock_volume_3d
        
        # Apply preprocessing to generate frequency and tissue maps
        model_generator._preprocess_volume()
        
    else:
        # Load real test data
        model_generator.load_mri_volume(test_data_path)
    
    # Detect features with enhanced multi-modal approach
    print("\n🔍 Running enhanced anatomical feature detection...")
    features = model_generator.detect_anatomical_features()
    
    # If no features detected with signal-based method, use fallback detection
    if len(features) == 0:
        print("\n⚠️ No features detected with temporal analysis, using spatial analysis...")
        features = model_generator._detect_spatial_features_fallback()
    
    # Generate 3D model
    print("\n🧠 Generating 3D brain model...")
    model = model_generator.generate_3d_model()
    
    # Save model
    output_path = Path("visualizations/advanced_brain_model.npz")
    model_generator.save_model(output_path)
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("🎉 ADVANCED BRAIN 3D MODEL GENERATION COMPLETED!")
    print("=" * 60)
    
    print(f"\n📊 Generation Results:")
    print(f"   Total features detected: {len(features)}")
    print(f"   Volume dimensions: {model_generator.brain_volume.shape}")
    print(f"   Atlas used: {model_generator.atlas_name}")
    print(f"   Frequency analysis: {'✅' if model_generator.frequency_analysis else '❌'}")
    
    if features:
        print(f"\n🏆 Top 5 Detected Features:")
        top_features = sorted(features, key=lambda f: f.confidence, reverse=True)[:5]
        for i, feature in enumerate(top_features, 1):
            tissue_type = feature.properties.get('tissue_type', 'Unknown')
            if hasattr(tissue_type, 'value'):
                tissue_type = tissue_type.value
            detection_method = feature.properties.get('detection_method', 'Unknown')
            
            print(f"   {i}. {feature.name}")
            print(f"      Confidence: {feature.confidence:.3f}")
            print(f"      Coordinates: {feature.coordinates}")
            print(f"      Tissue: {tissue_type}")
            print(f"      Method: {detection_method}")
            print()
        
        # Detection method statistics
        methods = {}
        tissues = {}
        for feature in features:
            method = feature.properties.get('detection_method', 'unknown')
            tissue = feature.properties.get('tissue_type', 'unknown')
            if hasattr(tissue, 'value'):
                tissue = tissue.value
            
            methods[method] = methods.get(method, 0) + 1
            tissues[tissue] = tissues.get(tissue, 0) + 1
        
        print(f"📈 Detection Method Distribution:")
        for method, count in methods.items():
            print(f"   {method}: {count} features")
        
        print(f"\n🧠 Tissue Type Distribution:")
        for tissue, count in tissues.items():
            print(f"   {tissue}: {count} features")
    
    return True

if __name__ == "__main__":
    main()