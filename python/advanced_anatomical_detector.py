#!/usr/bin/env python3
"""
Advanced Anatomical Feature Detector

This module provides advanced AI-powered anatomical feature detection using
signal analysis combined with brain atlas integration from OpenNeuro. It can
detect and label a maximum number of brain regions by analyzing signal
characteristics, frequency patterns, and anatomical priors.

Key Features:
- Multi-modal feature detection (signal, frequency, atlas-guided)
- Maximum brain region detection capabilities  
- Integration with OpenNeuro brain atlases
- Signal-based tissue classification
- Confidence scoring and feature merging
"""

import numpy as np
import scipy.ndimage
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings

# Local imports
try:
    from ai_roi_detector import SignalProcessingROIDetector, ROIDetection, NetworkState
    from brain_atlas_manager import BrainAtlasManager
    from frequency_brain_processor import FrequencyBrainProcessor
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    warnings.warn("Required dependencies not available - using fallback implementations")

class BrainTissueType(Enum):
    """Types of brain tissue for classification."""
    GRAY_MATTER = "gray_matter"
    WHITE_MATTER = "white_matter"
    CSF = "cerebrospinal_fluid"
    BLOOD_VESSEL = "blood_vessel"
    SKULL = "skull"
    UNKNOWN = "unknown"

@dataclass
class AnatomicalFeature:
    """Represents a detected anatomical feature in the brain."""
    name: str
    region_id: int
    coordinates: Tuple[int, int, int]
    volume: np.ndarray
    confidence: float
    atlas_label: str
    properties: Dict[str, Any]

class AdvancedAnatomicalDetector:
    """
    Advanced detector for anatomical features using multiple detection methods.
    """
    
    def __init__(self, 
                 atlas_name: str = 'harvard_oxford',
                 confidence_threshold: float = 0.6,
                 max_regions: int = 200):
        """
        Initialize the advanced anatomical detector.
        
        Args:
            atlas_name: Brain atlas to use for anatomical guidance
            confidence_threshold: Minimum confidence for feature detection
            max_regions: Maximum number of regions to detect
        """
        self.atlas_name = atlas_name
        self.confidence_threshold = confidence_threshold
        self.max_regions = max_regions
        
        # Initialize components
        self.atlas_manager = None
        self.roi_detector = None
        self.frequency_processor = None
        
        if HAS_DEPENDENCIES:
            try:
                self.atlas_manager = BrainAtlasManager(atlas_name)
                self.roi_detector = SignalProcessingROIDetector()
                self.frequency_processor = FrequencyBrainProcessor()
                print("✅ All components initialized successfully")
            except Exception as e:
                print(f"⚠️ Error initializing components: {e}")
        
        print(f"🧠 Advanced Anatomical Detector initialized")
        print(f"   Atlas: {atlas_name}")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Max regions: {max_regions}")
    
    def detect_anatomical_features(self, 
                                 brain_volume: np.ndarray,
                                 frequency_volume: Optional[Dict[str, np.ndarray]] = None,
                                 tissue_maps: Optional[Dict[str, np.ndarray]] = None) -> List[AnatomicalFeature]:
        """
        Detect anatomical features using multiple detection methods.
        
        Args:
            brain_volume: 3D brain volume data
            frequency_volume: Optional frequency-decomposed volume
            tissue_maps: Optional tissue probability maps
            
        Returns:
            List of detected anatomical features
        """
        print("🔍 Detecting anatomical features with advanced AI...")
        
        all_features = []
        
        # 1. Signal-based ROI detection
        signal_features = self._detect_signal_based_features(brain_volume)
        all_features.extend(signal_features)
        
        # 2. Frequency-based feature detection
        if frequency_volume:
            freq_features = self._detect_frequency_features(frequency_volume)
            all_features.extend(freq_features)
        
        # 3. Tissue-based anatomical segmentation
        if tissue_maps:
            tissue_features = self._detect_tissue_features(tissue_maps)
            all_features.extend(tissue_features)
        
        # 4. Atlas-guided detection
        atlas_features = self._detect_atlas_guided_features(brain_volume)
        all_features.extend(atlas_features)
        
        # 5. Advanced morphological analysis
        morph_features = self._detect_morphological_features(brain_volume)
        all_features.extend(morph_features)
        
        # Merge and filter features
        merged_features = self._merge_similar_features(all_features)
        final_features = self._filter_and_rank_features(merged_features)
        
        print(f"✅ Detected {len(final_features)} high-confidence anatomical features")
        return final_features
    
    def _detect_signal_based_features(self, brain_volume: np.ndarray) -> List[AnatomicalFeature]:
        """Detect features using signal processing ROI detector."""
        print("   🎯 Signal-based ROI detection...")
        
        features = []
        
        if self.roi_detector:
            try:
                # Create 4D volume for ROI detector
                volume_4d = np.expand_dims(brain_volume, axis=0)
                
                # Detect ROIs
                rois = self.roi_detector.detect_rois(volume_4d, 
                                                   confidence_threshold=self.confidence_threshold)
                
                # Convert to anatomical features
                for roi in rois:
                    feature = self._roi_to_anatomical_feature(roi, brain_volume)
                    if feature:
                        features.append(feature)
                
                print(f"     📊 Found {len(features)} signal-based features")
                
            except Exception as e:
                print(f"     ⚠️ Signal-based detection failed: {e}")
        
        return features
    
    def _detect_frequency_features(self, frequency_volume: Dict[str, np.ndarray]) -> List[AnatomicalFeature]:
        """Detect features based on frequency domain characteristics."""
        print("   🌊 Frequency-based feature detection...")
        
        features = []
        
        if self.frequency_processor:
            try:
                freq_features = self.frequency_processor.detect_frequency_based_features(frequency_volume)
                
                # Convert to anatomical features
                for i, feat in enumerate(freq_features):
                    x, y, z = feat['coordinates']
                    
                    feature = AnatomicalFeature(
                        name=feat['name'],
                        region_id=hash((feat['frequency_band'], x, y, z)) % 10000,
                        coordinates=(x, y, z),
                        volume=self._create_sphere_mask((x, y, z), frequency_volume[list(frequency_volume.keys())[0]].shape, radius=3),
                        confidence=feat['confidence'],
                        atlas_label=f"Frequency_{feat['frequency_band']}",
                        properties={
                            'tissue_type': BrainTissueType.UNKNOWN,
                            'frequency_band': feat['frequency_band'],
                            'signal_power': feat['signal_power'],
                            'detection_method': 'frequency_analysis'
                        }
                    )
                    features.append(feature)
                
                print(f"     📊 Found {len(features)} frequency-based features")
                
            except Exception as e:
                print(f"     ⚠️ Frequency-based detection failed: {e}")
        
        return features
    
    def _detect_tissue_features(self, tissue_maps: Dict[str, np.ndarray]) -> List[AnatomicalFeature]:
        """Detect features based on tissue probability maps."""
        print("   🧠 Tissue-based feature detection...")
        
        features = []
        
        for tissue_type, prob_map in tissue_maps.items():
            # Find high-probability regions
            threshold = np.percentile(prob_map, 90)
            high_prob_regions = prob_map > threshold
            
            # Label connected components
            labeled_regions, n_regions = scipy.ndimage.label(high_prob_regions)
            
            for region_id in range(1, min(n_regions + 1, 31)):  # Limit per tissue type
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
                    atlas_label=f"Tissue_{tissue_type}",
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
    
    def _detect_atlas_guided_features(self, brain_volume: np.ndarray) -> List[AnatomicalFeature]:
        """Detect features using brain atlas guidance."""
        print("   🗺️ Atlas-guided feature detection...")
        
        features = []
        
        if self.atlas_manager:
            try:
                # Get atlas coordinates and labels
                atlas_coords = self.atlas_manager.get_region_coordinates()
                atlas_labels = self.atlas_manager.atlas_labels
                
                volume_shape = brain_volume.shape
                
                for i, (coord, label) in enumerate(zip(atlas_coords, atlas_labels)):
                    if i >= 150:  # Limit atlas regions
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
                        region_mask = self._create_sphere_mask((x, y, z), volume_shape, radius=4)
                        
                        # Calculate signal characteristics
                        signal_strength = np.mean(brain_volume[region_mask])
                        signal_variance = np.var(brain_volume[region_mask])
                        
                        # Determine tissue type
                        tissue_type = self._classify_tissue_from_signal(signal_strength, signal_variance)
                        
                        # Calculate confidence
                        confidence = signal_strength * (1 - signal_variance)
                        
                        if confidence > self.confidence_threshold:
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
    
    def _detect_morphological_features(self, brain_volume: np.ndarray) -> List[AnatomicalFeature]:
        """Detect features using advanced morphological analysis."""
        print("   🔬 Morphological feature detection...")
        
        features = []
        
        # Calculate morphological features
        # 1. Local curvature analysis
        curvature = self._calculate_local_curvature(brain_volume)
        
        # 2. Structural tensor analysis
        structure_tensor = self._calculate_structure_tensor(brain_volume)
        
        # 3. Find morphologically interesting regions
        morph_interest = curvature + np.linalg.norm(structure_tensor, axis=-1)
        
        # Find peaks in morphological interest
        threshold = np.percentile(morph_interest, 95)
        peaks = scipy.ndimage.maximum_filter(morph_interest, size=5) == morph_interest
        significant_peaks = peaks & (morph_interest > threshold)
        
        coords = np.where(significant_peaks)
        
        for i in range(min(len(coords[0]), 50)):  # Limit morphological features
            x, y, z = coords[0][i], coords[1][i], coords[2][i]
            
            feature = AnatomicalFeature(
                name=f"Morphological_Feature_{i+1}",
                region_id=hash(('morphological', x, y, z)) % 10000,
                coordinates=(x, y, z),
                volume=self._create_sphere_mask((x, y, z), brain_volume.shape, radius=2),
                confidence=float(morph_interest[x, y, z] / np.max(morph_interest)),
                atlas_label="Morphological_Analysis",
                properties={
                    'tissue_type': BrainTissueType.UNKNOWN,
                    'curvature': float(curvature[x, y, z]),
                    'structure_response': float(np.linalg.norm(structure_tensor[x, y, z])),
                    'detection_method': 'morphological_analysis'
                }
            )
            features.append(feature)
        
        print(f"     📊 Found {len(features)} morphological features")
        return features
    
    def _calculate_local_curvature(self, volume: np.ndarray) -> np.ndarray:
        """Calculate local curvature of the volume."""
        # Simple curvature approximation using second derivatives
        grad_x = np.gradient(volume, axis=0)
        grad_y = np.gradient(volume, axis=1)
        grad_z = np.gradient(volume, axis=2)
        
        # Second derivatives
        grad_xx = np.gradient(grad_x, axis=0)
        grad_yy = np.gradient(grad_y, axis=1)
        grad_zz = np.gradient(grad_z, axis=2)
        
        # Mean curvature approximation
        curvature = np.abs(grad_xx) + np.abs(grad_yy) + np.abs(grad_zz)
        return curvature
    
    def _calculate_structure_tensor(self, volume: np.ndarray) -> np.ndarray:
        """Calculate structure tensor for each voxel."""
        # Gradients
        grad_x = np.gradient(volume, axis=0)
        grad_y = np.gradient(volume, axis=1)
        grad_z = np.gradient(volume, axis=2)
        
        # Structure tensor components
        tensor = np.zeros((*volume.shape, 3, 3))
        tensor[..., 0, 0] = grad_x * grad_x
        tensor[..., 0, 1] = grad_x * grad_y
        tensor[..., 0, 2] = grad_x * grad_z
        tensor[..., 1, 1] = grad_y * grad_y
        tensor[..., 1, 2] = grad_y * grad_z
        tensor[..., 2, 2] = grad_z * grad_z
        
        # Symmetric tensor
        tensor[..., 1, 0] = tensor[..., 0, 1]
        tensor[..., 2, 0] = tensor[..., 0, 2]
        tensor[..., 2, 1] = tensor[..., 1, 2]
        
        return tensor
    
    def _merge_similar_features(self, features: List[AnatomicalFeature]) -> List[AnatomicalFeature]:
        """Merge features that are spatially close and similar."""
        print("   🔄 Merging similar features...")
        
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
    
    def _filter_and_rank_features(self, features: List[AnatomicalFeature]) -> List[AnatomicalFeature]:
        """Filter features by confidence and rank by importance."""
        print("   🎯 Filtering and ranking features...")
        
        # Filter by confidence
        confident_features = [f for f in features if f.confidence >= self.confidence_threshold]
        
        # Sort by confidence
        confident_features.sort(key=lambda f: f.confidence, reverse=True)
        
        # Limit to max regions
        final_features = confident_features[:self.max_regions]
        
        print(f"     📊 Final features: {len(final_features)} (confidence ≥ {self.confidence_threshold})")
        return final_features
    
    def _create_sphere_mask(self, center: Tuple[int, int, int], shape: Tuple[int, int, int], radius: int) -> np.ndarray:
        """Create a spherical mask around a center point."""
        x, y, z = center
        mask = np.zeros(shape, dtype=bool)
        
        # Create spherical region
        xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
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
    
    def _roi_to_anatomical_feature(self, roi: 'ROIDetection', brain_volume: np.ndarray) -> Optional[AnatomicalFeature]:
        """Convert ROI detection to anatomical feature."""
        try:
            x, y, z = roi.coordinates
            
            # Create region mask
            region_mask = self._create_sphere_mask((x, y, z), brain_volume.shape, radius=3)
            
            # Determine tissue type from signal characteristics
            signal_strength = np.mean(brain_volume[region_mask])
            signal_variance = np.var(brain_volume[region_mask])
            tissue_type = self._classify_tissue_from_signal(signal_strength, signal_variance)
            
            feature = AnatomicalFeature(
                name=roi.region_name,
                region_id=roi.region_id,
                coordinates=roi.coordinates,
                volume=region_mask,
                confidence=roi.confidence,
                atlas_label=roi.region_name,
                properties={
                    'tissue_type': tissue_type,
                    'network_state': roi.network_state.value,
                    'activation_strength': roi.activation_strength,
                    'detection_method': 'signal_processing'
                }
            )
            
            return feature
        
        except Exception as e:
            print(f"     ⚠️ Failed to convert ROI to feature: {e}")
            return None

def main():
    """Test the advanced anatomical detector."""
    print("🧠 Testing Advanced Anatomical Detector")
    print("=" * 60)
    
    # Create detector
    detector = AdvancedAnatomicalDetector(max_regions=50)
    
    # Create mock brain volume
    volume_shape = (91, 109, 91)
    mock_volume = np.random.rand(*volume_shape)
    
    # Add some realistic brain structures
    # Gray matter cortex
    mock_volume[20:70, 20:90, 20:70] += 0.3
    # White matter core
    mock_volume[35:55, 35:75, 35:55] += 0.5
    # Ventricles (CSF)
    mock_volume[43:48, 50:60, 43:48] = 0.1
    
    print(f"📊 Processing volume: {volume_shape}")
    
    # Test feature detection
    features = detector.detect_anatomical_features(mock_volume)
    
    print(f"\n✅ Detection Results:")
    print(f"   Total features detected: {len(features)}")
    
    # Show top features
    print(f"\n🏆 Top 10 Features by Confidence:")
    for i, feature in enumerate(features[:10]):
        print(f"   {i+1}. {feature.name}")
        print(f"      Confidence: {feature.confidence:.3f}")
        print(f"      Coordinates: {feature.coordinates}")
        print(f"      Tissue: {feature.properties.get('tissue_type', 'Unknown')}")
        print(f"      Method: {feature.properties.get('detection_method', 'Unknown')}")
        print()
    
    # Show detection method distribution
    methods = {}
    for feature in features:
        method = feature.properties.get('detection_method', 'unknown')
        methods[method] = methods.get(method, 0) + 1
    
    print(f"📊 Detection Method Distribution:")
    for method, count in methods.items():
        print(f"   {method}: {count} features")
    
    print("\n🎉 Advanced Anatomical Detector test completed!")
    return True

if __name__ == "__main__":
    main()