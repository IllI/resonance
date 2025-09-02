#!/usr/bin/env python3
"""
4D Spatial-Temporal Brain Registration System

Implements advanced 4D registration to align 3D anatomical brain models with 
4D fMRI time series data. Allows the 3D model to "shift" through 4D space 
to maintain proper anatomical orientation at each timepoint.
"""

import numpy as np
import scipy.ndimage
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from pathlib import Path

@dataclass
class SpatialTransform:
    """Represents a 3D spatial transformation."""
    translation: np.ndarray  # [x, y, z] translation
    rotation: np.ndarray     # [rx, ry, rz] rotation angles (radians)
    scaling: np.ndarray      # [sx, sy, sz] scaling factors
    matrix: np.ndarray       # 4x4 transformation matrix

@dataclass
class TemporalRegistrationResult:
    """Results from temporal registration."""
    timepoint: int
    transform: SpatialTransform
    alignment_score: float
    anatomical_features: List[Dict]
    confidence: float

class SpatioTemporalBrainRegistrator:
    """
    Advanced 4D registration system for aligning 3D brain models with 4D fMRI data.
    """
    
    def __init__(self, registration_method: str = 'mutual_information'):
        """Initialize the registration system."""
        self.registration_method = registration_method
        self.reference_model = None
        self.temporal_transforms = {}
        
        print("🧠 Initializing 4D Spatial-Temporal Brain Registration System")
        print(f"   Registration method: {registration_method}")
    
    def register_model_to_fmri_sequence(self, 
                                       brain_model: Dict,
                                       fmri_4d: np.ndarray,
                                       anatomical_features: List[Dict]) -> List[TemporalRegistrationResult]:
        """
        Register 3D brain model to each timepoint of 4D fMRI sequence.
        
        Args:
            brain_model: 3D anatomical brain model
            fmri_4d: 4D fMRI data (time, x, y, z)
            anatomical_features: Known anatomical features from the model
            
        Returns:
            List of registration results for each timepoint
        """
        print(f"🔄 Registering 3D model to 4D fMRI sequence: {fmri_4d.shape}")
        
        # Extract reference anatomy from 3D model
        reference_volume = brain_model.get('volume', brain_model.get('labeled_volume'))
        if reference_volume is None:
            raise ValueError("Brain model must contain 'volume' or 'labeled_volume'")
        
        self.reference_model = reference_volume
        n_timepoints = fmri_4d.shape[0]
        
        results = []
        
        for t in range(n_timepoints):
            print(f"   📍 Registering timepoint {t+1}/{n_timepoints}...")
            
            # Extract current fMRI volume
            current_volume = fmri_4d[t]
            
            # Compute optimal transformation
            transform = self._compute_optimal_transform(
                reference_volume, current_volume, anatomical_features
            )
            
            # Apply transformation to anatomical features
            transformed_features = self._transform_anatomical_features(
                anatomical_features, transform
            )
            
            # Compute alignment quality
            alignment_score, confidence = self._evaluate_alignment(
                current_volume, transformed_features
            )
            
            result = TemporalRegistrationResult(
                timepoint=t,
                transform=transform,
                alignment_score=alignment_score,
                anatomical_features=transformed_features,
                confidence=confidence
            )
            
            results.append(result)
            self.temporal_transforms[t] = transform
            
            if (t + 1) % 20 == 0:
                avg_score = np.mean([r.alignment_score for r in results[-20:]])
                print(f"     📈 Average alignment score (last 20): {avg_score:.3f}")
        
        print(f"✅ 4D registration completed: {len(results)} timepoints processed")
        return results
    
    def _compute_optimal_transform(self, 
                                  reference: np.ndarray,
                                  target: np.ndarray,
                                  features: List[Dict]) -> SpatialTransform:
        """Compute optimal transformation between reference and target volumes."""
        
        # Initialize transformation parameters
        translation = np.array([0.0, 0.0, 0.0])
        rotation = np.array([0.0, 0.0, 0.0])
        scaling = np.array([1.0, 1.0, 1.0])
        
        # Feature-based registration using anatomical landmarks
        if features:
            translation, rotation = self._register_using_features(
                reference, target, features
            )
        
        # Intensity-based registration refinement
        translation, rotation, scaling = self._refine_with_intensity_matching(
            reference, target, translation, rotation, scaling
        )
        
        # Create transformation matrix
        matrix = self._create_transformation_matrix(translation, rotation, scaling)
        
        return SpatialTransform(
            translation=translation,
            rotation=rotation, 
            scaling=scaling,
            matrix=matrix
        )
    
    def _register_using_features(self,
                               reference: np.ndarray,
                               target: np.ndarray, 
                               features: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Register using anatomical feature landmarks."""
        
        # Extract landmark coordinates from features
        ref_landmarks = []
        for feature in features[:10]:  # Use top 10 features
            if 'coordinates' in feature:
                ref_landmarks.append(feature['coordinates'])
        
        if len(ref_landmarks) < 3:
            return np.zeros(3), np.zeros(3)
        
        ref_landmarks = np.array(ref_landmarks)
        
        # Find corresponding points in target volume
        target_landmarks = self._find_corresponding_landmarks(
            target, ref_landmarks
        )
        
        # Compute transformation using landmark pairs
        if len(target_landmarks) >= 3:
            translation, rotation = self._compute_landmark_transform(
                ref_landmarks, target_landmarks
            )
        else:
            translation, rotation = np.zeros(3), np.zeros(3)
        
        return translation, rotation
    
    def _find_corresponding_landmarks(self,
                                    target: np.ndarray,
                                    ref_landmarks: np.ndarray) -> np.ndarray:
        """Find corresponding landmark points in target volume."""
        
        target_landmarks = []
        
        for ref_point in ref_landmarks:
            # Search in neighborhood for best match
            best_match = self._find_best_match_point(target, ref_point)
            target_landmarks.append(best_match)
        
        return np.array(target_landmarks)
    
    def _find_best_match_point(self, 
                              target: np.ndarray,
                              ref_point: np.ndarray,
                              search_radius: int = 10) -> np.ndarray:
        """Find best matching point in target volume."""
        
        x, y, z = ref_point.astype(int)
        
        # Define search region
        x_min = max(0, x - search_radius)
        x_max = min(target.shape[0], x + search_radius + 1)
        y_min = max(0, y - search_radius)
        y_max = min(target.shape[1], y + search_radius + 1)
        z_min = max(0, z - search_radius)
        z_max = min(target.shape[2], z + search_radius + 1)
        
        # Find maximum intensity point in search region
        search_region = target[x_min:x_max, y_min:y_max, z_min:z_max]
        
        if search_region.size > 0:
            max_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
            best_point = np.array([
                x_min + max_idx[0],
                y_min + max_idx[1], 
                z_min + max_idx[2]
            ])
        else:
            best_point = ref_point
        
        return best_point
    
    def _compute_landmark_transform(self,
                                  ref_points: np.ndarray,
                                  target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute transformation from landmark correspondences."""
        
        # Compute centroids
        ref_centroid = np.mean(ref_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        # Translation
        translation = target_centroid - ref_centroid
        
        # Center points
        ref_centered = ref_points - ref_centroid
        target_centered = target_points - target_centroid
        
        # Compute rotation using SVD
        H = ref_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        
        # Convert rotation matrix to Euler angles
        rotation_obj = Rotation.from_matrix(R)
        rotation = rotation_obj.as_euler('xyz')
        
        return translation, rotation
    
    def _refine_with_intensity_matching(self,
                                      reference: np.ndarray,
                                      target: np.ndarray,
                                      init_translation: np.ndarray,
                                      init_rotation: np.ndarray,
                                      init_scaling: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refine transformation using intensity-based registration."""
        
        # Simple optimization using correlation
        best_translation = init_translation.copy()
        best_rotation = init_rotation.copy()
        best_scaling = init_scaling.copy()
        best_score = -np.inf
        
        # Grid search around initial values
        search_range = 2.0
        steps = 5
        
        for dx in np.linspace(-search_range, search_range, steps):
            for dy in np.linspace(-search_range, search_range, steps):
                for dz in np.linspace(-search_range, search_range, steps):
                    
                    trial_translation = init_translation + np.array([dx, dy, dz])
                    
                    # Apply transformation and compute similarity
                    transformed = self._apply_transform_to_volume(
                        reference, trial_translation, init_rotation, init_scaling
                    )
                    
                    score = self._compute_similarity_score(transformed, target)
                    
                    if score > best_score:
                        best_score = score
                        best_translation = trial_translation
        
        return best_translation, best_rotation, best_scaling
    
    def _apply_transform_to_volume(self,
                                 volume: np.ndarray,
                                 translation: np.ndarray,
                                 rotation: np.ndarray,
                                 scaling: np.ndarray) -> np.ndarray:
        """Apply spatial transformation to volume."""
        
        # Create transformation matrix
        transform_matrix = self._create_transformation_matrix(
            translation, rotation, scaling
        )
        
        # Apply affine transformation
        transformed = scipy.ndimage.affine_transform(
            volume,
            transform_matrix[:3, :3],
            offset=transform_matrix[:3, 3],
            output_shape=volume.shape,
            mode='constant',
            cval=0.0
        )
        
        return transformed
    
    def _create_transformation_matrix(self,
                                    translation: np.ndarray,
                                    rotation: np.ndarray,
                                    scaling: np.ndarray) -> np.ndarray:
        """Create 4x4 transformation matrix."""
        
        # Create rotation matrix
        rot_obj = Rotation.from_euler('xyz', rotation)
        rot_matrix = rot_obj.as_matrix()
        
        # Create scaling matrix
        scale_matrix = np.diag(scaling)
        
        # Combine rotation and scaling
        linear_transform = rot_matrix @ scale_matrix
        
        # Create 4x4 homogeneous transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = linear_transform
        transform_matrix[:3, 3] = translation
        
        return transform_matrix
    
    def _compute_similarity_score(self, 
                                volume1: np.ndarray,
                                volume2: np.ndarray) -> float:
        """Compute similarity score between volumes."""
        
        # Normalize volumes
        v1_norm = (volume1 - np.mean(volume1)) / (np.std(volume1) + 1e-8)
        v2_norm = (volume2 - np.mean(volume2)) / (np.std(volume2) + 1e-8)
        
        # Compute normalized cross-correlation
        correlation = np.corrcoef(v1_norm.flatten(), v2_norm.flatten())[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _transform_anatomical_features(self,
                                     features: List[Dict],
                                     transform: SpatialTransform) -> List[Dict]:
        """Apply spatial transformation to anatomical features."""
        
        transformed_features = []
        
        for feature in features:
            transformed_feature = feature.copy()
            
            # Transform coordinates
            if 'coordinates' in feature:
                coords = np.array(feature['coordinates'])
                coords_homo = np.append(coords, 1.0)  # Homogeneous coordinates
                transformed_coords = transform.matrix @ coords_homo
                transformed_feature['coordinates'] = transformed_coords[:3]
            
            # Transform volume mask if present
            if 'volume' in feature and feature['volume'] is not None:
                transformed_volume = self._apply_transform_to_volume(
                    feature['volume'].astype(float),
                    transform.translation,
                    transform.rotation,
                    transform.scaling
                )
                transformed_feature['volume'] = transformed_volume > 0.5
            
            transformed_features.append(transformed_feature)
        
        return transformed_features
    
    def _evaluate_alignment(self,
                          fmri_volume: np.ndarray,
                          transformed_features: List[Dict]) -> Tuple[float, float]:
        """Evaluate quality of anatomical-functional alignment."""
        
        if not transformed_features:
            return 0.0, 0.0
        
        alignment_scores = []
        
        for feature in transformed_features:
            if 'coordinates' in feature:
                coords = feature['coordinates'].astype(int)
                
                # Check if coordinates are within volume bounds
                if (0 <= coords[0] < fmri_volume.shape[0] and
                    0 <= coords[1] < fmri_volume.shape[1] and
                    0 <= coords[2] < fmri_volume.shape[2]):
                    
                    # Get functional signal at anatomical location
                    signal_strength = fmri_volume[coords[0], coords[1], coords[2]]
                    
                    # Score based on signal strength and feature confidence
                    feature_confidence = feature.get('confidence', 0.5)
                    alignment_score = signal_strength * feature_confidence
                    alignment_scores.append(alignment_score)
        
        if alignment_scores:
            mean_score = np.mean(alignment_scores)
            confidence = len(alignment_scores) / len(transformed_features)
        else:
            mean_score = 0.0
            confidence = 0.0
        
        return mean_score, confidence
    
    def get_anatomical_attribution(self,
                                 fmri_4d: np.ndarray,
                                 registration_results: List[TemporalRegistrationResult]) -> Dict[str, np.ndarray]:
        """
        Extract anatomical attribution of functional signals across time.
        
        Returns:
            Dictionary mapping anatomical regions to their temporal signals
        """
        print("🧬 Extracting anatomical attribution from 4D registration...")
        
        # Collect all unique anatomical regions
        all_regions = set()
        for result in registration_results:
            for feature in result.anatomical_features:
                if 'name' in feature:
                    all_regions.add(feature['name'])
        
        region_signals = {}
        
        for region_name in all_regions:
            temporal_signal = []
            
            for t, result in enumerate(registration_results):
                # Find this region in current timepoint
                region_signal = 0.0
                region_count = 0
                
                for feature in result.anatomical_features:
                    if feature.get('name') == region_name:
                        coords = feature.get('coordinates')
                        if coords is not None:
                            coords = coords.astype(int)
                            
                            # Extract signal from fMRI volume
                            if (0 <= coords[0] < fmri_4d.shape[1] and
                                0 <= coords[1] < fmri_4d.shape[2] and
                                0 <= coords[2] < fmri_4d.shape[3]):
                                
                                signal = fmri_4d[t, coords[0], coords[1], coords[2]]
                                region_signal += signal
                                region_count += 1
                
                # Average signal for this region at this timepoint
                if region_count > 0:
                    region_signal /= region_count
                
                temporal_signal.append(region_signal)
            
            region_signals[region_name] = np.array(temporal_signal)
        
        print(f"✅ Extracted signals for {len(region_signals)} anatomical regions")
        return region_signals
    
    def save_registration_results(self, 
                                results: List[TemporalRegistrationResult],
                                output_path: str):
        """Save registration results to file."""
        
        output_data = {
            'timepoints': len(results),
            'registration_method': self.registration_method,
            'results': []
        }
        
        for result in results:
            result_data = {
                'timepoint': result.timepoint,
                'transform': {
                    'translation': result.transform.translation.tolist(),
                    'rotation': result.transform.rotation.tolist(),
                    'scaling': result.transform.scaling.tolist(),
                    'matrix': result.transform.matrix.tolist()
                },
                'alignment_score': result.alignment_score,
                'confidence': result.confidence,
                'n_features': len(result.anatomical_features)
            }
            output_data['results'].append(result_data)
        
        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Registration results saved to {output_path}")


def demo_4d_registration():
    """Demonstrate 4D spatial-temporal registration."""
    print("🎬 4D Spatial-Temporal Brain Registration Demo")
    print("=" * 60)
    
    # Create mock 3D brain model
    print("🧠 Creating mock 3D brain model...")
    brain_model = {
        'volume': np.random.rand(64, 64, 48),
        'shape': (64, 64, 48),
        'features': []
    }
    
    # Add mock anatomical features
    anatomical_features = []
    for i in range(10):
        feature = {
            'name': f'Region_{i}',
            'coordinates': np.random.randint(10, 54, 3),
            'confidence': np.random.uniform(0.7, 0.95),
            'volume': np.random.rand(64, 64, 48) > 0.9
        }
        anatomical_features.append(feature)
    
    print(f"   Created model with {len(anatomical_features)} anatomical features")
    
    # Create mock 4D fMRI data
    print("📈 Creating mock 4D fMRI time series...")
    n_timepoints = 20
    fmri_4d = np.random.rand(n_timepoints, 64, 64, 48)
    
    # Add temporal dynamics
    for t in range(n_timepoints):
        # Add some spatial shift over time
        shift_x = int(2 * np.sin(2 * np.pi * t / n_timepoints))
        shift_y = int(2 * np.cos(2 * np.pi * t / n_timepoints))
        
        # Apply shifts to simulate brain motion
        if shift_x != 0 or shift_y != 0:
            fmri_4d[t] = np.roll(fmri_4d[t], shift_x, axis=0)
            fmri_4d[t] = np.roll(fmri_4d[t], shift_y, axis=1)
    
    print(f"   Created 4D fMRI data: {fmri_4d.shape}")
    
    # Initialize registration system
    print("🔄 Initializing registration system...")
    registrator = SpatioTemporalBrainRegistrator(registration_method='mutual_information')
    
    # Perform 4D registration
    print("⚡ Performing 4D spatial-temporal registration...")
    start_time = time.time()
    
    registration_results = registrator.register_model_to_fmri_sequence(
        brain_model, fmri_4d, anatomical_features
    )
    
    registration_time = time.time() - start_time
    
    # Extract anatomical attribution
    print("🧬 Extracting anatomical attribution...")
    region_signals = registrator.get_anatomical_attribution(fmri_4d, registration_results)
    
    # Display results
    print("\n📊 Registration Results:")
    print(f"   Timepoints processed: {len(registration_results)}")
    print(f"   Registration time: {registration_time:.2f}s")
    print(f"   Average alignment score: {np.mean([r.alignment_score for r in registration_results]):.3f}")
    print(f"   Average confidence: {np.mean([r.confidence for r in registration_results]):.3f}")
    
    print(f"\n🧬 Anatomical Attribution:")
    print(f"   Regions tracked: {len(region_signals)}")
    for region_name, signal in list(region_signals.items())[:5]:
        print(f"   {region_name}: signal std = {np.std(signal):.4f}")
    
    # Save results
    output_path = "demo_4d_registration_results.json"
    registrator.save_registration_results(registration_results, output_path)
    
    print(f"\n✅ 4D Registration Demo Completed!")
    print(f"   Results saved to: {output_path}")
    
    return registration_results, region_signals


if __name__ == "__main__":
    # Run demonstration
    results, signals = demo_4d_registration()