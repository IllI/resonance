#!/usr/bin/env python3
"""
Brain Atlas Manager

This module handles 3D brain atlases and spatial mapping for neural network
analysis results, providing anatomical context to connectivity changes.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt

# Optional imports with error handling
try:
    from nilearn import datasets, plotting, image
    from nilearn.regions import RegionExtractor
    HAS_NILEARN_PLOTTING = True
except ImportError:
    HAS_NILEARN_PLOTTING = False
    print("⚠️ Nilearn plotting not available - some visualization features limited")

class BrainAtlasManager:
    """
    Manager for brain atlases and spatial mapping of neural network results.
    """
    
    def __init__(self, atlas_name: str = 'harvard_oxford'):
        """
        Initialize the Brain Atlas Manager.
        
        Args:
            atlas_name: Name of the atlas to use ('harvard_oxford', 'aal', 'craddock', 'power')
        """
        self.atlas_name = atlas_name
        self.atlas_data = None
        self.atlas_labels = None
        self.region_coordinates = None
        self.available_atlases = [
            'harvard_oxford', 'aal', 'craddock_2012', 'power_2011'
        ]
        
        print(f"🧠 Initialized Brain Atlas Manager")
        print(f"   Atlas: {atlas_name}")
        
        # Load the specified atlas
        self.load_atlas()
    
    def load_atlas(self) -> bool:
        """
        Load the specified brain atlas.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"📥 Loading {self.atlas_name} atlas...")
        
        if not HAS_NILEARN_PLOTTING:
            print("⚠️ Nilearn not available - creating mock atlas for testing")
            self._create_mock_atlas()
            return True
        
        try:
            if self.atlas_name == 'harvard_oxford':
                atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
                self.atlas_data = atlas.maps
                self.atlas_labels = atlas.labels[1:]  # Skip background
                
            elif self.atlas_name == 'aal':
                atlas = datasets.fetch_atlas_aal()
                self.atlas_data = atlas.maps
                self.atlas_labels = atlas.labels
                
            elif self.atlas_name == 'craddock_2012':
                atlas = datasets.fetch_atlas_craddock_2012()
                self.atlas_data = atlas.scorr_mean  # Use correlation-based parcellation
                # Generate labels for regions
                n_regions = len(np.unique(nib.load(self.atlas_data).get_fdata())) - 1
                self.atlas_labels = [f"Region_{i:03d}" for i in range(1, n_regions + 1)]
                
            elif self.atlas_name == 'power_2011':
                atlas = datasets.fetch_coords_power_2011()
                self.region_coordinates = np.column_stack([
                    atlas.rois['x'], atlas.rois['y'], atlas.rois['z']
                ])
                self.atlas_labels = [f"Power_{i:03d}" for i in range(len(atlas.rois))]
                print(f"   Loaded {len(self.atlas_labels)} Power atlas coordinates")
                return True
                
            else:
                raise ValueError(f"Unknown atlas: {self.atlas_name}")
            
            # Load atlas image and extract basic info
            if self.atlas_data:
                atlas_img = nib.load(self.atlas_data)
                unique_regions = np.unique(atlas_img.get_fdata())
                unique_regions = unique_regions[unique_regions > 0]  # Remove background
                
                print(f"✅ Atlas loaded successfully:")
                print(f"   Regions: {len(self.atlas_labels)}")
                print(f"   Unique values: {len(unique_regions)}")
                print(f"   Shape: {atlas_img.shape}")
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to load atlas {self.atlas_name}: {e}")
            print("   Creating mock atlas for testing...")
            self._create_mock_atlas()
            return False
    
    def _create_mock_atlas(self):
        """Create a mock atlas for testing when nilearn is not available."""
        # Create mock data
        n_regions = 100
        self.atlas_labels = [f"Region_{i:03d}" for i in range(n_regions)]
        
        # Create mock coordinates (distributed in brain-like space)
        np.random.seed(42)
        self.region_coordinates = np.random.normal(0, 30, (n_regions, 3))
        
        print(f"✅ Mock atlas created:")
        print(f"   Regions: {len(self.atlas_labels)}")
        print(f"   Coordinates: {self.region_coordinates.shape}")
    
    def get_region_coordinates(self, atlas_img_path: Optional[str] = None) -> np.ndarray:
        """
        Extract center coordinates for all regions in the atlas.
        
        Args:
            atlas_img_path: Optional path to atlas image
            
        Returns:
            Array of coordinates (n_regions x 3)
        """
        if self.region_coordinates is not None:
            return self.region_coordinates
        
        print("📍 Extracting region coordinates...")
        
        if not HAS_NILEARN_PLOTTING or self.atlas_data is None:
            print("⚠️ Using mock coordinates")
            return self.region_coordinates if self.region_coordinates is not None else np.zeros((100, 3))
        
        try:
            # Load atlas image
            atlas_img = nib.load(atlas_img_path or self.atlas_data)
            atlas_data = atlas_img.get_fdata()
            
            # Extract center of mass for each region
            coordinates = []
            unique_regions = np.unique(atlas_data)
            unique_regions = unique_regions[unique_regions > 0]  # Remove background
            
            for region_id in unique_regions:
                # Find voxels belonging to this region
                region_mask = atlas_data == region_id
                if np.any(region_mask):
                    # Calculate center of mass
                    indices = np.where(region_mask)
                    center = [np.mean(indices[i]) for i in range(3)]
                    
                    # Convert to world coordinates
                    world_coords = nib.affines.apply_affine(atlas_img.affine, center)
                    coordinates.append(world_coords)
            
            self.region_coordinates = np.array(coordinates)
            print(f"✅ Extracted {len(coordinates)} region coordinates")
            
            return self.region_coordinates
            
        except Exception as e:
            print(f"⚠️ Error extracting coordinates: {e}")
            print("   Using mock coordinates")
            if self.region_coordinates is None:
                n_regions = len(self.atlas_labels)
                np.random.seed(42)
                self.region_coordinates = np.random.normal(0, 30, (n_regions, 3))
            return self.region_coordinates
    
    def map_connectivity_to_atlas(self, 
                                connectivity_matrix: np.ndarray,
                                region_indices: Optional[List[int]] = None) -> Dict:
        """
        Map connectivity results to anatomical atlas regions.
        
        Args:
            connectivity_matrix: Connectivity matrix to map
            region_indices: Optional list of region indices for mapping
            
        Returns:
            Dictionary with mapped connectivity information
        """
        print("🗺️ Mapping connectivity to atlas regions...")
        
        n_regions = connectivity_matrix.shape[0]
        
        # If no specific mapping provided, assume 1:1 mapping with first N atlas regions
        if region_indices is None:
            max_atlas_regions = len(self.atlas_labels)
            if n_regions <= max_atlas_regions:
                region_indices = list(range(n_regions))
            else:
                print(f"⚠️ More regions ({n_regions}) than atlas labels ({max_atlas_regions})")
                region_indices = list(range(min(n_regions, max_atlas_regions)))
        
        # Create mapping
        mapped_connectivity = {}
        
        for i, atlas_idx in enumerate(region_indices):
            if i < n_regions and atlas_idx < len(self.atlas_labels):
                region_name = self.atlas_labels[atlas_idx]
                
                # Extract connectivity for this region
                region_connections = connectivity_matrix[i, :]
                
                # Find strongest connections
                connection_strength = np.abs(region_connections)
                top_connections = np.argsort(connection_strength)[-5:][::-1]  # Top 5
                
                mapped_connectivity[region_name] = {
                    'region_index': i,
                    'atlas_index': atlas_idx,
                    'connectivity_vector': region_connections,
                    'mean_connectivity': np.mean(region_connections),
                    'max_connectivity': np.max(connection_strength),
                    'top_connections': [
                        {
                            'target_region': self.atlas_labels[region_indices[j]] if j < len(region_indices) else f"Region_{j}",
                            'strength': region_connections[j]
                        }
                        for j in top_connections if j < len(region_connections)
                    ]
                }
        
        print(f"✅ Mapped {len(mapped_connectivity)} regions to atlas")
        return mapped_connectivity
    
    def identify_network_hubs(self, 
                            connectivity_matrix: np.ndarray,
                            threshold: float = 0.8) -> List[Dict]:
        """
        Identify network hubs based on connectivity patterns.
        
        Args:
            connectivity_matrix: Connectivity matrix
            threshold: Threshold for defining hub regions (percentile)
            
        Returns:
            List of hub regions with their properties
        """
        print(f"🎯 Identifying network hubs (threshold={threshold})...")
        
        # Calculate node strength (sum of absolute connections)
        node_strength = np.sum(np.abs(connectivity_matrix), axis=1)
        
        # Identify hubs as regions above threshold percentile
        hub_threshold = np.percentile(node_strength, threshold * 100)
        hub_indices = np.where(node_strength >= hub_threshold)[0]
        
        hubs = []
        for hub_idx in hub_indices:
            hub_info = {
                'region_index': hub_idx,
                'region_name': self.atlas_labels[hub_idx] if hub_idx < len(self.atlas_labels) else f"Region_{hub_idx}",
                'node_strength': node_strength[hub_idx],
                'percentile': (node_strength[hub_idx] / np.max(node_strength)) * 100,
                'coordinates': self.region_coordinates[hub_idx] if self.region_coordinates is not None and hub_idx < len(self.region_coordinates) else [0, 0, 0]
            }
            hubs.append(hub_info)
        
        # Sort by node strength
        hubs.sort(key=lambda x: x['node_strength'], reverse=True)
        
        print(f"✅ Identified {len(hubs)} network hubs:")
        for i, hub in enumerate(hubs[:5]):  # Show top 5
            print(f"   {i+1}. {hub['region_name']}: {hub['node_strength']:.3f}")
        
        return hubs
    
    def analyze_connectivity_changes(self, 
                                   connectivity_matrices: List[np.ndarray],
                                   change_points: List[int]) -> Dict:
        """
        Analyze how connectivity changes relate to anatomical regions.
        
        Args:
            connectivity_matrices: List of connectivity matrices over time
            change_points: Time points where significant changes occurred
            
        Returns:
            Dictionary with change analysis results
        """
        print("📈 Analyzing connectivity changes across brain regions...")
        
        if len(connectivity_matrices) < 2:
            return {'error': 'Need at least 2 connectivity matrices'}
        
        # Calculate connectivity variability for each region pair
        n_regions = connectivity_matrices[0].shape[0]
        connectivity_variance = np.zeros((n_regions, n_regions))
        
        # Stack matrices and compute variance across time
        conn_stack = np.stack(connectivity_matrices, axis=2)
        connectivity_variance = np.var(conn_stack, axis=2)
        
        # Find most variable connections
        most_variable_connections = []
        var_threshold = np.percentile(connectivity_variance, 95)
        
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                if connectivity_variance[i, j] > var_threshold:
                    connection_info = {
                        'region_1': self.atlas_labels[i] if i < len(self.atlas_labels) else f"Region_{i}",
                        'region_2': self.atlas_labels[j] if j < len(self.atlas_labels) else f"Region_{j}",
                        'variance': connectivity_variance[i, j],
                        'mean_connectivity': np.mean([matrix[i, j] for matrix in connectivity_matrices]),
                        'change_magnitude': np.max([matrix[i, j] for matrix in connectivity_matrices]) - 
                                          np.min([matrix[i, j] for matrix in connectivity_matrices])
                    }
                    most_variable_connections.append(connection_info)
        
        # Sort by variance
        most_variable_connections.sort(key=lambda x: x['variance'], reverse=True)
        
        # Analyze changes at specific time points
        change_analysis = {}
        for change_point in change_points:
            if change_point < len(connectivity_matrices) - 1:
                before_matrix = connectivity_matrices[change_point - 1]
                after_matrix = connectivity_matrices[change_point]
                
                difference_matrix = after_matrix - before_matrix
                max_change_idx = np.unravel_index(np.argmax(np.abs(difference_matrix)), difference_matrix.shape)
                
                change_analysis[f'change_point_{change_point}'] = {
                    'max_change_regions': [
                        self.atlas_labels[max_change_idx[0]] if max_change_idx[0] < len(self.atlas_labels) else f"Region_{max_change_idx[0]}",
                        self.atlas_labels[max_change_idx[1]] if max_change_idx[1] < len(self.atlas_labels) else f"Region_{max_change_idx[1]}"
                    ],
                    'change_magnitude': difference_matrix[max_change_idx],
                    'mean_absolute_change': np.mean(np.abs(difference_matrix))
                }
        
        results = {
            'connectivity_variance': connectivity_variance,
            'most_variable_connections': most_variable_connections[:10],  # Top 10
            'change_point_analysis': change_analysis,
            'summary': {
                'n_matrices': len(connectivity_matrices),
                'n_change_points': len(change_points),
                'mean_variance': np.mean(connectivity_variance),
                'max_variance': np.max(connectivity_variance)
            }
        }
        
        print(f"✅ Change analysis completed:")
        print(f"   Most variable connections: {len(most_variable_connections)}")
        print(f"   Change points analyzed: {len(change_points)}")
        print(f"   Mean connectivity variance: {results['summary']['mean_variance']:.6f}")
        
        return results
    
    def create_anatomical_summary(self, 
                                analysis_results: Dict) -> str:
        """
        Create a human-readable anatomical summary of the analysis.
        
        Args:
            analysis_results: Results from connectivity analysis
            
        Returns:
            Formatted anatomical summary string
        """
        print("📋 Creating anatomical summary...")
        
        summary_lines = [
            f"🧠 Brain Network Analysis Summary - {self.atlas_name.upper()} Atlas",
            "=" * 60,
            "",
            f"📊 Atlas Information:",
            f"   Atlas: {self.atlas_name}",
            f"   Total regions: {len(self.atlas_labels)}",
            f"   Coordinate system: {'Available' if self.region_coordinates is not None else 'Not available'}",
            ""
        ]
        
        # Add connectivity information if available
        if 'most_variable_connections' in analysis_results:
            summary_lines.extend([
                "🔗 Most Variable Connections:",
                ""
            ])
            
            for i, conn in enumerate(analysis_results['most_variable_connections'][:5]):
                summary_lines.append(
                    f"   {i+1}. {conn['region_1']} ↔ {conn['region_2']}"
                )
                summary_lines.append(
                    f"      Variance: {conn['variance']:.6f}, Mean: {conn['mean_connectivity']:.3f}"
                )
                summary_lines.append("")
        
        # Add change point information
        if 'change_point_analysis' in analysis_results:
            summary_lines.extend([
                "📈 Temporal Changes:",
                ""
            ])
            
            for change_name, change_info in analysis_results['change_point_analysis'].items():
                summary_lines.append(f"   {change_name}:")
                summary_lines.append(f"      Regions: {change_info['max_change_regions'][0]} ↔ {change_info['max_change_regions'][1]}")
                summary_lines.append(f"      Magnitude: {change_info['change_magnitude']:.6f}")
                summary_lines.append("")
        
        summary_text = "\n".join(summary_lines)
        print("✅ Anatomical summary created")
        
        return summary_text

def main():
    """Test the Brain Atlas Manager."""
    print("🧠 Testing Brain Atlas Manager")
    print("=" * 60)
    
    try:
        # Initialize atlas manager
        atlas_manager = BrainAtlasManager('harvard_oxford')
        
        # Get region coordinates
        coordinates = atlas_manager.get_region_coordinates()
        print(f"\n📍 Region coordinates shape: {coordinates.shape}")
        
        # Create a mock connectivity matrix for testing
        n_regions = min(50, len(atlas_manager.atlas_labels))
        np.random.seed(42)
        connectivity_matrix = np.random.rand(n_regions, n_regions)
        connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(connectivity_matrix, 1.0)
        
        print(f"\n🔗 Testing with {n_regions}x{n_regions} connectivity matrix")
        
        # Map connectivity to atlas
        mapped_connectivity = atlas_manager.map_connectivity_to_atlas(connectivity_matrix)
        
        # Identify network hubs
        hubs = atlas_manager.identify_network_hubs(connectivity_matrix, threshold=0.8)
        
        # Create mock dynamic matrices for change analysis
        dynamic_matrices = []
        for i in range(5):
            matrix = connectivity_matrix + np.random.normal(0, 0.1, connectivity_matrix.shape)
            matrix = (matrix + matrix.T) / 2  # Keep symmetric
            np.fill_diagonal(matrix, 1.0)
            dynamic_matrices.append(matrix)
        
        # Analyze changes
        change_points = [1, 3]
        change_results = atlas_manager.analyze_connectivity_changes(dynamic_matrices, change_points)
        
        # Create summary
        summary = atlas_manager.create_anatomical_summary(change_results)
        
        print("\n" + "=" * 60)
        print("🎉 BRAIN ATLAS INTEGRATION COMPLETED!")
        print("=" * 60)
        
        print(f"\n📊 Results Summary:")
        print(f"   Atlas regions: {len(atlas_manager.atlas_labels)}")
        print(f"   Mapped connections: {len(mapped_connectivity)}")
        print(f"   Network hubs: {len(hubs)}")
        print(f"   Variable connections: {len(change_results['most_variable_connections'])}")
        
        print(f"\n🧠 Top Network Hubs:")
        for i, hub in enumerate(hubs[:3]):
            print(f"   {i+1}. {hub['region_name']}")
            print(f"      Strength: {hub['node_strength']:.3f}")
        
        print("\n" + summary)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 