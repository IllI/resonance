#!/usr/bin/env python3
"""
3D Brain Visualization System

Real-time 3D visualization of brain networks and connectivity changes
using the dual-GPU accelerated fMRI analysis results.

Phase 4: Advanced 3D visualization with interactive brain network display.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Optional imports for advanced visualization
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from mayavi import mlab
    HAS_MAYAVI = True
except ImportError:
    HAS_MAYAVI = False
    print("WARNING: Mayavi not available - using Plotly for 3D visualization")

# Import our analysis modules
try:
    from enhanced_dual_gpu_analyzer import EnhancedDualGPUBrainAnalyzer
    from brain_atlas_manager import BrainAtlasManager
    from neural_network_analyzer import BrainNetworkAnalyzer
    HAS_ANALYSIS_MODULES = True
except ImportError as e:
    print(f"WARNING: Analysis modules not available: {e}")
    HAS_ANALYSIS_MODULES = False

class Brain3DVisualizer:
    """
    Advanced 3D brain visualization system for dual-GPU analysis results.
    
    Features:
    - Real-time 3D brain network visualization
    - Dynamic connectivity animation
    - Interactive brain atlas exploration
    - Network hub highlighting
    - Change point visualization
    """
    
    def __init__(self, atlas_name: str = 'harvard_oxford'):
        """
        Initialize the 3D brain visualizer.
        
        Args:
            atlas_name: Brain atlas to use for anatomical localization
        """
        self.atlas_name = atlas_name
        self.brain_coords = None
        self.atlas_labels = None
        self.current_connectivity = None
        self.dynamic_matrices = []
        self.network_hubs = []
        
        print(" Initializing 3D Brain Visualizer...")
        print(f"   Atlas: {atlas_name}")
        
        if HAS_ANALYSIS_MODULES:
            try:
                self.atlas_manager = BrainAtlasManager(atlas_name)
                self.atlas_manager.load_atlas()
                self.brain_coords = self.atlas_manager.get_region_coordinates()
                self.atlas_labels = self.atlas_manager.atlas_labels
                print(f" Loaded {len(self.atlas_labels)} brain regions")
            except Exception as e:
                print(f"WARNING: Using mock brain coordinates: {e}")
                self._create_mock_brain_coordinates()
        else:
            self._create_mock_brain_coordinates()
        
        print(" 3D Brain Visualizer initialized")
    
    def _create_mock_brain_coordinates(self):
        """Create mock brain coordinates for demonstration."""
        print(" Creating mock brain coordinates...")
        
        # Create brain-like coordinate distribution
        n_regions = 100
        np.random.seed(42)
        
        # Generate coordinates in brain-like distribution
        # Cerebral cortex regions (outer shell)
        cortex_regions = 60
        phi = np.random.uniform(0, 2*np.pi, cortex_regions)
        theta = np.random.uniform(0, np.pi, cortex_regions)
        r = np.random.uniform(25, 35, cortex_regions)
        
        cortex_x = r * np.sin(theta) * np.cos(phi)
        cortex_y = r * np.sin(theta) * np.sin(phi)
        cortex_z = r * np.cos(theta)
        
        # Subcortical regions (inner regions)
        subcortex_regions = 40
        subcortex_x = np.random.uniform(-15, 15, subcortex_regions)
        subcortex_y = np.random.uniform(-15, 15, subcortex_regions)
        subcortex_z = np.random.uniform(-20, 20, subcortex_regions)
        
        # Combine coordinates
        x = np.concatenate([cortex_x, subcortex_x])
        y = np.concatenate([cortex_y, subcortex_y])
        z = np.concatenate([cortex_z, subcortex_z])
        
        self.brain_coords = np.column_stack([x, y, z])
        
        # Create mock labels
        cortex_labels = [f"Cortex_Region_{i+1}" for i in range(cortex_regions)]
        subcortex_labels = [f"Subcortex_Region_{i+1}" for i in range(subcortex_regions)]
        self.atlas_labels = cortex_labels + subcortex_labels
        
        print(f" Created {len(self.atlas_labels)} mock brain regions")
    
    def visualize_static_connectivity(self, 
                                    connectivity_matrix: np.ndarray,
                                    threshold: float = 0.3,
                                    title: str = "Brain Connectivity Network") -> go.Figure:
        """
        Create 3D visualization of static brain connectivity.
        
        Args:
            connectivity_matrix: Connectivity matrix (regions x regions)
            threshold: Connection strength threshold for visualization
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        print(f" Creating 3D static connectivity visualization...")
        print(f"   Matrix shape: {connectivity_matrix.shape}")
        print(f"   Threshold: {threshold}")
        
        if self.brain_coords is None:
            print(" Brain coordinates not available")
            return None
        
        # Ensure matrix size matches coordinates
        n_coords = min(len(self.brain_coords), connectivity_matrix.shape[0])
        coords = self.brain_coords[:n_coords]
        matrix = connectivity_matrix[:n_coords, :n_coords]
        
        fig = go.Figure()
        
        # Add brain regions as scatter points
        region_colors = np.random.rand(n_coords)  # Color by region ID
        
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1], 
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=region_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Region ID")
            ),
            text=[f"{self.atlas_labels[i]}<br>Coord: ({coords[i,0]:.1f}, {coords[i,1]:.1f}, {coords[i,2]:.1f})" 
                  for i in range(n_coords)],
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Brain Regions"
        ))
        
        # Add connections as lines
        connection_count = 0
        line_x, line_y, line_z = [], [], []
        
        for i in range(n_coords):
            for j in range(i+1, n_coords):
                if abs(matrix[i, j]) > threshold:
                    connection_count += 1
                    
                    # Add line between connected regions
                    line_x.extend([coords[i,0], coords[j,0], None])
                    line_y.extend([coords[i,1], coords[j,1], None])
                    line_z.extend([coords[i,2], coords[j,2], None])
        
        if connection_count > 0:
            fig.add_trace(go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode='lines',
                line=dict(
                    color='rgba(255,255,255,0.3)',
                    width=2
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br>Connections: {connection_count} (threshold: {threshold})",
                x=0.5
            ),
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                bgcolor='black',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800,
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        print(f" Created 3D visualization with {connection_count} connections")
        return fig
    
    def visualize_network_hubs(self,
                             connectivity_matrix: np.ndarray,
                             node_strength: np.ndarray,
                             hub_threshold: float = 0.8) -> go.Figure:
        """
        Create 3D visualization highlighting network hubs.
        
        Args:
            connectivity_matrix: Connectivity matrix
            node_strength: Node strength values
            hub_threshold: Threshold for hub identification
            
        Returns:
            Plotly figure with network hubs highlighted
        """
        print(f" Creating network hubs visualization...")
        
        if self.brain_coords is None:
            return None
        
        # Identify hubs
        hubs = node_strength > np.percentile(node_strength, hub_threshold * 100)
        hub_indices = np.where(hubs)[0]
        
        print(f"   Identified {len(hub_indices)} network hubs")
        
        n_coords = min(len(self.brain_coords), len(node_strength))
        coords = self.brain_coords[:n_coords]
        
        fig = go.Figure()
        
        # Regular regions
        regular_mask = ~hubs[:n_coords]
        if np.any(regular_mask):
            fig.add_trace(go.Scatter3d(
                x=coords[regular_mask, 0],
                y=coords[regular_mask, 1],
                z=coords[regular_mask, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color='lightblue',
                    opacity=0.6
                ),
                name="Regular Regions"
            ))
        
        # Network hubs
        hub_mask = hubs[:n_coords]
        if np.any(hub_mask):
            fig.add_trace(go.Scatter3d(
                x=coords[hub_mask, 0],
                y=coords[hub_mask, 1],
                z=coords[hub_mask, 2],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    opacity=0.9,
                    line=dict(width=2, color='white')
                ),
                text=[f"HUB: {self.atlas_labels[i]}<br>Strength: {node_strength[i]:.3f}" 
                      for i in hub_indices if i < len(self.atlas_labels)],
                name="Network Hubs"
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Brain Network Hubs<br>{len(hub_indices)} hubs identified (top {(1-hub_threshold)*100}%)",
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                bgcolor='black'
            ),
            paper_bgcolor='black',
            font=dict(color='white'),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_dynamic_connectivity_animation(self,
                                            dynamic_matrices: List[np.ndarray],
                                            time_points: List[float],
                                            threshold: float = 0.3) -> go.Figure:
        """
        Create animated visualization of dynamic connectivity changes.
        
        Args:
            dynamic_matrices: List of connectivity matrices over time
            time_points: Time points for each matrix
            threshold: Connection threshold
            
        Returns:
            Animated Plotly figure
        """
        print(f"🎬 Creating dynamic connectivity animation...")
        print(f"   Time points: {len(dynamic_matrices)}")
        
        if not dynamic_matrices or self.brain_coords is None:
            return None
        
        # Create frames for animation
        frames = []
        
        for i, matrix in enumerate(dynamic_matrices):
            n_coords = min(len(self.brain_coords), matrix.shape[0])
            coords = self.brain_coords[:n_coords]
            
            # Calculate node strength for this time point
            node_strength = np.sum(np.abs(matrix[:n_coords, :n_coords]), axis=1)
            max_strength = np.max(node_strength)
            
            # Create frame data
            frame_data = [
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5 + 10 * (node_strength / max_strength) if max_strength > 0 else 5,
                        color=node_strength,
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Node Strength")
                    ),
                    name=f"Time {time_points[i]:.1f}s"
                )
            ]
            
            frames.append(go.Frame(data=frame_data, name=str(i)))
        
        # Create initial figure
        fig = go.Figure(data=frames[0].data, frames=frames)
        
        # Add animation controls
        fig.update_layout(
            title="Dynamic Brain Connectivity Animation",
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)', 
                zaxis_title='Z (mm)',
                bgcolor='black'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[frame.name], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }],
                        'label': f"T={time_points[int(frame.name)]:.1f}s",
                        'method': 'animate'
                    }
                    for frame in frames
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Time: '},
                'transition': {'duration': 300},
                'x': 0.1,
                'len': 0.9
            }],
            paper_bgcolor='black',
            font=dict(color='white'),
            width=1000,
            height=800
        )
        
        print(" Dynamic connectivity animation created")
        return fig
    
    def create_comprehensive_brain_dashboard(self, analysis_results: Dict) -> go.Figure:
        """
        Create comprehensive dashboard with multiple brain visualizations.
        
        Args:
            analysis_results: Results from dual-GPU brain analysis
            
        Returns:
            Multi-panel dashboard figure
        """
        print(" Creating comprehensive brain analysis dashboard...")
        
        # Create subplot figure with multiple panels
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}]],
            subplot_titles=(
                '3D Brain Network',
                'Network Metrics Over Time', 
                'Connectivity Matrix',
                'Regional Analysis'
            )
        )
        
        # Panel 1: 3D Brain Network
        if 'correlation_matrix' in analysis_results and self.brain_coords is not None:
            matrix = analysis_results['correlation_matrix']
            n_coords = min(len(self.brain_coords), matrix.shape[0])
            coords = self.brain_coords[:n_coords]
            
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode='markers',
                    marker=dict(size=5, color='lightblue'),
                    name="Brain Regions"
                ),
                row=1, col=1
            )
        
        # Panel 2: Network metrics over time
        if 'dynamic_matrices' in analysis_results:
            n_timepoints = len(analysis_results['dynamic_matrices'])
            time_points = np.arange(n_timepoints) * 2.5  # Assuming 2.5s TR
            
            # Calculate global efficiency over time (mock)
            efficiency = np.random.rand(n_timepoints) * 0.5 + 0.3
            
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=efficiency,
                    mode='lines+markers',
                    name="Global Efficiency",
                    line=dict(color='orange')
                ),
                row=1, col=2
            )
        
        # Panel 3: Connectivity Matrix Heatmap
        if 'correlation_matrix' in analysis_results:
            matrix = analysis_results['correlation_matrix'][:50, :50]  # Show subset
            
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=True
                ),
                row=2, col=1
            )
        
        # Panel 4: Regional Analysis
        if 'network_metrics' in analysis_results:
            metrics = analysis_results['network_metrics']
            if 'node_strength' in metrics:
                strength = metrics['node_strength'][:20]  # Top 20 regions
                regions = [f"R{i+1}" for i in range(len(strength))]
                
                fig.add_trace(
                    go.Bar(
                        x=regions,
                        y=strength,
                        name="Node Strength",
                        marker=dict(color='green')
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Comprehensive Brain Analysis Dashboard",
            height=800,
            showlegend=True,
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        print(" Comprehensive dashboard created")
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: str, format: str = 'html'):
        """
        Save visualization to file.
        
        Args:
            fig: Plotly figure to save
            filename: Output filename
            format: Output format ('html', 'png', 'svg')
        """
        output_path = Path(f"visualizations/{filename}.{format}")
        output_path.parent.mkdir(exist_ok=True)
        
        if format == 'html':
            fig.write_html(str(output_path))
        elif format == 'png':
            fig.write_image(str(output_path))
        elif format == 'svg':
            fig.write_image(str(output_path))
        
        print(f" Visualization saved: {output_path}")

def main():
    """Demonstrate the 3D brain visualization capabilities."""
    print(" 3D Brain Visualization System Test")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = Brain3DVisualizer()
    
    if HAS_ANALYSIS_MODULES:
        print("\n Running dual-GPU analysis for visualization...")
        
        # Get analysis results
        analyzer = EnhancedDualGPUBrainAnalyzer()
        
        # Mock data for visualization
        n_regions, n_timepoints = 100, 200
        mock_data = np.random.randn(n_regions, n_timepoints)
        
        # Run analysis
        results = analyzer.analyze_fmri_data_dual_gpu(mock_data)
        
        print("\n Creating 3D visualizations...")
        
        # 1. Static connectivity visualization
        if 'correlation_matrix' in results:
            fig1 = visualizer.visualize_static_connectivity(
                results['correlation_matrix'],
                threshold=0.3,
                title="Dual-GPU Brain Connectivity Analysis"
            )
            
            if fig1:
                print(" Showing 3D brain connectivity...")
                fig1.show()
                visualizer.save_visualization(fig1, "brain_connectivity_3d")
        
        # 2. Network hubs visualization
        if 'network_metrics' in results and 'node_strength' in results['network_metrics']:
            fig2 = visualizer.visualize_network_hubs(
                results['correlation_matrix'],
                results['network_metrics']['node_strength']
            )
            
            if fig2:
                print(" Showing network hubs...")
                fig2.show()
                visualizer.save_visualization(fig2, "network_hubs_3d")
        
        # 3. Dynamic connectivity animation  
        if 'dynamic_matrices' in results:
            time_points = [i * 2.5 for i in range(len(results['dynamic_matrices']))]
            fig3 = visualizer.create_dynamic_connectivity_animation(
                results['dynamic_matrices'][:5],  # First 5 timepoints
                time_points[:5]
            )
            
            if fig3:
                print("🎬 Showing dynamic connectivity animation...")
                fig3.show()
                visualizer.save_visualization(fig3, "dynamic_connectivity_animation")
        
        # 4. Comprehensive dashboard
        fig4 = visualizer.create_comprehensive_brain_dashboard(results)
        if fig4:
            print(" Showing comprehensive dashboard...")
            fig4.show()
            visualizer.save_visualization(fig4, "brain_analysis_dashboard")
    
    else:
        print("\n Creating demo visualizations with mock data...")
        
        # Create mock connectivity matrix
        n_regions = 80
        mock_matrix = np.random.rand(n_regions, n_regions)
        mock_matrix = (mock_matrix + mock_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(mock_matrix, 1.0)
        
        # Demo static visualization
        fig = visualizer.visualize_static_connectivity(
            mock_matrix,
            threshold=0.7,
            title="Demo: 3D Brain Connectivity"
        )
        
        if fig:
            print(" Showing demo 3D brain connectivity...")
            fig.show()
            visualizer.save_visualization(fig, "demo_brain_connectivity")
    
    print("\n🎉 3D Brain Visualization System demonstration complete!")
    print("   Check the 'visualizations' folder for saved outputs")

if __name__ == "__main__":
    main() 