#!/usr/bin/env python3
"""
Neural Network Analyzer

This module implements algorithms for detecting and analyzing changes in brain
neural networks from fMRI time series data, specifically designed for Kevin O'Neill's
modernized fMRI brain analysis project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path

# Import optional packages with error handling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from sklearn.decomposition import PCA
    try:
        from sklearn.decomposition import ICA
    except ImportError:
        # ICA might not be available in older sklearn versions
        ICA = None
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    PCA = None
    ICA = None
    KMeans = None
    silhouette_score = None
    StandardScaler = None
    print("WARNING: scikit-learn not available - some features will be limited")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("⚠️ NetworkX not available - network metrics will be limited")

class BrainNetworkAnalyzer:
    """
    Advanced brain network analysis for detecting temporal changes in neural connectivity.
    """
    
    def __init__(self, sampling_rate: float = 0.4):
        """
        Initialize the Brain Network Analyzer.
        
        Args:
            sampling_rate: fMRI sampling rate in Hz (default 0.4 Hz = 2.5s TR)
        """
        self.sampling_rate = sampling_rate
        self.time_series = None
        self.connectivity_matrices = []
        self.network_metrics = {}
        self.change_points = []
        
        print(f"🧠 Initialized Brain Network Analyzer")
        print(f"   Sampling rate: {sampling_rate} Hz")
    
    def load_time_series(self, data: np.ndarray, roi_labels: Optional[List[str]] = None):
        """
        Load fMRI time series data.
        
        Args:
            data: Time series data (timepoints x regions/voxels)
            roi_labels: Optional labels for regions of interest
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data (timepoints x regions), got {data.shape}")
        
        self.time_series = data
        self.n_timepoints, self.n_regions = data.shape
        self.roi_labels = roi_labels or [f"Region_{i:03d}" for i in range(self.n_regions)]
        
        print(f"✅ Loaded time series data:")
        print(f"   Shape: {data.shape} (timepoints x regions)")
        print(f"   Duration: {self.n_timepoints / self.sampling_rate:.1f} seconds")
        print(f"   Regions: {self.n_regions}")
    
    def preprocess_time_series(self, 
                             detrend: bool = True,
                             standardize: bool = True,
                             bandpass_filter: Optional[Tuple[float, float]] = (0.01, 0.1),
                             remove_motion: bool = True) -> np.ndarray:
        """
        Advanced preprocessing of fMRI time series.
        
        Args:
            detrend: Remove linear trend from each region
            standardize: Z-score standardization
            bandpass_filter: Frequency range for bandpass filtering (Hz)
            remove_motion: Remove motion-related artifacts
            
        Returns:
            Preprocessed time series data
        """
        if self.time_series is None:
            raise ValueError("No time series data loaded. Call load_time_series() first.")
        
        print("🔧 Preprocessing time series data...")
        processed_data = self.time_series.copy()
        
        # Detrending
        if detrend:
            print("   📈 Removing linear trends...")
            for i in range(self.n_regions):
                processed_data[:, i] = signal.detrend(processed_data[:, i])
        
        # Bandpass filtering
        if bandpass_filter:
            low_freq, high_freq = bandpass_filter
            print(f"   🔊 Bandpass filtering: {low_freq}-{high_freq} Hz...")
            
            # Validate signal length for filtering
            min_required_length = 60  # Minimum length for stable filtering
            if processed_data.shape[0] < min_required_length:
                print(f"   ⚠️ Signal too short ({processed_data.shape[0]} samples) for bandpass filtering, skipping...")
            else:
                nyquist = self.sampling_rate / 2
                low_norm = low_freq / nyquist
                high_norm = high_freq / nyquist
                
                if high_norm >= 1.0:
                    high_norm = 0.99
                    
                # Use lower filter order for shorter signals
                filter_order = min(4, processed_data.shape[0] // 20)  # Adaptive filter order
                filter_order = max(2, filter_order)  # Minimum order of 2
                
                try:
                    b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')
                    
                    for i in range(self.n_regions):
                        # Add padding validation for filtfilt
                        if len(processed_data[:, i]) > 3 * filter_order:
                            processed_data[:, i] = signal.filtfilt(b, a, processed_data[:, i])
                        else:
                            print(f"   ⚠️ Region {i}: Signal too short for filtfilt, using simple filtering")
                            processed_data[:, i] = signal.lfilter(b, a, processed_data[:, i])
                            
                except Exception as e:
                    print(f"   ⚠️ Bandpass filtering failed: {e}, skipping filtering")
        
        # Standardization
        if standardize:
            print("   📊 Standardizing signals...")
            if HAS_SKLEARN:
                scaler = StandardScaler()
                processed_data = scaler.fit_transform(processed_data)
            else:
                # Manual z-score standardization
                processed_data = (processed_data - np.mean(processed_data, axis=0)) / np.std(processed_data, axis=0)
        
        # Motion artifact removal (simple version - global signal regression)
        if remove_motion:
            print("   🚫 Removing motion artifacts...")
            global_signal = np.mean(processed_data, axis=1)
            for i in range(self.n_regions):
                # Regress out global signal
                slope, intercept, _, _, _ = stats.linregress(global_signal, processed_data[:, i])
                processed_data[:, i] -= (slope * global_signal + intercept - np.mean(processed_data[:, i]))
        
        print("✅ Preprocessing completed")
        return processed_data
    
    def compute_connectivity_matrix(self, 
                                   data: Optional[np.ndarray] = None,
                                   method: str = 'pearson',
                                   window_size: Optional[int] = None) -> np.ndarray:
        """
        Compute functional connectivity matrix.
        
        Args:
            data: Time series data (if None, uses loaded data)
            method: Connectivity method ('pearson', 'partial', 'mutual_info')
            window_size: Sliding window size for dynamic connectivity
            
        Returns:
            Connectivity matrix (regions x regions)
        """
        if data is None:
            if self.time_series is None:
                raise ValueError("No data provided and no time series loaded")
            data = self.time_series
        
        print(f"🔗 Computing {method} connectivity matrix...")
        
        if method == 'pearson':
            connectivity = np.corrcoef(data.T)
        elif method == 'partial':
            # Partial correlation (requires inversion)
            try:
                precision_matrix = np.linalg.inv(np.cov(data.T))
                connectivity = -precision_matrix / np.sqrt(np.outer(np.diag(precision_matrix), 
                                                                   np.diag(precision_matrix)))
                np.fill_diagonal(connectivity, 1.0)
            except np.linalg.LinAlgError:
                print("⚠️ Singular matrix, falling back to Pearson correlation")
                connectivity = np.corrcoef(data.T)
        else:
            raise ValueError(f"Unknown connectivity method: {method}")
        
        # Remove NaN values
        connectivity = np.nan_to_num(connectivity)
        
        print(f"✅ Connectivity matrix computed: {connectivity.shape}")
        return connectivity
    
    def dynamic_connectivity_analysis(self, 
                                    data: Optional[np.ndarray] = None,
                                    window_size: int = 50,
                                    step_size: int = 10) -> List[np.ndarray]:
        """
        Perform dynamic connectivity analysis using sliding windows.
        
        Args:
            data: Time series data
            window_size: Size of sliding window (timepoints)
            step_size: Step size for sliding window
            
        Returns:
            List of connectivity matrices for each time window
        """
        if data is None:
            data = self.time_series
        
        print(f"🔄 Computing dynamic connectivity (window={window_size}, step={step_size})...")
        
        connectivity_matrices = []
        n_windows = (len(data) - window_size) // step_size + 1
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_data = data[start_idx:end_idx]
            
            conn_matrix = self.compute_connectivity_matrix(window_data, method='pearson')
            connectivity_matrices.append(conn_matrix)
        
        self.connectivity_matrices = connectivity_matrices
        print(f"✅ Computed {len(connectivity_matrices)} connectivity matrices")
        
        return connectivity_matrices
    
    def detect_network_changes(self, 
                             connectivity_matrices: Optional[List[np.ndarray]] = None,
                             threshold: float = 0.2) -> List[int]:
        """
        Detect significant changes in network connectivity over time.
        
        Args:
            connectivity_matrices: List of connectivity matrices
            threshold: Threshold for change detection
            
        Returns:
            List of change point indices
        """
        if connectivity_matrices is None:
            connectivity_matrices = self.connectivity_matrices
        
        if len(connectivity_matrices) < 2:
            print("⚠️ Need at least 2 connectivity matrices for change detection, returning no changes")
            return []  # Return empty list instead of raising error
        
        print(f"🔍 Detecting network changes (threshold={threshold})...")
        
        # Compute matrix differences
        changes = []
        for i in range(1, len(connectivity_matrices)):
            prev_matrix = connectivity_matrices[i-1]
            curr_matrix = connectivity_matrices[i]
            
            # Frobenius norm of difference
            diff_norm = np.linalg.norm(curr_matrix - prev_matrix, 'fro')
            changes.append(diff_norm)
        
        # Detect change points using threshold
        change_points = []
        changes = np.array(changes)
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        for i, change in enumerate(changes):
            if change > mean_change + threshold * std_change:
                change_points.append(i + 1)  # +1 because we started from index 1
        
        self.change_points = change_points
        print(f"✅ Detected {len(change_points)} significant changes at timepoints: {change_points}")
        
        return change_points
    
    def compute_network_metrics(self, 
                               connectivity_matrix: Optional[np.ndarray] = None,
                               threshold: float = 0.3) -> Dict[str, float]:
        """
        Compute graph-theoretic network metrics.
        
        Args:
            connectivity_matrix: Connectivity matrix
            threshold: Threshold for binarizing connections
            
        Returns:
            Dictionary of network metrics
        """
        if connectivity_matrix is None:
            if len(self.connectivity_matrices) > 0:
                connectivity_matrix = self.connectivity_matrices[0]
            else:
                raise ValueError("No connectivity matrix available")
        
        print(f"📊 Computing network metrics (threshold={threshold})...")
        
        if not HAS_NETWORKX:
            print("⚠️ NetworkX not available - computing basic metrics only")
            # Basic metrics without NetworkX
            binary_matrix = (np.abs(connectivity_matrix) > threshold).astype(int)
            np.fill_diagonal(binary_matrix, 0)
            
            n_nodes = binary_matrix.shape[0]
            n_edges = np.sum(binary_matrix) // 2  # Undirected graph
            density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
            
            metrics = {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': density
            }
            self.network_metrics = metrics
            return metrics
        
        # Threshold and binarize matrix
        binary_matrix = (np.abs(connectivity_matrix) > threshold).astype(int)
        np.fill_diagonal(binary_matrix, 0)  # Remove self-connections
        
        # Create NetworkX graph
        G = nx.from_numpy_array(binary_matrix)
        
        # Compute metrics
        metrics = {}
        
        # Basic metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Clustering and path metrics
        if nx.is_connected(G):
            metrics['avg_clustering'] = nx.average_clustering(G)
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
            metrics['global_efficiency'] = nx.global_efficiency(G)
        else:
            # Handle disconnected graphs
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc)
            metrics['avg_clustering'] = nx.average_clustering(G_largest)
            metrics['avg_path_length'] = nx.average_shortest_path_length(G_largest)
            metrics['global_efficiency'] = nx.global_efficiency(G_largest)
        
        # Centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        metrics['max_degree_centrality'] = max(degree_centrality.values())
        metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
        metrics['max_betweenness_centrality'] = max(betweenness_centrality.values())
        metrics['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
        
        # Small-world properties
        try:
            metrics['small_worldness'] = nx.sigma(G)
        except:
            metrics['small_worldness'] = np.nan
        
        self.network_metrics = metrics
        
        print("✅ Network metrics computed:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.3f}")
        
        return metrics
    
    def identify_network_modules(self, 
                               connectivity_matrix: Optional[np.ndarray] = None,
                               n_clusters: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Identify network modules/communities using clustering.
        
        Args:
            connectivity_matrix: Connectivity matrix
            n_clusters: Number of clusters (if None, determined automatically)
            
        Returns:
            Tuple of (cluster_labels, silhouette_score)
        """
        if connectivity_matrix is None:
            if len(self.connectivity_matrices) > 0:
                connectivity_matrix = self.connectivity_matrices[0]
            else:
                raise ValueError("No connectivity matrix available")
        
        print("🔍 Identifying network modules...")
        
        if not HAS_SKLEARN:
            print("⚠️ scikit-learn not available - using simple threshold-based clustering")
            # Simple threshold-based clustering
            abs_matrix = np.abs(connectivity_matrix)
            threshold = np.percentile(abs_matrix, 75)  # Use 75th percentile as threshold
            strong_connections = abs_matrix > threshold
            
            # Create simple modules based on strong connections
            cluster_labels = np.zeros(self.n_regions, dtype=int)
            current_cluster = 0
            visited = np.zeros(self.n_regions, dtype=bool)
            
            for i in range(self.n_regions):
                if not visited[i]:
                    # Find all regions strongly connected to this one
                    connected_regions = np.where(strong_connections[i])[0]
                    cluster_labels[connected_regions] = current_cluster
                    visited[connected_regions] = True
                    current_cluster += 1
            
            return cluster_labels, 0.0  # No silhouette score available
        
        # Use absolute values and remove diagonal
        abs_matrix = np.abs(connectivity_matrix)
        np.fill_diagonal(abs_matrix, 0)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            silhouette_scores = []
            cluster_range = range(2, min(10, self.n_regions // 2))
            
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(abs_matrix)
                score = silhouette_score(abs_matrix, cluster_labels)
                silhouette_scores.append(score)
            
            optimal_k = cluster_range[np.argmax(silhouette_scores)]
            print(f"   Optimal number of clusters: {optimal_k}")
        else:
            optimal_k = n_clusters
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(abs_matrix)
        score = silhouette_score(abs_matrix, cluster_labels)
        
        print(f"✅ Identified {optimal_k} modules with silhouette score: {score:.3f}")
        
        # Print module information
        for i in range(optimal_k):
            module_regions = np.where(cluster_labels == i)[0]
            print(f"   Module {i}: {len(module_regions)} regions")
        
        return cluster_labels, score
    
    def visualize_connectivity_matrix(self, 
                                    connectivity_matrix: Optional[np.ndarray] = None,
                                    title: str = "Functional Connectivity Matrix",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize connectivity matrix as a heatmap.
        
        Args:
            connectivity_matrix: Connectivity matrix to visualize
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if connectivity_matrix is None:
            if len(self.connectivity_matrices) > 0:
                connectivity_matrix = self.connectivity_matrices[0]
            else:
                raise ValueError("No connectivity matrix available")
        
        print("📊 Creating connectivity matrix visualization...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(connectivity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Brain Regions', fontsize=12)
        ax.set_ylabel('Brain Regions', fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', fontsize=12)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        
        return fig
    
    def generate_analysis_report(self) -> Dict:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Dictionary containing analysis summary
        """
        print("📋 Generating analysis report...")
        
        report = {
            'data_info': {
                'n_timepoints': self.n_timepoints,
                'n_regions': self.n_regions,
                'duration_seconds': self.n_timepoints / self.sampling_rate,
                'sampling_rate_hz': self.sampling_rate
            },
            'connectivity_analysis': {
                'n_dynamic_matrices': len(self.connectivity_matrices),
                'change_points': self.change_points,
                'n_changes_detected': len(self.change_points)
            },
            'network_metrics': self.network_metrics
        }
        
        print("✅ Analysis report generated")
        return report

def main():
    """Test the Neural Network Analyzer with sample data."""
    print("🧠 Testing Neural Network Analyzer")
    print("=" * 60)
    
    # Load sample data using our data loader
    from fmri_data_loader import fMRIDataLoader
    
    loader = fMRIDataLoader()
    try:
        # Get preprocessed sample data
        fmri_data, behavioral_data = loader.get_sample_analysis_data()
        
        # Subsample for faster testing (use first 1000 regions and 500 timepoints)
        sample_data = fmri_data[:1000, :500].T  # Transpose to timepoints x regions
        
        print(f"\n📊 Using sample data: {sample_data.shape}")
        
        # Initialize analyzer
        analyzer = BrainNetworkAnalyzer(sampling_rate=0.4)
        
        # Load and preprocess data
        analyzer.load_time_series(sample_data)
        processed_data = analyzer.preprocess_time_series(
            bandpass_filter=(0.01, 0.08),
            standardize=True
        )
        
        # Compute static connectivity
        static_connectivity = analyzer.compute_connectivity_matrix(processed_data)
        
        # Dynamic connectivity analysis
        dynamic_matrices = analyzer.dynamic_connectivity_analysis(
            processed_data, 
            window_size=30, 
            step_size=10
        )
        
        # Detect changes
        change_points = analyzer.detect_network_changes()
        
        # Compute network metrics
        metrics = analyzer.compute_network_metrics(static_connectivity)
        
        # Identify modules
        modules, silhouette = analyzer.identify_network_modules(static_connectivity)
        
        # Generate report
        report = analyzer.generate_analysis_report()
        
        print(f"\n📋 Analysis Summary:")
        print(f"   Data duration: {report['data_info']['duration_seconds']:.1f} seconds")
        print(f"   Dynamic windows: {report['connectivity_analysis']['n_dynamic_matrices']}")
        print(f"   Changes detected: {report['connectivity_analysis']['n_changes_detected']}")
        print(f"   Network density: {metrics.get('density', 'N/A'):.3f}")
        print(f"   Module silhouette: {silhouette:.3f}")
        
        # Create visualization
        try:
            fig = analyzer.visualize_connectivity_matrix(
                static_connectivity,
                title="Sample fMRI Connectivity Matrix",
                save_path="connectivity_matrix.png"
            )
            plt.show()
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
        
        print("\n🎉 Neural Network Analyzer test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 