#!/usr/bin/env python3
"""
ROI Simulator and 3D Model Generator GUI

This application allows users to:
1. Load fMRI/MRI volumes from training results.
2. Generate high-definition 3D models of brain regions (ROIs).
3. Reconstruct synaptic neural network representations using Blue Brain Atlas data.
4. Visualize the results.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
import time
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import core modules
try:
    from brain_3d_model_generator import Brain3DModelGenerator, AnatomicalFeature
    from brain_3d_visualizer import Brain3DVisualizer
    from blue_brain_atlas_integrator import BlueBrainAtlasIntegrator, CellType
except ImportError as e:
    print(f"Error importing core modules: {e}")

# Import Resonance Neural Network
try:
    from resonance_network import ResonanceNet, ResonanceConfig
    HAS_RESONANCE = True
except ImportError as e:
    print(f"ResonanceNet not available: {e}")
    HAS_RESONANCE = False

class ROISimulatorGUI:
    """GUI for ROI Simulation and 3D Model Generation."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(" ROI Simulator & 3D Model Generator")
        self.root.geometry("1400x900")
        self.root.configure(bg='#121212')  # Dark theme
        
        # State
        self.model_generator = None
        self.visualizer = None
        self.blue_brain_integrator = None
        self.current_volume_path = None
        self.detected_features = []
        self.enhanced_features = []
        self.is_processing = False
        self.resonance_net = None
        self.resonance_results = None
        
        # Initialize components
        self.setup_gui()
        self.initialize_systems()
        
    def initialize_systems(self):
        """Initialize the backend systems."""
        def init_thread():
            try:
                self.log("Initializing Brain 3D Model Generator...")
                self.model_generator = Brain3DModelGenerator(
                    use_gpu=True,
                    enable_blue_brain=True
                )
                self.log(" Model Generator initialized.")
                
                self.log("Initializing Blue Brain Atlas Integrator...")
                self.blue_brain_integrator = BlueBrainAtlasIntegrator(data_source="mock")
                self.log(" Blue Brain Integrator initialized.")
                
                self.log("Initializing 3D Visualizer...")
                self.visualizer = Brain3DVisualizer()
                self.log(" Visualizer initialized.")
                
                self.root.after(0, lambda: self.status_label.config(text="Systems Ready", fg="#00ff00"))
                
            except Exception as e:
                self.log(f" Initialization error: {e}")
                self.root.after(0, lambda: self.status_label.config(text="Initialization Failed", fg="#ff0000"))
        
        threading.Thread(target=init_thread, daemon=True).start()

    def setup_gui(self):
        """Setup the GUI layout."""
        # Main container
        main_container = tk.Frame(self.root, bg='#121212')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#121212')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(header_frame, text="ROI Simulator & 3D Model Generator", 
                font=("Segoe UI", 24, "bold"), bg='#121212', fg='#ffffff').pack(side=tk.LEFT)
        
        self.status_label = tk.Label(header_frame, text="Initializing...", 
                                   font=("Segoe UI", 12), bg='#121212', fg='#aaaaaa')
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # Content Area (Split Pane)
        content_pane = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, bg='#1e1e1e', sashwidth=4)
        content_pane.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel: Controls & Data
        left_panel = tk.Frame(content_pane, bg='#1e1e1e', width=400)
        content_pane.add(left_panel)
        
        # Right Panel: Visualization & Logs
        right_panel = tk.Frame(content_pane, bg='#1e1e1e')
        content_pane.add(right_panel)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        """Setup controls in the left panel."""
        # 1. Data Loading Section
        data_frame = tk.LabelFrame(parent, text="Data Source", bg='#1e1e1e', fg='#ffffff', font=("Segoe UI", 11, "bold"))
        data_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(data_frame, text=" Find fMRI Volumes", command=self.find_fmri_volumes,
                 bg='#2d2d2d', fg='#ffffff', font=("Segoe UI", 10), relief=tk.FLAT, padx=10, pady=5).pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(data_frame, text=" Browse File...", command=self.browse_file,
                 bg='#2d2d2d', fg='#ffffff', font=("Segoe UI", 10), relief=tk.FLAT, padx=10, pady=5).pack(fill=tk.X, padx=10, pady=5)

        self.file_label = tk.Label(data_frame, text="No file selected", bg='#1e1e1e', fg='#aaaaaa', wraplength=350)
        self.file_label.pack(padx=10, pady=5)

        # 2. ROI Generation Section
        roi_frame = tk.LabelFrame(parent, text="ROI Generation", bg='#1e1e1e', fg='#ffffff', font=("Segoe UI", 11, "bold"))
        roi_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.generate_btn = tk.Button(roi_frame, text=" Generate 3D Models & Detect ROIs", command=self.generate_models,
                                     bg='#007acc', fg='#ffffff', font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=10, pady=8, state=tk.DISABLED)
        self.generate_btn.pack(fill=tk.X, padx=10, pady=10)
        
        # Options
        opts_frame = tk.Frame(roi_frame, bg='#1e1e1e')
        opts_frame.pack(fill=tk.X, padx=10)
        
        self.use_blue_brain = tk.BooleanVar(value=True)
        tk.Checkbutton(opts_frame, text="Blue Brain Atlas Integration", variable=self.use_blue_brain,
                      bg='#1e1e1e', fg='#ffffff', selectcolor='#2d2d2d', activebackground='#1e1e1e', activeforeground='#ffffff').pack(anchor='w')
        
        self.use_freq_analysis = tk.BooleanVar(value=True)
        tk.Checkbutton(opts_frame, text="Frequency Analysis", variable=self.use_freq_analysis,
                      bg='#1e1e1e', fg='#ffffff', selectcolor='#2d2d2d', activebackground='#1e1e1e', activeforeground='#ffffff').pack(anchor='w')

        # 3. Network Reconstruction Section
        net_frame = tk.LabelFrame(parent, text="Network Reconstruction", bg='#1e1e1e', fg='#ffffff', font=("Segoe UI", 11, "bold"))
        net_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.visualize_btn = tk.Button(net_frame, text="Reconstruct Synaptic Network", command=self.visualize_network,
                                      bg='#28a745', fg='#ffffff', font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=10, pady=8, state=tk.DISABLED)

        # 4. Resonance Neural Network Section
        if HAS_RESONANCE:
            res_frame = tk.LabelFrame(parent, text="Resonance Network", bg='#1e1e1e', fg='#ffffff', font=("Segoe UI", 11, "bold"))
            res_frame.pack(fill=tk.X, padx=10, pady=10)

            self.resonate_btn = tk.Button(res_frame, text="Resonate on BOLD Data",
                                          command=self.start_resonance,
                                          bg='#9C27B0', fg='#ffffff', font=("Segoe UI", 10, "bold"),
                                          relief=tk.FLAT, padx=10, pady=8, state=tk.DISABLED)
            self.resonate_btn.pack(fill=tk.X, padx=10, pady=5)

            # Epoch slider
            epoch_frame = tk.Frame(res_frame, bg='#1e1e1e')
            epoch_frame.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(epoch_frame, text="Epochs:", bg='#1e1e1e', fg='#aaaaaa', font=("Segoe UI", 9)).pack(side=tk.LEFT)
            self.resonance_epochs = tk.IntVar(value=30)
            tk.Scale(epoch_frame, from_=5, to=200, orient=tk.HORIZONTAL, variable=self.resonance_epochs,
                     bg='#1e1e1e', fg='#ffffff', highlightbackground='#1e1e1e', troughcolor='#2d2d2d',
                     font=("Segoe UI", 8)).pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Neurons slider
            neuron_frame = tk.Frame(res_frame, bg='#1e1e1e')
            neuron_frame.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(neuron_frame, text="Neurons:", bg='#1e1e1e', fg='#aaaaaa', font=("Segoe UI", 9)).pack(side=tk.LEFT)
            self.resonance_neurons = tk.IntVar(value=64)
            tk.Scale(neuron_frame, from_=8, to=256, orient=tk.HORIZONTAL, variable=self.resonance_neurons,
                     bg='#1e1e1e', fg='#ffffff', highlightbackground='#1e1e1e', troughcolor='#2d2d2d',
                     font=("Segoe UI", 8)).pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.resonance_status = tk.Label(res_frame, text="Load a BOLD volume to begin",
                                             bg='#1e1e1e', fg='#aaaaaa', font=("Segoe UI", 9))
            self.resonance_status.pack(pady=5)

        # 5. Cellular View Section
        cell_frame = tk.LabelFrame(parent, text="Cellular View", bg='#1e1e1e', fg='#ffffff', font=("Segoe UI", 11, "bold"))
        cell_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.neuroglancer_btn = tk.Button(cell_frame, text=" Explore H01 Sample (Neuroglancer)", command=self.launch_neuroglancer,
                                         bg='#E91E63', fg='#ffffff', font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=10, pady=8)
        self.neuroglancer_btn.pack(fill=tk.X, padx=10, pady=10)
        
        self.cellular_info = tk.Label(cell_frame, text="View 8nm resolution EM data", bg='#1e1e1e', fg='#aaaaaa', font=("Segoe UI", 9))
        self.cellular_info.pack(pady=2)

    def setup_right_panel(self, parent):
        """Setup visualization and logs in the right panel."""
        # Tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Log Tab
        log_frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(log_frame, text=" Activity Log")
        
        self.log_text = tk.Text(log_frame, bg='#000000', fg='#00ff00', font=("Consolas", 10), state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results Tab
        results_frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(results_frame, text=" Detected ROIs")
        
        # Treeview with scrollbar
        tree_scroll = ttk.Scrollbar(results_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_tree = ttk.Treeview(results_frame, columns=("Name", "Confidence", "Type", "Cells"), show='headings', yscrollcommand=tree_scroll.set)
        tree_scroll.config(command=self.results_tree.yview)
        
        self.results_tree.heading("Name", text="Region Name")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.heading("Type", text="Tissue Type")
        self.results_tree.heading("Cells", text="Est. Cell Count")
        
        self.results_tree.column("Name", width=150)
        self.results_tree.column("Confidence", width=80)
        self.results_tree.column("Type", width=100)
        self.results_tree.column("Cells", width=100)
        
        self.results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Resonance Tab
        if HAS_RESONANCE:
            resonance_frame = tk.Frame(notebook, bg='#0a0a0a')
            notebook.add(resonance_frame, text=" Resonance")

            # Coherence display
            coh_header = tk.Frame(resonance_frame, bg='#0a0a0a')
            coh_header.pack(fill=tk.X, padx=10, pady=10)
            tk.Label(coh_header, text="Phase Coherence", font=("Segoe UI", 14, "bold"),
                     bg='#0a0a0a', fg='#BB86FC').pack(side=tk.LEFT)
            self.coherence_value_label = tk.Label(coh_header, text="--",
                                                   font=("Consolas", 24, "bold"),
                                                   bg='#0a0a0a', fg='#03DAC6')
            self.coherence_value_label.pack(side=tk.RIGHT, padx=20)

            # Resonance log
            self.resonance_log = tk.Text(resonance_frame, bg='#0a0a0a', fg='#BB86FC',
                                         font=("Consolas", 10), state=tk.DISABLED,
                                         height=8)
            self.resonance_log.pack(fill=tk.X, padx=10, pady=5)

            # Network state frame
            state_frame = tk.LabelFrame(resonance_frame, text="Network State",
                                        bg='#0a0a0a', fg='#ffffff',
                                        font=("Segoe UI", 11, "bold"))
            state_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            self.resonance_state_text = tk.Text(state_frame, bg='#0a0a0a', fg='#03DAC6',
                                                font=("Consolas", 9), state=tk.DISABLED)
            self.resonance_state_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def log(self, message):
        """Add message to log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update()

    def find_fmri_volumes(self):
        """Search for fMRI volumes in known directories."""
        self.log("Searching for fMRI volumes...")
        
        # Define search paths based on user's environment
        base_dir = Path.cwd().parent  # Assuming we are in python/
        search_patterns = [
            "**/ds000030/**/*.nii.gz",
            "**/ds000114/**/*.nii.gz",
            "**/*_training_results/**/*.nii.gz",
            "**/*.nii"
        ]
        
        found_files = []
        for pattern in search_patterns:
            try:
                found_files.extend(list(base_dir.glob(pattern)))
            except Exception as e:
                self.log(f"Error searching pattern {pattern}: {e}")
                
        # Filter out duplicates and non-files
        found_files = sorted(list(set([str(f) for f in found_files if f.is_file()])))
        
        if not found_files:
            self.log(" No fMRI volumes found.")
            messagebox.showinfo("No Files Found", "Could not automatically find fMRI volumes.\nPlease browse manually.")
            return
            
        self.log(f" Found {len(found_files)} potential volumes.")
        
        # Show selection dialog
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Volume")
        select_window.geometry("800x400")
        
        lb = tk.Listbox(select_window, width=100, height=20)
        lb.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        for f in found_files:
            lb.insert(tk.END, f)
            
        def on_select():
            selection = lb.curselection()
            if selection:
                file_path = lb.get(selection[0])
                self.load_volume(file_path)
                select_window.destroy()
                
        tk.Button(select_window, text="Load Selected", command=on_select).pack(pady=10)

    def browse_file(self):
        """Browse for a file manually."""
        file_path = filedialog.askopenfilename(
            title="Select fMRI/MRI Volume",
            filetypes=[("NIfTI Files", "*.nii.gz *.nii"), ("All Files", "*.*")]
        )
        if file_path:
            self.load_volume(file_path)

    def load_volume(self, file_path):
        """Load the selected volume."""
        self.current_volume_path = file_path
        self.file_label.config(text=f"Selected: {Path(file_path).name}")
        self.log(f" Selected volume: {file_path}")
        self.generate_btn.config(state=tk.NORMAL)
        if HAS_RESONANCE and hasattr(self, 'resonate_btn'):
            self.resonate_btn.config(state=tk.NORMAL)
            self.resonance_status.config(text="Ready to resonate", fg='#03DAC6')

    def generate_models(self):
        """Run the 3D model generation pipeline."""
        if not self.current_volume_path or not self.model_generator:
            return
            
        self.is_processing = True
        self.generate_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Processing...", fg="#ffff00")
        
        def process_thread():
            try:
                # 1. Load Volume
                self.log("Loading MRI volume...")
                success = self.model_generator.load_mri_volume(self.current_volume_path)
                if not success:
                    raise Exception("Failed to load volume")
                
                # 2. Detect Features
                self.log("Detecting anatomical features and ROIs...")
                self.model_generator.enable_blue_brain = self.use_blue_brain.get()
                self.model_generator.frequency_analysis = self.use_freq_analysis.get()
                
                self.detected_features = self.model_generator.detect_anatomical_features()
                
                self.log(f" Detected {len(self.detected_features)} features.")
                
                # 3. Enhance with Blue Brain Data (Isolate Neuron Types)
                if self.use_blue_brain.get() and self.blue_brain_integrator:
                    self.log(" Isolating neuron types using Blue Brain Atlas...")
                    
                    # Mock fMRI volume for registration enhancement
                    fmri_vol = np.zeros((10, 10, 10)) 
                    
                    enhancement_result = self.blue_brain_integrator.enhance_registration(
                        fmri_vol, 
                        [f.__dict__ for f in self.detected_features]
                    )
                    
                    if enhancement_result['enhanced']:
                        self.enhanced_features = enhancement_result['enhanced_features']
                        self.log(f" Enhanced {len(self.enhanced_features)} regions with cellular data.")
                    else:
                        self.log(" Enhancement failed or no matching regions found.")
                        self.enhanced_features = [f.__dict__ for f in self.detected_features]
                else:
                    self.enhanced_features = [f.__dict__ for f in self.detected_features]
                
                # Update UI
                self.root.after(0, self.update_results_ui)
                
            except Exception as e:
                self.log(f" Error during processing: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.status_label.config(text="Error", fg="#ff0000"))
            finally:
                self.is_processing = False
                self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
                
        threading.Thread(target=process_thread, daemon=True).start()

    def update_results_ui(self):
        """Update the results treeview."""
        self.results_tree.delete(*self.results_tree.get_children())
        
        for feature in self.enhanced_features:
            name = feature.get('name', 'Unknown')
            confidence = feature.get('confidence', 0.0)
            
            # Extract tissue type
            props = feature.get('properties', {})
            tissue_type = props.get('tissue_type', 'Unknown')
            if hasattr(tissue_type, 'name'):
                tissue_type = tissue_type.name
            
            # Extract cell count if available
            cell_counts = feature.get('cellular_composition', {})
            total_cells = sum(cell_counts.values()) if cell_counts else "N/A"
            
            self.results_tree.insert("", tk.END, values=(name, f"{confidence:.2f}", tissue_type, total_cells))
            
        self.visualize_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Processing Complete", fg="#00ff00")

    def visualize_network(self):
        """Reconstruct and visualize the network."""
        if not self.enhanced_features or not self.visualizer:
            return
            
        self.log("Reconstructing synaptic neural network representation...")
        
        try:
            # 1. Construct Connectivity Matrix
            n_features = len(self.enhanced_features)
            connectivity_matrix = np.zeros((n_features, n_features))
            
            coords = []
            labels = []
            
            for f in self.enhanced_features:
                c = f.get('coordinates')
                coords.append(c if c else (0,0,0))
                labels.append(f.get('name', 'Unknown'))
            
            coords = np.array(coords)
            
            # Simple distance-based connectivity
            from scipy.spatial.distance import cdist
            dists = cdist(coords, coords)
            max_dist = np.max(dists) if np.max(dists) > 0 else 1
            connectivity_matrix = 1 - (dists / max_dist)
            
            # Modulate connectivity based on neuron types (Synaptic Reconstruction Simulation)
            for i in range(n_features):
                for j in range(n_features):
                    if i == j: continue
                    
                    # Get cellular composition
                    cells_i = self.enhanced_features[i].get('cellular_composition', {})
                    cells_j = self.enhanced_features[j].get('cellular_composition', {})
                    
                    # Calculate excitatory/inhibitory ratio
                    exc_i = cells_i.get('excitatory_neuron', 0)
                    inh_i = cells_i.get('inhibitory_neuron', 0)
                    ratio_i = exc_i / (exc_i + inh_i) if (exc_i + inh_i) > 0 else 0.5
                    
                    # Adjust connection strength: higher excitatory ratio -> stronger output
                    connectivity_matrix[i, j] *= (0.5 + ratio_i)
            
            # Threshold
            connectivity_matrix[connectivity_matrix < 0.6] = 0
            
            # 2. Update Visualizer
            self.visualizer.brain_coords = coords
            self.visualizer.atlas_labels = labels
            
            # 3. Generate Visualization
            self.log("Generating 3D interactive visualization...")
            fig = self.visualizer.visualize_static_connectivity(
                connectivity_matrix, 
                threshold=0.6,
                title="Reconstructed Synaptic Neural Network"
            )
            
            if fig:
                fig.show()
                self.visualizer.save_visualization(fig, "reconstructed_synaptic_network")
                self.log(" Visualization opened in browser.")
                
        except Exception as e:
            self.log(f" Visualization error: {e}")
            import traceback
            traceback.print_exc()

    def launch_neuroglancer(self):
        """Launch Neuroglancer viewer."""
        self.log("")
        self.log(" Launching Cellular View...")
        
        def launch_thread():
            try:
                # 1. Check for data
                data_path = Path("data/h01_sample")
                if not (data_path / "image" / "info").exists():
                    self.log(" Data not found. Downloading/Generating...")
                    self.root.after(0, lambda: self.cellular_info.config(text="Generating data...", fg="yellow"))
                    
                    # Import download script dynamically
                    import sys
                    sys.path.append(str(Path.cwd() / "python"))
                    from download_h01_sample import download_h01_sample
                    
                    success = download_h01_sample()
                    if not success:
                        raise Exception("Failed to generate data")
                    self.log(" Data ready.")
                
                # 2. Start Server
                self.log(" Starting Neuroglancer server...")
                self.root.after(0, lambda: self.cellular_info.config(text="Starting server...", fg="yellow"))
                
                import sys
                sys.path.append(str(Path.cwd() / "python"))
                from neuroglancer_server import start_neuroglancer_server
                
                url = start_neuroglancer_server(str(data_path))
                
                # 3. Open Browser
                import webbrowser
                webbrowser.open(url)
                
                self.log(f" Neuroglancer running at: {url}")
                self.log("   Browser opened.")
                
                self.root.after(0, lambda: self.cellular_info.config(text="Running", fg="lightgreen"))
                
            except Exception as e:
                self.log(f" Failed to launch Neuroglancer: {e}")
                self.root.after(0, lambda: self.cellular_info.config(text="Error", fg="red"))
        
        threading.Thread(target=launch_thread, daemon=True).start()

    def start_resonance(self):
        """Start resonance process on loaded BOLD data."""
        if not self.current_volume_path or not HAS_RESONANCE:
            return

        self.is_processing = True
        self.resonate_btn.config(state=tk.DISABLED)
        self.resonance_status.config(text="Resonating...", fg='#FFD700')
        self.status_label.config(text="Resonating...", fg='#BB86FC')

        def resonance_thread():
            try:
                import nibabel as nib

                self.log("Loading BOLD volume for resonance...")
                img = nib.load(self.current_volume_path)
                data = img.get_fdata()
                self.log(f"  Volume shape: {data.shape}")

                # Apply mask if 4D
                if data.ndim == 4:
                    # Use variance-based voxel selection
                    flat = data.reshape(-1, data.shape[3])
                    variances = np.var(flat, axis=1)
                    top_n = min(300, len(variances))
                    top_idx = np.argsort(variances)[-top_n:]
                    masked_data = flat[top_idx]
                    n_tp = min(300, masked_data.shape[1])
                    masked_data = masked_data[:, :n_tp]
                else:
                    masked_data = data
                    n_tp = masked_data.shape[-1] if masked_data.ndim > 1 else len(masked_data)

                self.log(f"  Using {masked_data.shape[0]} voxels x {n_tp} timepoints")

                # Configure
                n_neurons = self.resonance_neurons.get()
                n_epochs = self.resonance_epochs.get()

                config = ResonanceConfig(
                    input_dim=n_tp,
                    n_complex_neurons=n_neurons,
                    n_spiking_neurons=n_neurons * 2,
                    n_harmonics=12,
                    n_resonance_epochs=n_epochs,
                    convergence_threshold=0.90,
                    coupling_strength=0.15,
                    learning_rate=0.02,
                    damping=0.02,
                    output_dir=str(Path(__file__).parent.parent / "resonance_results")
                )

                self.log(f"  ResonanceNet: {n_neurons} complex neurons, {n_epochs} epochs")
                self.resonance_net = ResonanceNet(config)

                # Run resonance with live updates
                self.log("  Beginning resonance...")
                time_series = self.resonance_net._extract_voxel_timeseries(masked_data)

                for epoch in range(n_epochs):
                    epoch_coherences = []
                    for i in range(time_series.shape[0]):
                        signal = time_series[i] if time_series.ndim > 1 else time_series
                        if np.std(signal) > 0:
                            signal = (signal - np.mean(signal)) / np.std(signal)
                        metrics = self.resonance_net.resonate_step(signal)
                        epoch_coherences.append(metrics['phase_coherence'])

                    mean_coh = np.mean(epoch_coherences)
                    self.resonance_net.resonance_history.append({
                        'epoch': epoch,
                        'mean_coherence': float(mean_coh),
                        'max_coherence': float(np.max(epoch_coherences)),
                        'min_coherence': float(np.min(epoch_coherences))
                    })

                    # Update UI
                    self.root.after(0, lambda c=mean_coh, e=epoch: self._update_resonance_ui(c, e, n_epochs))

                    if mean_coh >= 0.90:
                        self.log(f"  RESONANCE ACHIEVED at epoch {epoch+1}! Coherence: {mean_coh:.4f}")
                        break

                # Save
                self.resonance_net.save()
                final_coh = mean_coh
                self.log(f"  Resonance complete. Final coherence: {final_coh:.4f}")
                self.root.after(0, lambda: self._finalize_resonance(final_coh))

            except Exception as e:
                self.log(f"  Resonance error: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.resonance_status.config(text="Error", fg='#ff0000'))
            finally:
                self.is_processing = False
                self.root.after(0, lambda: self.resonate_btn.config(state=tk.NORMAL))

        threading.Thread(target=resonance_thread, daemon=True).start()

    def _update_resonance_ui(self, coherence, epoch, total_epochs):
        """Update resonance tab with live metrics."""
        if hasattr(self, 'coherence_value_label'):
            self.coherence_value_label.config(text=f"{coherence:.4f}")

        if hasattr(self, 'resonance_status'):
            pct = int((epoch + 1) / total_epochs * 100)
            self.resonance_status.config(text=f"Epoch {epoch+1}/{total_epochs} ({pct}%)", fg='#FFD700')

        if hasattr(self, 'resonance_log'):
            self.resonance_log.config(state=tk.NORMAL)
            bar_len = int(coherence * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            self.resonance_log.insert(tk.END, f"Epoch {epoch+1:3d}: [{bar}] {coherence:.4f}\n")
            self.resonance_log.see(tk.END)
            self.resonance_log.config(state=tk.DISABLED)

    def _finalize_resonance(self, final_coherence):
        """Finalize resonance UI after completion."""
        if hasattr(self, 'resonance_status'):
            self.resonance_status.config(text=f"Complete - Coherence: {final_coherence:.4f}", fg='#03DAC6')
        self.status_label.config(text="Resonance Complete", fg='#BB86FC')

        # Show network state
        if self.resonance_net and hasattr(self, 'resonance_state_text'):
            state = self.resonance_net._get_network_state()
            self.resonance_state_text.config(state=tk.NORMAL)
            self.resonance_state_text.delete('1.0', tk.END)
            self.resonance_state_text.insert(tk.END, f"Mean weight amplitude:  {state['mean_weight_amplitude']:.4f}\n")
            self.resonance_state_text.insert(tk.END, f"Mean natural frequency: {state['mean_natural_frequency']:.6f} Hz\n")
            self.resonance_state_text.insert(tk.END, f"Harmonic frequencies:   {[f'{f:.4f}' for f in state['harmonic_frequencies']]}\n")
            self.resonance_state_text.insert(tk.END, f"Astrocyte calcium:      {[f'{c:.4f}' for c in state['astrocyte_calcium']]}\n")
            self.resonance_state_text.insert(tk.END, f"Total epochs:           {state['total_resonance_epochs']}\n")
            self.resonance_state_text.config(state=tk.DISABLED)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ROISimulatorGUI()
    app.run()
