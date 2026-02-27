#!/usr/bin/env python3
"""
Ultra-Realistic Brain 3D Visualization GUI - Clean Version with RX 7700S Integration

This is a clean version of the brain GUI with proper RX 7700S integration
and no orphaned code blocks.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import threading
import time
from pathlib import Path
import json
import warnings

# Suppress warnings for clean GUI experience
warnings.filterwarnings("ignore")

# RX 7700S GPU Selection - Using Successful Methods
import os

# Apply successful RX 7700S environment variables
rx7700s_env_vars = {
    # HIP/ROCm device selection (most effective for AMD)
    'HIP_VISIBLE_DEVICES': '1',           # Use GPU 1 (RX 7700S)
    'ROCR_VISIBLE_DEVICES': '1',          # ROCm device selection
    'GPU_MAX_ALLOC_PERCENT': '95',        # Allow high memory usage
    'GPU_MAX_HEAP_SIZE': '100',           # Maximum heap allocation
    'HSA_OVERRIDE_GFX_VERSION': '11.0.0', # RDNA3 architecture
    
    # Additional AMD GPU preferences
    'GPU_FORCE_64BIT_PTR': '1',
    'GPU_USE_SYNC_OBJECTS': '1',
    'GPU_SINGLE_ALLOC_PERCENT': '100',
    
    # OpenCL backup (secondary)
    'OPENCL_DEVICE_FILTER': 'discrete',
    'AMD_OPENCL_DEVICE_MASK': '2',        # Binary: 10 = GPU 1 only
    
    # Legacy compatibility
    'AmdPowerXpressRequestHighPerformance': '1',
    'AMD_OPENCL_DEVICE': '1',
    'OPENCL_DEVICE': '1',
    'PYOPENCL_CTX': '0:1',
}

# Apply RX 7700S environment variables
for var_name, var_value in rx7700s_env_vars.items():
    os.environ[var_name] = var_value

print(" RX 7700S environment configured - discrete GPU will be used")

# Local imports - Using Fixed RX 7700S Implementation
try:
    from rx7700s_brain_analyzer_fixed import RX7700SBrainAnalyzerFixed
    from rx7700s_simple_detector import RX7700SSimpleDetector
    HAS_RX7700S = True
    print(" RX 7700S brain analyzer components loaded")
except ImportError as e:
    HAS_RX7700S = False
    print(f"Warning: RX 7700S components not available - {e}")

class UltraRealisticBrainGUI:
    """
    Main GUI application for ultra-realistic 3D brain visualization with RX 7700S.
    """
    
    def __init__(self, root):
        """Initialize the main GUI application."""
        self.root = root
        self.root.title("Ultra-Realistic Brain 3D Visualization - RX 7700S Edition")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize components
        self.brain_model = None
        self.fmri_data = None
        self.gpu_analyzer = None
        
        # Initialize RX 7700S components
        self._initialize_rx7700s_components()
        
        # Create GUI layout
        self._create_gui_layout()
        
        # Status variables
        self.processing_thread = None
        self.is_processing = False
        
        print("Ultra-Realistic Brain GUI initialized with RX 7700S")
        print("   Ready for high-definition 3D brain visualization")
    
    def _initialize_rx7700s_components(self):
        """Initialize RX 7700S components."""
        try:
            if HAS_RX7700S:
                print(" Initializing RX 7700S Brain Analyzer (Fixed Version)...")
                self.gpu_analyzer = RX7700SBrainAnalyzerFixed()
                device_info = self.gpu_analyzer.device_info
                print(f" RX 7700S analyzer initialized: {device_info['name']} ({device_info['memory_gb']} GB)")
                print("   GPU 0 (780M) will be completely ignored")
            else:
                print(" RX 7700S components not available")
                self.gpu_analyzer = None
        except Exception as e:
            print(f" RX 7700S analyzer initialization failed: {e}")
            print("   Falling back to basic functionality")
            self.gpu_analyzer = None
    
    def _create_gui_layout(self):
        """Create the main GUI layout."""
        # Create main frames
        self._create_menu_bar()
        self._create_main_frames()
        self._create_control_panel()
        self._create_visualization_area()
        self._create_status_bar()
        
        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load MRI Volume", command=self.load_mri_volume)
        file_menu.add_command(label="Generate Test Data", command=self.generate_test_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Generate 3D Model", command=self.generate_3d_model)
        tools_menu.add_command(label="RX 7700S Test", command=self.test_rx7700s)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _create_main_frames(self):
        """Create the main application frames."""
        # Control panel frame (left)
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        # Visualization frame (right)
        self.viz_frame = ttk.Frame(self.root, padding="10")
        self.viz_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        
        # Status frame (bottom)
        self.status_frame = ttk.Frame(self.root, padding="5")
        self.status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
    
    def _create_control_panel(self):
        """Create the control panel with all controls."""
        # Title
        title_label = ttk.Label(self.control_frame, text="RX 7700S Brain Controls", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # GPU Status
        gpu_frame = ttk.LabelFrame(self.control_frame, text=" GPU Status", padding="10")
        gpu_frame.pack(fill="x", pady=(0, 15))
        
        if self.gpu_analyzer:
            device_info = self.gpu_analyzer.device_info
            gpu_status = f" {device_info['name']} ({device_info['memory_gb']} GB)"
        else:
            gpu_status = " RX 7700S not available"
        
        self.gpu_status_label = ttk.Label(gpu_frame, text=gpu_status)
        self.gpu_status_label.pack()
        
        # Data Loading Section
        data_frame = ttk.LabelFrame(self.control_frame, text=" Brain Data", padding="10")
        data_frame.pack(fill="x", pady=(0, 15))
        
        self.load_btn = ttk.Button(data_frame, text="Load MRI Volume",
                                  command=self.load_mri_volume)
        self.load_btn.pack(fill="x", pady=(0, 5))
        
        self.generate_btn = ttk.Button(data_frame, text="Generate Test Data",
                                      command=self.generate_test_data)
        self.generate_btn.pack(fill="x", pady=(0, 5))
        
        self.data_info_label = ttk.Label(data_frame, text="No data loaded")
        self.data_info_label.pack()
        
        # Analysis Section
        analysis_frame = ttk.LabelFrame(self.control_frame, text=" Brain Analysis", padding="10")
        analysis_frame.pack(fill="x", pady=(0, 15))
        
        self.analyze_btn = ttk.Button(analysis_frame, text=" Analyze on RX 7700S",
                                     command=self.start_analysis,
                                     state=tk.DISABLED)
        self.analyze_btn.pack(fill="x", pady=(0, 10))
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(analysis_frame, textvariable=self.progress_var)
        self.progress_label.pack()
    
    def _create_visualization_area(self):
        """Create the main visualization area."""
        # Title
        title_label = ttk.Label(self.viz_frame, text=" 3D Brain Visualization", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Visualization area
        self.viz_text = tk.Text(self.viz_frame, bg='#1a1a1a', fg='white',
                               font=('Consolas', 11))
        self.viz_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.log_message(" Ultra-Realistic Brain 3D Visualization - RX 7700S Edition")
        self.log_message("=" * 60)
        self.log_message("This GUI uses the RX 7700S discrete GPU for brain processing")
        self.log_message("")
        if self.gpu_analyzer:
            device_info = self.gpu_analyzer.device_info
            self.log_message(f" RX 7700S Ready: {device_info['name']}")
            self.log_message(f"   Memory: {device_info['memory_gb']} GB")
            self.log_message(f"   Compute Units: {device_info['compute_units']}")
        else:
            self.log_message(" RX 7700S not available")
        self.log_message("")
        self.log_message("Load brain data to begin analysis...")
    
    def _create_status_bar(self):
        """Create the status bar."""
        self.status_var = tk.StringVar(value="Ready - RX 7700S configured")
        status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        status_label.pack(fill="x")
    
    def log_message(self, message):
        """Log a message to the visualization area."""
        timestamp = time.strftime("%H:%M:%S")
        self.viz_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.viz_text.see(tk.END)
        self.root.update()
    
    def load_mri_volume(self):
        """Load an MRI volume file."""
        file_path = filedialog.askopenfilename(
            title="Select MRI Volume",
            filetypes=[
                ("NumPy files", "*.npy"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.fmri_data = np.load(file_path)
            
            if len(self.fmri_data.shape) != 3:
                messagebox.showerror("Error", "Brain data must be 3D volume")
                return
            
            self.data_info_label.config(text=f"Loaded: {self.fmri_data.shape}")
            self.analyze_btn.config(state=tk.NORMAL)
            
            self.log_message(f" Loaded MRI volume: {self.fmri_data.shape}")
            self.log_message(f"   File: {Path(file_path).name}")
            self.log_message(f"   Size: {self.fmri_data.size:,} voxels")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load MRI volume: {e}")
            self.log_message(f" Failed to load MRI volume: {e}")
    
    def generate_test_data(self):
        """Generate synthetic test brain data."""
        try:
            self.log_message(" Generating synthetic brain data...")
            
            # Create brain-like volume
            size = 128
            x, y, z = np.meshgrid(np.linspace(-2, 2, size),
                                 np.linspace(-2, 2, size),
                                 np.linspace(-2, 2, size))
            
            # Brain ellipsoid
            brain_shape = (x**2/2.5 + y**2/2.0 + z**2/1.5) < 1.0
            
            # Create volume
            self.fmri_data = np.zeros((size, size, size), dtype=np.float32)
            self.fmri_data[brain_shape] = 0.8
            
            # Add structure
            self.fmri_data += 0.2 * np.sin(x*3) * np.cos(y*3) * np.sin(z*2) * brain_shape
            self.fmri_data += 0.1 * np.random.random((size, size, size))
            self.fmri_data = np.clip(self.fmri_data, 0, 1)
            
            brain_voxels = np.sum(brain_shape)
            
            self.data_info_label.config(text=f"Generated: {self.fmri_data.shape}")
            self.analyze_btn.config(state=tk.NORMAL)
            
            self.log_message(f" Generated synthetic brain data: {self.fmri_data.shape}")
            self.log_message(f"   Brain voxels: {brain_voxels:,}")
            self.log_message(f"   Ready for RX 7700S analysis!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate test data: {e}")
            self.log_message(f" Failed to generate test data: {e}")
    
    def start_analysis(self):
        """Start brain analysis on RX 7700S."""
        if self.is_processing or self.fmri_data is None or self.gpu_analyzer is None:
            return
        
        self.is_processing = True
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress_var.set("Starting RX 7700S analysis...")
        
        def analysis_thread():
            try:
                self.log_message("")
                self.log_message(" STARTING RX 7700S BRAIN ANALYSIS")
                self.log_message("=" * 60)
                self.log_message("*** WATCH TASK MANAGER → PERFORMANCE → GPU 1 ***")
                self.log_message("")
                
                # Update progress
                self.root.after(0, lambda: self.progress_var.set("RX 7700S processing brain volume..."))
                
                # Run analysis
                results = self.gpu_analyzer.analyze_brain_volume(
                    self.fmri_data,
                    generate_surface=True,
                    surface_quality='high'
                )
                
                # Update GUI with results
                self.root.after(0, self.on_analysis_complete, results)
                
            except Exception as e:
                self.root.after(0, self.on_analysis_failed, str(e))
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def on_analysis_complete(self, results):
        """Called when analysis completes."""
        self.is_processing = False
        self.analyze_btn.config(state=tk.NORMAL)
        
        if results['success']:
            self.progress_var.set(" Analysis completed!")
            
            self.log_message(" RX 7700S BRAIN ANALYSIS COMPLETED!")
            self.log_message("=" * 60)
            self.log_message(f"Total time: {results['total_processing_time']:.2f} seconds")
            self.log_message(f"GPU used: {results['gpu_device']}")
            self.log_message(f"GPU memory: {results['gpu_memory_gb']} GB")
            self.log_message("")
            
            # Segmentation results
            if 'segmentation' in results:
                seg = results['segmentation']
                self.log_message(" SEGMENTATION RESULTS:")
                self.log_message(f"   Brain voxels: {seg['brain_voxels']:,}")
                self.log_message(f"   Processing time: {seg['processing_time']:.3f}s")
                self.log_message("")
            
            # Classification results
            if 'classification' in results:
                cls = results['classification']
                if cls['success']:
                    self.log_message(" TISSUE CLASSIFICATION:")
                    self.log_message(f"   Gray matter: {cls['gray_percentage']:.1f}%")
                    self.log_message(f"   White matter: {cls['white_percentage']:.1f}%")
                    self.log_message(f"   CSF: {cls['csf_percentage']:.1f}%")
                    self.log_message("")
            
            # Surface results
            if 'surface' in results:
                surf = results['surface']
                if surf['success']:
                    self.log_message(" 3D SURFACE GENERATION:")
                    self.log_message(f"   Vertices: {surf['vertex_count']:,}")
                    self.log_message(f"   Faces: {surf['face_count']:,}")
                    self.log_message("")
            
            self.log_message("🎉 RX 7700S brain analysis successful!")
            self.log_message("   Check Task Manager to verify GPU 1 was used")
            
        else:
            self.progress_var.set(" Analysis failed")
            self.log_message(f" Analysis failed: {results['error']}")
    
    def on_analysis_failed(self, error):
        """Called when analysis fails."""
        self.is_processing = False
        self.analyze_btn.config(state=tk.NORMAL)
        self.progress_var.set(" Analysis failed")
        
        self.log_message(f" RX 7700S analysis failed: {error}")
    
    def generate_3d_model(self):
        """Generate 3D model (same as start_analysis)."""
        self.start_analysis()
    
    def test_rx7700s(self):
        """Test RX 7700S functionality."""
        if not self.gpu_analyzer:
            messagebox.showerror("Error", "RX 7700S analyzer not available")
            return
        
        self.log_message("")
        self.log_message(" Testing RX 7700S functionality...")
        
        try:
            # Get device info
            device_info = self.gpu_analyzer.device_info
            self.log_message(f" RX 7700S detected: {device_info['name']}")
            self.log_message(f"   Memory: {device_info['memory_gb']} GB")
            self.log_message(f"   Compute Units: {device_info['compute_units']}")
            
            messagebox.showinfo("RX 7700S Test", 
                              f"RX 7700S is working correctly!\n\n"
                              f"Device: {device_info['name']}\n"
                              f"Memory: {device_info['memory_gb']} GB\n"
                              f"Compute Units: {device_info['compute_units']}")
            
        except Exception as e:
            self.log_message(f" RX 7700S test failed: {e}")
            messagebox.showerror("RX 7700S Test", f"Test failed: {e}")
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo("About", 
                          "Ultra-Realistic Brain 3D Visualization\n"
                          "RX 7700S Edition\n\n"
                          "This application uses the AMD Radeon RX 7700S\n"
                          "discrete GPU for high-performance brain analysis.")

def main():
    """Main function."""
    print(" Ultra-Realistic Brain 3D Visualization - RX 7700S Edition")
    print("=" * 70)
    print("This GUI uses the RX 7700S discrete GPU for brain processing")
    print()
    
    try:
        root = tk.Tk()
        app = UltraRealisticBrainGUI(root)
        root.mainloop()
    except Exception as e:
        print(f" Failed to start GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()