#!/usr/bin/env python3
"""
RX 7700S Brain GUI - Fixed Version

This GUI uses the successful GPU selection methods to ensure
the RX 7700S discrete GPU is used for all brain processing.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
import time
from pathlib import Path

# Import our fixed analyzer
from rx7700s_brain_analyzer_fixed import RX7700SBrainAnalyzerFixed

class RX7700SBrainGUIFixed:
    """Fixed Brain GUI that reliably uses RX 7700S GPU."""
    
    def __init__(self):
        """Initialize the fixed brain GUI."""
        self.root = tk.Tk()
        self.root.title("🧠 RX 7700S Brain Analyzer - Fixed Version")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        
        # Initialize analyzer
        self.analyzer = None
        self.brain_data = None
        self.analysis_results = None
        self.is_processing = False
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize analyzer in background
        self.initialize_analyzer()
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a1a')
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title_label = tk.Label(title_frame, 
                              text="🧠 RX 7700S Brain Analyzer - Fixed Version",
                              bg='#1a1a1a', fg='#00ff88',
                              font=('Arial', 18, 'bold'))
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="Using successful GPU selection methods for reliable RX 7700S usage",
                                 bg='#1a1a1a', fg='#88ff88',
                                 font=('Arial', 12))
        subtitle_label.pack()
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#2a2a2a')
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame,
                                    text="Initializing RX 7700S GPU...",
                                    bg='#2a2a2a', fg='yellow',
                                    font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=10)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#2a2a2a', width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg='#2a2a2a')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_control_panel(left_panel)
        self.setup_results_panel(right_panel)
    
    def setup_control_panel(self, parent):
        """Setup the control panel."""
        # GPU Info Section
        gpu_frame = tk.LabelFrame(parent, text="🎯 GPU Information", 
                                 bg='#2a2a2a', fg='white',
                                 font=('Arial', 12, 'bold'))
        gpu_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.gpu_info_text = tk.Text(gpu_frame, height=6, bg='#1a1a1a', fg='#00ff88',
                                    font=('Consolas', 10))
        self.gpu_info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Data Loading Section
        data_frame = tk.LabelFrame(parent, text="📁 Brain Data", 
                                  bg='#2a2a2a', fg='white',
                                  font=('Arial', 12, 'bold'))
        data_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.load_btn = tk.Button(data_frame, text="Load Brain Volume",
                                 command=self.load_brain_data,
                                 bg='#4CAF50', fg='white',
                                 font=('Arial', 11, 'bold'),
                                 height=2)
        self.load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.generate_btn = tk.Button(data_frame, text="Generate Test Data",
                                     command=self.generate_test_data,
                                     bg='#2196F3', fg='white',
                                     font=('Arial', 11, 'bold'),
                                     height=2)
        self.generate_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.data_info_label = tk.Label(data_frame, text="No data loaded",
                                       bg='#2a2a2a', fg='gray',
                                       font=('Arial', 10))
        self.data_info_label.pack(pady=5)
        
        # Analysis Section
        analysis_frame = tk.LabelFrame(parent, text="🧠 Brain Analysis", 
                                      bg='#2a2a2a', fg='white',
                                      font=('Arial', 12, 'bold'))
        analysis_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Surface quality
        tk.Label(analysis_frame, text="Surface Quality:",
                bg='#2a2a2a', fg='white').pack(anchor='w', padx=5)
        
        self.quality_var = tk.StringVar(value='medium')
        quality_combo = ttk.Combobox(analysis_frame, textvariable=self.quality_var,
                                    values=['low', 'medium', 'high', 'ultra'])
        quality_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # Generate surface checkbox
        self.generate_surface_var = tk.BooleanVar(value=True)
        surface_check = tk.Checkbutton(analysis_frame, text="Generate 3D Surface",
                                      variable=self.generate_surface_var,
                                      bg='#2a2a2a', fg='white',
                                      selectcolor='#1a1a1a')
        surface_check.pack(anchor='w', padx=5, pady=5)
        
        # Analyze button
        self.analyze_btn = tk.Button(analysis_frame, text="🚀 Analyze on RX 7700S",
                                    command=self.start_analysis,
                                    bg='#ff6b35', fg='white',
                                    font=('Arial', 12, 'bold'),
                                    height=2, state=tk.DISABLED)
        self.analyze_btn.pack(fill=tk.X, padx=5, pady=10)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = tk.Label(analysis_frame, textvariable=self.progress_var,
                                      bg='#2a2a2a', fg='lightgreen',
                                      font=('Arial', 10))
        self.progress_label.pack(pady=5)
        
        # Monitor GPU button
        monitor_btn = tk.Button(parent, text="📊 Monitor GPU Usage",
                               command=self.open_task_manager,
                               bg='#9C27B0', fg='white',
                               font=('Arial', 11, 'bold'))
        monitor_btn.pack(fill=tk.X, padx=10, pady=10)
    
    def setup_results_panel(self, parent):
        """Setup the results panel."""
        # Results title
        results_title = tk.Label(parent, text="📊 Analysis Results",
                                bg='#2a2a2a', fg='white',
                                font=('Arial', 14, 'bold'))
        results_title.pack(pady=10)
        
        # Results text area
        self.results_text = tk.Text(parent, bg='#1a1a1a', fg='white',
                                   font=('Consolas', 11))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
        # Initial message
        self.log_result("🧠 RX 7700S Brain Analyzer - Fixed Version")
        self.log_result("=" * 50)
        self.log_result("This version uses successful GPU selection methods:")
        self.log_result("• HIP/ROCm environment variables")
        self.log_result("• Simple memory-based GPU detection")
        self.log_result("• Direct RX 7700S targeting")
        self.log_result("• Windows Graphics Settings integration")
        self.log_result("")
        self.log_result("Initializing RX 7700S GPU...")
    
    def log_result(self, message):
        """Log a message to the results panel."""
        timestamp = time.strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def initialize_analyzer(self):
        """Initialize the RX 7700S analyzer in background."""
        def init_thread():
            try:
                self.log_result("Configuring RX 7700S environment...")
                self.analyzer = RX7700SBrainAnalyzerFixed()
                
                # Get GPU info
                device_info = self.analyzer.device_info
                
                # Update GUI
                self.root.after(0, self.on_analyzer_initialized, device_info)
                
            except Exception as e:
                self.root.after(0, self.on_analyzer_failed, str(e))
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def on_analyzer_initialized(self, device_info):
        """Called when analyzer is successfully initialized."""
        self.status_label.config(text=f"✅ RX 7700S Ready: {device_info['name']}", fg='lightgreen')
        
        # Update GPU info panel
        self.gpu_info_text.delete(1.0, tk.END)
        self.gpu_info_text.insert(tk.END, f"Device: {device_info['name']}\n")
        self.gpu_info_text.insert(tk.END, f"Memory: {device_info['memory_gb']} GB\n")
        self.gpu_info_text.insert(tk.END, f"Compute Units: {device_info['compute_units']}\n")
        self.gpu_info_text.insert(tk.END, f"Max Frequency: {device_info['max_clock_frequency']} MHz\n")
        self.gpu_info_text.insert(tk.END, f"Driver: {device_info['driver_version']}\n")
        self.gpu_info_text.insert(tk.END, f"OpenCL: {device_info['opencl_version']}")
        
        # Enable controls
        self.load_btn.config(state=tk.NORMAL)
        self.generate_btn.config(state=tk.NORMAL)
        
        self.log_result("✅ RX 7700S analyzer initialized successfully!")
        self.log_result(f"   Device: {device_info['name']}")
        self.log_result(f"   Memory: {device_info['memory_gb']} GB")
        self.log_result(f"   Compute Units: {device_info['compute_units']}")
        self.log_result("")
        self.log_result("Ready for brain analysis!")
    
    def on_analyzer_failed(self, error):
        """Called when analyzer initialization fails."""
        self.status_label.config(text=f"❌ RX 7700S initialization failed", fg='red')
        
        self.gpu_info_text.delete(1.0, tk.END)
        self.gpu_info_text.insert(tk.END, f"ERROR: {error}\n\n")
        self.gpu_info_text.insert(tk.END, "Troubleshooting:\n")
        self.gpu_info_text.insert(tk.END, "1. Update AMD drivers\n")
        self.gpu_info_text.insert(tk.END, "2. Restart computer\n")
        self.gpu_info_text.insert(tk.END, "3. Check Device Manager")
        
        self.log_result(f"❌ RX 7700S initialization failed: {error}")
        self.log_result("")
        self.log_result("Troubleshooting steps:")
        self.log_result("1. Update AMD drivers to latest version")
        self.log_result("2. Restart your computer")
        self.log_result("3. Check Device Manager for GPU errors")
        self.log_result("4. Verify RX 7700S is enabled in BIOS")
    
    def load_brain_data(self):
        """Load brain data from file."""
        file_path = filedialog.askopenfilename(
            title="Select Brain Volume",
            filetypes=[
                ("NumPy files", "*.npy"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.brain_data = np.load(file_path)
            
            if len(self.brain_data.shape) != 3:
                messagebox.showerror("Error", "Brain data must be 3D volume")
                return
            
            self.data_info_label.config(text=f"Loaded: {self.brain_data.shape}")
            self.analyze_btn.config(state=tk.NORMAL)
            
            self.log_result(f"📁 Loaded brain data: {self.brain_data.shape}")
            self.log_result(f"   File: {Path(file_path).name}")
            self.log_result(f"   Size: {self.brain_data.size:,} voxels")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load brain data: {e}")
            self.log_result(f"❌ Failed to load brain data: {e}")
    
    def generate_test_data(self):
        """Generate synthetic test brain data."""
        try:
            self.log_result("🧪 Generating synthetic brain data...")
            
            # Create brain-like volume
            size = 128
            x, y, z = np.meshgrid(np.linspace(-2, 2, size),
                                 np.linspace(-2, 2, size),
                                 np.linspace(-2, 2, size))
            
            # Brain ellipsoid
            brain_shape = (x**2/2.5 + y**2/2.0 + z**2/1.5) < 1.0
            
            # Create volume
            self.brain_data = np.zeros((size, size, size), dtype=np.float32)
            self.brain_data[brain_shape] = 0.8
            
            # Add structure
            self.brain_data += 0.2 * np.sin(x*3) * np.cos(y*3) * np.sin(z*2) * brain_shape
            self.brain_data += 0.1 * np.random.random((size, size, size))
            self.brain_data = np.clip(self.brain_data, 0, 1)
            
            brain_voxels = np.sum(brain_shape)
            
            self.data_info_label.config(text=f"Generated: {self.brain_data.shape}")
            self.analyze_btn.config(state=tk.NORMAL)
            
            self.log_result(f"✅ Generated synthetic brain data: {self.brain_data.shape}")
            self.log_result(f"   Brain voxels: {brain_voxels:,}")
            self.log_result(f"   Ready for RX 7700S analysis!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate test data: {e}")
            self.log_result(f"❌ Failed to generate test data: {e}")
    
    def start_analysis(self):
        """Start brain analysis on RX 7700S."""
        if self.is_processing or self.brain_data is None or self.analyzer is None:
            return
        
        self.is_processing = True
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress_var.set("Starting RX 7700S analysis...")
        
        def analysis_thread():
            try:
                self.log_result("")
                self.log_result("🚀 STARTING RX 7700S BRAIN ANALYSIS")
                self.log_result("=" * 50)
                self.log_result("*** WATCH TASK MANAGER → PERFORMANCE → GPU 1 ***")
                self.log_result("")
                
                # Update progress
                self.root.after(0, lambda: self.progress_var.set("RX 7700S processing brain volume..."))
                
                # Run analysis
                results = self.analyzer.analyze_brain_volume(
                    self.brain_data,
                    generate_surface=self.generate_surface_var.get(),
                    surface_quality=self.quality_var.get()
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
        self.analysis_results = results
        
        if results['success']:
            self.progress_var.set("✅ Analysis completed!")
            
            self.log_result("✅ RX 7700S BRAIN ANALYSIS COMPLETED!")
            self.log_result("=" * 50)
            self.log_result(f"Total time: {results['total_processing_time']:.2f} seconds")
            self.log_result(f"GPU used: {results['gpu_device']}")
            self.log_result(f"GPU memory: {results['gpu_memory_gb']} GB")
            self.log_result("")
            
            # Segmentation results
            if 'segmentation' in results:
                seg = results['segmentation']
                self.log_result("🔍 SEGMENTATION RESULTS:")
                self.log_result(f"   Brain voxels: {seg['brain_voxels']:,}")
                self.log_result(f"   Processing time: {seg['processing_time']:.3f}s")
                self.log_result("")
            
            # Classification results
            if 'classification' in results:
                cls = results['classification']
                if cls['success']:
                    self.log_result("🧬 TISSUE CLASSIFICATION:")
                    self.log_result(f"   Gray matter: {cls['gray_percentage']:.1f}%")
                    self.log_result(f"   White matter: {cls['white_percentage']:.1f}%")
                    self.log_result(f"   CSF: {cls['csf_percentage']:.1f}%")
                    self.log_result(f"   Total volume: {cls['total_brain_volume_mm3']:.0f} mm³")
                    self.log_result("")
            
            # Surface results
            if 'surface' in results:
                surf = results['surface']
                if surf['success']:
                    self.log_result("🎨 3D SURFACE GENERATION:")
                    self.log_result(f"   Vertices: {surf['vertex_count']:,}")
                    self.log_result(f"   Faces: {surf['face_count']:,}")
                    self.log_result(f"   Quality: {surf['quality']}")
                    self.log_result("")
            
            # Metrics
            if 'metrics' in results:
                met = results['metrics']
                if met['success']:
                    self.log_result("📊 BRAIN METRICS:")
                    self.log_result(f"   Mean intensity: {met['mean_intensity']:.3f}")
                    self.log_result(f"   Brain volume: {met['brain_volume_percentage']:.1f}%")
                    self.log_result(f"   Intensity range: {met['min_intensity']:.3f} - {met['max_intensity']:.3f}")
                    self.log_result("")
            
            self.log_result("🎉 RX 7700S brain analysis successful!")
            self.log_result("   Check Task Manager to verify GPU 1 was used")
            
        else:
            self.progress_var.set("❌ Analysis failed")
            self.log_result(f"❌ Analysis failed: {results['error']}")
    
    def on_analysis_failed(self, error):
        """Called when analysis fails."""
        self.is_processing = False
        self.analyze_btn.config(state=tk.NORMAL)
        self.progress_var.set("❌ Analysis failed")
        
        self.log_result(f"❌ RX 7700S analysis failed: {error}")
    
    def open_task_manager(self):
        """Open Task Manager to monitor GPU usage."""
        try:
            import subprocess
            subprocess.Popen(['taskmgr'])
            self.log_result("📊 Task Manager opened - check Performance → GPU 1")
        except Exception as e:
            self.log_result(f"❌ Failed to open Task Manager: {e}")
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()

def main():
    """Main function."""
    print("🧠 RX 7700S Brain GUI - Fixed Version")
    print("=" * 50)
    print("This GUI uses successful GPU selection methods")
    print("to ensure RX 7700S is used for brain processing.")
    print()
    
    try:
        app = RX7700SBrainGUIFixed()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()