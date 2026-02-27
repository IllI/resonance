#!/usr/bin/env python3
"""
GPU-Accelerated Brain GUI
Uses actual RX 7700S GPU processing for brain visualization
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import time
import threading

# Force RX 7700S environment
os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
os.environ['AMD_OPENCL_DEVICE'] = '1'  # Device 1 = gfx1103 = RX 7700S
os.environ['OPENCL_DEVICE'] = '1'
os.environ['PYOPENCL_CTX'] = '0:1'

class GPUBrainProcessor:
    def __init__(self):
        self.context = None
        self.queue = None
        self.device = None
        self.initialized = False
        
    def initialize_rx7700s(self):
        """Initialize RX 7700S for GPU processing"""
        try:
            import pyopencl as cl
            
            # Get AMD platform
            platforms = cl.get_platforms()
            amd_platform = None
            
            for platform in platforms:
                if 'AMD' in platform.name:
                    amd_platform = platform
                    break
            
            if not amd_platform:
                return False, "No AMD platform found"
            
            # Get RX 7700S device (gfx1103)
            devices = amd_platform.get_devices()
            rx7700s_device = None
            
            for device in devices:
                if 'gfx1103' in device.name or device.global_mem_size >= 12 * 1024**3:
                    rx7700s_device = device
                    break
            
            if not rx7700s_device and len(devices) > 1:
                rx7700s_device = devices[1]  # Fallback to device 1
            
            if not rx7700s_device:
                return False, "RX 7700S not found"
            
            # Create context and queue
            self.context = cl.Context([rx7700s_device])
            self.queue = cl.CommandQueue(self.context)
            self.device = rx7700s_device
            self.initialized = True
            
            return True, f"RX 7700S initialized: {rx7700s_device.name}"
            
        except ImportError:
            return False, "PyOpenCL not available"
        except Exception as e:
            return False, f"GPU initialization failed: {e}"
    
    def process_brain_data_gpu(self, brain_data, progress_callback=None):
        """Process brain data using RX 7700S GPU"""
        if not self.initialized:
            return None, "GPU not initialized"
        
        try:
            import pyopencl as cl
            
            print(f" PROCESSING BRAIN DATA ON RX 7700S ")
            print(f"*** WATCH TASK MANAGER - GPU 0 SHOULD ACTIVATE ***")
            
            # Flatten brain data
            flat_data = brain_data.flatten().astype(np.float32)
            size = len(flat_data)
            
            print(f"Processing {size:,} brain voxels on GPU...")
            
            # Create buffers
            input_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_data)
            output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, flat_data.nbytes)
            
            # Brain processing kernel
            kernel_code = """
            __kernel void process_brain_voxels(__global const float* input,
                                             __global float* output) {
                int gid = get_global_id(0);
                
                // Heavy brain processing computation
                float voxel = input[gid];
                float result = 0.0f;
                
                // Simulate complex brain analysis
                for (int i = 0; i < 1000; i++) {
                    float temp = sin(voxel * i * 0.01f) * cos(voxel * i * 0.02f);
                    result += sqrt(fabs(temp));
                    result = fmod(result + tan(temp * 0.1f), 10.0f);
                }
                
                // Apply brain-specific transformations
                result = tanh(result) * (1.0f + voxel);
                output[gid] = result;
            }
            """
            
            program = cl.Program(self.context, kernel_code).build()
            kernel = program.process_brain_voxels
            
            # Process in chunks to show progress
            chunk_size = size // 10  # 10 chunks
            processed_data = np.zeros_like(flat_data)
            
            for i in range(10):
                if progress_callback:
                    progress_callback(f"GPU processing chunk {i+1}/10...", (i+1) * 10)
                
                start_idx = i * chunk_size
                end_idx = min((i+1) * chunk_size, size)
                chunk_size_actual = end_idx - start_idx
                
                if chunk_size_actual > 0:
                    # Execute kernel on chunk
                    kernel(self.queue, (chunk_size_actual,), None, input_buf, output_buf)
                    self.queue.finish()
                    
                    # Read chunk result
                    chunk_result = np.empty(chunk_size_actual, dtype=np.float32)
                    cl.enqueue_copy(self.queue, chunk_result, output_buf, 
                                  device_offset=start_idx * 4, size=chunk_size_actual * 4)
                    
                    processed_data[start_idx:end_idx] = chunk_result
                
                time.sleep(0.2)  # Brief pause to see GPU usage
            
            # Reshape back to original shape
            result = processed_data.reshape(brain_data.shape)
            
            print(f" GPU brain processing completed!")
            print(f"   Device: {self.device.name}")
            print(f"   Voxels processed: {size:,}")
            
            return result, "GPU processing successful"
            
        except Exception as e:
            return None, f"GPU processing failed: {e}"

class GPUBrainGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPU-Accelerated Brain Visualization - RX 7700S")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a1a')
        
        self.gpu_processor = GPUBrainProcessor()
        self.brain_data = None
        self.processed_data = None
        self.is_processing = False
        
        self.setup_gui()
        self.initialize_gpu()
        
    def setup_gui(self):
        """Setup GUI layout"""
        # Title
        title_label = tk.Label(self.root, text=" GPU-Accelerated Brain Visualization ",
                              bg='#1a1a1a', fg='#ff6b35',
                              font=('Arial', 18, 'bold'))
        title_label.pack(pady=10)
        
        # GPU Status
        self.gpu_status_label = tk.Label(self.root, text="Initializing RX 7700S GPU...",
                                        bg='#1a1a1a', fg='yellow',
                                        font=('Arial', 12))
        self.gpu_status_label.pack(pady=5)
        
        # Control Frame
        control_frame = tk.Frame(self.root, bg='#2a2a2a')
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Generate Brain Data button
        self.generate_btn = tk.Button(control_frame, text=" Generate Brain Data",
                                     command=self.generate_brain_data,
                                     bg='#4CAF50', fg='white',
                                     font=('Arial', 12, 'bold'),
                                     height=2)
        self.generate_btn.pack(side=tk.LEFT, padx=10)
        
        # Process on GPU button
        self.process_btn = tk.Button(control_frame, text=" Process on RX 7700S",
                                    command=self.process_on_gpu,
                                    bg='#ff6b35', fg='white',
                                    font=('Arial', 12, 'bold'),
                                    height=2, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=10)
        
        # Visualize button
        self.visualize_btn = tk.Button(control_frame, text=" Visualize Brain",
                                      command=self.visualize_brain,
                                      bg='#2196F3', fg='white',
                                      font=('Arial', 12, 'bold'),
                                      height=2, state=tk.DISABLED)
        self.visualize_btn.pack(side=tk.LEFT, padx=10)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = tk.Label(self.root, textvariable=self.progress_var,
                                      bg='#1a1a1a', fg='lightgreen',
                                      font=('Arial', 11))
        self.progress_label.pack(pady=5)
        
        # Info display
        self.info_text = tk.Text(self.root, height=15, bg='#2a2a2a', fg='white',
                                font=('Consolas', 10))
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(self.info_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.info_text.yview)
        
    def log_info(self, message):
        """Log information to the text display"""
        self.info_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.info_text.see(tk.END)
        self.root.update()
        
    def initialize_gpu(self):
        """Initialize GPU in background"""
        def init_thread():
            success, message = self.gpu_processor.initialize_rx7700s()
            
            if success:
                self.gpu_status_label.config(text=f" {message}", fg='lightgreen')
                self.log_info(f"GPU initialized successfully: {message}")
                self.generate_btn.config(state=tk.NORMAL)
            else:
                self.gpu_status_label.config(text=f" {message}", fg='red')
                self.log_info(f"GPU initialization failed: {message}")
        
        threading.Thread(target=init_thread, daemon=True).start()
        
    def generate_brain_data(self):
        """Generate synthetic brain data"""
        try:
            self.log_info("Generating synthetic brain volume data...")
            self.progress_var.set("Generating brain data...")
            
            # Create 3D brain-like volume
            size = 128  # 128x128x128 voxels
            x, y, z = np.meshgrid(np.linspace(-2, 2, size),
                                 np.linspace(-2, 2, size),
                                 np.linspace(-2, 2, size))
            
            # Brain-like ellipsoid with structure
            brain_mask = (x**2/2.5 + y**2/2.0 + z**2/1.5) < 1.0
            
            # Create brain data
            self.brain_data = np.zeros((size, size, size), dtype=np.float32)
            self.brain_data[brain_mask] = 1.0
            
            # Add brain structure and noise
            structure = 0.3 * (np.sin(x*3) * np.cos(y*3) * np.sin(z*2))
            noise = 0.1 * np.random.random((size, size, size))
            
            self.brain_data += structure + noise
            self.brain_data = np.clip(self.brain_data, 0, 2)
            
            voxel_count = np.sum(self.brain_data > 0.1)
            
            self.log_info(f"Brain data generated: {size}³ voxels")
            self.log_info(f"Active brain voxels: {voxel_count:,}")
            self.log_info(f"Data size: {self.brain_data.nbytes / 1024**2:.1f} MB")
            
            self.progress_var.set("Brain data ready for GPU processing")
            self.process_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log_info(f"Error generating brain data: {e}")
            
    def process_on_gpu(self):
        """Process brain data on RX 7700S GPU"""
        if self.is_processing or self.brain_data is None:
            return
            
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        
        def process_thread():
            try:
                self.log_info(" STARTING GPU PROCESSING ON RX 7700S ")
                self.log_info("*** WATCH TASK MANAGER - GPU 0 SHOULD ACTIVATE ***")
                
                def progress_callback(message, percent):
                    self.progress_var.set(f"{message} ({percent}%)")
                    self.log_info(message)
                
                start_time = time.time()
                
                result, message = self.gpu_processor.process_brain_data_gpu(
                    self.brain_data, progress_callback)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                if result is not None:
                    self.processed_data = result
                    self.log_info(f" GPU processing completed in {processing_time:.2f}s")
                    self.log_info(f"   Processed {self.brain_data.size:,} voxels")
                    self.log_info(f"   Average: {self.brain_data.size/processing_time:.0f} voxels/sec")
                    self.progress_var.set("GPU processing completed - ready to visualize")
                    self.visualize_btn.config(state=tk.NORMAL)
                else:
                    self.log_info(f" GPU processing failed: {message}")
                    self.progress_var.set("GPU processing failed")
                    
            except Exception as e:
                self.log_info(f"Error in GPU processing: {e}")
                self.progress_var.set("GPU processing error")
                
            finally:
                self.is_processing = False
                self.process_btn.config(state=tk.NORMAL)
        
        threading.Thread(target=process_thread, daemon=True).start()
        
    def visualize_brain(self):
        """Visualize processed brain data"""
        if self.processed_data is None:
            return
            
        try:
            self.log_info("Creating brain visualization...")
            
            # Create visualization window
            viz_window = tk.Toplevel(self.root)
            viz_window.title("GPU-Processed Brain Visualization")
            viz_window.geometry("800x600")
            viz_window.configure(bg='#1a1a1a')
            
            # Use matplotlib for visualization
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            fig = plt.figure(figsize=(10, 8))
            fig.patch.set_facecolor('#1a1a1a')
            
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('#1a1a1a')
            
            # Create brain surface from processed data
            threshold = np.percentile(self.processed_data, 85)  # Top 15% of values
            
            # Find surface points
            surface_mask = self.processed_data > threshold
            z_coords, y_coords, x_coords = np.where(surface_mask)
            
            # Sample points for visualization (limit for performance)
            n_points = min(5000, len(x_coords))
            indices = np.random.choice(len(x_coords), n_points, replace=False)
            
            x_sample = x_coords[indices]
            y_sample = y_coords[indices]
            z_sample = z_coords[indices]
            colors = self.processed_data[surface_mask][indices]
            
            # Create 3D scatter plot
            scatter = ax.scatter(x_sample, y_sample, z_sample, 
                               c=colors, cmap='hot', alpha=0.6, s=20)
            
            ax.set_xlabel('X', color='white')
            ax.set_ylabel('Y', color='white')
            ax.set_zlabel('Z', color='white')
            ax.set_title('GPU-Processed Brain Visualization\n(RX 7700S Accelerated)', 
                        color='white', fontsize=14)
            
            # Style axes
            ax.tick_params(colors='white')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, shrink=0.8)
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.log_info(f" Brain visualization created with {n_points:,} points")
            
        except Exception as e:
            self.log_info(f"Error creating visualization: {e}")
    
    def run(self):
        """Run the GUI"""
        self.log_info("GPU-Accelerated Brain GUI started")
        self.log_info("This GUI will use your RX 7700S for brain processing")
        self.log_info("Watch Task Manager > Performance > GPU 0 during processing")
        self.root.mainloop()

def main():
    """Main function"""
    print(" GPU-Accelerated Brain Visualization ")
    print("=" * 50)
    print("This GUI will use your RX 7700S GPU for brain processing")
    print("Watch Task Manager during GPU processing")
    print()
    
    try:
        app = GPUBrainGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()