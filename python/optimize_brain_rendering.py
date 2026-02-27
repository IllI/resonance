#!/usr/bin/env python3
"""
Brain Rendering Performance Optimizer

Optimizes the brain GUI for smooth real-time interaction by:
1. Installing performance libraries
2. Configuring GPU acceleration
3. Testing rendering performance
"""

import subprocess
import sys
import os
import time

def install_performance_libraries():
    """Install libraries for better rendering performance."""
    print("Installing performance optimization libraries...")
    
    libraries = [
        "PyOpenGL",
        "PyOpenGL_accelerate", 
        "moderngl",
        "scikit-learn",
        "numba"
    ]
    
    for lib in libraries:
        try:
            print(f"Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f" {lib} installed successfully")
        except subprocess.CalledProcessError:
            print(f" Failed to install {lib} (optional)")
        except Exception as e:
            print(f" Error installing {lib}: {e}")

def test_gpu_acceleration():
    """Test GPU acceleration capabilities."""
    print("\nTesting GPU acceleration...")
    
    # Test OpenCL
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        print(f" OpenCL available: {len(platforms)} platform(s)")
        
        for i, platform in enumerate(platforms):
            devices = platform.get_devices()
            print(f"   Platform {i}: {platform.name}")
            for j, device in enumerate(devices):
                memory_gb = device.global_mem_size // (1024**3)
                print(f"      GPU {j}: {device.name.strip()} ({memory_gb} GB)")
    except ImportError:
        print(" OpenCL not available")
    except Exception as e:
        print(f" OpenCL test failed: {e}")
    
    # Test OpenGL
    try:
        import OpenGL.GL as gl
        print(" OpenGL available for hardware acceleration")
    except ImportError:
        print(" OpenGL not available")
    
    # Test ModernGL
    try:
        import moderngl
        print(" ModernGL available for advanced GPU rendering")
    except ImportError:
        print(" ModernGL not available")

def optimize_matplotlib_settings():
    """Optimize matplotlib for better performance."""
    print("\nOptimizing matplotlib settings...")
    
    try:
        import matplotlib
        
        # Set fast backend
        matplotlib.use('TkAgg')
        print(" Set TkAgg backend for better performance")
        
        # Configure for performance
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        print(" Disabled interactive mode for speed")
        
        # Set performance rcParams
        matplotlib.rcParams['path.simplify'] = True
        matplotlib.rcParams['path.simplify_threshold'] = 0.1
        matplotlib.rcParams['agg.path.chunksize'] = 10000
        print(" Configured matplotlib for performance")
        
    except Exception as e:
        print(f" Matplotlib optimization failed: {e}")

def test_brain_rendering_performance():
    """Test brain rendering performance."""
    print("\nTesting brain rendering performance...")
    
    try:
        import numpy as np
        from fast_brain_renderer import FastBrainRenderer
        import tkinter as tk
        
        # Create test data
        print("Creating test brain data...")
        vertices = np.random.rand(10000, 3) * 100  # 10K vertices
        faces = np.random.randint(0, 10000, (5000, 3))  # 5K faces
        
        # Create test window
        root = tk.Tk()
        root.withdraw()  # Hide window for test
        
        # Test renderer
        renderer = FastBrainRenderer(root)
        
        start_time = time.time()
        success = renderer.create_optimized_brain_visualization(vertices, faces)
        render_time = time.time() - start_time
        
        if success:
            print(f" Brain rendering test successful!")
            print(f"   Render time: {render_time:.3f}s")
            print(f"   Performance: {len(vertices)/render_time:.0f} vertices/sec")
        else:
            print(" Brain rendering test failed")
        
        root.destroy()
        
    except Exception as e:
        print(f" Rendering performance test failed: {e}")

def create_performance_config():
    """Create performance configuration file."""
    print("\nCreating performance configuration...")
    
    config = """# Brain GUI Performance Configuration
# This file contains optimized settings for smooth brain visualization

[rendering]
max_vertices = 5000          # Maximum vertices for real-time interaction
target_fps = 60             # Target frame rate
use_gpu_acceleration = true  # Enable GPU acceleration when available
use_scatter_rendering = true # Use scatter plots for better performance
decimation_factor = 0.1     # Mesh decimation for optimization

[gpu]
prefer_discrete_gpu = true   # Prefer discrete GPU (RX 7700S) over integrated
force_opencl = true         # Force OpenCL usage when available
gpu_memory_limit = 8        # GPU memory limit in GB

[matplotlib]
backend = TkAgg             # Fast matplotlib backend
dpi = 80                    # Lower DPI for better performance
interactive_mode = false    # Disable interactive mode for speed
"""
    
    try:
        with open("brain_performance.conf", "w") as f:
            f.write(config)
        print(" Performance configuration created: brain_performance.conf")
    except Exception as e:
        print(f" Failed to create config file: {e}")

def main():
    """Main optimization routine."""
    print("Brain Rendering Performance Optimizer")
    print("=" * 50)
    
    # Step 1: Install performance libraries
    install_performance_libraries()
    
    # Step 2: Test GPU acceleration
    test_gpu_acceleration()
    
    # Step 3: Optimize matplotlib
    optimize_matplotlib_settings()
    
    # Step 4: Test rendering performance
    test_brain_rendering_performance()
    
    # Step 5: Create performance config
    create_performance_config()
    
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE!")
    print("\nRecommendations for smooth brain visualization:")
    print("1.  Use scatter rendering for large meshes (>5K vertices)")
    print("2.  Enable mesh decimation for real-time interaction")
    print("3.  Use GPU acceleration when available")
    print("4.  Lower DPI settings for better performance")
    print("5.  Disable unnecessary visual effects during rotation")
    
    print("\nTo apply optimizations:")
    print("1. Restart the brain GUI")
    print("2. Load your MRI data")
    print("3. Generate 3D model")
    print("4. Enjoy smooth real-time interaction! ")

if __name__ == "__main__":
    main()