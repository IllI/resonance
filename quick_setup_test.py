#!/usr/bin/env python3
"""
Quick Setup Test - RX 7700S Integration

This script applies RX 7700S environment configuration and runs your 3D analyzer.
"""

import os
import sys

def setup_rx7700s_environment():
    """Apply RX 7700S environment variables that we know work."""
    print("🎯 Configuring RX 7700S environment...")
    
    # Apply the successful environment variables from our working implementation
    rx7700s_env_vars = {
        'HIP_VISIBLE_DEVICES': '1',           # Use GPU 1 (RX 7700S)
        'ROCR_VISIBLE_DEVICES': '1',          # ROCm device selection
        'GPU_MAX_ALLOC_PERCENT': '95',        # Allow high memory usage
        'GPU_MAX_HEAP_SIZE': '100',           # Maximum heap allocation
        'HSA_OVERRIDE_GFX_VERSION': '11.0.0', # RDNA3 architecture
        'GPU_FORCE_64BIT_PTR': '1',
        'GPU_USE_SYNC_OBJECTS': '1',
        'GPU_SINGLE_ALLOC_PERCENT': '100',
        'OPENCL_DEVICE_FILTER': 'discrete',
        'AMD_OPENCL_DEVICE_MASK': '2',        # Binary: 10 = GPU 1 only
        'AmdPowerXpressRequestHighPerformance': '1',
        'AMD_OPENCL_DEVICE': '1',
        'OPENCL_DEVICE': '1',
        'PYOPENCL_CTX': '0:1',
    }
    
    # Apply environment variables
    for var_name, var_value in rx7700s_env_vars.items():
        os.environ[var_name] = var_value
    
    print(f"✅ Applied {len(rx7700s_env_vars)} RX 7700S environment variables")
    print("*** RX 7700S will now be used for GPU processing ***")
    print("*** Monitor Task Manager → Performance → GPU 1 ***")

def main():
    """Main function."""
    print("🧠 Quick Setup Test - RX 7700S Integration")
    print("=" * 50)
    
    # Setup RX 7700S environment
    setup_rx7700s_environment()
    
    print("\n🚀 Launching 3D Brain Visualizer with RX 7700S...")
    
    try:
        # Add python directory to path and launch the visualizer
        import sys
        from pathlib import Path
        
        python_dir = Path("python")
        if python_dir.exists():
            sys.path.insert(0, str(python_dir))
        
        # Try different working GUIs in order of preference
        gui_launched = False
        
        # Try the RX 7700S fixed GUI first
        try:
            from python.rx7700s_brain_gui_fixed import main as launch_rx7700s_gui
            print("Using RX 7700S Brain GUI (recommended)")
            launch_rx7700s_gui()
            gui_launched = True
        except Exception as e:
            print(f"RX 7700S GUI failed: {e}")
        
        # Try the GPU accelerated brain GUI
        if not gui_launched:
            try:
                from python.gpu_accelerated_brain_gui import main as launch_gpu_gui
                print("Using GPU Accelerated Brain GUI")
                launch_gpu_gui()
                gui_launched = True
            except Exception as e:
                print(f"GPU GUI failed: {e}")
        
        # Try the clean ultra realistic GUI
        if not gui_launched:
            try:
                from python.ultra_realistic_brain_gui_clean import main as launch_clean_gui
                print("Using Clean Ultra Realistic Brain GUI")
                launch_clean_gui()
                gui_launched = True
            except Exception as e:
                print(f"Clean GUI failed: {e}")
        
        if not gui_launched:
            raise Exception("No working GUI found")
        
    except Exception as e:
        print(f"❌ Failed to launch visualizer: {e}")
        print("Try running manually: python python/ultra_realistic_brain_gui.py")

if __name__ == "__main__":
    main()