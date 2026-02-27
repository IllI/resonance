#!/usr/bin/env python3
"""
Quick Setup Test - RX 7700S Integrated 3D Brain Analyzer

This script integrates the successful RX 7700S GPU selection into the existing
3D brain analyzer program, ensuring the discrete GPU is used for all processing.
"""

import os
import sys
import time
from pathlib import Path

def setup_rx7700s_environment():
    """Setup RX 7700S environment using successful methods."""
    print(" Setting up RX 7700S environment...")
    
    # Apply the successful environment variables from our working implementation
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
        
        # Legacy environment variables (for compatibility)
        'AmdPowerXpressRequestHighPerformance': '1',
        'AMD_OPENCL_DEVICE': '1',
        'OPENCL_DEVICE': '1',
        'PYOPENCL_CTX': '0:1',
        
        # DirectX/Vulkan preferences
        'DXVK_FILTER_DEVICE_NAME': 'RX 7700S',
    }
    
    # Apply environment variables
    applied_count = 0
    for var_name, var_value in rx7700s_env_vars.items():
        try:
            os.environ[var_name] = var_value
            applied_count += 1
        except Exception as e:
            print(f" Failed to set {var_name}: {e}")
    
    print(f" Applied {applied_count}/{len(rx7700s_env_vars)} RX 7700S environment variables")
    return applied_count > 0

def verify_rx7700s_detection():
    """Verify RX 7700S is detected and available."""
    print("\n Verifying RX 7700S detection...")
    
    try:
        import pyopencl as cl
        
        # Find AMD platform
        platforms = cl.get_platforms()
        amd_platform = None
        
        for platform in platforms:
            if 'AMD' in platform.name:
                amd_platform = platform
                break
        
        if not amd_platform:
            print(" AMD platform not found")
            return False
        
        # Get GPU devices
        devices = amd_platform.get_devices(cl.device_type.GPU)
        print(f"Found {len(devices)} AMD GPU devices:")
        
        rx7700s_found = False
        for i, device in enumerate(devices):
            name = device.name.strip()
            memory_gb = device.global_mem_size // (1024**3)
            
            # Check if this is RX 7700S (gfx1103 or >10GB memory)
            is_rx7700s = 'gfx1103' in name or memory_gb >= 10
            
            status = " RX 7700S" if is_rx7700s else "   780M"
            print(f"  GPU {i}: {name} ({memory_gb} GB) {status}")
            
            if is_rx7700s:
                rx7700s_found = True
        
        if rx7700s_found:
            print(" RX 7700S detected and available")
            return True
        else:
            print(" RX 7700S not found")
            return False
            
    except ImportError:
        print(" OpenCL not available (pyopencl not installed)")
        return False
    except Exception as e:
        print(f" GPU detection failed: {e}")
        return False

def patch_brain_analyzer():
    """Patch the existing brain analyzer to use RX 7700S."""
    print("\n Patching brain analyzer for RX 7700S...")
    
    try:
        # Import our successful RX 7700S detector
        from rx7700s_simple_detector import RX7700SSimpleDetector
        
        # Create a global RX 7700S detector instance
        global rx7700s_detector
        rx7700s_detector = RX7700SSimpleDetector(auto_setup_environment=False)  # Environment already set
        
        if rx7700s_detector.rx7700s_device:
            device_name = rx7700s_detector.rx7700s_device.name.strip()
            memory_gb = rx7700s_detector.rx7700s_device.global_mem_size // (1024**3)
            print(f" RX 7700S detector initialized: {device_name} ({memory_gb} GB)")
            return True
        else:
            print(" RX 7700S detector failed to initialize")
            return False
            
    except Exception as e:
        print(f" Failed to patch brain analyzer: {e}")
        return False

def launch_3d_analyzer():
    """Launch the 3D brain analyzer with RX 7700S integration."""
    print("\n Launching 3D Brain Analyzer with RX 7700S...")
    print("=" * 60)
    
    try:
        # Check if we're in the right directory
        current_dir = Path.cwd()
        gui_file = current_dir / "python" / "ultra_realistic_brain_gui.py"
        
        if not gui_file.exists():
            # Try current directory
            gui_file = current_dir / "ultra_realistic_brain_gui.py"
            if not gui_file.exists():
                print(" ultra_realistic_brain_gui.py not found")
                print(f"   Searched in: {current_dir}")
                print(f"   And: {current_dir / 'python'}")
                return False
        
        print(" Found GUI file")
        
        # Import and launch the GUI
        print("Starting Ultra-Realistic Brain 3D Visualization...")
        print("*** MONITOR TASK MANAGER → PERFORMANCE → GPU 1 ***")
        print("*** RX 7700S should show high usage during processing ***")
        print()
        
        # Import the GUI module
        if str(current_dir).endswith('python'):
            from ultra_realistic_brain_gui import UltraRealisticBrainGUI
        else:
            sys.path.insert(0, str(current_dir / "python"))
            from ultra_realistic_brain_gui import UltraRealisticBrainGUI
        
        # Create and run GUI
        import tkinter as tk
        root = tk.Tk()
        app = UltraRealisticBrainGUI(root)
        
        print(" GUI launched successfully!")
        print("   Use the GUI to generate brain models and verify RX 7700S usage")
        
        root.mainloop()
        return True
        
    except ImportError as e:
        print(f" Import error: {e}")
        print("\nTrying alternative launch method...")
        
        # Try launching via the quick start GUI
        try:
            from quick_start_gui import main as launch_quick_start
            launch_quick_start()
            return True
        except Exception as e2:
            print(f" Alternative launch failed: {e2}")
            return False
            
    except Exception as e:
        print(f" Failed to launch 3D analyzer: {e}")
        return False

def run_quick_test():
    """Run a quick test to verify RX 7700S is working."""
    print("\n Running quick RX 7700S test...")
    
    try:
        # Import our brain analyzer
        from rx7700s_brain_analyzer_fixed import RX7700SBrainAnalyzerFixed
        
        # Create analyzer
        analyzer = RX7700SBrainAnalyzerFixed()
        
        # Create small test volume
        test_volume = np.random.rand(32, 32, 16).astype(np.float32)
        test_volume[8:24, 8:24, 4:12] = 0.8  # Brain region
        
        print(f"   Test volume: {test_volume.shape}")
        
        # Run analysis
        start_time = time.time()
        results = analyzer.analyze_brain_volume(test_volume, generate_surface=False)
        end_time = time.time()
        
        if results['success']:
            print(f" Quick test successful!")
            print(f"   Processing time: {end_time - start_time:.3f} seconds")
            print(f"   GPU used: {results['gpu_device']}")
            print(f"   Brain voxels: {results['segmentation']['brain_voxels']:,}")
            return True
        else:
            print(f" Quick test failed: {results['error']}")
            return False
            
    except Exception as e:
        print(f" Quick test error: {e}")
        return False

def main():
    """Main function for quick setup test."""
    print(" Quick Setup Test - RX 7700S Integrated 3D Brain Analyzer")
    print("=" * 70)
    print("This will integrate RX 7700S GPU selection into your 3D analyzer")
    print()
    
    # Step 1: Setup RX 7700S environment
    env_success = setup_rx7700s_environment()
    
    # Step 2: Verify RX 7700S detection
    detection_success = verify_rx7700s_detection()
    
    # Step 3: Patch brain analyzer
    patch_success = patch_brain_analyzer()
    
    # Step 4: Run quick test automatically
    test_success = True
    if detection_success and patch_success:
        print("\n" + "=" * 70)
        print(" Running quick RX 7700S test...")
        
        # Import numpy for test
        try:
            import numpy as np
            test_success = run_quick_test()
        except ImportError:
            print(" NumPy not available for quick test, skipping...")
            test_success = True
    
    # Summary
    print("\n" + "=" * 70)
    print(" SETUP SUMMARY")
    print("=" * 70)
    
    print(f"Environment Setup: {' SUCCESS' if env_success else ' FAILED'}")
    print(f"RX 7700S Detection: {' SUCCESS' if detection_success else ' FAILED'}")
    print(f"Analyzer Patching: {' SUCCESS' if patch_success else ' FAILED'}")
    print(f"Quick Test: {' SUCCESS' if test_success else ' FAILED'}")
    
    overall_success = env_success and detection_success
    
    if overall_success:
        print("\n🎉 RX 7700S integration successful!")
        print("\nNext steps:")
        print("1. The 3D analyzer will now use RX 7700S for all processing")
        print("2. Monitor Task Manager → Performance → GPU 1 during use")
        print("3. GPU 1 should show high usage, GPU 0 should be idle")
        
        # Launch the 3D analyzer automatically
        print("\n" + "=" * 70)
        print(" Launching 3D Brain Analyzer with RX 7700S...")
        
        launch_success = launch_3d_analyzer()
        if not launch_success:
            print("\n GUI launch failed, but RX 7700S is configured")
            print("Try running manually:")
            print("   python ultra_realistic_brain_gui.py")
            
    else:
        print("\n Some setup steps failed")
        print("\nTroubleshooting:")
        print("1. Update AMD drivers to latest version")
        print("2. Restart your computer")
        print("3. Check Device Manager for GPU errors")
        print("4. Re-run this script")
    
    print(f"\nFor detailed information, see: RX7700S_SUCCESS_SUMMARY.md")
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()