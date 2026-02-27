#!/usr/bin/env python3
"""
OpenCL Device Reorder - Nuclear Solution

This script forces OpenCL to see the RX 7700S as GPU 0 (first device)
by manipulating the OpenCL device enumeration order.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def create_opencl_wrapper():
    """Create OpenCL wrapper that reorders devices."""
    print("Creating OpenCL device reorder wrapper...")
    
    wrapper_code = '''#!/usr/bin/env python3
"""
OpenCL Device Wrapper - Forces RX 7700S as Primary

This wrapper intercepts OpenCL device enumeration and ensures
the RX 7700S appears as GPU 0 (first device).
"""

import pyopencl as cl
import sys

# Store original functions
_original_get_devices = cl.Platform.get_devices

def _reordered_get_devices(self, device_type=None):
    """Get devices with RX 7700S first."""
    try:
        # Get original device list
        original_devices = _original_get_devices(self, device_type)
        
        if device_type == cl.device_type.GPU or device_type is None:
            # Separate devices by type
            rx7700s_devices = []
            other_devices = []
            
            for device in original_devices:
                device_name = device.name.strip()
                memory_gb = device.global_mem_size // (1024**3)
                
                # Check if this is RX 7700S
                if 'gfx1103' in device_name or memory_gb >= 10 or '7700' in device_name:
                    rx7700s_devices.append(device)
                    print(f"OpenCL Wrapper: Found RX 7700S - {device_name} ({memory_gb} GB)")
                else:
                    other_devices.append(device)
                    print(f"OpenCL Wrapper: Found other GPU - {device_name} ({memory_gb} GB)")
            
            # Return RX 7700S devices first, then others
            reordered_devices = rx7700s_devices + other_devices
            
            if reordered_devices != original_devices:
                print("OpenCL Wrapper: Reordered devices - RX 7700S is now GPU 0")
            
            return reordered_devices
        else:
            return original_devices
            
    except Exception as e:
        print(f"OpenCL Wrapper error: {e}")
        return _original_get_devices(self, device_type)

# Monkey patch the get_devices method
cl.Platform.get_devices = _reordered_get_devices

print("OpenCL Device Wrapper: RX 7700S will appear as GPU 0")
'''
    
    try:
        wrapper_file = Path("opencl_wrapper.py")
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        print(f"   Created OpenCL wrapper: {wrapper_file}")
        return wrapper_file
        
    except Exception as e:
        print(f"   Failed to create wrapper: {e}")
        return None

def create_brain_gui_launcher_with_wrapper():
    """Create brain GUI launcher that uses the OpenCL wrapper."""
    print("Creating brain GUI launcher with OpenCL wrapper...")
    
    launcher_code = '''#!/usr/bin/env python3
"""
Brain GUI Launcher with OpenCL Device Reordering

This launcher ensures the RX 7700S appears as GPU 0 in OpenCL.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

# Import OpenCL wrapper FIRST (before any other imports)
try:
    import opencl_wrapper  # This reorders the devices
    print(" OpenCL device wrapper loaded - RX 7700S prioritized")
except ImportError as e:
    print(f" OpenCL wrapper not available: {e}")

def main():
    """Launch brain GUI with device reordering."""
    print("Brain GUI with OpenCL Device Reordering")
    print("=" * 50)
    print("RX 7700S should now appear as GPU 0 in OpenCL")
    print()
    
    try:
        # Import and launch brain GUI
        from ultra_realistic_brain_gui import main as launch_gui
        launch_gui()
        
    except Exception as e:
        print(f"Error launching brain GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
'''
    
    try:
        launcher_file = Path("brain_gui_rx7700s.py")
        with open(launcher_file, 'w') as f:
            f.write(launcher_code)
        
        print(f"   Created launcher: {launcher_file}")
        return launcher_file
        
    except Exception as e:
        print(f"   Failed to create launcher: {e}")
        return None

def test_device_reordering():
    """Test if device reordering works."""
    print("Testing OpenCL device reordering...")
    
    try:
        # Import the wrapper
        sys.path.insert(0, str(Path.cwd()))
        import opencl_wrapper
        
        # Test OpenCL device order
        import pyopencl as cl
        
        platforms = cl.get_platforms()
        for platform in platforms:
            if 'AMD' in platform.name:
                devices = platform.get_devices(cl.device_type.GPU)
                
                print(f"\nDevice order after wrapper:")
                for i, device in enumerate(devices):
                    device_name = device.name.strip()
                    memory_gb = device.global_mem_size // (1024**3)
                    
                    if 'gfx1103' in device_name or memory_gb >= 10:
                        gpu_type = "RX 7700S"
                    else:
                        gpu_type = "780M"
                    
                    print(f"  GPU {i}: {device_name} ({memory_gb} GB) - {gpu_type}")
                
                # Check if RX 7700S is now GPU 0
                if devices and len(devices) > 0:
                    first_device = devices[0]
                    first_name = first_device.name.strip()
                    first_memory = first_device.global_mem_size // (1024**3)
                    
                    if 'gfx1103' in first_name or first_memory >= 10:
                        print(f"   SUCCESS: RX 7700S is now GPU 0!")
                        return True
                    else:
                        print(f"   FAILED: GPU 0 is still {first_name}")
                        return False
        
        return False
        
    except Exception as e:
        print(f"Device reordering test failed: {e}")
        return False

def main():
    """Main OpenCL device reorder function."""
    print("OpenCL Device Reorder - Nuclear Solution")
    print("=" * 60)
    print("This will force OpenCL to see RX 7700S as GPU 0")
    print()
    
    # Step 1: Create OpenCL wrapper
    wrapper_file = create_opencl_wrapper()
    
    # Step 2: Create brain GUI launcher
    launcher_file = create_brain_gui_launcher_with_wrapper()
    
    # Step 3: Test device reordering
    if wrapper_file:
        reorder_success = test_device_reordering()
    else:
        reorder_success = False
    
    print("\n" + "=" * 60)
    print("DEVICE REORDERING RESULTS:")
    print(f"  OpenCL Wrapper: {'' if wrapper_file else ''}")
    print(f"  Brain GUI Launcher: {'' if launcher_file else ''}")
    print(f"  Device Reordering: {'' if reorder_success else ''}")
    
    if reorder_success and launcher_file:
        print("\n🎉 SUCCESS: OpenCL device reordering works!")
        print("\nNow run the brain GUI with device reordering:")
        print(f"  python {launcher_file.name}")
        print("\nExpected result:")
        print("  RX 7700S should now be used as GPU 0")
        print("  Task Manager should show RX 7700S usage")
        
    elif wrapper_file and launcher_file:
        print("\n Device reordering created but not tested")
        print("Try running the launcher anyway:")
        print(f"  python {launcher_file.name}")
        
    else:
        print("\n Device reordering failed")
        print("The brain GUI will continue using the integrated GPU")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()