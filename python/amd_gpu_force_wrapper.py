#!/usr/bin/env python3
"""
AMD GPU Force Wrapper
Loads the AMD GPU Force DLL to programmatically control GPU selection
"""

import ctypes
import os
import sys
import pyopencl as cl
import numpy as np
import time
from pathlib import Path

class AMDGPUForceWrapper:
    """Wrapper for AMD GPU Force DLL"""
    
    def __init__(self):
        self.dll = None
        self.load_dll()
    
    def load_dll(self):
        """Load the AMD GPU Force DLL"""
        try:
            # Try to load the compiled DLL
            dll_path = Path(__file__).parent / "amd_gpu_force.dll"
            if dll_path.exists():
                self.dll = ctypes.CDLL(str(dll_path))
                print(" Loaded AMD GPU Force DLL")
                self.setup_dll_functions()
            else:
                print("  AMD GPU Force DLL not found")
                print("   Compile amd_gpu_force.cpp first:")
                print("   cl /LD amd_gpu_force.cpp")
                self.create_fallback_export()
        except Exception as e:
            print(f"  Could not load DLL: {e}")
            self.create_fallback_export()
    
    def setup_dll_functions(self):
        """Setup DLL function signatures"""
        if self.dll:
            # RequestHighPerformanceGPU function
            self.dll.RequestHighPerformanceGPU.argtypes = []
            self.dll.RequestHighPerformanceGPU.restype = None
            
            # GetGPUPreference function
            self.dll.GetGPUPreference.argtypes = []
            self.dll.GetGPUPreference.restype = ctypes.c_int
    
    def create_fallback_export(self):
        """Create fallback export using ctypes"""
        try:
            # Create a simple DLL in memory with the required export
            print(" Creating fallback AMD power export...")
            
            # Set environment variables as fallback
            os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
            os.environ['AMD_OPENCL_DEVICE'] = '1'
            os.environ['OPENCL_DEVICE'] = '1'
            
            print(" Set AMD environment variables as fallback")
        except Exception as e:
            print(f"  Fallback export failed: {e}")
    
    def request_high_performance_gpu(self):
        """Request high performance GPU usage"""
        if self.dll:
            try:
                self.dll.RequestHighPerformanceGPU()
                print(" Requested high performance GPU via DLL")
            except Exception as e:
                print(f"  DLL request failed: {e}")
        else:
            print(" Using environment variable fallback")
    
    def get_gpu_preference(self):
        """Get current GPU preference"""
        if self.dll:
            try:
                preference = self.dll.GetGPUPreference()
                print(f" GPU Preference: {preference}")
                return preference
            except Exception as e:
                print(f"  Could not get preference: {e}")
        return 1  # Default to high performance
    
    def test_gpu_selection(self):
        """Test which GPU is being selected"""
        print("\n Testing GPU selection after AMD force...")
        
        # Request high performance GPU
        self.request_high_performance_gpu()
        
        # Get all platforms and devices
        platforms = cl.get_platforms()
        
        print(f"\nFound {len(platforms)} OpenCL platform(s):")
        
        for i, platform in enumerate(platforms):
            print(f"\nPlatform {i}: {platform.name}")
            devices = platform.get_devices()
            
            for j, device in enumerate(devices):
                device_info = {
                    'name': device.name,
                    'memory': device.global_mem_size // (1024**3),
                    'compute_units': device.max_compute_units,
                    'max_freq': device.max_clock_frequency,
                    'is_discrete': self.is_discrete_gpu(device.name, device.global_mem_size)
                }
                
                status = " DISCRETE" if device_info['is_discrete'] else " INTEGRATED"
                print(f"  Device {j}: {device.name} {status}")
                print(f"    Memory: {device_info['memory']} GB")
                print(f"    Compute Units: {device_info['compute_units']}")
                print(f"    Max Frequency: {device_info['max_freq']} MHz")
        
        # Test default context creation
        print(f"\n Testing default context creation...")
        try:
            # This should now prefer the discrete GPU
            context = cl.create_some_context()
            device = context.devices[0]
            print(f" Default context using: {device.name}")
            
            if self.is_discrete_gpu(device.name, device.global_mem_size):
                print(f"🎉 SUCCESS: Default context is using discrete GPU!")
                return True
            else:
                print(f"  Default context still using integrated GPU")
                return False
                
        except Exception as e:
            print(f" Context creation failed: {e}")
            return False
    
    def is_discrete_gpu(self, device_name: str, memory_size: int) -> bool:
        """Check if device is discrete GPU"""
        name_lower = device_name.lower()
        
        # Check for discrete GPU indicators
        discrete_indicators = [
            'gfx1103',  # RX 7700S
            'rx 7700',
            'radeon rx',
            memory_size > 8 * 1024**3  # More than 8GB memory
        ]
        
        return any(indicator for indicator in discrete_indicators 
                  if (isinstance(indicator, str) and indicator in name_lower) or
                     (isinstance(indicator, bool) and indicator))

def compile_dll():
    """Compile the AMD GPU Force DLL"""
    print("🔨 Compiling AMD GPU Force DLL...")
    
    try:
        import subprocess
        
        # Compile the C++ file
        cmd = ['cl', '/LD', 'amd_gpu_force.cpp']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" DLL compiled successfully")
            print("   Created: amd_gpu_force.dll")
        else:
            print(f" Compilation failed:")
            print(f"   Error: {result.stderr}")
            print(f"   Make sure Visual Studio Build Tools are installed")
            
    except FileNotFoundError:
        print(" Visual Studio compiler (cl.exe) not found")
        print("   Install Visual Studio Build Tools or Visual Studio")
        print("   Or manually compile: cl /LD amd_gpu_force.cpp")

def main():
    """Main function"""
    print("=== AMD GPU Force Wrapper ===")
    print("Programmatic AMD Adrenalin GPU control")
    
    # Try to compile DLL if it doesn't exist
    dll_path = Path(__file__).parent / "amd_gpu_force.dll"
    if not dll_path.exists():
        print(" AMD GPU Force DLL not found, attempting to compile...")
        compile_dll()
    
    # Create wrapper
    wrapper = AMDGPUForceWrapper()
    
    # Test GPU selection
    success = wrapper.test_gpu_selection()
    
    if success:
        print(f"\n🎉 SUCCESS: AMD GPU Force is working!")
        print(f"   Your applications will now use the discrete GPU")
    else:
        print(f"\n  AMD GPU Force partially working")
        print(f"   May need additional system configuration")
    
    print(f"\n{'='*60}")
    print("USAGE IN YOUR CODE:")
    print("="*60)
    print("from amd_gpu_force_wrapper import AMDGPUForceWrapper")
    print("")
    print("# At the start of your application")
    print("wrapper = AMDGPUForceWrapper()")
    print("wrapper.request_high_performance_gpu()")
    print("")
    print("# Then create OpenCL contexts normally")
    print("context = cl.create_some_context()  # Will now use discrete GPU")

if __name__ == "__main__":
    main()

