#!/usr/bin/env python3
"""
Simple GPU Force - Direct solution to force RX 7700S usage
"""

import os
import sys
import subprocess
import winreg
from pathlib import Path

def set_windows_graphics_settings():
    """Set Windows Graphics Settings to force discrete GPU"""
    print(" Setting Windows Graphics Settings...")
    
    python_path = sys.executable
    print(f"Python path: {python_path}")
    
    try:
        # Registry path for Windows Graphics Settings
        reg_path = r"SOFTWARE\Microsoft\DirectX\UserGpuPreferences"
        
        # Open or create the registry key
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
            # Set Python to use high performance GPU (GpuPreference=2)
            winreg.SetValueEx(key, python_path, 0, winreg.REG_SZ, "GpuPreference=2;")
            print(" Set Windows Graphics preference for Python")
            
            # Also set for common Python variants
            python_dir = Path(python_path).parent
            for variant in ["python.exe", "python3.exe", "pythonw.exe", "python3w.exe"]:
                variant_path = python_dir / variant
                if variant_path.exists():
                    winreg.SetValueEx(key, str(variant_path), 0, winreg.REG_SZ, "GpuPreference=2;")
                    print(f" Set preference for {variant}")
                    
    except Exception as e:
        print(f" Failed to set Windows Graphics preferences: {e}")

def set_environment_variables():
    """Set environment variables"""
    print(" Setting environment variables...")
    
    env_vars = {
        'AmdPowerXpressRequestHighPerformance': '1',
        'AMD_OPENCL_DEVICE': '0',  # Device 0 (RX 7700S)
        'OPENCL_DEVICE': '0',
        'GPU_DEVICE_ORDINAL': '0',
        'AMD_DISABLE_INTEGRATED_GPU': '1',
        'OPENCL_DISABLE_DEVICE_1': '1'  # Disable device 1 (780M)
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"   {var}={value}")

def test_gpu_usage():
    """Test which GPU is being used"""
    print("\n Testing GPU usage...")
    
    try:
        import pyopencl as cl
        import numpy as np
        import time
        
        # Get all devices
        platforms = cl.get_platforms()
        
        for platform in platforms:
            devices = platform.get_devices()
            for i, device in enumerate(devices):
                device_name = device.name
                print(f"Device {i}: {device_name}")
                
                # Test each device
                try:
                    context = cl.Context([device])
                    queue = cl.CommandQueue(context)
                    
                    # Large workload
                    size = 1024 * 1024 * 4
                    a = np.random.rand(size).astype(np.float32)
                    b = np.random.rand(size).astype(np.float32)
                    
                    a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                    b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                    c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
                    
                    kernel_code = """
                    __kernel void test_kernel(__global const float* a,
                                            __global const float* b,
                                            __global float* c) {
                        int gid = get_global_id(0);
                        float temp = a[gid] * b[gid];
                        
                        for(int i = 0; i < 10000; i++) {
                            temp = sqrt(temp);
                            temp = sin(temp) + cos(temp);
                            temp = temp * temp;
                        }
                        c[gid] = temp;
                    }
                    """
                    
                    program = cl.Program(context, kernel_code).build()
                    kernel = program.test_kernel
                    
                    print(f" Testing {device_name}...")
                    print(" Check Task Manager NOW!")
                    print("   GPU 0 (RX 7700S) should show activity")
                    print("   GPU 1 (780M) should NOT show activity")
                    
                    for iteration in range(3):
                        print(f"   Iteration {iteration+1}/3...")
                        kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
                        queue.finish()
                        time.sleep(2)
                    
                    print(f" Completed test on {device_name}")
                    print(" Check Task Manager - which GPU showed activity?")
                    
                except Exception as e:
                    print(f" Test failed for {device_name}: {e}")
                    
    except Exception as e:
        print(f" GPU test failed: {e}")

def main():
    """Main function"""
    print("=== Simple GPU Force ===")
    print("Forcing Windows to use RX 7700S")
    
    # Set configurations
    set_environment_variables()
    set_windows_graphics_settings()
    
    # Test GPU usage
    test_gpu_usage()
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print("="*60)
    print("1. Check Task Manager while the test runs")
    print("2. If GPU 1 (780M) still shows activity:")
    print("   - Restart Python/IDE")
    print("   - Or manually set Windows Graphics Settings:")
    print("     Win + I -> System -> Display -> Graphics settings")
    print("     Add python.exe -> Set to 'High performance'")
    print("3. If GPU 0 (RX 7700S) shows activity:")
    print("   - SUCCESS! Your brain analysis will use RX 7700S")

if __name__ == "__main__":
    main()

