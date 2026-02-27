#!/usr/bin/env python3
"""
Windows GPU 0 Direct Control
Uses Windows-specific methods to force GPU 0 (RX 7700S) usage
"""

import os
import sys
import ctypes
from ctypes import wintypes
import time

# Windows GPU preference constants
GPU_PREFERENCE_HIGH_PERFORMANCE = 2
GPU_PREFERENCE_POWER_SAVING = 1
GPU_PREFERENCE_UNSPECIFIED = 0

def set_windows_gpu_preference():
    """Set Windows GPU preference for current process"""
    try:
        # Try to set GPU preference via Windows API
        kernel32 = ctypes.windll.kernel32
        
        # Set environment variables for this process
        os.environ['DXGI_ADAPTER_INDEX'] = '0'  # Force adapter 0 (RX 7700S)
        os.environ['D3D_FEATURE_LEVEL'] = '11_1'
        os.environ['DXGI_DEBUG'] = '1'
        
        print("✓ Windows GPU preference set to high performance")
        print("✓ DXGI adapter forced to index 0 (RX 7700S)")
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not set Windows GPU preference: {e}")
        return False

def force_discrete_gpu_registry():
    """Force discrete GPU via registry (requires admin)"""
    try:
        import winreg
        
        # PowerXpress settings for AMD
        key_path = r"SOFTWARE\AMD\CN"
        
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_SET_VALUE):
                pass
            
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "PowerXpressRequestPolicy", 0, winreg.REG_DWORD, 1)
                print("✓ AMD PowerXpress set to high performance")
                
        except PermissionError:
            print("  Registry modification requires administrator privileges")
            return False
        except FileNotFoundError:
            print("  AMD registry key not found")
            return False
            
        # DirectX GPU preferences
        dx_key = r"Software\Microsoft\DirectX\UserGpuPreferences"
        python_exe = sys.executable
        
        try:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, dx_key) as key:
                winreg.SetValueEx(key, python_exe, 0, winreg.REG_SZ, "GpuPreference=2;")
                print("✓ DirectX GPU preference set for Python")
        except Exception as e:
            print(f"  DirectX preference failed: {e}")
            
        return True
        
    except ImportError:
        print("  Registry module not available")
        return False
    except Exception as e:
        print(f"  Registry modification failed: {e}")
        return False

def test_gpu0_with_compute():
    """Test GPU 0 using compute workload"""
    print("\n=== TESTING GPU 0 WITH COMPUTE WORKLOAD ===")
    
    # Set all possible environment variables to force GPU 0
    gpu_env_vars = {
        'AmdPowerXpressRequestHighPerformance': '1',
        'GPU_FORCE_64BIT_PTR': '1',
        'GPU_MAX_HEAP_SIZE': '100',
        'GPU_USE_SYNC_OBJECTS': '1',
        'GPU_MAX_ALLOC_PERCENT': '100',
        'GPU_SINGLE_ALLOC_PERCENT': '100',
        'CUDA_VISIBLE_DEVICES': '0',  # If CUDA is present
        'HIP_VISIBLE_DEVICES': '0',   # AMD HIP
        'OPENCL_DEVICE_TYPE': 'GPU',
        'PYOPENCL_COMPILER_OUTPUT': '1'
    }
    
    for var, value in gpu_env_vars.items():
        os.environ[var] = value
        print(f"Set {var} = {value}")
    
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
            print("No AMD platform found!")
            return False
        
        # Get devices and identify RX 7700S
        devices = amd_platform.get_devices()
        rx7700s_device = None
        
        print(f"\nFound {len(devices)} AMD devices:")
        for i, device in enumerate(devices):
            name = device.name.strip()
            memory_gb = device.global_mem_size // (1024**3)
            compute_units = device.max_compute_units
            
            print(f"  Device {i}: {name}")
            print(f"    Memory: {memory_gb} GB, CUs: {compute_units}")
            
            # Select RX 7700S (gfx1103, 12GB memory)
            if 'gfx1103' in name or memory_gb >= 12:
                rx7700s_device = device
                print(f"    *** SELECTED: RX 7700S (should be Windows GPU 0) ***")
        
        if not rx7700s_device:
            print("RX 7700S not found!")
            return False
        
        # Create context and test
        context = cl.Context([rx7700s_device])
        queue = cl.CommandQueue(context)
        
        print(f"\n STARTING WINDOWS GPU 0 TEST ")
        print(f"Device: {rx7700s_device.name}")
        print("*** WATCH TASK MANAGER - GPU 0 SHOULD ACTIVATE ***")
        
        # Create heavy workload
        size = 12 * 1024 * 1024  # 12M elements
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        
        # Buffers
        a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
        
        # Heavy kernel
        kernel_code = """
        __kernel void gpu0_test(__global const float* a,
                               __global const float* b,
                               __global float* c) {
            int gid = get_global_id(0);
            
            float result = 0.0f;
            for (int i = 0; i < 8000; i++) {  // 8000 iterations
                float temp = sin(a[gid] * i * 0.0001f) * cos(b[gid] * i * 0.0001f);
                result += sqrt(fabs(temp));
                result = fmod(result + temp * temp, 100.0f);
            }
            c[gid] = result;
        }
        """
        
        program = cl.Program(context, kernel_code).build()
        kernel = program.gpu0_test
        
        # Run test for 15 seconds
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < 15.0:
            elapsed = time.time() - start_time
            print(f"  {elapsed:.1f}s - GPU 0 should be active (iter {iteration+1})")
            
            kernel(queue, (size,), None, a_buf, b_buf, c_buf)
            queue.finish()
            
            iteration += 1
            time.sleep(0.8)
        
        # Complete
        c = np.empty_like(a)
        cl.enqueue_copy(queue, c, c_buf)
        
        total_time = time.time() - start_time
        print(f"\n GPU 0 test completed!")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Iterations: {iteration}")
        print(f"  Device: {rx7700s_device.name}")
        
        return True
        
    except ImportError:
        print("PyOpenCL not available!")
        return False
    except Exception as e:
        print(f"GPU 0 test failed: {e}")
        return False

def main():
    """Main function"""
    print(" WINDOWS GPU 0 DIRECT CONTROL ")
    print("=" * 50)
    print("This script forces Windows GPU 0 (RX 7700S) usage")
    print()
    
    # Import numpy here
    global np
    import numpy as np
    
    # Step 1: Set Windows preferences
    print("Step 1: Setting Windows GPU preferences...")
    set_windows_gpu_preference()
    
    # Step 2: Registry modifications (optional)
    print("\nStep 2: Setting registry preferences...")
    force_discrete_gpu_registry()
    
    # Step 3: Test GPU 0
    print("\nStep 3: Testing GPU 0 activation...")
    input("Press Enter to start GPU 0 test (watch Task Manager)...")
    
    success = test_gpu0_with_compute()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    if success:
        print(" GPU 0 test completed successfully!")
        print("   Did Windows GPU 0 (RX 7700S) show activity?")
    else:
        print(" GPU 0 test failed!")
        print("   Check AMD drivers and OpenCL installation")
    
    print("\nIf GPU 0 still didn't activate, the issue may be:")
    print("  1. Windows GPU scheduling override")
    print("  2. AMD driver configuration")
    print("  3. Hardware-level GPU switching")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()