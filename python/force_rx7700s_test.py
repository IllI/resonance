#!/usr/bin/env python3
"""
Force RX 7700S Test - Direct GPU Activation
This will definitely activate your RX 7700S GPU
"""

import os
import time
import numpy as np

# Force RX 7700S environment
os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
os.environ['AMD_OPENCL_DEVICE'] = '1'  # Device 1 = gfx1103 = RX 7700S
os.environ['OPENCL_DEVICE'] = '1'
os.environ['PYOPENCL_CTX'] = '0:1'  # Platform 0, Device 1

def force_rx7700s_activation():
    """Force RX 7700S activation with heavy workload"""
    print("=== FORCE RX 7700S ACTIVATION ===")
    print("*** WATCH TASK MANAGER - GPU 0 (RX 7700S) MUST ACTIVATE ***")
    
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
            print("No AMD platform found!")
            return False
        
        print(f"Using AMD platform: {amd_platform.name}")
        
        # Get devices
        devices = amd_platform.get_devices()
        print(f"Found {len(devices)} devices:")
        
        rx7700s_device = None
        for i, device in enumerate(devices):
            name = device.name.strip()
            memory_gb = device.global_mem_size // (1024**3)
            compute_units = device.max_compute_units
            
            print(f"  Device {i}: {name}")
            print(f"    Memory: {memory_gb} GB")
            print(f"    Compute Units: {compute_units}")
            
            # Select RX 7700S (gfx1103 with 12GB)
            if 'gfx1103' in name or memory_gb >= 12:
                rx7700s_device = device
                print(f"    *** SELECTED: RX 7700S ***")
        
        if not rx7700s_device:
            print("RX 7700S not found! Using device 1 as fallback...")
            if len(devices) > 1:
                rx7700s_device = devices[1]
            else:
                print("No device 1 available!")
                return False
        
        print(f"\nForcing heavy workload on: {rx7700s_device.name}")
        print("*** CHECK TASK MANAGER NOW - GPU 0 SHOULD ACTIVATE ***")
        
        # Create context and queue
        context = cl.Context([rx7700s_device])
        queue = cl.CommandQueue(context)
        
        # Create massive workload
        size = 8 * 1024 * 1024  # 8M elements
        print(f"Creating {size:,} element arrays...")
        
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        
        # Create buffers
        a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
        
        # Extremely heavy kernel
        kernel_code = """
        __kernel void extreme_workload(__global const float* a,
                                      __global const float* b,
                                      __global float* c) {
            int gid = get_global_id(0);
            
            // EXTREMELY heavy computation
            float result = 0.0f;
            for (int i = 0; i < 5000; i++) {  // 5000 iterations per element
                float temp = sin(a[gid] * i) * cos(b[gid] * i);
                result += sqrt(fabs(temp));
                result = fmod(result + tan(temp), 100.0f);
            }
            c[gid] = result;
        }
        """
        
        program = cl.Program(context, kernel_code).build()
        kernel = program.extreme_workload
        
        print("\n STARTING EXTREME GPU WORKLOAD ")
        print("*** GPU 0 (RX 7700S) SHOULD BE AT 100% NOW ***")
        print("Running for 15 seconds...")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < 15.0:  # 15 seconds of heavy load
            print(f"  Iteration {iteration+1} - GPU should be at 100%")
            
            # Execute heavy kernel
            kernel(queue, (size,), None, a_buf, b_buf, c_buf)
            queue.finish()
            
            iteration += 1
            time.sleep(0.5)  # Brief pause to see GPU usage
        
        # Read result to ensure completion
        c = np.empty_like(a)
        cl.enqueue_copy(queue, c, c_buf)
        
        total_time = time.time() - start_time
        print(f"\n EXTREME WORKLOAD COMPLETED!")
        print(f"  Device: {rx7700s_device.name}")
        print(f"  Iterations: {iteration}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Elements processed: {size * iteration:,}")
        
        print("\n*** DID GPU 0 (RX 7700S) SHOW 100% USAGE? ***")
        
        return True
        
    except ImportError:
        print("PyOpenCL not available!")
        return False
    except Exception as e:
        print(f"GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(" FORCE RX 7700S ACTIVATION TEST ")
    print("=" * 50)
    print("This will run EXTREME workload on RX 7700S for 15 seconds")
    print("Watch Task Manager > Performance > GPU 0 during test")
    print("GPU 0 should reach 100% usage")
    print()
    
    input("Press Enter to start EXTREME GPU test...")
    
    success = force_rx7700s_activation()
    
    print("\n" + "=" * 50)
    if success:
        print(" RX 7700S activation test completed!")
        print("   Check if GPU 0 showed high usage in Task Manager")
    else:
        print(" RX 7700S activation test failed!")
        print("   GPU may not be properly configured")
    
    input("\nPress Enter to exit...")