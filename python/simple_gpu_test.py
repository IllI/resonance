#!/usr/bin/env python3
"""
Simple GPU Test - Force RX 7700S Usage
Tests if we can actually get the RX 7700S to do work
"""

import os
import time
import numpy as np

# Set environment variables to force RX 7700S
os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
os.environ['AMD_OPENCL_DEVICE'] = '1'
os.environ['OPENCL_DEVICE'] = '1'
os.environ['PYOPENCL_CTX'] = '0:1'

def test_rx7700s_usage():
    """Test if we can force RX 7700S usage"""
    print("=== RX 7700S GPU Test ===")
    print("*** WATCH TASK MANAGER - GPU 0 (RX 7700S) should activate ***")
    
    try:
        import pyopencl as cl
        
        # Get all platforms and devices
        platforms = cl.get_platforms()
        print(f"Found {len(platforms)} OpenCL platforms")
        
        target_device = None
        for platform in platforms:
            if 'AMD' in platform.name:
                print(f"AMD Platform: {platform.name}")
                devices = platform.get_devices()
                
                for i, device in enumerate(devices):
                    name = device.name.strip()
                    memory_gb = device.global_mem_size // (1024**3)
                    compute_units = device.max_compute_units
                    
                    print(f"  Device {i}: {name}")
                    print(f"    Memory: {memory_gb} GB")
                    print(f"    Compute Units: {compute_units}")
                    
                    # Look for RX 7700S (gfx1103 with 12GB memory)
                    if 'gfx1103' in name or (memory_gb >= 12):  # RX 7700S characteristics
                        target_device = device
                        print(f"    *** SELECTED AS TARGET DEVICE (RX 7700S) ***")
                    elif memory_gb > 8:  # Fallback for discrete GPU with high memory
                        target_device = device
                        print(f"    *** SELECTED AS TARGET DEVICE (High Memory) ***")
        
        if not target_device:
            print("No suitable discrete GPU found!")
            return False
        
        print(f"\nUsing device: {target_device.name}")
        print("Creating heavy GPU workload...")
        
        # Create OpenCL context and queue
        context = cl.Context([target_device])
        queue = cl.CommandQueue(context)
        
        # Create large arrays for heavy computation
        size = 4 * 1024 * 1024  # 4M elements
        print(f"Processing {size} elements...")
        
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        
        # Create OpenCL buffers
        a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
        
        # Heavy computation kernel
        kernel_code = """
        __kernel void heavy_compute(__global const float* a,
                                   __global const float* b,
                                   __global float* c) {
            int gid = get_global_id(0);
            
            // Heavy computation to stress GPU
            float result = 0.0f;
            for (int i = 0; i < 1000; i++) {
                result += sin(a[gid] * i) * cos(b[gid] * i);
                result = sqrt(fabs(result));
            }
            c[gid] = result;
        }
        """
        
        program = cl.Program(context, kernel_code).build()
        kernel = program.heavy_compute
        
        print("*** GPU WORKLOAD STARTING - CHECK TASK MANAGER NOW ***")
        print("Running heavy computation for 10 seconds...")
        
        # Run computation multiple times to stress GPU
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < 10.0:  # Run for 10 seconds
            kernel(queue, (size,), None, a_buf, b_buf, c_buf)
            queue.finish()
            iterations += 1
            
            if iterations % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  {iterations} iterations completed ({elapsed:.1f}s)")
        
        # Read result
        c = np.empty_like(a)
        cl.enqueue_copy(queue, c, c_buf)
        
        total_time = time.time() - start_time
        print(f"\n✓ GPU computation completed!")
        print(f"  Total iterations: {iterations}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Performance: {iterations/total_time:.1f} iterations/sec")
        print(f"  Device used: {target_device.name}")
        
        print("\n*** CHECK TASK MANAGER - Did GPU 0 (RX 7700S) show activity? ***")
        
        return True
        
    except ImportError:
        print("PyOpenCL not available")
        return False
    except Exception as e:
        print(f"GPU test failed: {e}")
        return False

def test_cpu_fallback():
    """Test CPU computation as fallback"""
    print("\n=== CPU Fallback Test ===")
    
    try:
        size = 1024 * 1024  # 1M elements
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        
        print(f"CPU processing {size} elements...")
        start_time = time.time()
        
        # Heavy CPU computation
        result = np.zeros_like(a)
        for i in range(100):  # Reduced iterations for CPU
            result += np.sin(a * i) * np.cos(b * i)
            result = np.sqrt(np.abs(result))
        
        cpu_time = time.time() - start_time
        print(f"✓ CPU computation completed in {cpu_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"CPU test failed: {e}")
        return False

if __name__ == "__main__":
    print("Simple GPU Test - Force RX 7700S Usage")
    print("=" * 50)
    print("This test will stress your RX 7700S GPU for 10 seconds")
    print("Watch Task Manager > Performance > GPU during the test")
    print()
    
    input("Press Enter to start GPU test...")
    
    gpu_success = test_rx7700s_usage()
    cpu_success = test_cpu_fallback()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"GPU Test: {'✓ PASS' if gpu_success else '✗ FAIL'}")
    print(f"CPU Test: {'✓ PASS' if cpu_success else '✗ FAIL'}")
    
    if gpu_success:
        print("\n RX 7700S GPU test successful!")
        print("   Your discrete GPU should have shown activity in Task Manager")
    else:
        print("\n RX 7700S GPU test failed")
        print("   Check your AMD drivers and OpenCL installation")
    
    input("\nPress Enter to exit...")