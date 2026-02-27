#!/usr/bin/env python3
"""
Force Windows GPU 0 (RX 7700S) Test
This test specifically targets what Windows calls "GPU 0" in Task Manager
"""

import os
import time
import numpy as np

# Set environment to force discrete GPU
os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
os.environ['GPU_FORCE_64BIT_PTR'] = '1'
os.environ['GPU_MAX_HEAP_SIZE'] = '100'
os.environ['GPU_USE_SYNC_OBJECTS'] = '1'

def force_windows_gpu0():
    """Force Windows GPU 0 (RX 7700S) activation"""
    print("=== FORCE WINDOWS GPU 0 (RX 7700S) TEST ===")
    print("*** WATCH TASK MANAGER - GPU 0 SHOULD ACTIVATE, NOT GPU 1 ***")
    
    try:
        import pyopencl as cl
        
        # Get all platforms and devices
        platforms = cl.get_platforms()
        print("OpenCL Platform and Device Mapping:")
        
        all_devices = []
        for p_idx, platform in enumerate(platforms):
            if 'AMD' in platform.name:
                print(f"Platform {p_idx}: {platform.name}")
                devices = platform.get_devices()
                
                for d_idx, device in enumerate(devices):
                    name = device.name.strip()
                    memory_gb = device.global_mem_size // (1024**3)
                    compute_units = device.max_compute_units
                    
                    device_info = {
                        'platform': p_idx,
                        'device': d_idx,
                        'name': name,
                        'memory_gb': memory_gb,
                        'compute_units': compute_units,
                        'cl_device': device
                    }
                    all_devices.append(device_info)
                    
                    print(f"  OpenCL Device {d_idx}: {name}")
                    print(f"    Memory: {memory_gb} GB")
                    print(f"    Compute Units: {compute_units}")
                    
                    # Identify which is which
                    if 'gfx1103' in name or memory_gb >= 12:
                        print(f"    *** THIS IS RX 7700S (Windows GPU 0) ***")
                    elif 'gfx1102' in name or compute_units >= 16:
                        print(f"    *** THIS IS 780M (Windows GPU 1) ***")
        
        # Find RX 7700S device
        rx7700s_device = None
        for dev in all_devices:
            if 'gfx1103' in dev['name'] or dev['memory_gb'] >= 12:
                rx7700s_device = dev
                break
        
        if not rx7700s_device:
            print("RX 7700S not found!")
            return False
        
        print(f"\nTargeting RX 7700S: {rx7700s_device['name']}")
        print("This should activate Windows GPU 0 in Task Manager")
        
        # Create context with RX 7700S
        context = cl.Context([rx7700s_device['cl_device']])
        queue = cl.CommandQueue(context)
        
        # Create MASSIVE workload to ensure GPU 0 activation
        size = 16 * 1024 * 1024  # 16M elements
        print(f"Creating MASSIVE workload: {size:,} elements")
        
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        
        # Create buffers
        a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
        
        # EXTREME workload kernel
        kernel_code = """
        __kernel void extreme_gpu_load(__global const float* a,
                                      __global const float* b,
                                      __global float* c) {
            int gid = get_global_id(0);
            
            // EXTREME computation to max out GPU
            float result = 0.0f;
            for (int i = 0; i < 10000; i++) {  // 10,000 iterations per element!
                float temp1 = sin(a[gid] * i * 0.001f);
                float temp2 = cos(b[gid] * i * 0.001f);
                float temp3 = tan(temp1 * temp2);
                result += sqrt(fabs(temp3));
                result = fmod(result + exp(temp1 * 0.01f), 1000.0f);
            }
            c[gid] = result;
        }
        """
        
        program = cl.Program(context, kernel_code).build()
        kernel = program.extreme_gpu_load
        
        print("\n STARTING EXTREME GPU 0 WORKLOAD ")
        print("*** WINDOWS GPU 0 (RX 7700S) SHOULD HIT 100% NOW ***")
        print("*** GPU 1 (780M) SHOULD STAY AT 0% ***")
        print("Running for 20 seconds...")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < 20.0:  # 20 seconds of extreme load
            current_time = time.time() - start_time
            print(f"  {current_time:.1f}s - GPU 0 should be at 100% (iteration {iteration+1})")
            
            # Execute extreme kernel
            kernel(queue, (size,), None, a_buf, b_buf, c_buf)
            queue.finish()
            
            iteration += 1
            time.sleep(1.0)  # 1 second pause to clearly see GPU usage
        
        # Read result to ensure completion
        c = np.empty_like(a)
        cl.enqueue_copy(queue, c, c_buf)
        
        total_time = time.time() - start_time
        total_operations = size * iteration * 10000  # elements * iterations * kernel loops
        
        print(f"\n EXTREME GPU 0 WORKLOAD COMPLETED!")
        print(f"  Device used: {rx7700s_device['name']}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Iterations: {iteration}")
        print(f"  Total operations: {total_operations:,}")
        print(f"  Operations/sec: {total_operations/total_time:.0f}")
        
        print(f"\n*** RESULT CHECK ***")
        print(f"Did Windows GPU 0 (RX 7700S) show 100% usage?")
        print(f"Did Windows GPU 1 (780M) stay at 0%?")
        
        return True
        
    except ImportError:
        print("PyOpenCL not available!")
        return False
    except Exception as e:
        print(f"GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_both_gpus_separately():
    """Test both GPUs separately to see Windows mapping"""
    print("\n=== TESTING BOTH GPUS SEPARATELY ===")
    
    try:
        import pyopencl as cl
        
        platforms = cl.get_platforms()
        amd_platform = None
        
        for platform in platforms:
            if 'AMD' in platform.name:
                amd_platform = platform
                break
        
        if not amd_platform:
            print("No AMD platform found!")
            return
        
        devices = amd_platform.get_devices()
        
        for i, device in enumerate(devices):
            name = device.name.strip()
            memory_gb = device.global_mem_size // (1024**3)
            
            print(f"\n--- Testing OpenCL Device {i}: {name} ---")
            
            if 'gfx1103' in name:
                print("This should activate Windows GPU 0 (RX 7700S)")
            elif 'gfx1102' in name:
                print("This should activate Windows GPU 1 (780M)")
            
            # Create context for this specific device
            context = cl.Context([device])
            queue = cl.CommandQueue(context)
            
            # Moderate workload
            size = 2 * 1024 * 1024  # 2M elements
            a = np.random.rand(size).astype(np.float32)
            
            a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
            b_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
            
            kernel_code = """
            __kernel void test_load(__global const float* a, __global float* b) {
                int gid = get_global_id(0);
                float result = 0.0f;
                for (int i = 0; i < 1000; i++) {
                    result += sin(a[gid] * i);
                }
                b[gid] = result;
            }
            """
            
            program = cl.Program(context, kernel_code).build()
            kernel = program.test_load
            
            print(f"Running 5-second test on {name}...")
            print("*** WATCH TASK MANAGER NOW ***")
            
            start_time = time.time()
            while time.time() - start_time < 5.0:
                kernel(queue, (size,), None, a_buf, b_buf)
                queue.finish()
                time.sleep(0.1)
            
            print(f"Test completed for {name}")
            time.sleep(2)  # Pause between tests
            
    except Exception as e:
        print(f"Error in separate GPU test: {e}")

if __name__ == "__main__":
    print(" FORCE WINDOWS GPU 0 (RX 7700S) TEST ")
    print("=" * 60)
    print("This test will definitively activate Windows GPU 0")
    print("Watch Task Manager > Performance > GPU during test")
    print("GPU 0 should hit 100%, GPU 1 should stay at 0%")
    print()
    
    input("Press Enter to start GPU 0 test...")
    
    success = force_windows_gpu0()
    
    print("\n" + "=" * 60)
    if success:
        print(" GPU 0 test completed!")
    else:
        print(" GPU 0 test failed!")
    
    print("\nWould you like to test both GPUs separately to confirm mapping?")
    response = input("Type 'y' for yes, any other key to exit: ")
    
    if response.lower() == 'y':
        test_both_gpus_separately()
    
    input("\nPress Enter to exit...")