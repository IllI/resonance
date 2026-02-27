#!/usr/bin/env python3
"""
Automated GPU 0 Test - No User Input Required
"""

import os
import time
import numpy as np

# Force all GPU environment variables
os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
os.environ['GPU_FORCE_64BIT_PTR'] = '1'
os.environ['GPU_MAX_HEAP_SIZE'] = '100'
os.environ['DXGI_ADAPTER_INDEX'] = '0'

def auto_test_gpu0():
    """Automatically test GPU 0 without user input"""
    print(" AUTO GPU 0 TEST - RX 7700S ACTIVATION ")
    print("*** WATCH TASK MANAGER - GPU 0 SHOULD ACTIVATE ***")
    
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
            print(" No AMD platform found!")
            return False
        
        # Get devices
        devices = amd_platform.get_devices()
        print(f"Found {len(devices)} AMD devices:")
        
        rx7700s_device = None
        for i, device in enumerate(devices):
            name = device.name.strip()
            memory_gb = device.global_mem_size // (1024**3)
            
            print(f"  Device {i}: {name} ({memory_gb} GB)")
            
            # Select RX 7700S (gfx1103 with 12GB)
            if 'gfx1103' in name or memory_gb >= 12:
                rx7700s_device = device
                print(f"    *** SELECTED: RX 7700S ***")
        
        if not rx7700s_device:
            print(" RX 7700S not found!")
            return False
        
        # Create context
        context = cl.Context([rx7700s_device])
        queue = cl.CommandQueue(context)
        
        print(f"\n ACTIVATING {rx7700s_device.name} ")
        print("*** GPU 0 SHOULD SHOW ACTIVITY IN TASK MANAGER ***")
        
        # Create massive workload
        size = 20 * 1024 * 1024  # 20M elements
        a = np.random.rand(size).astype(np.float32)
        
        # Buffers
        a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
        
        # Extreme kernel
        kernel_code = """
        __kernel void extreme_load(__global const float* a, __global float* b) {
            int gid = get_global_id(0);
            
            float result = 0.0f;
            for (int i = 0; i < 15000; i++) {  // 15,000 iterations per element!
                result += sin(a[gid] * i * 0.0001f);
                result = sqrt(fabs(result));
            }
            b[gid] = result;
        }
        """
        
        program = cl.Program(context, kernel_code).build()
        kernel = program.extreme_load
        
        # Run for 10 seconds
        print("Running extreme workload for 10 seconds...")
        start_time = time.time()
        
        for i in range(10):
            print(f"  Second {i+1}/10 - GPU 0 should be at 100%")
            kernel(queue, (size,), None, a_buf, b_buf)
            queue.finish()
            time.sleep(1)
        
        # Read result
        b = np.empty_like(a)
        cl.enqueue_copy(queue, b, b_buf)
        
        total_time = time.time() - start_time
        total_ops = size * 10 * 15000  # elements * iterations * kernel loops
        
        print(f"\n EXTREME WORKLOAD COMPLETED!")
        print(f"  Device: {rx7700s_device.name}")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Operations: {total_ops:,}")
        print(f"  Ops/sec: {total_ops/total_time:.0f}")
        
        print(f"\n*** RESULT: Did GPU 0 activate in Task Manager? ***")
        
        return True
        
    except Exception as e:
        print(f" Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting automated GPU 0 test...")
    print("This will run for 10 seconds")
    print("Watch Task Manager > Performance > GPU 0")
    print()
    
    success = auto_test_gpu0()
    
    if success:
        print("\n Test completed successfully!")
    else:
        print("\n Test failed!")
    
    print("\nTest finished.")