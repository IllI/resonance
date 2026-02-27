#!/usr/bin/env python3
"""
Direct GPU 0 Test - Force RX 7700S usage with intensive workload
This creates a workload that should definitely show up in Task Manager
"""

import pyopencl as cl
import numpy as np
import time
import os
import sys

def force_gpu0_usage():
    """Force GPU 0 (RX 7700S) usage with intensive workload"""
    print("=== Direct GPU 0 (RX 7700S) Test ===")
    print("Creating intensive workload to force GPU 0 usage")
    
    # Set environment variables
    os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
    os.environ['AMD_OPENCL_DEVICE'] = '0'  # Try device 0 first
    os.environ['OPENCL_DEVICE'] = '0'
    
    print(" Set environment variables for GPU 0")
    
    # Get platforms and devices
    platforms = cl.get_platforms()
    
    for i, platform in enumerate(platforms):
        print(f"\nPlatform {i}: {platform.name}")
        devices = platform.get_devices()
        
        for j, device in enumerate(devices):
            device_name = device.name
            print(f"  Device {j}: {device_name}")
            
            # Try to use each device and see which one shows activity
            try:
                print(f"\n Testing Device {j}: {device_name}")
                print("    Watch Task Manager - this should show GPU activity")
                
                # Create context
                context = cl.Context([device])
                queue = cl.CommandQueue(context)
                
                # Create very large workload
                size = 1024 * 1024 * 8  # 8M elements
                print(f"    Creating {size:,} element workload...")
                
                a = np.random.rand(size).astype(np.float32)
                b = np.random.rand(size).astype(np.float32)
                
                # Create buffers
                a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
                
                # Very intensive kernel
                kernel_code = """
                __kernel void intensive_compute(__global const float* a,
                                              __global const float* b,
                                              __global float* c) {
                    int gid = get_global_id(0);
                    float temp = a[gid] * b[gid];
                    
                    // Very intensive computation
                    for(int i = 0; i < 10000; i++) {
                        temp = sqrt(temp);
                        temp = sin(temp) + cos(temp);
                        temp = temp * temp;
                        temp = log(temp + 1.0f);
                        temp = exp(temp);
                    }
                    c[gid] = temp;
                }
                """
                
                program = cl.Program(context, kernel_code).build()
                kernel = program.intensive_compute
                
                print(f"    Running intensive computation on {device_name}...")
                print(f"    Check Task Manager NOW - GPU activity should be visible")
                
                # Run multiple iterations to ensure GPU activity
                for iteration in range(5):
                    print(f"    Iteration {iteration + 1}/5...")
                    
                    start_time = time.time()
                    kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
                    queue.finish()
                    end_time = time.time()
                    
                    execution_time = (end_time - start_time) * 1000
                    print(f"      Time: {execution_time:.2f} ms")
                    
                    # Small delay to see activity in Task Manager
                    time.sleep(0.5)
                
                print(f"    Completed intensive workload on {device_name}")
                print(f"    Check Task Manager - this GPU should have shown activity")
                
                # Ask user which GPU showed activity
                print(f"\n❓ Which GPU showed activity in Task Manager?")
                print(f"   - GPU 0 (RX 7700S) - 12GB memory")
                print(f"   - GPU 1 (780M) - 7GB memory")
                
                return True
                
            except Exception as e:
                print(f"    Failed to test {device_name}: {e}")
                continue
    
    return False

def test_specific_gpu(device_index):
    """Test a specific GPU by index"""
    print(f"\n Testing specific GPU {device_index}...")
    
    platforms = cl.get_platforms()
    
    for platform in platforms:
        devices = platform.get_devices()
        if device_index < len(devices):
            device = devices[device_index]
            print(f"Testing Device {device_index}: {device.name}")
            
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
                    
                    for(int i = 0; i < 5000; i++) {
                        temp = sqrt(temp);
                        temp = sin(temp) + cos(temp);
                        temp = temp * temp;
                    }
                    c[gid] = temp;
                }
                """
                
                program = cl.Program(context, kernel_code).build()
                kernel = program.test_kernel
                
                print(f" Running workload on {device.name}")
                print(" Check Task Manager NOW!")
                
                for i in range(3):
                    print(f"   Iteration {i+1}/3...")
                    kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
                    queue.finish()
                    time.sleep(1)
                
                print(f" Completed test on {device.name}")
                return True
                
            except Exception as e:
                print(f" Test failed: {e}")
                return False
    
    return False

def main():
    """Main function"""
    print("=== Direct GPU 0 Test ===")
    print("This will create intensive workloads to test which GPU is actually used")
    print("Watch Task Manager while this runs!")
    
    # Test all devices
    success = force_gpu0_usage()
    
    if success:
        print(f"\n{'='*60}")
        print("ANALYSIS:")
        print("="*60)
        print("1. Check which GPU showed activity in Task Manager")
        print("2. If GPU 1 (780M) showed activity:")
        print("   - The system is still using integrated GPU")
        print("   - Need to configure Windows Graphics Settings")
        print("3. If GPU 0 (RX 7700S) showed activity:")
        print("   - Success! Discrete GPU is being used")
        print("   - Your brain analysis will use RX 7700S")
        
        print(f"\n{'='*60}")
        print("MANUAL CONFIGURATION (if needed):")
        print("="*60)
        print("1. Win + I → System → Display → Graphics settings")
        print("2. Click 'Browse' and find python.exe")
        print("3. Set to 'High performance'")
        print("4. Restart Python/IDE")
        print("5. Run this test again")

if __name__ == "__main__":
    main()

