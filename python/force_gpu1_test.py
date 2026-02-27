#!/usr/bin/env python3
"""
Force GPU 1 Usage Test

This script forces heavy usage on GPU 1 (RX 7700S) to verify it shows up in Task Manager.
"""

import numpy as np
import time
import sys
import os

def force_gpu1_heavy_usage():
    """Force heavy usage on GPU 1 to make it visible in Task Manager."""
    print("Force GPU 1 (RX 7700S) Heavy Usage Test")
    print("=" * 50)
    
    try:
        import pyopencl as cl
        
        # Find and FORCE select GPU 1 (RX 7700S) by memory size
        platforms = cl.get_platforms()
        gpu1_device = None
        
        for platform in platforms:
            if 'AMD' in platform.name:
                devices = platform.get_devices(cl.device_type.GPU)
                print(f"Found {len(devices)} AMD GPU devices:")
                
                # Sort devices by memory size (largest first)
                devices_with_memory = []
                for i, device in enumerate(devices):
                    device_name = device.name.strip()
                    memory_gb = device.global_mem_size // (1024**3)
                    devices_with_memory.append((device, i, device_name, memory_gb))
                    print(f"   GPU {i}: {device_name} ({memory_gb} GB)")
                
                # Sort by memory size (descending) to get the discrete GPU first
                devices_with_memory.sort(key=lambda x: x[3], reverse=True)
                
                # Select the GPU with the most memory (should be RX 7700S)
                if devices_with_memory:
                    gpu1_device, original_index, device_name, memory_gb = devices_with_memory[0]
                    print(f"   ✓ FORCE SELECTED LARGEST GPU: {device_name} ({memory_gb} GB)")
                    print(f"   Original index: {original_index}")
                break
        
        if gpu1_device is None:
            print("ERROR: GPU 1 not found")
            return False
        
        # Create context for GPU 1
        context = cl.Context([gpu1_device])
        queue = cl.CommandQueue(context)
        
        print(f"\nCreated GPU 1 context: {gpu1_device.name.strip()}")
        print("Starting heavy workload...")
        print("*** WATCH TASK MANAGER - GPU 1 should show HIGH USAGE ***")
        
        # Create massive workload
        workload_size = 2000000  # 2 million operations
        input_data = np.random.rand(workload_size).astype(np.float32)
        
        # Very heavy computation kernel
        heavy_kernel = """
        __kernel void extreme_gpu1_workload(__global const float* input,
                                           __global float* output,
                                           const int size) {
            int idx = get_global_id(0);
            if (idx < size) {
                float result = input[idx];
                // EXTREMELY heavy mathematical operations
                for(int i = 0; i < 200; i++) {
                    result = sqrt(result + 0.1f);
                    result = sin(result) * cos(result);
                    result = exp(result * 0.005f);
                    result = log(result + 1.0f);
                    result = pow(result, 1.1f);
                }
                output[idx] = result;
            }
        }
        """
        
        # Execute heavy workload on GPU 1
        mf = cl.mem_flags
        input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
        output_buf = cl.Buffer(context, mf.WRITE_ONLY, input_data.nbytes)
        
        program = cl.Program(context, heavy_kernel).build()
        kernel = program.extreme_gpu1_workload
        
        print("\nExecuting heavy workload on GPU 1...")
        print("Duration: 30 seconds of continuous heavy processing")
        
        start_time = time.time()
        
        # Run for 30 seconds with heavy load
        iterations = 0
        while time.time() - start_time < 30:
            kernel(queue, (workload_size,), None, input_buf, output_buf, np.int32(workload_size))
            queue.finish()
            iterations += 1
            
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] GPU 1 heavy workload iteration {iterations} - CHECK TASK MANAGER")
            
        total_time = time.time() - start_time
        print(f"\nHeavy workload completed!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Iterations: {iterations}")
        print(f"Operations: {iterations * workload_size:,}")
        
        print("\n*** GPU 1 (RX 7700S) should have shown HIGH USAGE in Task Manager ***")
        return True
        
    except ImportError:
        print("ERROR: OpenCL not available")
        print("Install with: pip install pyopencl")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("This test will force heavy usage on GPU 1 (RX 7700S)")
    print("Open Task Manager and watch GPU 1 usage during the test")
    print()
    
    input("Press Enter to start the heavy GPU 1 workload test...")
    
    success = force_gpu1_heavy_usage()
    
    if success:
        print("\n GPU 1 heavy workload test completed successfully!")
        print("If GPU 1 didn't show usage in Task Manager, there may be a driver issue.")
    else:
        print("\n GPU 1 heavy workload test failed!")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()