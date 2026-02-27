#!/usr/bin/env python3
"""
Direct GPU 1 Test - Windows Specific

Uses Windows-specific methods to force GPU 1 (RX 7700S) usage.
This bypasses OpenCL's default GPU selection.
"""

import os
import sys
import time

def set_windows_gpu_preference():
    """Set Windows GPU preference to high-performance (discrete GPU)."""
    try:
        # Set environment variables to prefer discrete GPU
        os.environ['GPU_FORCE_64BIT_PTR'] = '1'
        os.environ['GPU_MAX_HEAP_SIZE'] = '100'
        os.environ['GPU_USE_SYNC_OBJECTS'] = '1'
        os.environ['GPU_MAX_ALLOC_PERCENT'] = '100'
        os.environ['GPU_SINGLE_ALLOC_PERCENT'] = '100'
        
        # AMD-specific environment variables
        os.environ['AMD_VULKAN_ICD'] = 'DISABLE'  # Force OpenCL over Vulkan
        os.environ['OPENCL_VENDOR_PATH'] = r'C:\Windows\System32\DriverStore\FileRepository'
        
        print("Set Windows GPU preference environment variables")
        return True
    except Exception as e:
        print(f"Error setting GPU preferences: {e}")
        return False

def force_discrete_gpu_selection():
    """Force selection of discrete GPU using device properties."""
    try:
        import pyopencl as cl
        import numpy as np
        
        print("Direct GPU 1 (RX 7700S) Test")
        print("=" * 40)
        
        # Get all platforms and devices
        platforms = cl.get_platforms()
        all_devices = []
        
        for platform in platforms:
            if 'AMD' in platform.name:
                devices = platform.get_devices(cl.device_type.GPU)
                for i, device in enumerate(devices):
                    device_info = {
                        'device': device,
                        'platform': platform,
                        'index': i,
                        'name': device.name.strip(),
                        'memory': device.global_mem_size // (1024**3),
                        'compute_units': device.max_compute_units,
                        'max_work_group_size': device.max_work_group_size,
                        'vendor': device.vendor.strip()
                    }
                    all_devices.append(device_info)
                    
                    print(f"GPU {i}: {device_info['name']}")
                    print(f"   Memory: {device_info['memory']} GB")
                    print(f"   Compute Units: {device_info['compute_units']}")
                    print(f"   Max Work Group: {device_info['max_work_group_size']}")
                    print()
        
        # Select discrete GPU (highest memory + compute units)
        if not all_devices:
            print("ERROR: No AMD GPUs found")
            return False
        
        # Sort by memory first, then compute units
        discrete_gpu = max(all_devices, key=lambda x: (x['memory'], x['compute_units']))
        
        print(f"Selected discrete GPU: {discrete_gpu['name']}")
        print(f"Memory: {discrete_gpu['memory']} GB")
        print(f"Compute Units: {discrete_gpu['compute_units']}")
        
        # Create context with ONLY the discrete GPU
        context = cl.Context([discrete_gpu['device']])
        queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        print(f"\nCreated exclusive context for: {discrete_gpu['name']}")
        print("Starting intensive workload...")
        print("*** WATCH TASK MANAGER - Discrete GPU should show HIGH USAGE ***")
        
        # Create massive workload to force GPU usage
        workload_size = 5000000  # 5 million operations
        input_data = np.random.rand(workload_size).astype(np.float32)
        
        # Extremely heavy kernel
        intensive_kernel = """
        __kernel void intensive_discrete_gpu_work(__global const float* input,
                                                __global float* output,
                                                const int size) {
            int idx = get_global_id(0);
            if (idx < size) {
                float result = input[idx];
                // MASSIVE computation to force GPU usage
                for(int i = 0; i < 500; i++) {
                    result = sqrt(fabs(result) + 0.1f);
                    result = sin(result) * cos(result);
                    result = exp(result * 0.001f);
                    result = log(fabs(result) + 1.0f);
                    result = pow(fabs(result), 1.01f);
                    result = atan(result) * tan(result * 0.1f);
                }
                output[idx] = result;
            }
        }
        """
        
        # Execute on discrete GPU
        mf = cl.mem_flags
        input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)
        output_buf = cl.Buffer(context, mf.WRITE_ONLY, input_data.nbytes)
        
        program = cl.Program(context, intensive_kernel).build()
        kernel = program.intensive_discrete_gpu_work
        
        print("\nExecuting intensive workload on discrete GPU...")
        print("Duration: 60 seconds of maximum GPU load")
        
        start_time = time.time()
        iterations = 0
        
        # Run for 60 seconds with maximum load
        while time.time() - start_time < 60:
            event = kernel(queue, (workload_size,), None, input_buf, output_buf, np.int32(workload_size))
            event.wait()
            iterations += 1
            
            elapsed = time.time() - start_time
            if iterations % 5 == 0:  # Print every 5 iterations
                print(f"[{elapsed:.1f}s] Iteration {iterations} - GPU: {discrete_gpu['name']}")
        
        total_time = time.time() - start_time
        print(f"\nIntensive workload completed!")
        print(f"GPU Used: {discrete_gpu['name']}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Iterations: {iterations}")
        print(f"Total operations: {iterations * workload_size:,}")
        
        # Check if it was the high-memory GPU (RX 7700S)
        if discrete_gpu['memory'] >= 10:
            print(f"\n SUCCESS: Used discrete GPU with {discrete_gpu['memory']} GB memory")
            print("This should be the RX 7700S showing usage in Task Manager")
        else:
            print(f"\n WARNING: Used GPU with only {discrete_gpu['memory']} GB memory")
            print("This might still be the integrated GPU")
        
        return True
        
    except ImportError:
        print("ERROR: OpenCL not available")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Direct GPU 1 (RX 7700S) Force Test")
    print("This test will force usage of the discrete GPU")
    print("Open Task Manager and watch GPU usage")
    print()
    
    # Set Windows GPU preferences
    set_windows_gpu_preference()
    
    input("Press Enter to start the intensive discrete GPU test...")
    
    success = force_discrete_gpu_selection()
    
    if success:
        print("\n Discrete GPU test completed!")
        print("Check Task Manager - the discrete GPU should have shown high usage")
    else:
        print("\n Discrete GPU test failed!")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()