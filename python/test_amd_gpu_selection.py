#!/usr/bin/env python3
"""
Test AMD GPU Selection After Configuration
"""

import pyopencl as cl
import numpy as np
import time

def find_target_gpu():
    """Find the target GPU (RX 7700S) and return its device info"""
    platforms = cl.get_platforms()
    target_device = None
    
    for i, platform in enumerate(platforms):
        devices = platform.get_devices()
        for j, device in enumerate(devices):
            device_name = device.name.lower()
            # Look for RX 7700S specifically or AMD GPU codenames
            # gfx1103 is typically RX 7700S, gfx1102 is typically 780M
            if ('rx' in device_name and '7700' in device_name) or 'gfx1103' in device_name:
                target_device = {
                    'platform_id': i,
                    'device_id': j,
                    'name': device.name,
                    'type': cl.device_type.to_string(device.type),
                    'memory': device.global_mem_size // (1024**3),  # GB
                    'compute_units': device.max_compute_units,
                    'max_freq': device.max_clock_frequency,
                    'device': device,
                    'platform': platform
                }
                break
        if target_device:
            break
    
    return target_device

def test_gpu_selection():
    """Test which GPU is being used for OpenCL operations"""
    print("=== AMD GPU Selection Test ===")
    
    # Get all OpenCL platforms and devices
    platforms = cl.get_platforms()
    
    print(f"\nFound {len(platforms)} OpenCL platform(s):")
    
    all_devices = []
    for i, platform in enumerate(platforms):
        print(f"\nPlatform {i}: {platform.name}")
        devices = platform.get_devices()
        
        for j, device in enumerate(devices):
            device_info = {
                'platform_id': i,
                'device_id': j,
                'name': device.name,
                'type': cl.device_type.to_string(device.type),
                'memory': device.global_mem_size // (1024**3),  # GB
                'compute_units': device.max_compute_units,
                'max_freq': device.max_clock_frequency,
                'device': device,
                'platform': platform
            }
            all_devices.append(device_info)
            
            print(f"  Device {j}: {device.name}")
            print(f"    Type: {device_info['type']}")
            print(f"    Memory: {device_info['memory']} GB")
            print(f"    Compute Units: {device_info['compute_units']}")
            print(f"    Max Frequency: {device_info['max_freq']} MHz")
    
    if not all_devices:
        print("No OpenCL devices found!")
        return
    
    # Find target GPU (RX 7700S)
    target_gpu = find_target_gpu()
    if target_gpu:
        print(f"\n TARGET GPU FOUND: {target_gpu['name']}")
        print(f"   Platform: {target_gpu['platform_id']}, Device: {target_gpu['device_id']}")
        print(f"   Memory: {target_gpu['memory']} GB")
        print(f"   Compute Units: {target_gpu['compute_units']}")
    else:
        print(f"\n  TARGET GPU NOT FOUND: Looking for RX 7700S")
        print(f"   Available devices:")
        for device in all_devices:
            print(f"   - {device['name']}")
    
    # Test performance on each device
    print(f"\n=== Performance Test ===")
    
    # Create test data
    size = 1024 * 1024  # 1M elements
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    
    results = []
    
    # Test target GPU first if found
    if target_gpu:
        print(f"\n Testing TARGET GPU: {target_gpu['name']}")
        test_device_performance(target_gpu, a, b, size, results)
    
    # Test all other devices
    for device_info in all_devices:
        # Skip if already tested as target GPU
        if target_gpu and (device_info['platform_id'] == target_gpu['platform_id'] and 
                          device_info['device_id'] == target_gpu['device_id']):
            continue
        test_device_performance(device_info, a, b, size, results)

def test_device_performance(device_info, a, b, size, results):
    """Test performance on a specific device"""
    device = device_info['device']
    print(f"\nTesting {device.name}...")
    
    try:
        # Create context and queue
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        # Create buffers
        a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
        
        # Simple kernel for vector addition
        kernel_code = """
        __kernel void vector_add(__global const float* a,
                               __global const float* b,
                               __global float* c) {
            int gid = get_global_id(0);
            c[gid] = a[gid] + b[gid];
        }
        """
        
        program = cl.Program(context, kernel_code).build()
        kernel = program.vector_add
        
        # Time the operation
        start_time = time.time()
        
        # Execute kernel
        kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
        queue.finish()
        
        # Read result
        c = np.empty_like(a)
        cl.enqueue_copy(queue, c, c_buffer)
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # ms
        
        results.append({
            'device': device.name,
            'time_ms': execution_time,
            'performance': size / execution_time,  # operations per ms
            'is_target': ('rx' in device.name.lower() and '7700' in device.name.lower()) or 'gfx1103' in device.name.lower()
        })
        
        print(f"  ✓ Execution time: {execution_time:.2f} ms")
        print(f"  ✓ Performance: {size/execution_time:.0f} ops/ms")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append({
            'device': device.name,
            'time_ms': float('inf'),
            'performance': 0,
            'is_target': ('rx' in device.name.lower() and '7700' in device.name.lower()) or 'gfx1103' in device.name.lower()
        })
    
    # Show results
    print(f"\n=== Results Summary ===")
    results.sort(key=lambda x: x['time_ms'])
    
    for i, result in enumerate(results):
        status = " FASTEST" if i == 0 else f"#{i+1}"
        target_indicator = "  TARGET" if result.get('is_target', False) else ""
        print(f"{status}: {result['device']}{target_indicator}")
        if result['time_ms'] != float('inf'):
            print(f"  Time: {result['time_ms']:.2f} ms")
            print(f"  Performance: {result['performance']:.0f} ops/ms")
        else:
            print(f"  Status: FAILED")
    
    # Check if target GPU is being used
    fastest = results[0] if results else None
    target_gpu_result = next((r for r in results if r.get('is_target', False)), None)
    
    if target_gpu_result and target_gpu_result['time_ms'] != float('inf'):
        if fastest and fastest.get('is_target', False):
            print(f"\n SUCCESS: RX 7700S is the fastest GPU!")
            print(f"Your AMD GPU selection is working correctly.")
        else:
            print(f"\n  WARNING: RX 7700S found but not fastest")
            print(f"Target GPU performance: {target_gpu_result['time_ms']:.2f} ms")
            print(f"Fastest GPU: {fastest['device']} ({fastest['time_ms']:.2f} ms)")
            print(f"Consider system-level GPU configuration.")
    elif target_gpu_result:
        print(f"\n ERROR: RX 7700S found but failed to execute")
    else:
        print(f"\n ERROR: RX 7700S not found in available devices")
        print(f"Available devices:")
        for device in results:
            print(f"  - {device['device']}")
    
    # Provide configuration recommendations
    print(f"\n=== Configuration Recommendations ===")
    print(f"1. Windows Graphics Settings:")
    print(f"   - Win + I → System → Display → Graphics settings")
    print(f"   - Add python.exe and set to 'High performance'")
    print(f"2. AMD Radeon Settings:")
    print(f"   - Right-click desktop → AMD Radeon Settings")
    print(f"   - System → Switchable Graphics → Set python.exe to High Performance")
    print(f"3. Environment Variable (if supported):")
    print(f"   - Set AMD_OPENCL_DEVICE=1 (try different values)")
    print(f"4. Programmatic Selection:")
    print(f"   - Use the find_target_gpu() function to select specific device")

if __name__ == "__main__":
    test_gpu_selection()