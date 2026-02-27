#!/usr/bin/env python3
"""
Comprehensive GPU Test - Better workload to demonstrate discrete GPU advantages
"""

import pyopencl as cl
import numpy as np
import time
from amd_gpu_selector import AMDGPUSelector

def create_compute_intensive_kernel():
    """Create a more compute-intensive kernel that benefits from discrete GPU"""
    kernel_code = """
    __kernel void matrix_multiply(__global const float* A,
                                 __global const float* B,
                                 __global float* C,
                                 const int N) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        
        if (row < N && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    
    __kernel void vector_operations(__global const float* a,
                                   __global const float* b,
                                   __global float* c,
                                   const int size) {
        int gid = get_global_id(0);
        if (gid < size) {
            // Multiple operations to stress the GPU
            float temp = a[gid] * b[gid];
            temp = sqrt(temp);
            temp = sin(temp) + cos(temp);
            temp = temp * temp;
            c[gid] = temp;
        }
    }
    """
    return kernel_code

def test_gpu_with_workload(device_info, workload_type="matrix"):
    """Test GPU performance with different workloads"""
    device = device_info['device']
    print(f"\n Testing {device.name} with {workload_type} workload...")
    
    try:
        # Create context and queue
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        if workload_type == "matrix":
            # Matrix multiplication test
            N = 512  # 512x512 matrix
            size = N * N
            
            # Create test matrices
            A = np.random.rand(size).astype(np.float32)
            B = np.random.rand(size).astype(np.float32)
            C = np.zeros(size, dtype=np.float32)
            
            # Create buffers
            A_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
            B_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
            C_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes)
            
            # Build program
            program = cl.Program(context, create_compute_intensive_kernel()).build()
            kernel = program.matrix_multiply
            
            # Time the operation
            start_time = time.time()
            kernel(queue, (N, N), None, A_buffer, B_buffer, C_buffer, np.int32(N))
            queue.finish()
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000  # ms
            operations = N * N * N * 2  # Multiply and add for each element
            performance = operations / execution_time  # ops/ms
            
        else:  # vector operations
            size = 1024 * 1024 * 4  # 4M elements
            
            # Create test vectors
            a = np.random.rand(size).astype(np.float32)
            b = np.random.rand(size).astype(np.float32)
            c = np.zeros(size, dtype=np.float32)
            
            # Create buffers
            a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
            b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
            c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, c.nbytes)
            
            # Build program
            program = cl.Program(context, create_compute_intensive_kernel()).build()
            kernel = program.vector_operations
            
            # Time the operation
            start_time = time.time()
            kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer, np.int32(size))
            queue.finish()
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000  # ms
            operations = size * 6  # Multiple operations per element
            performance = operations / execution_time  # ops/ms
        
        result = {
            'device': device.name,
            'time_ms': execution_time,
            'performance': performance,
            'is_target': device_info.get('is_target', False),
            'memory': device_info['memory'],
            'compute_units': device_info['compute_units'],
            'max_freq': device_info['max_freq']
        }
        
        print(f"  ✓ Execution time: {execution_time:.2f} ms")
        print(f"  ✓ Performance: {performance:.0f} ops/ms")
        print(f"  ✓ Memory: {device_info['memory']} GB")
        print(f"  ✓ Compute Units: {device_info['compute_units']}")
        print(f"  ✓ Max Frequency: {device_info['max_freq']} MHz")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            'device': device.name,
            'time_ms': float('inf'),
            'performance': 0,
            'is_target': device_info.get('is_target', False),
            'error': str(e)
        }

def main():
    """Main comprehensive GPU test"""
    print("=== Comprehensive GPU Performance Test ===")
    print("This test uses more compute-intensive workloads to better demonstrate")
    print("the advantages of discrete GPUs over integrated ones.\n")
    
    selector = AMDGPUSelector()
    
    # List devices
    selector.list_devices()
    
    # Get target GPU
    target_gpu = selector.find_target_gpu()
    if not target_gpu:
        print(" Target GPU (RX 7700S) not found!")
        return
    
    print(f"\n Target GPU: {target_gpu['name']}")
    print(f"   Memory: {target_gpu['memory']} GB")
    print(f"   Compute Units: {target_gpu['compute_units']}")
    print(f"   Max Frequency: {target_gpu['max_freq']} MHz")
    
    # Test different workloads
    workloads = ["vector", "matrix"]
    results = {}
    
    for workload in workloads:
        print(f"\n{'='*50}")
        print(f"Testing {workload.upper()} workload")
        print(f"{'='*50}")
        
        workload_results = []
        
        # Test target GPU first
        target_result = test_gpu_with_workload(target_gpu, workload)
        workload_results.append(target_result)
        
        # Test other devices
        for device_info in selector.devices:
            if device_info['is_target']:
                continue
            result = test_gpu_with_workload(device_info, workload)
            workload_results.append(result)
        
        results[workload] = workload_results
        
        # Show results for this workload
        print(f"\n--- {workload.upper()} Workload Results ---")
        workload_results.sort(key=lambda x: x['time_ms'])
        
        for i, result in enumerate(workload_results):
            status = " FASTEST" if i == 0 else f"#{i+1}"
            target_indicator = "  TARGET" if result.get('is_target', False) else ""
            print(f"{status}: {result['device']}{target_indicator}")
            if result['time_ms'] != float('inf'):
                print(f"  Time: {result['time_ms']:.2f} ms")
                print(f"  Performance: {result['performance']:.0f} ops/ms")
            else:
                print(f"  Status: FAILED - {result.get('error', 'Unknown error')}")
    
    # Overall analysis
    print(f"\n{'='*50}")
    print("OVERALL ANALYSIS")
    print(f"{'='*50}")
    
    target_performance = {}
    for workload, workload_results in results.items():
        target_result = next((r for r in workload_results if r.get('is_target', False)), None)
        if target_result and target_result['time_ms'] != float('inf'):
            target_performance[workload] = target_result['performance']
    
    if target_performance:
        print(f" Target GPU (RX 7700S) performance:")
        for workload, perf in target_performance.items():
            print(f"   {workload.capitalize()}: {perf:.0f} ops/ms")
        
        # Check if target GPU is fastest in any workload
        fastest_in_any = False
        for workload, workload_results in results.items():
            if workload_results and workload_results[0].get('is_target', False):
                fastest_in_any = True
                break
        
        if fastest_in_any:
            print(f"\n🎉 SUCCESS: RX 7700S is fastest in at least one workload!")
            print(f"   The discrete GPU shows its advantages with compute-intensive tasks.")
        else:
            print(f"\n  RX 7700S found but not fastest in any workload.")
            print(f"   This might indicate:")
            print(f"   - Power management limiting discrete GPU performance")
            print(f"   - Workload not complex enough to show discrete GPU advantages")
            print(f"   - System configuration needed to force discrete GPU usage")
    else:
        print(f" Target GPU failed to execute any workload")
    
    # Recommendations
    print(f"\n{'='*50}")
    print("RECOMMENDATIONS")
    print(f"{'='*50}")
    print("1. For better discrete GPU performance:")
    print("   - Set Windows power plan to 'High Performance'")
    print("   - Configure GPU in AMD Radeon Settings")
    print("   - Use the amd_gpu_selector.py utility in your code")
    print("2. For complex workloads:")
    print("   - Use larger matrix sizes (1024x1024 or higher)")
    print("   - Implement multi-kernel pipelines")
    print("   - Utilize GPU memory bandwidth advantages")

if __name__ == "__main__":
    main()

