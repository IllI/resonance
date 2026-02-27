#!/usr/bin/env python3
"""
Test GPU identification to ensure RX 7700S is correctly assigned as GPU 1.
"""

from enhanced_dual_gpu_analyzer import EnhancedDualGPUBrainAnalyzer
import pyopencl as cl
import numpy as np
import time

def test_gpu_identification():
    """Test which GPU gets assigned as GPU 1 (should be RX 7700S)."""
    print("TESTING GPU IDENTIFICATION AND CONTEXT ASSIGNMENT")
    print("=" * 60)
    
    analyzer = EnhancedDualGPUBrainAnalyzer()
    
    if analyzer.gpu1_context:
        # Get device info for GPU 1
        device = analyzer.gpu1_context.devices[0]
        device_name = device.get_info(cl.device_info.NAME)
        memory_gb = device.get_info(cl.device_info.GLOBAL_MEM_SIZE) // (1024**3)
        
        print(f"GPU 1 Context Device: {device_name}")
        print(f"GPU 1 Memory: {memory_gb} GB")
        
        if "gfx1103" in device_name and memory_gb >= 12:
            print("SUCCESS: RX 7700S (gfx1103) correctly assigned as GPU 1")
            return True
        else:
            print("FAILED: Wrong GPU assigned as GPU 1")
            return False
    else:
        print("FAILED: GPU 1 context not initialized")
        return False

def force_rx7700s_workload():
    """Force heavy workload on RX 7700S to test Task Manager visibility."""
    print("\nFORCING HEAVY WORKLOAD ON RX 7700S")
    print("*** WATCH TASK MANAGER - GPU 1 should spike ***")
    
    analyzer = EnhancedDualGPUBrainAnalyzer()
    
    if not analyzer.gpu1_context:
        print("ERROR: RX 7700S context not available")
        return False
    
    # Heavy workload specifically on GPU 1 (RX 7700S)
    context = analyzer.gpu1_context
    queue = analyzer.gpu1_queue
    
    # Create massive dataset
    workload_size = 16 * 1024 * 1024  # 16M elements
    data = np.random.rand(workload_size).astype(np.float32)
    
    print(f"Processing {workload_size:,} elements on RX 7700S...")
    
    # Heavy computation kernel
    kernel_code = """
    __kernel void max_gpu_utilization(__global const float* input,
                                     __global float* output,
                                     const int size) {
        int idx = get_global_id(0);
        if (idx < size) {
            float result = input[idx];
            
            // Maximum GPU utilization loop
            for(int iter = 0; iter < 2000; iter++) {
                result = sqrt(fabs(result) + 0.001f);
                result = sin(result) * cos(result * 2.0f);
                result = exp(result * 0.001f);
                result = log(fabs(result) + 0.001f);
                result = pow(fabs(result) + 0.001f, 1.1f);
            }
            
            output[idx] = result;
        }
    }
    """
    
    # Execute on RX 7700S
    mf = cl.mem_flags
    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, data.nbytes)
    
    program = cl.Program(context, kernel_code).build()
    kernel = program.max_gpu_utilization
    
    start_time = time.time()
    
    print("Launching maximum GPU workload...")
    kernel(queue, (workload_size,), None, input_buf, output_buf, np.int32(workload_size))
    queue.finish()
    
    processing_time = time.time() - start_time
    
    print(f"RX 7700S processing completed in {processing_time:.3f} seconds")
    print(f"Throughput: {workload_size/processing_time/1e6:.2f} M elements/sec")
    
    return True

if __name__ == "__main__":
    # Test 1: GPU identification
    identification_success = test_gpu_identification()
    
    # Test 2: Force RX 7700S utilization
    if identification_success:
        force_rx7700s_workload()
    
    print("\n" + "=" * 60)
    print("GPU identification test completed")
    if identification_success:
        print("SUCCESS: RX 7700S should now be properly targeted for heavy workloads")
    else:
        print("FAILED: GPU identification failed - needs debugging")
