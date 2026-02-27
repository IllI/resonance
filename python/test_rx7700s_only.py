#!/usr/bin/env python3
"""
Test RX 7700S-only approach to ensure integrated GPU is completely ignored.
"""

from enhanced_dual_gpu_analyzer import EnhancedDualGPUBrainAnalyzer
import numpy as np
import time

def test_rx7700s_exclusive():
    """Test that only RX 7700S is used - no integrated GPU."""
    print("TESTING RX 7700S EXCLUSIVE APPROACH")
    print("=" * 60)
    
    analyzer = EnhancedDualGPUBrainAnalyzer()
    
    # Verify only RX 7700S context exists
    print("\nGPU Context Status:")
    print(f"  Main GPU context: {'ACTIVE' if analyzer.gpu_context else 'INACTIVE'}")
    print(f"  GPU 0 (780M): {'ACTIVE' if analyzer.gpu0_context else 'IGNORED'}")
    print(f"  GPU 1 (RX 7700S): {'ACTIVE' if analyzer.gpu1_context else 'IGNORED'}")
    
    if analyzer.gpu_context and not analyzer.gpu0_context:
        print("\nSUCCESS: Only RX 7700S context active, integrated GPU ignored")
        return True
    else:
        print("\nFAILED: Dual-GPU approach still active")
        return False

def force_massive_rx7700s_workload():
    """Force maximum workload exclusively on RX 7700S."""
    print("\nFORCING MASSIVE RX 7700S WORKLOAD")
    print("*** WATCH TASK MANAGER - ONLY GPU 1 should spike, GPU 0 should be idle ***")
    
    analyzer = EnhancedDualGPUBrainAnalyzer()
    
    if not analyzer.gpu_context:
        print("ERROR: RX 7700S context not available")
        return False
    
    # Use the main RX 7700S context
    context = analyzer.gpu_context
    queue = analyzer.gpu_queue
    
    # Create enormous workload - 32M elements
    workload_size = 32 * 1024 * 1024
    data = np.random.rand(workload_size).astype(np.float32)
    
    print(f"Processing {workload_size:,} elements EXCLUSIVELY on RX 7700S...")
    
    # Maximum intensity kernel
    kernel_code = """
    __kernel void exclusive_rx7700s_processing(__global const float* input,
                                              __global float* output,
                                              const int size) {
        int idx = get_global_id(0);
        if (idx < size) {
            float result = input[idx];
            
            // Extreme computational load
            for(int iter = 0; iter < 3000; iter++) {
                result = sqrt(fabs(result) + 0.0001f);
                result = sin(result) * cos(result * 3.14159f);
                result = exp(result * 0.0001f);
                result = log(fabs(result) + 0.0001f);
                result = pow(fabs(result) + 0.0001f, 1.2f);
                result = fabs(result * 1.1f);
            }
            
            output[idx] = result;
        }
    }
    """
    
    # Execute ONLY on RX 7700S
    import pyopencl as cl
    mf = cl.mem_flags
    
    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, data.nbytes)
    
    program = cl.Program(context, kernel_code).build()
    kernel = program.exclusive_rx7700s_processing
    
    print("Launching EXCLUSIVE RX 7700S workload...")
    print("GPU 0 (780M) should remain at 0%")
    print("GPU 1 (RX 7700S) should spike to 100%")
    
    start_time = time.time()
    kernel(queue, (workload_size,), None, input_buf, output_buf, np.int32(workload_size))
    queue.finish()
    processing_time = time.time() - start_time
    
    print(f"\nRX 7700S EXCLUSIVE processing completed!")
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Throughput: {workload_size/processing_time/1e6:.2f} M elements/sec")
    
    return True

def test_brain_processing_rx7700s_only():
    """Test brain processing using RX 7700S exclusively."""
    print("\nTESTING BRAIN PROCESSING - RX 7700S ONLY")
    print("=" * 60)
    
    analyzer = EnhancedDualGPUBrainAnalyzer()
    
    # Create test brain volume
    brain_volume = np.random.rand(128, 128, 64).astype(np.float32)
    print(f"Test brain volume: {brain_volume.shape}")
    
    # Process using RX 7700S only
    start_time = time.time()
    result = analyzer.process_brain_volume_gpu(brain_volume, use_discrete_gpu=True)
    processing_time = time.time() - start_time
    
    if result and 'brain_surface' in result:
        vertices = len(result['brain_surface']['vertices'])
        faces = len(result['brain_surface']['faces'])
        
        print(f"SUCCESS: Brain processed exclusively on RX 7700S")
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Generated: {vertices} vertices, {faces} faces")
        return True
    else:
        print("FAILED: Brain processing failed")
        return False

if __name__ == "__main__":
    print("RX 7700S EXCLUSIVE TESTING")
    print("Integrated GPU (780M) will be completely ignored")
    
    # Test 1: Verify exclusive RX 7700S setup
    exclusive_test = test_rx7700s_exclusive()
    
    # Test 2: Force massive workload on RX 7700S only
    if exclusive_test:
        workload_test = force_massive_rx7700s_workload()
        
        # Test 3: Brain processing
        brain_test = test_brain_processing_rx7700s_only()
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS:")
        print(f"RX 7700S Exclusive Setup: {'PASS' if exclusive_test else 'FAIL'}")
        print(f"Massive Workload Test: {'PASS' if workload_test else 'FAIL'}")
        print(f"Brain Processing Test: {'PASS' if brain_test else 'FAIL'}")
        
        if exclusive_test and workload_test and brain_test:
            print("\nSUCCESS: RX 7700S exclusive approach working!")
            print("   Integrated GPU (780M) should show 0% utilization")
            print("   RX 7700S should show high utilization")
        else:
            print("\nSome tests failed - check implementation")
    
    print("=" * 60)
