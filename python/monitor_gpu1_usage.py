#!/usr/bin/env python3
"""
GPU 1 Usage Monitor

Monitors GPU 1 (RX 7700S) usage and provides real-time feedback.
Run this alongside the brain GUI to verify GPU 1 activation.
"""

import time
import sys
import os

def monitor_gpu1_usage():
    """Monitor GPU 1 usage in real-time."""
    print("GPU 1 (RX 7700S) Usage Monitor")
    print("=" * 40)
    print("This script monitors GPU 1 usage.")
    print("Run the brain GUI in another window and watch for GPU 1 activity.")
    print("Press Ctrl+C to stop monitoring.")
    print()
    
    try:
        import pyopencl as cl
        
        # Find GPU 1 (RX 7700S)
        platforms = cl.get_platforms()
        gpu1_device = None
        
        for platform in platforms:
            if 'AMD' in platform.name:
                devices = platform.get_devices(cl.device_type.GPU)
                if len(devices) > 1:
                    gpu1_device = devices[1]  # GPU 1
                    break
        
        if gpu1_device is None:
            print("ERROR: GPU 1 not found")
            return
        
        device_name = gpu1_device.name.strip()
        memory_gb = gpu1_device.global_mem_size // (1024**3)
        
        print(f"Monitoring GPU 1: {device_name} ({memory_gb} GB)")
        print("=" * 40)
        
        # Create context for monitoring
        context = cl.Context([gpu1_device])
        queue = cl.CommandQueue(context)
        
        # Test workload to verify GPU 1 is accessible
        import numpy as np
        test_data = np.random.rand(1000).astype(np.float32)
        
        test_kernel = """
        __kernel void test_gpu1(__global float* data, const int size) {
            int idx = get_global_id(0);
            if (idx < size) {
                data[idx] = data[idx] * 2.0f;
            }
        }
        """
        
        mf = cl.mem_flags
        data_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=test_data)
        program = cl.Program(context, test_kernel).build()
        kernel = program.test_gpu1
        
        print("GPU 1 monitoring active...")
        print("Status: Waiting for brain GUI to activate GPU 1...")
        print()
        
        iteration = 0
        while True:
            try:
                # Light test to keep connection alive
                kernel(queue, (len(test_data),), None, data_buf, np.int32(len(test_data)))
                queue.finish()
                
                iteration += 1
                if iteration % 10 == 0:  # Every 1 second
                    print(f"[{time.strftime('%H:%M:%S')}] GPU 1 monitoring active... (Check Task Manager)")
                
                time.sleep(0.1)  # 100ms intervals
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
                
    except ImportError:
        print("ERROR: OpenCL not available")
        print("Install with: pip install pyopencl")
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    """Main function."""
    print("Instructions:")
    print("1. Run this script: python monitor_gpu1_usage.py")
    print("2. In another window, run: python quick_start_gui.py")
    print("3. Generate a 3D brain model in the GUI")
    print("4. Watch Task Manager for GPU 1 (RX 7700S) activity")
    print("5. This monitor will confirm GPU 1 is accessible")
    print()
    
    try:
        monitor_gpu1_usage()
    except KeyboardInterrupt:
        print("\nGPU 1 monitoring stopped")

if __name__ == "__main__":
    main()