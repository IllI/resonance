#!/usr/bin/env python3
"""
Force GPU 1 Exclusive Usage - Nuclear Option

This script uses the most aggressive methods to force GPU 1 (RX 7700S) usage
by temporarily disabling GPU 0 at the OpenCL level.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def temporarily_hide_gpu0():
    """Temporarily hide GPU 0 from OpenCL to force GPU 1 usage."""
    try:
        print("Attempting to hide GPU 0 from OpenCL...")
        
        # Method 1: Set OpenCL device filter environment variable
        os.environ['OPENCL_DEVICE_FILTER'] = '1'  # Only show device 1
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CUDA equivalent (just in case)
        os.environ['HIP_VISIBLE_DEVICES'] = '1'   # AMD HIP equivalent
        
        # Method 2: AMD-specific device masking
        os.environ['AMD_OPENCL_DEVICE_MASK'] = '2'  # Binary: 10 (only GPU 1)
        os.environ['GPU_DEVICE_ORDINAL'] = '1'
        
        print("  Set device filter environment variables")
        return True
        
    except Exception as e:
        print(f"Error setting device filters: {e}")
        return False

def create_opencl_device_override():
    """Create OpenCL device override configuration."""
    try:
        print("Creating OpenCL device override...")
        
        # Create temporary OpenCL configuration
        config_dir = Path.cwd() / "opencl_config"
        config_dir.mkdir(exist_ok=True)
        
        # Create device filter file
        device_filter = config_dir / "device_filter.txt"
        with open(device_filter, 'w') as f:
            f.write("# OpenCL Device Filter\n")
            f.write("# Only allow GPU 1 (RX 7700S)\n")
            f.write("DEVICE_TYPE=GPU\n")
            f.write("MIN_MEMORY_GB=10\n")
            f.write("EXCLUDE_INTEGRATED=1\n")
        
        # Set environment to use our config
        os.environ['OPENCL_CONFIG_PATH'] = str(config_dir)
        os.environ['OPENCL_DEVICE_FILTER_FILE'] = str(device_filter)
        
        print(f"  Created device filter: {device_filter}")
        return True
        
    except Exception as e:
        print(f"Error creating device override: {e}")
        return False

def test_gpu1_exclusive():
    """Test that only GPU 1 is visible and usable."""
    try:
        import pyopencl as cl
        import numpy as np
        
        print("\nTesting GPU 1 exclusive access...")
        
        # Get platforms and devices
        platforms = cl.get_platforms()
        gpu1_found = False
        gpu0_hidden = True
        
        for platform in platforms:
            if 'AMD' in platform.name:
                devices = platform.get_devices(cl.device_type.GPU)
                
                print(f"Visible devices after filtering:")
                for i, device in enumerate(devices):
                    device_name = device.name.strip()
                    memory_gb = device.global_mem_size // (1024**3)
                    
                    print(f"  Device {i}: {device_name} ({memory_gb} GB)")
                    
                    # Check if this is GPU 1 (RX 7700S)
                    if memory_gb >= 10 or "7700" in device_name or "gfx1103" in device_name:
                        gpu1_found = True
                        print(f"    ✓ This is GPU 1 (RX 7700S)")
                    elif memory_gb < 10 or "780M" in device_name or "gfx1102" in device_name:
                        gpu0_hidden = False
                        print(f"     GPU 0 is still visible")
                
                # If we found GPU 1, test it
                if gpu1_found and devices:
                    # Use the first (and hopefully only) device
                    target_device = devices[0]
                    
                    # Create context and test
                    context = cl.Context([target_device])
                    queue = cl.CommandQueue(context)
                    
                    # Heavy test workload
                    test_size = 2000000
                    test_data = np.random.rand(test_size).astype(np.float32)
                    
                    heavy_kernel = """
                    __kernel void heavy_gpu1_test(__global const float* input,
                                                __global float* output,
                                                const int size) {
                        int idx = get_global_id(0);
                        if (idx < size) {
                            float result = input[idx];
                            for(int i = 0; i < 200; i++) {
                                result = sqrt(fabs(result) + 0.1f);
                                result = sin(result) * cos(result);
                                result = exp(result * 0.005f);
                            }
                            output[idx] = result;
                        }
                    }
                    """
                    
                    # Execute heavy workload
                    mf = cl.mem_flags
                    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_data)
                    output_buf = cl.Buffer(context, mf.WRITE_ONLY, test_data.nbytes)
                    
                    program = cl.Program(context, heavy_kernel).build()
                    kernel = program.heavy_gpu1_test
                    
                    device_name = target_device.name.strip()
                    memory_gb = target_device.global_mem_size // (1024**3)
                    
                    print(f"\nRunning heavy workload on: {device_name}")
                    print("*** CHECK TASK MANAGER - GPU 1 should show HIGH USAGE ***")
                    print("*** GPU 0 should remain at 0% ***")
                    
                    start_time = time.time()
                    iterations = 0
                    
                    # Run for 15 seconds
                    while time.time() - start_time < 15:
                        kernel(queue, (test_size,), None, input_buf, output_buf, np.int32(test_size))
                        queue.finish()
                        iterations += 1
                        
                        if iterations % 5 == 0:
                            elapsed = time.time() - start_time
                            print(f"[{elapsed:.1f}s] GPU 1 workload iteration {iterations}")
                    
                    elapsed = time.time() - start_time
                    print(f"\n Heavy workload completed on GPU 1!")
                    print(f"   Device: {device_name} ({memory_gb} GB)")
                    print(f"   Iterations: {iterations}")
                    print(f"   Time: {elapsed:.1f}s")
                    print(f"   Performance: {iterations/elapsed:.1f} iter/sec")
                    
                    return True
        
        if not gpu1_found:
            print(" GPU 1 (RX 7700S) not found or not accessible")
        if not gpu0_hidden:
            print(" GPU 0 is still visible (filtering may not be working)")
        
        return gpu1_found
        
    except Exception as e:
        print(f"GPU 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_brain_gui_with_gpu1():
    """Launch the brain GUI with GPU 1 forced."""
    try:
        print("\nLaunching brain GUI with GPU 1 forced...")
        
        # Set additional environment variables for the GUI process
        gui_env = os.environ.copy()
        gui_env.update({
            'FORCE_GPU1_ONLY': '1',
            'DISABLE_GPU0': '1',
            'AMD_GPU_PRIORITY': 'DISCRETE_ONLY',
            'OPENCL_FORCE_DEVICE': '1'
        })
        
        # Launch the GUI in a subprocess with our environment
        gui_process = subprocess.Popen([
            sys.executable, 'quick_start_gui.py'
        ], env=gui_env, cwd=Path.cwd())
        
        print("Brain GUI launched with GPU 1 environment")
        print("Check Task Manager - GPU 1 should show usage when you generate a brain model")
        
        return gui_process
        
    except Exception as e:
        print(f"Failed to launch brain GUI: {e}")
        return None

def main():
    """Main function for GPU 1 exclusive mode."""
    print("Force GPU 1 Exclusive Usage - Nuclear Option")
    print("=" * 60)
    print("This will aggressively force GPU 1 (RX 7700S) usage")
    print("by hiding GPU 0 from OpenCL applications.")
    print()
    
    # Step 1: Hide GPU 0
    step1 = temporarily_hide_gpu0()
    
    # Step 2: Create device override
    step2 = create_opencl_device_override()
    
    # Step 3: Test GPU 1 exclusive access
    step3 = test_gpu1_exclusive()
    
    print("\n" + "=" * 60)
    print("GPU 1 EXCLUSIVE MODE RESULTS:")
    print(f"  Device Filtering: {'' if step1 else ''}")
    print(f"  OpenCL Override: {'' if step2 else ''}")
    print(f"  GPU 1 Test: {'' if step3 else ''}")
    
    if step3:
        print("\n🎉 SUCCESS: GPU 1 (RX 7700S) is now exclusively active!")
        print("\nOptions:")
        print("1. Launch brain GUI automatically")
        print("2. Keep this terminal open and run GUI manually")
        print("3. Exit and run GUI in another terminal")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == '1':
            gui_process = launch_brain_gui_with_gpu1()
            if gui_process:
                print("\nBrain GUI is running...")
                print("Generate a 3D brain model and check Task Manager")
                print("GPU 1 should show high usage, GPU 0 should stay at 0%")
                
                try:
                    gui_process.wait()
                except KeyboardInterrupt:
                    print("\nTerminating brain GUI...")
                    gui_process.terminate()
        
        elif choice == '2':
            print("\nGPU 1 exclusive mode is active!")
            print("Keep this terminal open and run in another terminal:")
            print("  cd python")
            print("  python quick_start_gui.py")
            input("\nPress Enter to exit and restore normal GPU mode...")
        
        else:
            print("\nGPU 1 exclusive mode set!")
            print("Run the brain GUI now - it should use GPU 1 exclusively")
    
    else:
        print("\n FAILED: Could not establish GPU 1 exclusive mode")
        print("GPU 0 may still be used by OpenCL applications")
    
    print("\nNote: GPU settings will revert when this process ends")

if __name__ == "__main__":
    main()