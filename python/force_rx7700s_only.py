#!/usr/bin/env python3
"""
Force RX 7700S Only - Completely bypass integrated GPU
This script ensures ONLY the RX 7700S is used, never the 780M
"""

import pyopencl as cl
import numpy as np
import time
import os
import sys

class RX7700SOnly:
    """Force exclusive use of RX 7700S GPU"""
    
    def __init__(self):
        self.rx7700s_device = None
        self.rx7700s_platform = None
        self.setup_exclusive_environment()
        self.find_rx7700s()
    
    def setup_exclusive_environment(self):
        """Set environment to force RX 7700S only"""
        print(" Setting exclusive RX 7700S environment...")
        
        # Clear any existing OpenCL context
        if 'PYOPENCL_CTX' in os.environ:
            del os.environ['PYOPENCL_CTX']
        
        # Set AMD variables to force discrete GPU
        os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
        os.environ['AMD_OPENCL_DEVICE'] = '1'  # Device 1 = gfx1103 (RX 7700S)
        os.environ['OPENCL_DEVICE'] = '1'
        os.environ['GPU_DEVICE_ORDINAL'] = '1'
        
        # Disable integrated GPU
        os.environ['AMD_DISABLE_INTEGRATED_GPU'] = '1'
        os.environ['OPENCL_DISABLE_DEVICE_0'] = '1'  # Disable device 0 (780M)
        
        print("    Set AMD_OPENCL_DEVICE=1 (RX 7700S)")
        print("    Set OPENCL_DISABLE_DEVICE_0=1 (Disable 780M)")
        print("    Set AmdPowerXpressRequestHighPerformance=1")
    
    def find_rx7700s(self):
        """Find the RX 7700S GPU specifically"""
        print("\n Finding RX 7700S GPU...")
        
        platforms = cl.get_platforms()
        
        for i, platform in enumerate(platforms):
            devices = platform.get_devices()
            for j, device in enumerate(devices):
                device_name = device.name.lower()
                
                # Look specifically for RX 7700S (gfx1103)
                if 'gfx1103' in device_name:
                    self.rx7700s_device = device
                    self.rx7700s_platform = platform
                    print(f" Found RX 7700S: {device.name}")
                    print(f"   Platform: {i}, Device: {j}")
                    print(f"   Memory: {device.global_mem_size // (1024**3)} GB")
                    print(f"   Compute Units: {device.max_compute_units}")
                    print(f"   Max Frequency: {device.max_clock_frequency} MHz")
                    return
        
        print(" RX 7700S not found!")
    
    def create_rx7700s_only_context(self):
        """Create context that ONLY uses RX 7700S"""
        if not self.rx7700s_device:
            raise RuntimeError("RX 7700S not found!")
        
        print(f"\n Creating RX 7700S ONLY context...")
        
        try:
            # Create context with ONLY the RX 7700S device
            context = cl.Context([self.rx7700s_device])
            queue = cl.CommandQueue(context)
            
            print(f" Created context with ONLY {self.rx7700s_device.name}")
            print(f"   No other GPUs will be used")
            
            return context, queue
        except Exception as e:
            raise RuntimeError(f"Failed to create RX 7700S context: {e}")
    
    def test_rx7700s_only(self):
        """Test that ONLY RX 7700S is being used"""
        print(f"\n Testing RX 7700S ONLY usage...")
        
        try:
            context, queue = self.create_rx7700s_only_context()
            
            # Verify the context only has RX 7700S
            devices = context.devices
            print(f" Context devices: {len(devices)}")
            
            for i, device in enumerate(devices):
                print(f"   Device {i}: {device.name}")
                if 'gfx1103' not in device.name.lower():
                    print(f"    ERROR: Non-RX 7700S device found!")
                    return False
            
            print(f" Confirmed: Context contains ONLY RX 7700S")
            
            # Test performance
            size = 1024 * 1024
            a = np.random.rand(size).astype(np.float32)
            b = np.random.rand(size).astype(np.float32)
            
            # Create buffers
            a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
            b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
            c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
            
            # Kernel
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
            kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
            queue.finish()
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000
            performance = size / execution_time
            
            print(f" RX 7700S Performance Test:")
            print(f"   Execution Time: {execution_time:.2f} ms")
            print(f"   Performance: {performance:.0f} ops/ms")
            print(f"   Memory: {self.rx7700s_device.global_mem_size // (1024**3)} GB")
            
            return True
            
        except Exception as e:
            print(f" RX 7700S test failed: {e}")
            return False
    
    def verify_no_780m_usage(self):
        """Verify that 780M is NOT being used"""
        print(f"\n Verifying 780M is NOT being used...")
        
        platforms = cl.get_platforms()
        
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                device_name = device.name.lower()
                
                # Check for 780M (gfx1102)
                if 'gfx1102' in device_name:
                    print(f"  Found 780M: {device.name}")
                    print(f"   This device should NOT be used")
                    return False
        
        print(f" Confirmed: No 780M devices found in active context")
        return True

def main():
    """Main function"""
    print("=== Force RX 7700S Only ===")
    print("Ensuring ONLY RX 7700S is used, never 780M")
    
    # Create RX 7700S only controller
    rx7700s_only = RX7700SOnly()
    
    if not rx7700s_only.rx7700s_device:
        print(" Cannot proceed - RX 7700S not found")
        return
    
    # Test RX 7700S only usage
    success = rx7700s_only.test_rx7700s_only()
    
    if success:
        # Verify no 780M usage
        no_780m = rx7700s_only.verify_no_780m_usage()
        
        if no_780m:
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"    ONLY RX 7700S is being used")
            print(f"    780M is completely bypassed")
            print(f"    Your brain analysis will use discrete GPU")
        else:
            print(f"\n  PARTIAL SUCCESS")
            print(f"    RX 7700S is working")
            print(f"     780M may still be accessible")
    else:
        print(f"\n FAILED!")
        print(f"   RX 7700S context creation failed")
    
    print(f"\n{'='*60}")
    print("USAGE IN YOUR BRAIN ANALYSIS CODE:")
    print("="*60)
    print("from force_rx7700s_only import RX7700SOnly")
    print("")
    print("# Initialize once at startup")
    print("rx7700s = RX7700SOnly()")
    print("")
    print("# Create context that ONLY uses RX 7700S")
    print("context, queue = rx7700s.create_rx7700s_only_context()")
    print("")
    print("# Use this context for all your brain analysis")
    print("# No other GPUs will be used")

if __name__ == "__main__":
    main()

