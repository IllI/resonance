#!/usr/bin/env python3
"""
Final GPU Force Solution
Complete solution to force RX 7700S usage in all scenarios
"""

import pyopencl as cl
import numpy as np
import time
import os
import sys

class FinalGPUForceSolution:
    """Complete solution for forcing discrete GPU usage"""
    
    def __init__(self):
        self.discrete_device = None
        self.discrete_platform = None
        self.setup_environment()
        self.find_discrete_gpu()
    
    def setup_environment(self):
        """Set all possible environment variables"""
        print(" Setting comprehensive environment variables...")
        
        # AMD-specific variables
        amd_vars = {
            'AmdPowerXpressRequestHighPerformance': '1',
            'AMD_OPENCL_DEVICE': '1',
            'OPENCL_DEVICE': '1',
            'GPU_DEVICE_ORDINAL': '1',
            'AMD_VULKAN_DEVICE': '1',
            'AMD_GPU_DEVICE': '1',
            'DISPLAY_GPU': '1'
        }
        
        for var, value in amd_vars.items():
            os.environ[var] = value
            print(f"   {var}={value}")
        
        # Set PyOpenCL context to use discrete GPU
        os.environ['PYOPENCL_CTX'] = '0:1'  # Platform 0, Device 1 (gfx1103)
        print(f"   PYOPENCL_CTX=0:1 (Platform 0, Device 1)")
    
    def find_discrete_gpu(self):
        """Find and store the discrete GPU"""
        print("\n Finding discrete GPU...")
        
        platforms = cl.get_platforms()
        
        for i, platform in enumerate(platforms):
            devices = platform.get_devices()
            for j, device in enumerate(devices):
                device_name = device.name.lower()
                
                # Look for RX 7700S (gfx1103)
                if 'gfx1103' in device_name:
                    self.discrete_device = device
                    self.discrete_platform = platform
                    print(f" Found discrete GPU: {device.name}")
                    print(f"   Platform: {i}, Device: {j}")
                    print(f"   Memory: {device.global_mem_size // (1024**3)} GB")
                    print(f"   Compute Units: {device.max_compute_units}")
                    return
        
        print(" Discrete GPU not found!")
    
    def create_discrete_context(self):
        """Create OpenCL context explicitly with discrete GPU"""
        if not self.discrete_device:
            raise RuntimeError("Discrete GPU not found!")
        
        print(f"\n Creating context with {self.discrete_device.name}...")
        
        try:
            context = cl.Context([self.discrete_device])
            queue = cl.CommandQueue(context)
            print(f" Successfully created discrete GPU context")
            return context, queue
        except Exception as e:
            raise RuntimeError(f"Failed to create context: {e}")
    
    def test_default_context(self):
        """Test if default context creation now uses discrete GPU"""
        print(f"\n Testing default context creation...")
        
        try:
            # This should now use the discrete GPU due to PYOPENCL_CTX
            context = cl.create_some_context()
            device = context.devices[0]
            
            print(f" Default context using: {device.name}")
            
            if device == self.discrete_device:
                print(f"🎉 SUCCESS: Default context is using discrete GPU!")
                return True
            else:
                print(f"  Default context using: {device.name} (not discrete)")
                return False
                
        except Exception as e:
            print(f" Default context creation failed: {e}")
            return False
    
    def test_explicit_context(self):
        """Test explicit discrete GPU context creation"""
        print(f"\n Testing explicit discrete GPU context...")
        
        try:
            context, queue = self.create_discrete_context()
            
            # Test with a simple workload
            size = 1024 * 1024
            a = np.random.rand(size).astype(np.float32)
            b = np.random.rand(size).astype(np.float32)
            
            # Create buffers
            a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
            b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
            c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
            
            # Simple kernel
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
            
            print(f" Explicit context test successful!")
            print(f"   Device: {self.discrete_device.name}")
            print(f"   Time: {execution_time:.2f} ms")
            print(f"   Performance: {performance:.0f} ops/ms")
            
            return True
            
        except Exception as e:
            print(f" Explicit context test failed: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all methods"""
        print("="*60)
        print("COMPREHENSIVE GPU FORCE TEST")
        print("="*60)
        
        if not self.discrete_device:
            print(" Cannot proceed - discrete GPU not found")
            return False
        
        # Test 1: Default context
        default_success = self.test_default_context()
        
        # Test 2: Explicit context
        explicit_success = self.test_explicit_context()
        
        # Overall result
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print("="*60)
        
        if default_success and explicit_success:
            print("🎉 COMPLETE SUCCESS!")
            print("   Both default and explicit contexts use discrete GPU")
            print("   Your applications will now use RX 7700S by default")
            return True
        elif explicit_success:
            print(" PARTIAL SUCCESS!")
            print("   Explicit context works - use create_discrete_context()")
            print("   Default context may need additional configuration")
            return True
        else:
            print(" FAILED!")
            print("   Neither method is working")
            print("   Check AMD drivers and system configuration")
            return False

def main():
    """Main function"""
    print("=== Final GPU Force Solution ===")
    print("Complete solution for forcing RX 7700S usage")
    
    # Create solution
    solution = FinalGPUForceSolution()
    
    # Run comprehensive test
    success = solution.run_comprehensive_test()
    
    if success:
        print(f"\n{'='*60}")
        print("USAGE IN YOUR CODE:")
        print("="*60)
        print("from final_gpu_force_solution import FinalGPUForceSolution")
        print("")
        print("# Initialize once at startup")
        print("gpu_force = FinalGPUForceSolution()")
        print("")
        print("# Method 1: Use default context (should now work)")
        print("context = cl.create_some_context()")
        print("")
        print("# Method 2: Use explicit context (guaranteed to work)")
        print("context, queue = gpu_force.create_discrete_context()")
        print("")
        print("# Method 3: Use the GPU selector utility")
        print("from amd_gpu_selector import AMDGPUSelector")
        print("selector = AMDGPUSelector()")
        print("context, queue, device = selector.force_target_gpu()")
    else:
        print(f"\n{'='*60}")
        print("TROUBLESHOOTING:")
        print("="*60)
        print("1. Update AMD drivers to latest version")
        print("2. Set Windows power plan to 'High Performance'")
        print("3. Configure AMD Radeon Settings:")
        print("   - Right-click desktop → AMD Radeon Settings")
        print("   - System → Switchable Graphics")
        print("   - Add python.exe → Set to High Performance")
        print("4. Configure Windows Graphics Settings:")
        print("   - Win + I → System → Display → Graphics settings")
        print("   - Add python.exe → Set to High Performance")

if __name__ == "__main__":
    main()

