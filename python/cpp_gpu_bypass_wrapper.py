#!/usr/bin/env python3
"""
C++ GPU Bypass Wrapper - Python wrapper for C++ GPU bypass
This uses C++ to work around Windows GPU selection limitations
"""

import ctypes
import os
import sys
import subprocess
from pathlib import Path

class CppGPUBypassWrapper:
    """Python wrapper for C++ GPU bypass"""
    
    def __init__(self):
        self.dll = None
        self.load_dll()
    
    def load_dll(self):
        """Load the C++ GPU bypass DLL"""
        try:
            # Try to load the compiled DLL
            dll_path = Path(__file__).parent / "cpp_gpu_bypass.dll"
            if dll_path.exists():
                self.dll = ctypes.CDLL(str(dll_path))
                print(" Loaded C++ GPU Bypass DLL")
                self.setup_dll_functions()
            else:
                print("  C++ GPU Bypass DLL not found")
                print("   Compile cpp_gpu_bypass.cpp first:")
                print("   cl /LD cpp_gpu_bypass.cpp")
                self.create_fallback_export()
        except Exception as e:
            print(f"  Could not load DLL: {e}")
            self.create_fallback_export()
    
    def setup_dll_functions(self):
        """Setup DLL function signatures"""
        if self.dll:
            # RequestHighPerformanceGPU function
            self.dll.RequestHighPerformanceGPU.argtypes = []
            self.dll.RequestHighPerformanceGPU.restype = None
            
            # GetGPUPreference function
            self.dll.GetGPUPreference.argtypes = []
            self.dll.GetGPUPreference.restype = ctypes.c_int
            
            # SetWindowsGraphicsPreference function
            self.dll.SetWindowsGraphicsPreference.argtypes = [ctypes.c_char_p]
            self.dll.SetWindowsGraphicsPreference.restype = None
    
    def create_fallback_export(self):
        """Create fallback export using ctypes"""
        try:
            print(" Creating fallback AMD power export...")
            
            # Set environment variables as fallback
            os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
            os.environ['AMD_OPENCL_DEVICE'] = '0'
            os.environ['OPENCL_DEVICE'] = '0'
            os.environ['GPU_DEVICE_ORDINAL'] = '0'
            os.environ['AMD_DISABLE_INTEGRATED_GPU'] = '1'
            os.environ['OPENCL_DISABLE_DEVICE_1'] = '1'
            
            print(" Set AMD environment variables as fallback")
        except Exception as e:
            print(f"  Fallback export failed: {e}")
    
    def request_high_performance_gpu(self):
        """Request high performance GPU usage"""
        if self.dll:
            try:
                self.dll.RequestHighPerformanceGPU()
                print(" Requested high performance GPU via C++ DLL")
            except Exception as e:
                print(f"  DLL request failed: {e}")
        else:
            print(" Using environment variable fallback")
    
    def get_gpu_preference(self):
        """Get current GPU preference"""
        if self.dll:
            try:
                preference = self.dll.GetGPUPreference()
                print(f" GPU Preference: {preference}")
                return preference
            except Exception as e:
                print(f"  Could not get preference: {e}")
        return 1  # Default to high performance
    
    def set_windows_graphics_preference(self, app_path):
        """Set Windows Graphics preference for an application"""
        if self.dll:
            try:
                self.dll.SetWindowsGraphicsPreference(app_path.encode('utf-8'))
                print(f" Set Windows Graphics preference for: {app_path}")
            except Exception as e:
                print(f"  Could not set Windows Graphics preference: {e}")
    
    def test_gpu_usage(self):
        """Test which GPU is actually being used"""
        print("\n Testing GPU usage with C++ bypass...")
        
        # Request high performance GPU
        self.request_high_performance_gpu()
        
        # Set Windows Graphics preference for Python
        python_path = sys.executable
        self.set_windows_graphics_preference(python_path)
        
        try:
            import pyopencl as cl
            import numpy as np
            import time
            
            # Get all devices
            platforms = cl.get_platforms()
            
            for platform in platforms:
                devices = platform.get_devices()
                for i, device in enumerate(devices):
                    device_name = device.name
                    print(f"Device {i}: {device_name}")
                    
                    # Test each device
                    try:
                        context = cl.Context([device])
                        queue = cl.CommandQueue(context)
                        
                        # Large workload
                        size = 1024 * 1024 * 6
                        a = np.random.rand(size).astype(np.float32)
                        b = np.random.rand(size).astype(np.float32)
                        
                        a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                        b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                        c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
                        
                        kernel_code = """
                        __kernel void test_kernel(__global const float* a,
                                                __global const float* b,
                                                __global float* c) {
                            int gid = get_global_id(0);
                            float temp = a[gid] * b[gid];
                            
                            for(int i = 0; i < 50000; i++) {
                                temp = sqrt(temp);
                                temp = sin(temp) + cos(temp);
                                temp = temp * temp;
                            }
                            c[gid] = temp;
                        }
                        """
                        
                        program = cl.Program(context, kernel_code).build()
                        kernel = program.test_kernel
                        
                        print(f" Testing {device_name}...")
                        print(" Check Task Manager NOW!")
                        print("   GPU 0 (RX 7700S) should show activity")
                        print("   GPU 1 (780M) should NOT show activity")
                        
                        for iteration in range(5):
                            print(f"   Iteration {iteration+1}/5...")
                            kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
                            queue.finish()
                            time.sleep(2)
                        
                        print(f" Completed test on {device_name}")
                        print(" Check Task Manager - which GPU showed activity?")
                        
                    except Exception as e:
                        print(f" Test failed for {device_name}: {e}")
                        
        except Exception as e:
            print(f" GPU test failed: {e}")

def compile_cpp_dll():
    """Compile the C++ DLL"""
    print("🔨 Compiling C++ GPU Bypass DLL...")
    
    try:
        import subprocess
        
        # Compile the C++ file
        cmd = ['cl', '/LD', 'cpp_gpu_bypass.cpp']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" C++ DLL compiled successfully")
            print("   Created: cpp_gpu_bypass.dll")
        else:
            print(f" Compilation failed:")
            print(f"   Error: {result.stderr}")
            print(f"   Make sure Visual Studio Build Tools are installed")
            
    except FileNotFoundError:
        print(" Visual Studio compiler (cl.exe) not found")
        print("   Install Visual Studio Build Tools or Visual Studio")
        print("   Or manually compile: cl /LD cpp_gpu_bypass.cpp")

def main():
    """Main function"""
    print("=== C++ GPU Bypass Wrapper ===")
    print("Using C++ to work around Windows GPU selection limitations")
    
    # Try to compile DLL if it doesn't exist
    dll_path = Path(__file__).parent / "cpp_gpu_bypass.dll"
    if not dll_path.exists():
        print(" C++ GPU Bypass DLL not found, attempting to compile...")
        compile_cpp_dll()
    
    # Create wrapper
    wrapper = CppGPUBypassWrapper()
    
    # Test GPU usage
    wrapper.test_gpu_usage()
    
    print(f"\n{'='*60}")
    print("C++ GPU BYPASS RESULTS:")
    print("="*60)
    print("1. Check Task Manager while the test runs")
    print("2. If GPU 1 (780M) still shows activity:")
    print("   - C++ approach may not be sufficient")
    print("   - Try manual Windows Graphics Settings")
    print("   - Check display connections")
    print("3. If GPU 0 (RX 7700S) shows activity:")
    print("   - SUCCESS! C++ approach is working")
    print("   - Your brain analysis will use RX 7700S")
    
    print(f"\n{'='*60}")
    print("WHY C++ CAN HELP:")
    print("="*60)
    print("1. Direct hardware access")
    print("2. Bypass Windows GPU selection")
    print("3. Direct3D context creation")
    print("4. Registry manipulation")
    print("5. Environment variable control")
    
    print(f"\n{'='*60}")
    print("USAGE IN YOUR CODE:")
    print("="*60)
    print("from cpp_gpu_bypass_wrapper import CppGPUBypassWrapper")
    print("")
    print("# Initialize C++ GPU bypass")
    print("gpu_bypass = CppGPUBypassWrapper()")
    print("gpu_bypass.request_high_performance_gpu()")
    print("")
    print("# Then use OpenCL normally")
    print("import pyopencl as cl")
    print("context = cl.create_some_context()  # Should now use RX 7700S")

if __name__ == "__main__":
    main()

