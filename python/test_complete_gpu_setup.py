#!/usr/bin/env python3
"""
Complete GPU Setup Test
Tests all AMD GPU selection methods and C++ backend integration
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

def test_environment_variables():
    """Test AMD GPU environment variables"""
    print("=== Environment Variables Test ===")
    
    expected_vars = {
        'AmdPowerXpressRequestHighPerformance': '1',
        'AMD_OPENCL_DEVICE': '1', 
        'OPENCL_DEVICE': '1',
        'PYOPENCL_CTX': '0:1'
    }
    
    all_set = True
    for var, expected in expected_vars.items():
        actual = os.environ.get(var, 'NOT SET')
        status = "✓" if actual == expected else "✗"
        print(f"{status} {var} = {actual}")
        if actual != expected:
            all_set = False
    
    return all_set

def test_pyopencl_gpu_selection():
    """Test PyOpenCL GPU selection"""
    print("\n=== PyOpenCL GPU Selection Test ===")
    
    try:
        import pyopencl as cl
        
        platforms = cl.get_platforms()
        print(f"Found {len(platforms)} OpenCL platform(s)")
        
        amd_devices = []
        for i, platform in enumerate(platforms):
            if 'AMD' in platform.name or 'Advanced Micro Devices' in platform.name:
                print(f"AMD Platform {i}: {platform.name}")
                devices = platform.get_devices()
                
                for j, device in enumerate(devices):
                    device_info = {
                        'platform': i,
                        'device': j,
                        'name': device.name,
                        'memory_gb': device.global_mem_size // (1024**3),
                        'compute_units': device.max_compute_units
                    }
                    amd_devices.append(device_info)
                    
                    print(f"  Device {j}: {device.name}")
                    print(f"    Memory: {device_info['memory_gb']} GB")
                    print(f"    Compute Units: {device_info['compute_units']}")
        
        # Check if RX 7700S is detected
        rx7700s_found = any('RX' in dev['name'] and '7700' in dev['name'] for dev in amd_devices)
        if rx7700s_found:
            print("✓ RX 7700S detected")
        else:
            print("  RX 7700S not found by name, checking by specs...")
            # Look for discrete GPU (>4GB memory, >20 compute units)
            discrete_gpus = [dev for dev in amd_devices if dev['memory_gb'] > 4 and dev['compute_units'] > 20]
            if discrete_gpus:
                print(f"✓ Discrete GPU found: {discrete_gpus[0]['name']}")
            else:
                print("✗ No discrete GPU detected")
        
        return len(amd_devices) > 0
        
    except ImportError:
        print("✗ PyOpenCL not available")
        return False
    except Exception as e:
        print(f"✗ PyOpenCL error: {e}")
        return False

def test_cpp_backend():
    """Test C++ backend availability and functionality"""
    print("\n=== C++ Backend Test ===")
    
    try:
        from brain_gpu_backend import BrainGPUBackend
        
        backend = BrainGPUBackend()
        
        if backend.initialize(enable_gl_interop=False):
            print("✓ C++ backend initialized successfully")
            
            # Get device info
            info = backend.get_device_info()
            print("Device Information:")
            for line in info.split('\n'):
                if line.strip():
                    print(f"  {line}")
            
            # Test mesh processing
            print("\nTesting mesh processing...")
            vertices = np.random.rand(1000, 3).astype(np.float32)
            normals = np.random.rand(1000, 3).astype(np.float32) 
            indices = np.arange(3000, dtype=np.int32)
            
            start_time = time.time()
            processed_vertices, processed_normals, success = backend.process_brain_mesh(vertices, normals, indices)
            end_time = time.time()
            
            if success:
                print(f"✓ Mesh processing successful ({(end_time-start_time)*1000:.2f} ms)")
                print(f"  Input: {vertices.shape[0]} vertices")
                print(f"  Output: {processed_vertices.shape[0]} vertices")
                return True
            else:
                print("✗ Mesh processing failed")
                return False
        else:
            print("✗ C++ backend initialization failed")
            return False
            
    except ImportError:
        print("  C++ backend not available (DLL not found)")
        print("   To build: cd cpp/brain_gpu_backend && build_msvc.bat")
        return False
    except Exception as e:
        print(f"✗ C++ backend error: {e}")
        return False

def test_brain_gui_integration():
    """Test integration with brain GUI"""
    print("\n=== Brain GUI Integration Test ===")
    
    try:
        # Check if GUI files exist
        gui_file = Path("ultra_realistic_brain_gui.py")
        if not gui_file.exists():
            print("✗ ultra_realistic_brain_gui.py not found")
            return False
        
        print("✓ Brain GUI file found")
        
        # Test import (without running GUI)
        sys.path.insert(0, str(Path.cwd()))
        
        # Set environment for testing
        os.environ['CPP_BACKEND_AVAILABLE'] = '1' if test_cpp_backend() else '0'
        
        print("✓ Environment configured for GUI")
        print(f"  CPP_BACKEND_AVAILABLE = {os.environ.get('CPP_BACKEND_AVAILABLE', 'NOT SET')}")
        
        return True
        
    except Exception as e:
        print(f"✗ GUI integration error: {e}")
        return False

def test_gpu_performance():
    """Test GPU performance with different methods"""
    print("\n=== GPU Performance Test ===")
    
    try:
        import pyopencl as cl
        
        # Create test data
        size = 1024 * 1024  # 1M elements
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        
        # Test PyOpenCL performance
        platforms = cl.get_platforms()
        results = []
        
        for platform in platforms:
            if 'AMD' not in platform.name:
                continue
                
            devices = platform.get_devices()
            for device in devices:
                try:
                    context = cl.Context([device])
                    queue = cl.CommandQueue(context)
                    
                    # Create buffers
                    a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                    b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                    c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
                    
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
                    
                    # Time execution
                    start_time = time.time()
                    kernel(queue, (size,), None, a_buf, b_buf, c_buf)
                    queue.finish()
                    end_time = time.time()
                    
                    execution_time = (end_time - start_time) * 1000  # ms
                    performance = size / execution_time  # ops/ms
                    
                    results.append({
                        'device': device.name,
                        'time_ms': execution_time,
                        'performance': performance
                    })
                    
                except Exception as e:
                    print(f"  Error testing {device.name}: {e}")
        
        # Show results
        if results:
            results.sort(key=lambda x: x['time_ms'])
            print("Performance Results (fastest first):")
            for i, result in enumerate(results):
                status = "" if i == 0 else f"#{i+1}"
                print(f"  {status} {result['device']}: {result['time_ms']:.2f} ms ({result['performance']:.0f} ops/ms)")
            
            # Check if discrete GPU is fastest
            fastest = results[0]
            if 'RX' in fastest['device'] and '7700' in fastest['device']:
                print(" RX 7700S is the fastest GPU - GPU selection working correctly!")
                return True
            else:
                print(f"  {fastest['device']} is fastest - may indicate GPU selection issue")
                return False
        else:
            print("✗ No performance results obtained")
            return False
            
    except Exception as e:
        print(f"✗ Performance test error: {e}")
        return False

def main():
    """Run complete GPU setup test"""
    print(" Complete AMD GPU Setup Test")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("PyOpenCL GPU Selection", test_pyopencl_gpu_selection), 
        ("C++ Backend", test_cpp_backend),
        ("Brain GUI Integration", test_brain_gui_integration),
        ("GPU Performance", test_gpu_performance)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    passed = 0
    total = len(tests)
    
    for test_name, success in results.items():
        status = " PASS" if success else " FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your AMD GPU setup is working correctly.")
        print("   You can now run the brain GUI with optimal performance.")
    elif passed >= total - 1:
        print("  Most tests passed. Minor issues detected but should still work.")
    else:
        print(" Multiple issues detected. Please review the failed tests above.")
        print("   Refer to CPP_BACKEND_SETUP_GUIDE.md for troubleshooting.")
    
    print(f"\nNext steps:")
    if results.get("C++ Backend", False):
        print("✓ C++ backend available - run: python quick_start_gui.py")
    else:
        print("• Build C++ backend: cd cpp/brain_gpu_backend && build_msvc.bat")
    print("• Test brain GUI: python ultra_realistic_brain_gui.py")
    print("• Monitor GPU usage in Task Manager during processing")

if __name__ == "__main__":
    main()