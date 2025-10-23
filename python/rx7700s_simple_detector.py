#!/usr/bin/env python3
"""
RX 7700S Simple GPU Detector

Based on successful GPU selection guide, this uses simple memory-based
heuristics to detect and select the RX 7700S discrete GPU.
"""

import numpy as np
import time
from typing import Optional, Dict, Any, Tuple

# Import environment setup
from rx7700s_environment_setup import setup_rx7700s_environment

# OpenCL for GPU detection
HAS_OPENCL = False
try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    print("Warning: OpenCL not available")

class RX7700SSimpleDetector:
    """Simple, reliable RX 7700S detection using memory heuristics."""
    
    def __init__(self, auto_setup_environment=True):
        """Initialize RX 7700S detector."""
        print("🎯 RX 7700S Simple Detector")
        print("=" * 40)
        
        # Setup environment for RX 7700S if requested
        if auto_setup_environment:
            print("Setting up RX 7700S environment...")
            setup_rx7700s_environment()
            print()
        
        self.rx7700s_device = None
        self.rx7700s_context = None
        self.rx7700s_queue = None
        
        if HAS_OPENCL:
            self._detect_rx7700s()
        else:
            raise Exception("OpenCL required for GPU detection")
    
    def _detect_rx7700s(self):
        """Detect RX 7700S using simple memory-based heuristic."""
        try:
            print("🔍 Detecting RX 7700S using memory heuristic...")
            
            # Get AMD platform
            platforms = cl.get_platforms()
            amd_platform = None
            
            for platform in platforms:
                if 'AMD' in platform.name:
                    amd_platform = platform
                    print(f"Found AMD platform: {platform.name}")
                    break
            
            if not amd_platform:
                raise Exception("AMD platform not found")
            
            # Get all GPU devices
            devices = amd_platform.get_devices(cl.device_type.GPU)
            print(f"Found {len(devices)} AMD GPU devices:")
            
            # Analyze devices by memory size (simple heuristic from guide)
            device_info = []
            for i, device in enumerate(devices):
                name = device.name.strip()
                memory_gb = device.global_mem_size // (1024**3)
                compute_units = device.max_compute_units
                
                device_info.append({
                    'index': i,
                    'device': device,
                    'name': name,
                    'memory_gb': memory_gb,
                    'compute_units': compute_units
                })
                
                print(f"  GPU {i}: {name}")
                print(f"    Memory: {memory_gb} GB")
                print(f"    Compute Units: {compute_units}")
                print()
            
            # Select discrete GPU using memory heuristic (>4GB = discrete)
            discrete_gpus = [d for d in device_info if d['memory_gb'] > 4]
            
            if discrete_gpus:
                # Select the discrete GPU with most memory (should be RX 7700S)
                rx7700s_info = max(discrete_gpus, key=lambda x: x['memory_gb'])
                self.rx7700s_device = rx7700s_info['device']
                
                print(f"🎯 SELECTED RX 7700S:")
                print(f"   GPU {rx7700s_info['index']}: {rx7700s_info['name']}")
                print(f"   Memory: {rx7700s_info['memory_gb']} GB")
                print(f"   Compute Units: {rx7700s_info['compute_units']}")
                
                # Create OpenCL context for RX 7700S only
                self.rx7700s_context = cl.Context([self.rx7700s_device])
                self.rx7700s_queue = cl.CommandQueue(self.rx7700s_context)
                
                print("✅ RX 7700S context created successfully")
                return True
                
            else:
                print("❌ No discrete GPU found (all GPUs have ≤4GB memory)")
                print("Available GPUs:")
                for info in device_info:
                    print(f"  - {info['name']}: {info['memory_gb']} GB")
                return False
                
        except Exception as e:
            print(f"❌ RX 7700S detection failed: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the selected RX 7700S device."""
        if not self.rx7700s_device:
            return {'error': 'RX 7700S not detected'}
        
        try:
            device = self.rx7700s_device
            
            return {
                'name': device.name.strip(),
                'memory_gb': device.global_mem_size // (1024**3),
                'memory_mb': device.global_mem_size // (1024**2),
                'compute_units': device.max_compute_units,
                'max_work_group_size': device.max_work_group_size,
                'max_clock_frequency': device.max_clock_frequency,
                'vendor': device.vendor,
                'version': device.version,
                'driver_version': device.driver_version,
                'opencl_version': device.opencl_c_version,
                'available': True
            }
            
        except Exception as e:
            return {'error': f'Failed to get device info: {e}'}
    
    def test_rx7700s_compute(self, test_size: int = 1024*1024) -> Dict[str, Any]:
        """Test RX 7700S compute performance."""
        if not self.rx7700s_context or not self.rx7700s_queue:
            return {'error': 'RX 7700S not available'}
        
        try:
            print(f"🧪 Testing RX 7700S compute performance...")
            print(f"   Test size: {test_size:,} elements")
            
            # Create test data
            a = np.random.rand(test_size).astype(np.float32)
            b = np.random.rand(test_size).astype(np.float32)
            
            # Create OpenCL buffers
            a_buffer = cl.Buffer(self.rx7700s_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
            b_buffer = cl.Buffer(self.rx7700s_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
            c_buffer = cl.Buffer(self.rx7700s_context, cl.mem_flags.WRITE_ONLY, a.nbytes)
            
            # Simple compute kernel
            kernel_code = """
            __kernel void vector_add(__global const float* a,
                                   __global const float* b,
                                   __global float* c) {
                int gid = get_global_id(0);
                
                // Heavy computation to stress GPU
                float result = 0.0f;
                for (int i = 0; i < 100; i++) {
                    result += sin(a[gid] * i) * cos(b[gid] * i);
                }
                c[gid] = result;
            }
            """
            
            # Compile and execute
            program = cl.Program(self.rx7700s_context, kernel_code).build()
            kernel = program.vector_add
            
            # Time the execution
            start_time = time.time()
            
            kernel(self.rx7700s_queue, (test_size,), None, a_buffer, b_buffer, c_buffer)
            self.rx7700s_queue.finish()
            
            # Read result
            c = np.empty_like(a)
            cl.enqueue_copy(self.rx7700s_queue, c, c_buffer)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate performance metrics
            operations_per_second = (test_size * 100) / execution_time  # 100 ops per element
            
            result = {
                'success': True,
                'execution_time_ms': execution_time * 1000,
                'operations_per_second': operations_per_second,
                'test_size': test_size,
                'device_name': self.rx7700s_device.name.strip(),
                'performance_rating': 'Excellent' if operations_per_second > 1e9 else 'Good' if operations_per_second > 1e8 else 'Fair'
            }
            
            print(f"✅ RX 7700S compute test completed:")
            print(f"   Execution time: {execution_time*1000:.2f} ms")
            print(f"   Performance: {operations_per_second/1e6:.1f} M ops/sec")
            print(f"   Rating: {result['performance_rating']}")
            
            return result
            
        except Exception as e:
            print(f"❌ RX 7700S compute test failed: {e}")
            return {'error': str(e), 'success': False}
    
    def process_brain_data_simple(self, brain_data: np.ndarray) -> Dict[str, Any]:
        """Process brain data using simple RX 7700S compute."""
        if not self.rx7700s_context:
            return {'error': 'RX 7700S not available'}
        
        try:
            print(f"🧠 Processing brain data on RX 7700S...")
            print(f"   Data shape: {brain_data.shape}")
            print(f"   Data size: {brain_data.size:,} voxels")
            
            # Flatten and prepare data
            brain_flat = brain_data.flatten().astype(np.float32)
            
            # Create buffers
            input_buffer = cl.Buffer(self.rx7700s_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=brain_flat)
            output_buffer = cl.Buffer(self.rx7700s_context, cl.mem_flags.WRITE_ONLY, brain_flat.nbytes)
            
            # Brain processing kernel
            kernel_code = """
            __kernel void process_brain(__global const float* input,
                                      __global float* output) {
                int gid = get_global_id(0);
                
                float voxel = input[gid];
                
                // Brain-specific processing
                float processed = 0.0f;
                
                // Simulate complex brain analysis
                for (int i = 0; i < 50; i++) {
                    processed += sin(voxel * i * 0.1f) * cos(voxel * i * 0.05f);
                }
                
                // Apply brain tissue enhancement
                processed = tanh(processed) * (1.0f + voxel * 0.5f);
                output[gid] = processed;
            }
            """
            
            # Execute brain processing
            program = cl.Program(self.rx7700s_context, kernel_code).build()
            kernel = program.process_brain
            
            start_time = time.time()
            
            kernel(self.rx7700s_queue, (brain_flat.size,), None, input_buffer, output_buffer)
            self.rx7700s_queue.finish()
            
            # Read processed data
            processed_flat = np.empty_like(brain_flat)
            cl.enqueue_copy(self.rx7700s_queue, processed_flat, output_buffer)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Reshape back to original shape
            processed_brain = processed_flat.reshape(brain_data.shape)
            
            result = {
                'success': True,
                'processed_data': processed_brain,
                'processing_time': processing_time,
                'voxels_processed': brain_data.size,
                'voxels_per_second': brain_data.size / processing_time,
                'device_used': self.rx7700s_device.name.strip()
            }
            
            print(f"✅ Brain processing completed:")
            print(f"   Processing time: {processing_time:.3f} seconds")
            print(f"   Voxels/second: {result['voxels_per_second']:,.0f}")
            print(f"   Device: {result['device_used']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Brain processing failed: {e}")
            return {'error': str(e), 'success': False}

def test_rx7700s_detection():
    """Test RX 7700S detection and basic functionality."""
    print("🧪 RX 7700S Detection Test")
    print("=" * 50)
    
    try:
        # Create detector
        detector = RX7700SSimpleDetector()
        
        # Get device info
        device_info = detector.get_device_info()
        if 'error' not in device_info:
            print("\n📊 RX 7700S Device Information:")
            for key, value in device_info.items():
                print(f"   {key}: {value}")
        
        # Test compute performance
        print("\n" + "="*50)
        compute_result = detector.test_rx7700s_compute()
        
        # Test brain data processing
        print("\n" + "="*50)
        test_brain = np.random.rand(64, 64, 32).astype(np.float32)
        brain_result = detector.process_brain_data_simple(test_brain)
        
        # Summary
        print("\n" + "="*50)
        print("🎯 RX 7700S TEST SUMMARY")
        print("="*50)
        
        detection_ok = detector.rx7700s_device is not None
        compute_ok = compute_result.get('success', False)
        brain_ok = brain_result.get('success', False)
        
        print(f"RX 7700S Detection: {'✅' if detection_ok else '❌'}")
        print(f"Compute Test:       {'✅' if compute_ok else '❌'}")
        print(f"Brain Processing:   {'✅' if brain_ok else '❌'}")
        
        if detection_ok and compute_ok and brain_ok:
            print("\n🎉 SUCCESS: RX 7700S is working correctly!")
            print("   Monitor Task Manager → Performance → GPU 1 during processing")
        else:
            print("\n⚠️ Issues detected with RX 7700S setup")
        
        return detection_ok and compute_ok and brain_ok
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_rx7700s_detection()
    input("\nPress Enter to exit...")