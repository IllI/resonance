#!/usr/bin/env python3
"""
Test RX 7700S Integration - Automated Test

This script automatically tests the RX 7700S integration without user input.
"""

import os
import sys
import time

def setup_rx7700s_environment():
    """Setup RX 7700S environment using successful methods."""
    print("🔧 Setting up RX 7700S environment...")
    
    # Apply the successful environment variables
    rx7700s_env_vars = {
        'HIP_VISIBLE_DEVICES': '1',
        'ROCR_VISIBLE_DEVICES': '1',
        'GPU_MAX_ALLOC_PERCENT': '95',
        'GPU_MAX_HEAP_SIZE': '100',
        'HSA_OVERRIDE_GFX_VERSION': '11.0.0',
        'GPU_FORCE_64BIT_PTR': '1',
        'GPU_USE_SYNC_OBJECTS': '1',
        'GPU_SINGLE_ALLOC_PERCENT': '100',
        'OPENCL_DEVICE_FILTER': 'discrete',
        'AMD_OPENCL_DEVICE_MASK': '2',
        'AmdPowerXpressRequestHighPerformance': '1',
        'AMD_OPENCL_DEVICE': '1',
        'OPENCL_DEVICE': '1',
        'PYOPENCL_CTX': '0:1',
    }
    
    for var_name, var_value in rx7700s_env_vars.items():
        os.environ[var_name] = var_value
    
    print(f"✅ Applied {len(rx7700s_env_vars)} RX 7700S environment variables")
    return True

def test_rx7700s_detection():
    """Test RX 7700S detection."""
    print("\n🔍 Testing RX 7700S detection...")
    
    try:
        import pyopencl as cl
        
        platforms = cl.get_platforms()
        amd_platform = None
        
        for platform in platforms:
            if 'AMD' in platform.name:
                amd_platform = platform
                break
        
        if not amd_platform:
            print("❌ AMD platform not found")
            return False
        
        devices = amd_platform.get_devices(cl.device_type.GPU)
        print(f"Found {len(devices)} AMD GPU devices:")
        
        rx7700s_found = False
        for i, device in enumerate(devices):
            name = device.name.strip()
            memory_gb = device.global_mem_size // (1024**3)
            
            is_rx7700s = 'gfx1103' in name or memory_gb >= 10
            status = "🎯 RX 7700S" if is_rx7700s else "   780M"
            print(f"  GPU {i}: {name} ({memory_gb} GB) {status}")
            
            if is_rx7700s:
                rx7700s_found = True
        
        if rx7700s_found:
            print("✅ RX 7700S detected and available")
            return True
        else:
            print("❌ RX 7700S not found")
            return False
            
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

def test_brain_processing():
    """Test brain processing on RX 7700S."""
    print("\n🧠 Testing brain processing on RX 7700S...")
    
    try:
        # Add python directory to path
        sys.path.insert(0, 'python')
        
        from rx7700s_brain_analyzer_fixed import RX7700SBrainAnalyzerFixed
        import numpy as np
        
        # Create analyzer
        analyzer = RX7700SBrainAnalyzerFixed()
        
        # Create test volume
        test_volume = np.random.rand(32, 32, 16).astype(np.float32)
        test_volume[8:24, 8:24, 4:12] = 0.8  # Brain region
        
        print(f"   Test volume: {test_volume.shape}")
        
        # Run analysis
        start_time = time.time()
        results = analyzer.analyze_brain_volume(test_volume, generate_surface=False)
        end_time = time.time()
        
        if results['success']:
            print(f"✅ Brain processing successful!")
            print(f"   Processing time: {end_time - start_time:.3f} seconds")
            print(f"   GPU used: {results['gpu_device']}")
            print(f"   Brain voxels: {results['segmentation']['brain_voxels']:,}")
            return True
        else:
            print(f"❌ Brain processing failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Brain processing test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 RX 7700S Integration Test - Automated")
    print("=" * 60)
    
    # Run tests
    env_success = setup_rx7700s_environment()
    detection_success = test_rx7700s_detection()
    processing_success = test_brain_processing()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS")
    print("=" * 60)
    
    print(f"Environment Setup: {'✅ SUCCESS' if env_success else '❌ FAILED'}")
    print(f"RX 7700S Detection: {'✅ SUCCESS' if detection_success else '❌ FAILED'}")
    print(f"Brain Processing: {'✅ SUCCESS' if processing_success else '❌ FAILED'}")
    
    overall_success = env_success and detection_success and processing_success
    
    if overall_success:
        print("\n🎉 RX 7700S integration is working perfectly!")
        print("\nYour 3D analyzer is now configured to use RX 7700S:")
        print("   python python/quick_start_gui.py")
        print("   python python/ultra_realistic_brain_gui.py")
        print("\nMonitor Task Manager → Performance → GPU 1 during processing")
    else:
        print("\n⚠️ Some tests failed - check the errors above")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    print(f"\nTest completed: {'SUCCESS' if success else 'FAILED'}")