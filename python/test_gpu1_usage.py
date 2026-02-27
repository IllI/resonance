#!/usr/bin/env python3
"""
Test GPU 1 (RX 7700S) Usage

Verifies that the brain GUI is using GPU 1 (RX 7700S) instead of GPU 0 (780M).
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gpu1_selection():
    """Test that GPU 1 (RX 7700S) is selected."""
    print("Testing GPU 1 (RX 7700S) Selection...")
    
    try:
        from rx7700s_only_brain_analyzer import RX7700SOnlyBrainAnalyzer
        
        # Create analyzer
        analyzer = RX7700SOnlyBrainAnalyzer()
        
        # Check which GPU was selected
        if hasattr(analyzer, 'rx7700s_device'):
            device_name = analyzer.rx7700s_device.name.strip()
            memory_gb = analyzer.rx7700s_device.global_mem_size // (1024**3)
            
            print(f" GPU Selected: {device_name}")
            print(f"   Memory: {memory_gb} GB")
            
            # Verify it's GPU 1 (not GPU 0)
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                if 'AMD' in platform.name:
                    devices = platform.get_devices(cl.device_type.GPU)
                    for i, device in enumerate(devices):
                        if device == analyzer.rx7700s_device:
                            if i == 1:
                                print(f" CONFIRMED: Using GPU 1 (RX 7700S)")
                                return True
                            else:
                                print(f" ERROR: Using GPU {i} instead of GPU 1")
                                return False
        else:
            print(" ERROR: No RX 7700S device found")
            return False
            
    except Exception as e:
        print(f" ERROR: Test failed: {e}")
        return False

def test_gpu1_processing():
    """Test brain processing on GPU 1."""
    print("\nTesting Brain Processing on GPU 1...")
    
    try:
        from rx7700s_only_brain_analyzer import RX7700SOnlyBrainAnalyzer
        
        # Create analyzer
        analyzer = RX7700SOnlyBrainAnalyzer()
        
        # Create test brain volume
        test_volume = np.random.rand(32, 32, 16).astype(np.float32)
        test_volume[8:24, 8:24, 4:12] = 0.8  # Brain region
        
        print(f"Processing test volume on GPU 1...")
        
        # Process with GPU 1
        result = analyzer.process_brain_volume_gpu(test_volume, use_discrete_gpu=True)
        
        if result and result.get('success', False):
            device_used = result.get('gpu_device', 'Unknown')
            processing_time = result.get('processing_time', 0)
            
            print(f" SUCCESS: GPU 1 processing completed")
            print(f"   Device: {device_used}")
            print(f"   Time: {processing_time:.3f}s")
            
            # Check if it's the RX 7700S
            if 'gfx1103' in device_used or '7700' in device_used:
                print(f" CONFIRMED: RX 7700S (GPU 1) was used")
                return True
            else:
                print(f"  WARNING: Device name doesn't match RX 7700S")
                return False
        else:
            print(f" ERROR: GPU 1 processing failed")
            return False
            
    except Exception as e:
        print(f" ERROR: Processing test failed: {e}")
        return False

def main():
    """Main test function."""
    print("GPU 1 (RX 7700S) Usage Test")
    print("=" * 40)
    
    # Test 1: GPU selection
    test1_passed = test_gpu1_selection()
    
    # Test 2: GPU processing
    test2_passed = test_gpu1_processing()
    
    print("\n" + "=" * 40)
    print("TEST RESULTS:")
    print(f"  GPU 1 Selection: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  GPU 1 Processing: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n ALL TESTS PASSED - GPU 1 (RX 7700S) is being used!")
        print("The brain GUI should now run on the discrete GPU.")
    else:
        print("\n SOME TESTS FAILED - Check GPU configuration.")
    
    print("\nNow run the GUI and check Task Manager:")
    print("  GPU 0 (780M): Should be 0% usage")
    print("  GPU 1 (RX 7700S): Should show activity")

if __name__ == "__main__":
    main()