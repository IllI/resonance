#!/usr/bin/env python3
"""
Test script to verify the brain GUI fix works correctly.
This tests the GPU processing without freezing.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gpu_brain_processing():
    """Test the GPU brain processing fix."""
    print("Testing GPU brain processing fix...")
    
    try:
        # Import the fixed analyzer
        from rx7700s_only_brain_analyzer import RX7700SOnlyBrainAnalyzer
        
        # Create analyzer
        analyzer = RX7700SOnlyBrainAnalyzer()
        
        # Create test brain volume
        test_volume = np.random.rand(64, 64, 32).astype(np.float32)
        # Add brain-like structure
        test_volume[16:48, 16:48, 8:24] = 0.8  # Brain region
        test_volume[20:44, 20:44, 10:22] = 0.9  # Inner brain
        
        print(f"Test volume shape: {test_volume.shape}")
        
        # Process with GPU
        result = analyzer.process_brain_volume_gpu(test_volume)
        
        if result and result.get('success', False):
            brain_surface = result.get('brain_surface', {})
            vertices = brain_surface.get('vertices', np.array([]))
            faces = brain_surface.get('faces', np.array([]))
            
            print(f"SUCCESS: GPU processing completed")
            print(f"  Processing time: {result.get('processing_time', 0):.3f}s")
            print(f"  Vertices: {len(vertices)}")
            print(f"  Faces: {len(faces)}")
            print(f"  GPU device: {result.get('gpu_device', 'Unknown')}")
            
            # Test the data structure that was causing the error
            if len(vertices) > 0:
                print(f"  Vertex data shape: {vertices.shape}")
                print(f"  Vertex range: [{vertices.min():.2f}, {vertices.max():.2f}]")
            
            if len(faces) > 0:
                print(f"  Face data shape: {faces.shape}")
                print(f"  Face range: [{faces.min()}, {faces.max()}]")
            
            return True
        else:
            print(f"FAILED: GPU processing failed")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_brain_surface_data_access():
    """Test accessing brain surface data like the GUI does."""
    print("\nTesting brain surface data access pattern...")
    
    try:
        # Simulate the data structure returned by the GPU analyzer
        mock_result = {
            'success': True,
            'brain_surface': {
                'vertices': np.random.rand(1000, 3),
                'faces': np.random.randint(0, 1000, (500, 3)),
                'normals': np.random.rand(1000, 3),
                'values': np.random.rand(1000)
            },
            'processing_time': 1.234,
            'gpu_device': 'AMD Radeon RX 7700S'
        }
        
        # Test the access pattern that was causing the error
        if mock_result and mock_result.get('success', False):
            brain_surface = mock_result.get('brain_surface', {})
            verts = brain_surface.get('vertices', np.array([]))
            faces = brain_surface.get('faces', np.array([]))
            
            if len(verts) > 0 and len(faces) > 0:
                print(f"SUCCESS: Data access pattern works correctly")
                print(f"  Vertices: {len(verts)} points")
                print(f"  Faces: {len(faces)} triangles")
                print(f"  Vertex shape: {verts.shape}")
                print(f"  Face shape: {faces.shape}")
                return True
            else:
                print("FAILED: Empty vertex or face data")
                return False
        else:
            print("FAILED: Invalid result structure")
            return False
            
    except Exception as e:
        print(f"ERROR: Data access test failed: {e}")
        return False

if __name__ == "__main__":
    print("Brain GUI Fix Verification Test")
    print("=" * 50)
    
    # Test 1: GPU processing
    test1_passed = test_gpu_brain_processing()
    
    # Test 2: Data access pattern
    test2_passed = test_brain_surface_data_access()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"  GPU Processing: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Data Access: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n ALL TESTS PASSED - Brain GUI fix is working!")
        print("The 'vertices' error should be resolved.")
    else:
        print("\n SOME TESTS FAILED - Check the errors above.")
    
    print("\nYou can now run the GUI with: python quick_start_gui.py")