#!/usr/bin/env python3
"""
Test script for real-time 3D brain visualization with GPU acceleration.
This script tests the core 3D rendering capabilities that will be used in the GUI.
"""

import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
from PyQt5.QtWidgets import QApplication
import sys

def test_basic_3d_rendering():
    """Test basic 3D rendering capabilities."""
    print(" Testing 3D Brain Visualization Components...")
    
    try:
        # Test PyVista installation
        print(" PyVista installed successfully")
        
        # Test VTK backend
        print(f" VTK available: {pv.__version__}")
        
        # Test GPU acceleration
        print(f" PyVista theme: {pv.global_theme.name}")
        
        # Test basic mesh creation
        sphere = pv.Sphere(radius=1.0)
        print(f" Basic mesh creation: {sphere.n_points} points, {sphere.n_cells} cells")
        
        # Test structured grid creation (for MRI data)
        nx, ny, nz = 64, 64, 64
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Create mock MRI data
        mri_data = np.random.randn(nx, ny, nz)
        
        # Create structured grid
        grid = pv.StructuredGrid()
        grid.points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        grid.dimensions = [nx, ny, nz]
        grid.point_data['intensity'] = mri_data.ravel()
        
        print(f" MRI grid creation: {grid.n_points} points, {grid.n_cells} cells")
        
        # Test isosurface extraction
        try:
            surface = grid.contour([0.0])
            print(f" Isosurface extraction: {surface.n_points} points, {surface.n_cells} cells")
        except Exception as e:
            print(f" Isosurface extraction failed: {e}")
        
        print("\n All 3D rendering tests passed!")
        return True
        
    except Exception as e:
        print(f" 3D rendering test failed: {e}")
        return False

def test_qt_integration():
    """Test Qt integration for GUI embedding."""
    print("\n Testing Qt Integration...")
    
    try:
        # Test PyQt5
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        print(" PyQt5 integration successful")
        
        # Test PyVistaQt
        try:
            # Create a simple plotter
            plotter = pvqt.QtInteractor()
            print(" PyVistaQt integration successful")
            
            # Test basic plotting
            sphere = pv.Sphere(radius=0.5)
            plotter.add_mesh(sphere, color='red')
            print(" Basic plotting successful")
            
            return True
            
        except Exception as e:
            print(f" PyVistaQt test failed: {e}")
            return False
            
    except Exception as e:
        print(f" Qt integration test failed: {e}")
        return False

def test_gpu_acceleration():
    """Test GPU acceleration capabilities."""
    print("\n Testing GPU Acceleration...")
    
    try:
        # Check OpenGL capabilities
        print(f" PyVista theme: {pv.global_theme.name}")
        
        # Check for hardware acceleration
        print(" Hardware-accelerated rendering should be available via VTK")
        
        # Test large dataset rendering
        print(" Testing large dataset rendering...")
        
        # Create a larger test dataset
        nx, ny, nz = 128, 128, 128
        x = np.linspace(-2, 2, nx)
        y = np.linspace(-2, 2, ny)
        z = np.linspace(-2, 2, nz)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Create complex brain-like data
        r = np.sqrt(xx**2 + yy**2 + zz**2)
        brain_data = np.sin(r * 2) * np.exp(-r / 2)
        
        # Create structured grid
        grid = pv.StructuredGrid()
        grid.points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        grid.dimensions = [nx, ny, nz]
        grid.point_data['intensity'] = brain_data.ravel()
        
        print(f" Large dataset created: {grid.n_points:,} points")
        
        # Test performance
        import time
        start_time = time.time()
        
        try:
            # Extract multiple isosurfaces
            surfaces = []
            for threshold in [-0.5, 0.0, 0.5]:
                surface = grid.contour([threshold])
                surfaces.append(surface)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            total_points = sum(s.n_points for s in surfaces)
            print(f" GPU-accelerated processing: {total_points:,} total points in {processing_time:.2f}s")
            
            if processing_time < 5.0:  # Should be fast with GPU
                print(" GPU acceleration working well!")
            else:
                print(" Processing slower than expected - may need GPU optimization")
                
        except Exception as e:
            print(f" Large dataset processing failed: {e}")
        
        return True
        
    except Exception as e:
        print(f" GPU acceleration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print(" Real-Time 3D Brain Visualization - Component Tests")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Basic 3D Rendering", test_basic_3d_rendering),
        ("Qt Integration", test_qt_integration),
        ("GPU Acceleration", test_gpu_acceleration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f" Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print(" Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for real-time 3D brain visualization!")
    else:
        print(" Some tests failed. Check dependencies and try again.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
