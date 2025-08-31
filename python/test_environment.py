#!/usr/bin/env python3
"""
Test script to verify fMRI analysis environment setup.
This script tests all core dependencies and system capabilities.
"""

import sys
import platform
from pathlib import Path

def test_core_libraries():
    """Test core scientific computing libraries."""
    print("🧪 Testing Core Libraries...")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import scipy
        print(f"✅ SciPy {scipy.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        import nibabel as nib
        print(f"✅ NiBabel {nib.__version__}")
        
        import nilearn
        print(f"✅ Nilearn {nilearn.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False

def test_system_info():
    """Display system information."""
    print("\n💻 System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.architecture()}")

def test_gpu_detection():
    """Test GPU detection capabilities."""
    print("\n🖥️ GPU Detection Test:")
    
    # Test basic GPU info
    try:
        import psutil
        print(f"Total RAM: {psutil.virtual_memory().total // (1024**3)} GB")
    except ImportError:
        print("psutil not available - install for system monitoring")
    
    # Test if we can import OpenCL
    try:
        import pyopencl as cl
        print("✅ PyOpenCL available")
        
        # Get platforms and devices
        platforms = cl.get_platforms()
        print(f"OpenCL Platforms found: {len(platforms)}")
        
        for i, platform in enumerate(platforms):
            devices = platform.get_devices()
            print(f"  Platform {i}: {platform.name}")
            for j, device in enumerate(devices):
                print(f"    Device {j}: {device.name} ({device.type})")
                
    except ImportError:
        print("⚠️ PyOpenCL not installed - GPU acceleration not available yet")
    except Exception as e:
        print(f"⚠️ OpenCL Error: {e}")

def test_neuroimaging_capabilities():
    """Test neuroimaging-specific functionality."""
    print("\n🧠 Neuroimaging Capabilities:")
    
    try:
        import nibabel as nib
        print("✅ NiBabel - NIfTI file support available")
        
        import nilearn
        from nilearn import datasets
        print("✅ Nilearn - Machine learning for neuroimaging available")
        
                 # Test sample data access
         print("📊 Testing sample data access...")
         try:
             # This will attempt to download a small sample dataset
             haxby = datasets.fetch_haxby(subjects=[1], fetch_stimuli=False)
             print("✅ Sample fMRI data download successful")
             
             # Load and inspect the data
             func_img = nib.load(haxby.func[0])
             print(f"✅ Sample data shape: {func_img.shape}")
             print(f"✅ Sample data dtype: {func_img.get_fdata().dtype}")
             
         except Exception as e:
             print(f"⚠️ Sample data access failed: {e}")
        
    except ImportError as e:
        print(f"❌ Neuroimaging library error: {e}")

def test_file_formats():
    """Test various file format support."""
    print("\n📁 File Format Support:")
    
    formats = [
        ("NIfTI", "nibabel"),
        ("DICOM", "pydicom"), 
        ("HDF5", "h5py"),
        ("MATLAB", "scipy.io")
    ]
    
    for format_name, module_name in formats:
        try:
            __import__(module_name)
            print(f"✅ {format_name} support available")
        except ImportError:
            print(f"⚠️ {format_name} support not installed")

def main():
    """Run all tests."""
    print("🚀 fMRI Analysis Environment Test")
    print("=" * 50)
    
    success = test_core_libraries()
    test_system_info()
    test_gpu_detection()
    test_neuroimaging_capabilities()
    test_file_formats()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Environment test completed!")
        print("💡 Ready for fMRI analysis development")
    else:
        print("❌ Some issues detected - check requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main() 