#!/usr/bin/env python3
"""
Neuroglancer Environment Setup for AMD 7700S
"""

import os
import sys
import subprocess
import platform

def configure_env():
    """Configure environment variables for AMD 7700S."""
    print("Configuring environment for AMD Radeon RX 7700S...")
    
    # Set environment variables for HIP/ROCm
    os.environ["HIP_VISIBLE_DEVICES"] = "1"
    os.environ["ROCR_VISIBLE_DEVICES"] = "1"
    os.environ["GPU_MAX_ALLOC_PERCENT"] = "95"
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    
    print("Environment variables set:")
    print(f"   HIP_VISIBLE_DEVICES={os.environ['HIP_VISIBLE_DEVICES']}")
    print(f"   HSA_OVERRIDE_GFX_VERSION={os.environ['HSA_OVERRIDE_GFX_VERSION']}")

def install_dependencies():
    """Install required dependencies."""
    required_packages = [
        "neuroglancer",
        "cloud-volume",
        "tornado",
        "numpy",
        "requests",
        "imageio"
    ]
    
    print("\nChecking dependencies...")
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   {package} already installed")
        except ImportError:
            print(f"   Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"   Failed to install {package}: {e}")
                return False
    
    return True

def main():
    print("Neuroglancer Environment Setup")
    print("=" * 40)
    
    configure_env()
    
    if install_dependencies():
        print("\nSetup complete! You can now run the Neuroglancer tools.")
    else:
        print("\nSetup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
