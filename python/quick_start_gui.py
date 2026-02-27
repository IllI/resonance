#!/usr/bin/env python3
"""
Quick Start GUI Launcher

Simple launcher for the Ultra-Realistic Brain 3D Visualization GUI.
This script provides a one-click solution to get started with brain visualization.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Quick start launcher."""
    print("Ultra-Realistic Brain 3D Visualization - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "ultra_realistic_brain_gui.py").exists():
        print("Please run this script from the 'python' directory")
        print(f"   Current directory: {current_dir}")
        print("   Expected: python/ultra_realistic_brain_gui.py")
        input("\nPress Enter to exit...")
        return False
    
    print("Found GUI files")
    print(f"   Working directory: {current_dir}")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Python 3.8+ required")
        print(f"   Current version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        input("\nPress Enter to exit...")
        return False
    
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Try to launch the GUI
    print("\nLaunching Ultra-Realistic Brain GUI...")
    print("   This may take a few seconds...")
    
    try:
        # Import and launch
        # UPDATED: Launch the fixed version for AMD 7700S support
        # UPDATED: Launch the ROI Simulator GUI
        from roi_simulator_gui import ROISimulatorGUI
        app = ROISimulatorGUI()
        app.run()
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nAttempting to fix dependencies...")
        
        # Try to install missing packages
        missing_packages = []
        
        if "numpy" in str(e):
            missing_packages.append("numpy")
        if "plotly" in str(e):
            missing_packages.append("plotly")
        if "tkinter" in str(e):
            print("tkinter not available - this is usually built into Python")
            print("   Try reinstalling Python or use a different Python distribution")
            input("\nPress Enter to exit...")
            return False
        
        if missing_packages:
            print(f"Installing missing packages: {', '.join(missing_packages)}")
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"{package} installed successfully")
                except subprocess.CalledProcessError:
                    print(f"Failed to install {package}")
                    input("\nPress Enter to exit...")
                    return False
            
            # Try launching again
            print("\nRetrying GUI launch...")
            try:
                from roi_simulator_gui import ROISimulatorGUI
                app = ROISimulatorGUI()
                app.run()
                return True
            except Exception as e2:
                print(f"Still failed: {e2}")
                input("\nPress Enter to exit...")
                return False
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("\nTroubleshooting tips:")
        print("   1. Make sure you're in the 'python' directory")
        print("   2. Check that all required files are present")
        print("   3. Try running: python launch_ultra_realistic_brain_gui.py")
        print("   4. Check the documentation for setup instructions")
        input("\nPress Enter to exit...")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n GUI launched successfully!")
    else:
        print("\n Quick start failed. Check the error messages above.")
        print("   For help, see ULTRA_REALISTIC_BRAIN_GUI_GUIDE.md")
