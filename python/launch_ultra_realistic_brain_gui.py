#!/usr/bin/env python3
"""
Ultra-Realistic Brain GUI Launcher

Simple launcher script that checks dependencies and launches the brain visualization GUI.
"""

import sys
import subprocess
import importlib.util

def check_dependency(module_name, package_name=None):
    """Check if a Python module is available."""
    if package_name is None:
        package_name = module_name
    
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {package_name} not found")
        return False
    else:
        print(f"✅ {package_name} available")
        return True

def install_dependency(package_name):
    """Install a Python package."""
    print(f"📦 Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def main():
    """Main launcher function."""
    print("🧠 Ultra-Realistic Brain GUI Launcher")
    print("=" * 50)
    
    # Check required dependencies
    required_deps = [
        ("tkinter", "tkinter (built-in)"),
        ("numpy", "numpy"),
        ("plotly", "plotly"),
        ("pathlib", "pathlib (built-in)"),
        ("json", "json (built-in)"),
        ("threading", "threading (built-in)"),
        ("time", "time (built-in)"),
        ("warnings", "warnings (built-in)")
    ]
    
    print("\n📋 Checking dependencies...")
    missing_deps = []
    
    for module, package in required_deps:
        if not check_dependency(module, package):
            if package not in ["tkinter (built-in)", "pathlib (built-in)", "json (built-in)", "threading (built-in)", "time (built-in)", "warnings (built-in)"]:
                missing_deps.append(package)
    
    # Install missing dependencies
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("Attempting to install...")
        
        for dep in missing_deps:
            if not install_dependency(dep):
                print(f"\n❌ Failed to install {dep}")
                print("Please install manually: pip install " + dep)
                return False
    
    # Check optional dependencies
    print("\n🔍 Checking optional dependencies...")
    optional_deps = [
        ("brain_3d_model_generator", "brain_3d_model_generator.py"),
        ("brain_3d_interface", "brain_3d_interface.py"),
        ("blue_brain_atlas_integrator", "blue_brain_atlas_integrator.py"),
        ("enhanced_dual_gpu_analyzer", "enhanced_dual_gpu_analyzer.py"),
        ("brain_atlas_manager", "brain_atlas_manager.py")
    ]
    
    for module, file_name in optional_deps:
        check_dependency(module, file_name)
    
    # Launch the GUI
    print("\n🚀 Launching Ultra-Realistic Brain GUI...")
    try:
        from ultra_realistic_brain_gui import main as launch_gui
        launch_gui()
    except ImportError as e:
        print(f"❌ Failed to launch GUI: {e}")
        print("Make sure ultra_realistic_brain_gui.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Launcher failed. Check the error messages above.")
        input("Press Enter to exit...")
    else:
        print("\n✅ GUI launched successfully!")
