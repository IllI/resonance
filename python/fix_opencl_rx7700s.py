#!/usr/bin/env python3
"""
Fix OpenCL RX 7700S Detection

This script attempts to fix OpenCL detection of the RX 7700S discrete GPU
so that the brain processing can use the powerful discrete GPU instead of
the integrated 780M.
"""

import os
import sys
import subprocess
import winreg
from pathlib import Path

def check_amd_opencl_installation():
    """Check AMD OpenCL installation status."""
    print("Checking AMD OpenCL Installation...")
    
    # Check for AMD OpenCL files
    opencl_files = [
        r"C:\Windows\System32\amdocl64.dll",
        r"C:\Windows\System32\amdocl12cl64.dll", 
        r"C:\Windows\System32\OpenCL.dll",
        r"C:\Program Files\AMD\ROCm\bin\amdhip64.dll"
    ]
    
    found_files = []
    missing_files = []
    
    for file_path in opencl_files:
        if Path(file_path).exists():
            found_files.append(file_path)
            print(f"   Found: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"   Missing: {file_path}")
    
    return found_files, missing_files

def check_opencl_registry():
    """Check OpenCL registry entries."""
    print("\nChecking OpenCL Registry Entries...")
    
    try:
        # Check OpenCL vendors registry
        vendors_key = r"SOFTWARE\Khronos\OpenCL\Vendors"
        
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, vendors_key) as key:
                print(f"   OpenCL Vendors registry key exists")
                
                # Enumerate vendor entries
                i = 0
                while True:
                    try:
                        name, value, type = winreg.EnumValue(key, i)
                        print(f"    {name} = {value}")
                        i += 1
                    except WindowsError:
                        break
                        
        except FileNotFoundError:
            print(f"   OpenCL Vendors registry key missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"   Registry check failed: {e}")
        return False

def reinstall_amd_opencl():
    """Attempt to reinstall AMD OpenCL support."""
    print("\nAttempting to reinstall AMD OpenCL support...")
    
    try:
        # Method 1: Reinstall AMD Software
        print("Method 1: Reinstalling AMD Software OpenCL components...")
        
        amd_installer_paths = [
            r"C:\Program Files\AMD\CNext\CNext\AMDSoftware.exe",
            r"C:\Program Files\AMD\CNext\CNext\RadeonSoftware.exe"
        ]
        
        amd_software = None
        for path in amd_installer_paths:
            if Path(path).exists():
                amd_software = path
                break
        
        if amd_software:
            print(f"  Found AMD Software: {amd_software}")
            print("  You can reinstall AMD drivers to fix OpenCL support")
        else:
            print("   AMD Software not found")
        
        # Method 2: Download and install OpenCL runtime
        print("\nMethod 2: Install Intel OpenCL Runtime (compatible with AMD)...")
        print("  Download: https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html")
        
        # Method 3: Install AMD ROCm (if available)
        print("\nMethod 3: Install AMD ROCm for OpenCL support...")
        print("  Download: https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html")
        
        return True
        
    except Exception as e:
        print(f"Reinstall attempt failed: {e}")
        return False

def fix_opencl_icd_registration():
    """Fix OpenCL ICD registration for AMD GPUs."""
    print("\nAttempting to fix OpenCL ICD registration...")
    
    try:
        # Find AMD OpenCL driver
        amd_opencl_paths = [
            r"C:\Windows\System32\amdocl64.dll",
            r"C:\Windows\System32\amdocl12cl64.dll"
        ]
        
        amd_opencl_dll = None
        for path in amd_opencl_paths:
            if Path(path).exists():
                amd_opencl_dll = path
                break
        
        if not amd_opencl_dll:
            print("   AMD OpenCL driver not found")
            return False
        
        print(f"   Found AMD OpenCL driver: {amd_opencl_dll}")
        
        # Register the OpenCL driver
        vendors_key = r"SOFTWARE\Khronos\OpenCL\Vendors"
        
        try:
            with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, vendors_key) as key:
                # Register AMD OpenCL driver
                winreg.SetValueEx(key, amd_opencl_dll, 0, winreg.REG_DWORD, 0)
                print(f"   Registered AMD OpenCL driver in registry")
                return True
                
        except PermissionError:
            print("   Permission denied - run as Administrator")
            return False
            
    except Exception as e:
        print(f"   ICD registration failed: {e}")
        return False

def test_opencl_after_fix():
    """Test OpenCL detection after attempting fixes."""
    print("\nTesting OpenCL detection after fixes...")
    
    try:
        # Reload OpenCL module
        if 'pyopencl' in sys.modules:
            del sys.modules['pyopencl']
        
        import pyopencl as cl
        
        platforms = cl.get_platforms()
        print(f"Found {len(platforms)} OpenCL platform(s)")
        
        total_gpus = 0
        rx7700s_found = False
        
        for platform in platforms:
            print(f"\nPlatform: {platform.name}")
            
            try:
                devices = platform.get_devices(cl.device_type.GPU)
                total_gpus += len(devices)
                
                for i, device in enumerate(devices):
                    device_name = device.name.strip()
                    memory_gb = device.global_mem_size // (1024**3)
                    
                    print(f"  GPU {i}: {device_name} ({memory_gb} GB)")
                    
                    # Check if this is the RX 7700S
                    if memory_gb >= 10 or '7700' in device_name or 'gfx1103' in device_name:
                        rx7700s_found = True
                        print(f"     RX 7700S detected!")
                    elif memory_gb < 10:
                        print(f"    (Integrated GPU)")
                        
            except Exception as e:
                print(f"  Error getting devices: {e}")
        
        print(f"\nTotal GPUs detected: {total_gpus}")
        
        if rx7700s_found:
            print(" SUCCESS: RX 7700S is now visible to OpenCL!")
            print("The brain GUI should now be able to use the discrete GPU")
            return True
        else:
            print(" RX 7700S still not detected by OpenCL")
            return False
            
    except Exception as e:
        print(f"OpenCL test failed: {e}")
        return False

def provide_manual_fix_instructions():
    """Provide manual fix instructions."""
    print("\nMANUAL FIX INSTRUCTIONS:")
    print("=" * 50)
    
    print("\n1. UPDATE AMD DRIVERS (Recommended):")
    print("   - Download latest AMD Adrenalin drivers")
    print("   - https://www.amd.com/en/support/graphics/amd-radeon-rx-7000-series/amd-radeon-rx-7700s")
    print("   - Uninstall current drivers (use DDU if needed)")
    print("   - Install fresh drivers")
    print("   - Restart computer")
    
    print("\n2. INSTALL OPENCL RUNTIME:")
    print("   - Download Intel OpenCL Runtime")
    print("   - https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html")
    print("   - Install and restart")
    
    print("\n3. CHECK DEVICE MANAGER:")
    print("   - Device Manager → Display adapters")
    print("   - Ensure RX 7700S is enabled and working")
    print("   - Update driver if there's a warning icon")
    
    print("\n4. REGISTRY FIX (Advanced):")
    print("   - Run this script as Administrator")
    print("   - It will attempt to register AMD OpenCL drivers")
    
    print("\n5. ALTERNATIVE SOLUTION:")
    print("   - Temporarily disable integrated GPU (780M)")
    print("   - This forces all operations to use RX 7700S")
    print("   - Device Manager → Disable AMD Radeon 780M")

def main():
    """Main OpenCL fix function."""
    print("Fix OpenCL RX 7700S Detection")
    print("=" * 50)
    print("This script will attempt to fix OpenCL detection of the RX 7700S")
    print("so that the brain processing can use the discrete GPU.")
    print()
    
    # Step 1: Check current OpenCL installation
    found_files, missing_files = check_amd_opencl_installation()
    
    # Step 2: Check registry
    registry_ok = check_opencl_registry()
    
    # Step 3: Test current OpenCL detection
    print("\nCurrent OpenCL Detection:")
    opencl_working = test_opencl_after_fix()
    
    if opencl_working:
        print("\n🎉 OpenCL is already working correctly!")
        print("The RX 7700S should be available for brain processing.")
        return
    
    # Step 4: Attempt fixes
    print("\nAttempting automatic fixes...")
    
    if missing_files:
        print(f"Missing {len(missing_files)} OpenCL files - driver reinstall needed")
    
    if not registry_ok:
        print("OpenCL registry entries missing - attempting fix...")
        fix_opencl_icd_registration()
    
    # Step 5: Test again
    print("\nTesting OpenCL after fixes...")
    fixed = test_opencl_after_fix()
    
    if fixed:
        print("\n🎉 SUCCESS: OpenCL fixes worked!")
        print("The brain GUI should now use the RX 7700S for processing.")
    else:
        print("\n Automatic fixes didn't work.")
        provide_manual_fix_instructions()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()