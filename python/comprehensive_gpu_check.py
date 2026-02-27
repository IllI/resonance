#!/usr/bin/env python3
"""
Comprehensive GPU Check

This script uses multiple methods to detect all GPUs in the system,
including Windows WMI, Device Manager, and OpenCL detection.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_windows_gpus():
    """Check GPUs using Windows WMI (most comprehensive)."""
    try:
        print("Checking GPUs using Windows WMI...")
        
        # PowerShell command to get detailed GPU info
        ps_command = '''
        Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*AMD*" } | ForEach-Object {
            $memoryGB = if ($_.AdapterRAM) { [math]::Round($_.AdapterRAM / 1GB, 1) } else { "Unknown" }
            Write-Host "GPU: $($_.Name)"
            Write-Host "  Memory: $memoryGB GB"
            Write-Host "  Status: $($_.Status)"
            Write-Host "  Availability: $($_.Availability)"
            Write-Host "  Driver Version: $($_.DriverVersion)"
            Write-Host "  Driver Date: $($_.DriverDate)"
            Write-Host "  Device ID: $($_.DeviceID)"
            Write-Host "  PNP Device ID: $($_.PNPDeviceID)"
            Write-Host ""
        }
        '''
        
        result = subprocess.run([
            'powershell', '-Command', ps_command
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print("Windows GPU Detection Results:")
            print(result.stdout)
            return True
        else:
            print("Windows GPU detection failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Windows GPU check failed: {e}")
        return False

def check_device_manager_gpus():
    """Check GPU status in Device Manager."""
    try:
        print("Checking Device Manager GPU status...")
        
        # PowerShell command to check device status
        ps_command = '''
        Get-PnpDevice | Where-Object { $_.Class -eq "Display" -and $_.FriendlyName -like "*AMD*" } | ForEach-Object {
            Write-Host "Device: $($_.FriendlyName)"
            Write-Host "  Status: $($_.Status)"
            Write-Host "  Problem: $($_.Problem)"
            Write-Host "  Instance ID: $($_.InstanceId)"
            Write-Host ""
        }
        '''
        
        result = subprocess.run([
            'powershell', '-Command', ps_command
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print("Device Manager Results:")
            print(result.stdout)
            return True
        else:
            print("Device Manager check failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Device Manager check failed: {e}")
        return False

def check_opencl_gpus():
    """Check GPUs visible to OpenCL."""
    try:
        print("Checking OpenCL GPU detection...")
        
        import pyopencl as cl
        
        platforms = cl.get_platforms()
        opencl_gpus = []
        
        for platform in platforms:
            print(f"Platform: {platform.name}")
            print(f"  Vendor: {platform.vendor}")
            print(f"  Version: {platform.version}")
            
            try:
                devices = platform.get_devices(cl.device_type.GPU)
                
                for i, device in enumerate(devices):
                    device_name = device.name.strip()
                    memory_gb = device.global_mem_size // (1024**3)
                    compute_units = device.max_compute_units
                    max_work_group = device.max_work_group_size
                    
                    gpu_info = {
                        'platform': platform.name,
                        'index': i,
                        'name': device_name,
                        'memory': memory_gb,
                        'compute_units': compute_units,
                        'max_work_group': max_work_group,
                        'device': device
                    }
                    
                    opencl_gpus.append(gpu_info)
                    
                    print(f"  GPU {i}: {device_name}")
                    print(f"    Memory: {memory_gb} GB")
                    print(f"    Compute Units: {compute_units}")
                    print(f"    Max Work Group: {max_work_group}")
                    print(f"    Vendor: {device.vendor}")
                    print(f"    Version: {device.version}")
                    print()
                    
            except Exception as e:
                print(f"  Error getting devices: {e}")
        
        return opencl_gpus
        
    except ImportError:
        print("OpenCL not available (pyopencl not installed)")
        return []
    except Exception as e:
        print(f"OpenCL check failed: {e}")
        return []

def check_amd_drivers():
    """Check AMD driver installation."""
    try:
        print("Checking AMD driver installation...")
        
        # Check for AMD software
        amd_paths = [
            r"C:\Program Files\AMD\CNext\CNext\RadeonSoftware.exe",
            r"C:\Program Files\AMD\CNext\CNext\AMDSoftware.exe",
            r"C:\AMD\CNext\CNext\RadeonSoftware.exe"
        ]
        
        amd_software_found = False
        for path in amd_paths:
            if Path(path).exists():
                print(f"   AMD Software found: {path}")
                amd_software_found = True
                break
        
        if not amd_software_found:
            print("   AMD Software not found")
        
        # Check for OpenCL drivers
        opencl_paths = [
            r"C:\Windows\System32\amdocl64.dll",
            r"C:\Windows\System32\OpenCL.dll"
        ]
        
        opencl_found = False
        for path in opencl_paths:
            if Path(path).exists():
                print(f"   OpenCL driver found: {path}")
                opencl_found = True
            else:
                print(f"   OpenCL driver missing: {path}")
        
        return amd_software_found and opencl_found
        
    except Exception as e:
        print(f"AMD driver check failed: {e}")
        return False

def analyze_gpu_situation():
    """Analyze the GPU situation and provide recommendations."""
    print("\n" + "="*60)
    print("GPU SITUATION ANALYSIS")
    print("="*60)
    
    # Check all detection methods
    windows_gpus = check_windows_gpus()
    print()
    
    device_manager_gpus = check_device_manager_gpus()
    print()
    
    opencl_gpus = check_opencl_gpus()
    print()
    
    amd_drivers_ok = check_amd_drivers()
    print()
    
    # Analyze results
    print("ANALYSIS RESULTS:")
    print("="*40)
    
    if windows_gpus:
        print(" Windows can see AMD GPUs")
    else:
        print(" Windows cannot detect AMD GPUs properly")
    
    if device_manager_gpus:
        print(" Device Manager shows AMD GPUs")
    else:
        print(" Device Manager has issues with AMD GPUs")
    
    if opencl_gpus:
        print(f" OpenCL can see {len(opencl_gpus)} AMD GPU(s)")
        
        # Check for RX 7700S specifically
        rx7700s_found = False
        for gpu in opencl_gpus:
            if gpu['memory'] >= 10 or '7700' in gpu['name'] or 'gfx1103' in gpu['name']:
                rx7700s_found = True
                print(f" RX 7700S found in OpenCL: {gpu['name']} ({gpu['memory']} GB)")
                break
        
        if not rx7700s_found:
            print(" RX 7700S not found in OpenCL")
            print("   This explains why the brain GUI uses GPU 0")
    else:
        print(" OpenCL cannot see any AMD GPUs")
    
    if amd_drivers_ok:
        print(" AMD drivers appear to be installed")
    else:
        print(" AMD drivers may be missing or incomplete")
    
    # Provide recommendations
    print("\nRECOMMENDations:")
    print("="*40)
    
    if not opencl_gpus or len(opencl_gpus) < 2:
        print(" ISSUE: OpenCL cannot see RX 7700S")
        print("   Solutions:")
        print("   1. Update AMD drivers to latest version")
        print("   2. Reinstall AMD Adrenalin software")
        print("   3. Check Device Manager for disabled/error GPUs")
        print("   4. Restart computer after driver update")
        print()
        
        print("   Driver download:")
        print("   https://www.amd.com/en/support/graphics/amd-radeon-rx-7000-series/amd-radeon-rx-7700s")
    
    elif len(opencl_gpus) >= 2:
        print(" Both GPUs visible to OpenCL")
        print("   The brain GUI should be able to use GPU 1")
        print("   Try running: python quick_start_gui.py")
    
    print("\n   Manual GPU disable method still available:")
    print("   Device Manager → Display adapters → Disable AMD Radeon 780M")

def main():
    """Main comprehensive GPU check."""
    print("Comprehensive GPU Check")
    print("="*60)
    print("This will check GPU detection using multiple methods")
    print("to diagnose why OpenCL cannot see the RX 7700S")
    print()
    
    analyze_gpu_situation()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()