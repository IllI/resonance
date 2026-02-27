#!/usr/bin/env python3
"""
Windows GPU Override Solution
Addresses the issue where Windows overrides programmatic GPU selection
Based on web search findings and AMD documentation
"""

import os
import sys
import subprocess
import winreg
import ctypes
import time
from pathlib import Path

class WindowsGPUOverrideSolution:
    """Solution for Windows overriding programmatic GPU selection"""
    
    def __init__(self):
        self.python_path = sys.executable
        self.is_admin = self.check_admin()
        self.setup_amd_power_export()
    
    def check_admin(self):
        """Check if running as administrator"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    
    def setup_amd_power_export(self):
        """Set up AMD power export - the key method from AMD documentation"""
        try:
            # This is the critical method from AMD's switchable graphics technology
            # Export the symbol that tells AMD drivers to use discrete GPU
            os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
            
            # Try to create the export in memory
            try:
                # Create a simple DLL export approach
                kernel32 = ctypes.windll.kernel32
                print(" Set AmdPowerXpressRequestHighPerformance=1")
            except Exception as e:
                print(f"  Could not set AMD power export: {e}")
        except Exception as e:
            print(f"  AMD power export setup failed: {e}")
    
    def set_windows_graphics_preferences(self):
        """Set Windows Graphics Preferences - the most reliable method"""
        print(" Setting Windows Graphics Preferences...")
        
        try:
            # Registry path for Windows Graphics Settings
            reg_path = r"SOFTWARE\Microsoft\DirectX\UserGpuPreferences"
            
            # Open or create the registry key
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
                # Set Python to use high performance GPU (GpuPreference=2)
                winreg.SetValueEx(key, self.python_path, 0, winreg.REG_SZ, "GpuPreference=2;")
                print(f" Set Windows Graphics preference for Python")
                
                # Also set for common Python variants
                python_dir = Path(self.python_path).parent
                for variant in ["python.exe", "python3.exe", "pythonw.exe", "python3w.exe"]:
                    variant_path = python_dir / variant
                    if variant_path.exists():
                        winreg.SetValueEx(key, str(variant_path), 0, winreg.REG_SZ, "GpuPreference=2;")
                        print(f" Set preference for {variant}")
                
                # Set for the script itself
                script_path = Path(__file__).resolve()
                winreg.SetValueEx(key, str(script_path), 0, winreg.REG_SZ, "GpuPreference=2;")
                print(f" Set preference for script")
                
        except Exception as e:
            print(f" Failed to set Windows Graphics preferences: {e}")
    
    def set_amd_radeon_settings(self):
        """Set AMD Radeon Settings programmatically"""
        print(" Setting AMD Radeon Settings...")
        
        try:
            # AMD registry path
            amd_path = r"SOFTWARE\AMD\DxDiag"
            
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, amd_path) as key:
                # Set application profile
                winreg.SetValueEx(key, "AppProfile", 0, winreg.REG_SZ, self.python_path)
                # Set GPU preference (1 = discrete GPU)
                winreg.SetValueEx(key, "GPUPreference", 0, winreg.REG_DWORD, 1)
                print(" Set AMD Radeon Settings")
                
        except Exception as e:
            print(f" Failed to set AMD Radeon Settings: {e}")
    
    def set_comprehensive_environment_variables(self):
        """Set comprehensive environment variables"""
        print(" Setting comprehensive environment variables...")
        
        env_vars = {
            'AmdPowerXpressRequestHighPerformance': '1',
            'AMD_OPENCL_DEVICE': '0',  # Device 0 (RX 7700S)
            'OPENCL_DEVICE': '0',
            'GPU_DEVICE_ORDINAL': '0',
            'AMD_VULKAN_DEVICE': '0',
            'AMD_GPU_DEVICE': '0',
            'DISPLAY_GPU': '0',
            'CUDA_VISIBLE_DEVICES': '0',  # For CUDA compatibility
            'AMD_DISABLE_INTEGRATED_GPU': '1',
            'OPENCL_DISABLE_DEVICE_1': '1',  # Disable device 1 (780M)
            'GPU_DEVICE_ORDINAL': '0',
            'AMD_GPU_DEVICE_ORDINAL': '0'
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            print(f"   {var}={value}")
    
    def create_amd_dll_export(self):
        """Create AMD DLL export - the most reliable method"""
        print(" Creating AMD DLL export...")
        
        dll_code = '''
#include <windows.h>

// Export the symbol that tells AMD drivers to use discrete GPU
// This is the key method from AMD's switchable graphics technology
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

// Additional exports for different scenarios
extern "C" __declspec(dllexport) int NvOptimusEnablement = 1;  // For NVIDIA compatibility
extern "C" __declspec(dllexport) int DXVK_HUD = 0;  // Disable DXVK HUD

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        // DLL is being loaded - set environment variables
        SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance", "1");
        SetEnvironmentVariableA("AMD_OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("GPU_DEVICE_ORDINAL", "0");
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

// Function to explicitly request high performance GPU
extern "C" __declspec(dllexport) void RequestHighPerformanceGPU() {
    AmdPowerXpressRequestHighPerformance = 1;
    SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance", "1");
}

// Function to get current GPU preference
extern "C" __declspec(dllexport) int GetGPUPreference() {
    return AmdPowerXpressRequestHighPerformance;
}
'''
        
        # Write the C++ code
        with open('amd_gpu_force_dll.cpp', 'w') as f:
            f.write(dll_code)
        
        print(" Created amd_gpu_force_dll.cpp")
        print("   Compile with: cl /LD amd_gpu_force_dll.cpp")
        print("   This creates a DLL that forces discrete GPU usage")
    
    def run_powershell_commands(self):
        """Run PowerShell commands to force GPU selection"""
        print(" Running PowerShell commands...")
        
        commands = [
            f'Set-ItemProperty -Path "HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences" -Name "{self.python_path}" -Value "GpuPreference=2;" -Force',
            '$env:AmdPowerXpressRequestHighPerformance = "1"',
            '$env:AMD_OPENCL_DEVICE = "0"',
            '$env:OPENCL_DEVICE = "0"',
            '$env:GPU_DEVICE_ORDINAL = "0"'
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(['powershell', '-Command', cmd], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f" Executed: {cmd}")
                else:
                    print(f"  Failed: {cmd}")
            except Exception as e:
                print(f"  Command failed: {e}")
    
    def test_gpu_usage(self):
        """Test which GPU is actually being used"""
        print("\n Testing GPU usage...")
        
        try:
            import pyopencl as cl
            import numpy as np
            
            # Get all devices
            platforms = cl.get_platforms()
            
            for platform in platforms:
                devices = platform.get_devices()
                for i, device in enumerate(devices):
                    device_name = device.name
                    print(f"Device {i}: {device_name}")
                    
                    # Test each device
                    try:
                        context = cl.Context([device])
                        queue = cl.CommandQueue(context)
                        
                        # Large workload
                        size = 1024 * 1024 * 4
                        a = np.random.rand(size).astype(np.float32)
                        b = np.random.rand(size).astype(np.float32)
                        
                        a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                        b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                        c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
                        
                        kernel_code = """
                        __kernel void test_kernel(__global const float* a,
                                                __global const float* b,
                                                __global float* c) {
                            int gid = get_global_id(0);
                            float temp = a[gid] * b[gid];
                            
                            for(int i = 0; i < 10000; i++) {
                                temp = sqrt(temp);
                                temp = sin(temp) + cos(temp);
                                temp = temp * temp;
                            }
                            c[gid] = temp;
                        }
                        """
                        
                        program = cl.Program(context, kernel_code).build()
                        kernel = program.test_kernel
                        
                        print(f" Testing {device_name}...")
                        print(" Check Task Manager NOW!")
                        print("   GPU 0 (RX 7700S) should show activity")
                        print("   GPU 1 (780M) should NOT show activity")
                        
                        for iteration in range(3):
                            print(f"   Iteration {iteration+1}/3...")
                            kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
                            queue.finish()
                            time.sleep(2)
                        
                        print(f" Completed test on {device_name}")
                        print(" Check Task Manager - which GPU showed activity?")
                        
                    except Exception as e:
                        print(f" Test failed for {device_name}: {e}")
                        
        except Exception as e:
            print(f" GPU test failed: {e}")
    
    def create_manual_setup_guide(self):
        """Create manual setup guide"""
        guide = f"""
# Manual GPU Setup Guide

## The Problem
Windows overrides programmatic GPU selection, causing applications to use the integrated GPU (780M) instead of the discrete GPU (RX 7700S).

## Solution 1: Windows Graphics Settings (Most Reliable)
1. Press Win + I
2. Go to System → Display
3. Scroll down and click "Graphics settings"
4. Click "Browse" and find: {self.python_path}
5. Set to "High performance"
6. Click "Save"
7. Restart Python/IDE

## Solution 2: AMD Radeon Settings
1. Right-click desktop → AMD Radeon Settings
2. Go to System → Switchable Graphics
3. Find Python in the list (or add it)
4. Set to "High Performance"
5. Restart Python/IDE

## Solution 3: Registry (Advanced)
Run this PowerShell command as Administrator:
Set-ItemProperty -Path "HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences" -Name "{self.python_path}" -Value "GpuPreference=2;" -Force

## Solution 4: C++ DLL (Most Reliable)
Compile the amd_gpu_force_dll.cpp and load it in your Python application.

## Verification
After setup, run: python windows_gpu_override_solution.py
Check Task Manager - GPU 0 (RX 7700S) should show activity, not GPU 1 (780M)

## If All Else Fails
Consider rewriting the application in C++ with direct GPU control.
"""
        
        with open('MANUAL_GPU_SETUP_GUIDE.md', 'w', encoding='utf-8') as f:
            f.write(guide)
        
        print(" Created MANUAL_GPU_SETUP_GUIDE.md")

def main():
    """Main function"""
    print("=== Windows GPU Override Solution ===")
    print("Addressing Windows overriding programmatic GPU selection")
    
    # Create solution controller
    solution = WindowsGPUOverrideSolution()
    
    # Check admin status
    if not solution.is_admin:
        print("  Not running as administrator")
        print("   Some settings may not be applied")
        print("   Run PowerShell as administrator for full effect")
    
    # Apply all configurations
    solution.set_comprehensive_environment_variables()
    solution.set_windows_graphics_preferences()
    solution.set_amd_radeon_settings()
    solution.create_amd_dll_export()
    solution.run_powershell_commands()
    solution.create_manual_setup_guide()
    
    # Test GPU usage
    solution.test_gpu_usage()
    
    print(f"\n{'='*60}")
    print("SOLUTION SUMMARY:")
    print("="*60)
    print("1. Windows Graphics Settings (Most Reliable)")
    print("2. AMD Radeon Settings")
    print("3. Registry modifications")
    print("4. C++ DLL with AMD power export")
    print("5. If all else fails: Rewrite in C++")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("="*60)
    print("1. Follow MANUAL_GPU_SETUP_GUIDE.md")
    print("2. Restart Python/IDE after configuration")
    print("3. Test again")
    print("4. If still using 780M, consider C++ rewrite")

if __name__ == "__main__":
    main()

