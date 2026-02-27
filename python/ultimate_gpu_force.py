#!/usr/bin/env python3
"""
Ultimate GPU Force - Force Windows to use RX 7700S at system level
This addresses the core issue: Windows routing workloads to integrated GPU
"""

import os
import sys
import subprocess
import winreg
import ctypes
from pathlib import Path

class UltimateGPUForce:
    """Ultimate solution to force RX 7700S usage"""
    
    def __init__(self):
        self.python_path = sys.executable
        self.script_path = Path(__file__).resolve()
        self.is_admin = self.check_admin()
    
    def check_admin(self):
        """Check if running as administrator"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    
    def set_windows_graphics_preferences(self):
        """Set Windows Graphics Preferences to force discrete GPU"""
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
                winreg.SetValueEx(key, str(self.script_path), 0, winreg.REG_SZ, "GpuPreference=2;")
                print(f" Set preference for script")
                
        except Exception as e:
            print(f" Failed to set Windows Graphics preferences: {e}")
    
    def set_amd_radeon_settings(self):
        """Set AMD Radeon Settings"""
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
    
    def set_environment_variables(self):
        """Set comprehensive environment variables"""
        print(" Setting environment variables...")
        
        env_vars = {
            'AmdPowerXpressRequestHighPerformance': '1',
            'AMD_OPENCL_DEVICE': '0',  # Try device 0 (RX 7700S)
            'OPENCL_DEVICE': '0',
            'GPU_DEVICE_ORDINAL': '0',
            'AMD_VULKAN_DEVICE': '0',
            'AMD_GPU_DEVICE': '0',
            'DISPLAY_GPU': '0',
            'CUDA_VISIBLE_DEVICES': '0',
            'AMD_DISABLE_INTEGRATED_GPU': '1',
            'OPENCL_DISABLE_DEVICE_1': '1'  # Disable device 1 (780M)
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            print(f"   {var}={value}")
    
    def create_gpu_force_dll(self):
        """Create a DLL that exports AmdPowerXpressRequestHighPerformance"""
        print(" Creating GPU force DLL...")
        
        dll_code = '''
#include <windows.h>

// Export the symbol that tells AMD drivers to use discrete GPU
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        // Set environment variables when DLL is loaded
        SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance", "1");
        SetEnvironmentVariableA("AMD_OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("OPENCL_DEVICE", "0");
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
'''
        
        # Write the C++ code
        with open('ultimate_gpu_force.cpp', 'w') as f:
            f.write(dll_code)
        
        print(" Created ultimate_gpu_force.cpp")
        print("   Compile with: cl /LD ultimate_gpu_force.cpp")
    
    def run_powershell_commands(self):
        """Run PowerShell commands to force GPU selection"""
        print(" Running PowerShell commands...")
        
        commands = [
            f'Set-ItemProperty -Path "HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences" -Name "{self.python_path}" -Value "GpuPreference=2;" -Force',
            '$env:AmdPowerXpressRequestHighPerformance = "1"',
            '$env:AMD_OPENCL_DEVICE = "0"',
            '$env:OPENCL_DEVICE = "0"'
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
    
    def create_manual_setup_guide(self):
        """Create a manual setup guide"""
        guide = '''
# Manual GPU Setup Guide

## Method 1: Windows Graphics Settings (Recommended)
1. Press Win + I
2. Go to System → Display
3. Scroll down and click "Graphics settings"
4. Click "Browse" and find: ''' + self.python_path + '''
5. Set to "High performance"
6. Click "Save"
7. Restart Python/IDE

## Method 2: AMD Radeon Settings
1. Right-click desktop → AMD Radeon Settings
2. Go to System → Switchable Graphics
3. Find Python in the list (or add it)
4. Set to "High Performance"
5. Restart Python/IDE

## Method 3: Registry (Advanced)
Run this PowerShell command as Administrator:
Set-ItemProperty -Path "HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences" -Name "''' + self.python_path + '''" -Value "GpuPreference=2;" -Force

## Verification
After setup, run: python direct_gpu0_test.py
Check Task Manager - GPU 0 (RX 7700S) should show activity, not GPU 1 (780M)
'''
        
        with open('MANUAL_GPU_SETUP.md', 'w') as f:
            f.write(guide)
        
        print(" Created MANUAL_GPU_SETUP.md")
    
    def test_gpu_usage(self):
        """Test which GPU is actually being used"""
        print("\n Testing GPU usage...")
        
        try:
            import pyopencl as cl
            import numpy as np
            import time
            
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

def main():
    """Main function"""
    print("=== Ultimate GPU Force ===")
    print("Forcing Windows to use RX 7700S at system level")
    
    # Create ultimate GPU force controller
    gpu_force = UltimateGPUForce()
    
    # Check admin status
    if not gpu_force.is_admin:
        print("  Not running as administrator")
        print("   Some settings may not be applied")
        print("   Run PowerShell as administrator for full effect")
    
    # Apply all configurations
    gpu_force.set_environment_variables()
    gpu_force.set_windows_graphics_preferences()
    gpu_force.set_amd_radeon_settings()
    gpu_force.create_gpu_force_dll()
    gpu_force.run_powershell_commands()
    gpu_force.create_manual_setup_guide()
    
    # Test GPU usage
    gpu_force.test_gpu_usage()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("="*60)
    print("1. Check Task Manager while the test runs")
    print("2. If still using GPU 1 (780M):")
    print("   - Follow MANUAL_GPU_SETUP.md")
    print("   - Or run PowerShell as administrator")
    print("   - Or manually set Windows Graphics Settings")
    print("3. Restart Python/IDE after configuration")
    print("4. Run this test again")
    print("5. GPU 0 (RX 7700S) should show activity")

if __name__ == "__main__":
    main()

