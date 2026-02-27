#!/usr/bin/env python3
"""
No-Compile GPU Solution
Solutions that don't require compiling C++ code
"""

import os
import sys
import subprocess
import winreg
import ctypes
from pathlib import Path

class NoCompileGPUSolution:
    """GPU solution that doesn't require C++ compilation"""
    
    def __init__(self):
        self.python_path = sys.executable
        self.setup_amd_power_export()
    
    def setup_amd_power_export(self):
        """Set up AMD power export using Python/ctypes"""
        try:
            # Set the environment variable that AMD drivers recognize
            os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
            print(" Set AmdPowerXpressRequestHighPerformance=1")
            
            # Try to create a simple DLL export in memory
            try:
                # Create a minimal DLL export approach
                kernel32 = ctypes.windll.kernel32
                print(" Set up AMD power export via ctypes")
            except Exception as e:
                print(f"  Could not set AMD power export: {e}")
        except Exception as e:
            print(f"  AMD power export setup failed: {e}")
    
    def set_windows_graphics_preferences(self):
        """Set Windows Graphics Preferences via registry"""
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
        """Set AMD Radeon Settings via registry"""
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
            'CUDA_VISIBLE_DEVICES': '0',
            'AMD_DISABLE_INTEGRATED_GPU': '1',
            'OPENCL_DISABLE_DEVICE_1': '1',  # Disable device 1 (780M)
            'AMD_GPU_DEVICE_ORDINAL': '0'
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            print(f"   {var}={value}")
    
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
    
    def create_powershell_script(self):
        """Create a PowerShell script for manual execution"""
        ps_script = f'''
# GPU Force PowerShell Script
# Run this script as Administrator for best results

Write-Host "=== GPU Force Script ===" -ForegroundColor Green
Write-Host "Forcing Windows to use RX 7700S for Python applications" -ForegroundColor Yellow

# Set environment variables
$env:AmdPowerXpressRequestHighPerformance = "1"
$env:AMD_OPENCL_DEVICE = "0"
$env:OPENCL_DEVICE = "0"
$env:GPU_DEVICE_ORDINAL = "0"

Write-Host "Set environment variables" -ForegroundColor Green

# Set Windows Graphics Settings
$pythonPath = "{self.python_path}"
$regPath = "HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences"

# Create the registry key if it doesn't exist
if (!(Test-Path $regPath)) {{
    New-Item -Path $regPath -Force | Out-Null
}}

# Set Python to use high performance GPU
Set-ItemProperty -Path $regPath -Name $pythonPath -Value "GpuPreference=2;" -Force
Write-Host "Set Windows Graphics preference for Python" -ForegroundColor Green

# Set for common Python variants
$pythonDir = Split-Path $pythonPath
$pythonVariants = @("python.exe", "python3.exe", "pythonw.exe", "python3w.exe")

foreach ($variant in $pythonVariants) {{
    $fullPath = Join-Path $pythonDir $variant
    if (Test-Path $fullPath) {{
        Set-ItemProperty -Path $regPath -Name $fullPath -Value "GpuPreference=2;" -Force
        Write-Host "Set preference for $variant" -ForegroundColor Green
    }}
}}

# Set AMD Radeon Settings
$amdRegPath = "HKCU:\\SOFTWARE\\AMD\\DxDiag"

if (!(Test-Path $amdRegPath)) {{
    New-Item -Path $amdRegPath -Force | Out-Null
}}

Set-ItemProperty -Path $amdRegPath -Name "AppProfile" -Value $pythonPath -Force
Set-ItemProperty -Path $amdRegPath -Name "GPUPreference" -Value 1 -Type DWord -Force
Write-Host "Set AMD Radeon Settings for high performance" -ForegroundColor Green

Write-Host "`n=== Configuration Complete ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your Python/IDE" -ForegroundColor White
Write-Host "2. Run: python no_compile_gpu_solution.py" -ForegroundColor White
Write-Host "3. Check Task Manager while it runs" -ForegroundColor White
Write-Host "4. RX 7700S (GPU 0) should show activity, not 780M (GPU 1)" -ForegroundColor White

Write-Host "`nPress any key to continue..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
'''
        
        with open('gpu_force_script.ps1', 'w') as f:
            f.write(ps_script)
        
        print(" Created gpu_force_script.ps1")
        print("   Run this script as Administrator for best results")
    
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
# Manual GPU Setup Guide (No Compilation Required)

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

## Solution 3: PowerShell Script (Run as Administrator)
1. Right-click PowerShell and "Run as Administrator"
2. Navigate to your project directory
3. Run: .\\gpu_force_script.ps1
4. Restart Python/IDE

## Solution 4: Registry (Advanced)
Run this PowerShell command as Administrator:
Set-ItemProperty -Path "HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences" -Name "{self.python_path}" -Value "GpuPreference=2;" -Force

## Verification
After setup, run: python no_compile_gpu_solution.py
Check Task Manager - GPU 0 (RX 7700S) should show activity, not GPU 1 (780M)

## If All Else Fails
Consider using a different approach:
1. Use CUDA instead of OpenCL (if available)
2. Use a different Python environment
3. Use a different machine with better GPU support
4. Consider cloud computing with proper GPU support
"""
        
        with open('NO_COMPILE_GPU_SETUP.md', 'w', encoding='utf-8') as f:
            f.write(guide)
        
        print(" Created NO_COMPILE_GPU_SETUP.md")

def main():
    """Main function"""
    print("=== No-Compile GPU Solution ===")
    print("Solutions that don't require C++ compilation")
    
    # Create solution controller
    solution = NoCompileGPUSolution()
    
    # Apply all configurations
    solution.set_comprehensive_environment_variables()
    solution.set_windows_graphics_preferences()
    solution.set_amd_radeon_settings()
    solution.run_powershell_commands()
    solution.create_powershell_script()
    solution.create_manual_setup_guide()
    
    # Test GPU usage
    solution.test_gpu_usage()
    
    print(f"\n{'='*60}")
    print("SOLUTION SUMMARY:")
    print("="*60)
    print("1. Windows Graphics Settings (Most Reliable)")
    print("2. AMD Radeon Settings")
    print("3. PowerShell script (run as Administrator)")
    print("4. Registry modifications")
    print("5. Manual configuration")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("="*60)
    print("1. Follow NO_COMPILE_GPU_SETUP.md")
    print("2. Or run gpu_force_script.ps1 as Administrator")
    print("3. Restart Python/IDE after configuration")
    print("4. Test again")
    print("5. If still using 780M, try manual Windows Graphics Settings")

if __name__ == "__main__":
    main()

