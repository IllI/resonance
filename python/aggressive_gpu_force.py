#!/usr/bin/env python3
"""
Aggressive GPU Force - More aggressive approach to force RX 7700S usage
"""

import os
import sys
import subprocess
import winreg
import ctypes
import time
from pathlib import Path

class AggressiveGPUForce:
    """Aggressive approach to force RX 7700S usage"""
    
    def __init__(self):
        self.python_path = sys.executable
        self.setup_aggressive_environment()
    
    def setup_aggressive_environment(self):
        """Set up aggressive environment variables"""
        print(" Setting up aggressive GPU forcing...")
        
        # Set all possible AMD environment variables
        amd_vars = {
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
            'AMD_GPU_DEVICE_ORDINAL': '0',
            'GPU_DEVICE_ORDINAL': '0',
            'AMD_OPENCL_DEVICE_ORDINAL': '0',
            'OPENCL_DEVICE_ORDINAL': '0',
            'AMD_GPU_INDEX': '0',
            'GPU_INDEX': '0',
            'AMD_DISCRETE_GPU': '1',
            'USE_DISCRETE_GPU': '1',
            'FORCE_DISCRETE_GPU': '1'
        }
        
        for var, value in amd_vars.items():
            os.environ[var] = value
            print(f"   {var}={value}")
    
    def set_aggressive_registry_settings(self):
        """Set aggressive registry settings"""
        print(" Setting aggressive registry settings...")
        
        try:
            # Windows Graphics Settings
            reg_path = r"SOFTWARE\Microsoft\DirectX\UserGpuPreferences"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
                # Set Python to use high performance GPU
                winreg.SetValueEx(key, self.python_path, 0, winreg.REG_SZ, "GpuPreference=2;")
                
                # Set for all Python variants
                python_dir = Path(self.python_path).parent
                for variant in ["python.exe", "python3.exe", "pythonw.exe", "python3w.exe"]:
                    variant_path = python_dir / variant
                    if variant_path.exists():
                        winreg.SetValueEx(key, str(variant_path), 0, winreg.REG_SZ, "GpuPreference=2;")
                
                # Set for the script itself
                script_path = Path(__file__).resolve()
                winreg.SetValueEx(key, str(script_path), 0, winreg.REG_SZ, "GpuPreference=2;")
            
            # AMD Radeon Settings
            amd_path = r"SOFTWARE\AMD\DxDiag"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, amd_path) as key:
                winreg.SetValueEx(key, "AppProfile", 0, winreg.REG_SZ, self.python_path)
                winreg.SetValueEx(key, "GPUPreference", 0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(key, "DiscreteGPU", 0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(key, "ForceDiscreteGPU", 0, winreg.REG_DWORD, 1)
            
            print(" Set aggressive registry settings")
            
        except Exception as e:
            print(f" Failed to set registry settings: {e}")
    
    def run_aggressive_powershell_commands(self):
        """Run aggressive PowerShell commands"""
        print(" Running aggressive PowerShell commands...")
        
        commands = [
            f'Set-ItemProperty -Path "HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences" -Name "{self.python_path}" -Value "GpuPreference=2;" -Force',
            f'Set-ItemProperty -Path "HKCU:\\SOFTWARE\\AMD\\DxDiag" -Name "GPUPreference" -Value 1 -Type DWord -Force',
            f'Set-ItemProperty -Path "HKCU:\\SOFTWARE\\AMD\\DxDiag" -Name "DiscreteGPU" -Value 1 -Type DWord -Force',
            f'Set-ItemProperty -Path "HKCU:\\SOFTWARE\\AMD\\DxDiag" -Name "ForceDiscreteGPU" -Value 1 -Type DWord -Force',
            '$env:AmdPowerXpressRequestHighPerformance = "1"',
            '$env:AMD_OPENCL_DEVICE = "0"',
            '$env:OPENCL_DEVICE = "0"',
            '$env:GPU_DEVICE_ORDINAL = "0"',
            '$env:AMD_DISABLE_INTEGRATED_GPU = "1"',
            '$env:OPENCL_DISABLE_DEVICE_1 = "1"'
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
    
    def test_with_explicit_device_selection(self):
        """Test with explicit device selection"""
        print("\n Testing with explicit device selection...")
        
        try:
            import pyopencl as cl
            import numpy as np
            
            # Get all platforms and devices
            platforms = cl.get_platforms()
            
            for i, platform in enumerate(platforms):
                print(f"Platform {i}: {platform.name}")
                devices = platform.get_devices()
                
                for j, device in enumerate(devices):
                    device_name = device.name
                    print(f"  Device {j}: {device_name}")
                    
                    # Try to force specific device usage
                    try:
                        print(f"\n Forcing Device {j}: {device_name}")
                        
                        # Create context with only this device
                        context = cl.Context([device])
                        queue = cl.CommandQueue(context)
                        
                        # Large workload
                        size = 1024 * 1024 * 6  # 6M elements
                        a = np.random.rand(size).astype(np.float32)
                        b = np.random.rand(size).astype(np.float32)
                        
                        a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                        b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                        c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
                        
                        # Very intensive kernel
                        kernel_code = """
                        __kernel void intensive_kernel(__global const float* a,
                                                     __global const float* b,
                                                     __global float* c) {
                            int gid = get_global_id(0);
                            float temp = a[gid] * b[gid];
                            
                            // Very intensive computation
                            for(int i = 0; i < 50000; i++) {
                                temp = sqrt(temp);
                                temp = sin(temp) + cos(temp);
                                temp = temp * temp;
                                temp = log(temp + 1.0f);
                                temp = exp(temp);
                            }
                            c[gid] = temp;
                        }
                        """
                        
                        program = cl.Program(context, kernel_code).build()
                        kernel = program.intensive_kernel
                        
                        print(f" Running intensive workload on {device_name}...")
                        print(" Check Task Manager NOW!")
                        print("   This should show GPU activity")
                        
                        # Run multiple iterations
                        for iteration in range(5):
                            print(f"    Iteration {iteration+1}/5...")
                            
                            start_time = time.time()
                            kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
                            queue.finish()
                            end_time = time.time()
                            
                            execution_time = (end_time - start_time) * 1000
                            print(f"      Time: {execution_time:.2f} ms")
                            
                            # Small delay to see activity in Task Manager
                            time.sleep(1)
                        
                        print(f" Completed intensive workload on {device_name}")
                        print(" Check Task Manager - which GPU showed activity?")
                        
                        # Ask user which GPU showed activity
                        print(f"\n❓ Which GPU showed activity in Task Manager?")
                        print(f"   - GPU 0 (RX 7700S) - 12GB memory")
                        print(f"   - GPU 1 (780M) - 7GB memory")
                        
                    except Exception as e:
                        print(f" Test failed for {device_name}: {e}")
                        continue
                        
        except Exception as e:
            print(f" GPU test failed: {e}")
    
    def create_system_restart_script(self):
        """Create a script that suggests system restart"""
        script = '''
# System Restart Script for GPU Configuration
# Sometimes GPU configuration changes require a system restart

Write-Host "=== GPU Configuration Restart Script ===" -ForegroundColor Green
Write-Host "GPU configuration changes may require a system restart" -ForegroundColor Yellow

Write-Host "`nCurrent GPU Configuration:" -ForegroundColor Cyan
Write-Host "1. Windows Graphics Settings configured" -ForegroundColor White
Write-Host "2. AMD Radeon Settings configured" -ForegroundColor White
Write-Host "3. Registry settings applied" -ForegroundColor White
Write-Host "4. Environment variables set" -ForegroundColor White

Write-Host "`nRecommendations:" -ForegroundColor Cyan
Write-Host "1. Restart your computer" -ForegroundColor Yellow
Write-Host "2. After restart, run the GPU test again" -ForegroundColor Yellow
Write-Host "3. Check Task Manager during GPU tests" -ForegroundColor Yellow

Write-Host "`nIf restart doesn't help:" -ForegroundColor Cyan
Write-Host "1. Check AMD driver updates" -ForegroundColor White
Write-Host "2. Try different Python environment" -ForegroundColor White
Write-Host "3. Consider using CUDA instead of OpenCL" -ForegroundColor White
Write-Host "4. Use cloud computing with proper GPU support" -ForegroundColor White

Write-Host "`nPress any key to continue..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
'''
        
        with open('gpu_restart_script.ps1', 'w') as f:
            f.write(script)
        
        print(" Created gpu_restart_script.ps1")

def main():
    """Main function"""
    print("=== Aggressive GPU Force ===")
    print("More aggressive approach to force RX 7700S usage")
    
    # Create aggressive GPU force controller
    gpu_force = AggressiveGPUForce()
    
    # Apply aggressive configurations
    gpu_force.set_aggressive_registry_settings()
    gpu_force.run_aggressive_powershell_commands()
    gpu_force.create_system_restart_script()
    
    # Test with explicit device selection
    gpu_force.test_with_explicit_device_selection()
    
    print(f"\n{'='*60}")
    print("AGGRESSIVE SOLUTION SUMMARY:")
    print("="*60)
    print("1. Set all possible AMD environment variables")
    print("2. Applied aggressive registry settings")
    print("3. Ran aggressive PowerShell commands")
    print("4. Tested with explicit device selection")
    print("5. Created system restart script")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("="*60)
    print("1. Check Task Manager while the test runs")
    print("2. If still using 780M:")
    print("   - Restart your computer")
    print("   - Run gpu_restart_script.ps1")
    print("   - Check AMD driver updates")
    print("3. If GPU 0 (RX 7700S) shows activity:")
    print("   - SUCCESS! Aggressive approach worked")
    print("   - Your brain analysis will use RX 7700S")

if __name__ == "__main__":
    main()

