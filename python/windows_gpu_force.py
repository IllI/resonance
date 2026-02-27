#!/usr/bin/env python3
"""
Windows GPU Force - Force Windows to use RX 7700S at the system level
This addresses the issue where OpenCL selects the right GPU but Windows still uses the integrated GPU
"""

import os
import sys
import ctypes
import winreg
import subprocess
import time
from pathlib import Path

class WindowsGPUForce:
    """Force Windows to use RX 7700S at the system level"""
    
    def __init__(self):
        self.python_path = sys.executable
        self.script_path = Path(__file__).resolve()
        self.setup_amd_power_export()
    
    def setup_amd_power_export(self):
        """Set up AMD power export at the process level"""
        try:
            # This is the key method from AMD documentation
            # Export the symbol that tells AMD drivers to use discrete GPU
            if hasattr(ctypes, 'windll'):
                # Set environment variable
                os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
                
                # Try to export the symbol directly
                try:
                    # Create a simple DLL export in memory
                    kernel32 = ctypes.windll.kernel32
                    # This is a simplified approach - the full DLL method would be better
                    print(" Set AmdPowerXpressRequestHighPerformance=1")
                except Exception as e:
                    print(f"  Could not set AMD power export: {e}")
        except Exception as e:
            print(f"  AMD power export setup failed: {e}")
    
    def set_windows_graphics_settings(self):
        """Set Windows Graphics Settings to force discrete GPU"""
        print(" Configuring Windows Graphics Settings...")
        
        try:
            # Registry path for Windows Graphics Settings
            graphics_key_path = r"SOFTWARE\Microsoft\DirectX\UserGpuPreferences"
            
            # Create or open the key
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, graphics_key_path) as key:
                # Set Python to use high performance GPU
                python_exe = self.python_path
                winreg.SetValueEx(key, python_exe, 0, winreg.REG_SZ, "GpuPreference=2;")
                print(f" Set Windows Graphics preference for Python: {python_exe}")
                
                # Also set for the script itself
                script_exe = str(self.script_path)
                winreg.SetValueEx(key, script_exe, 0, winreg.REG_SZ, "GpuPreference=2;")
                print(f" Set Windows Graphics preference for script: {script_exe}")
                
        except Exception as e:
            print(f"  Could not set Windows Graphics Settings: {e}")
    
    def set_amd_radeon_settings(self):
        """Set AMD Radeon Settings programmatically"""
        print(" Configuring AMD Radeon Settings...")
        
        try:
            # AMD Radeon Settings registry path
            amd_key_path = r"SOFTWARE\AMD\DxDiag"
            
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, amd_key_path) as key:
                # Set application profile for high performance
                winreg.SetValueEx(key, "AppProfile", 0, winreg.REG_SZ, self.python_path)
                winreg.SetValueEx(key, "GPUPreference", 0, winreg.REG_DWORD, 1)  # 1 = discrete GPU
                print(" Set AMD Radeon Settings for high performance")
                
        except Exception as e:
            print(f"  Could not set AMD Radeon Settings: {e}")
    
    def set_environment_variables(self):
        """Set comprehensive environment variables"""
        print(" Setting comprehensive environment variables...")
        
        env_vars = {
            'AmdPowerXpressRequestHighPerformance': '1',
            'AMD_OPENCL_DEVICE': '1',
            'OPENCL_DEVICE': '1',
            'GPU_DEVICE_ORDINAL': '1',
            'AMD_VULKAN_DEVICE': '1',
            'AMD_GPU_DEVICE': '1',
            'DISPLAY_GPU': '1',
            'CUDA_VISIBLE_DEVICES': '1',  # For CUDA compatibility
            'AMD_DISABLE_INTEGRATED_GPU': '1',
            'OPENCL_DISABLE_DEVICE_0': '1'
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            print(f"   {var}={value}")
    
    def create_gpu_force_script(self):
        """Create a PowerShell script to force GPU selection"""
        ps_script = '''
# Force GPU Selection PowerShell Script
Write-Host "Forcing RX 7700S GPU usage..."

# Set environment variables
$env:AmdPowerXpressRequestHighPerformance = "1"
$env:AMD_OPENCL_DEVICE = "1"
$env:OPENCL_DEVICE = "1"

# Set Windows Graphics Settings via PowerShell
$pythonPath = "' + self.python_path + '"
$regPath = "HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences"
Set-ItemProperty -Path $regPath -Name $pythonPath -Value "GpuPreference=2;" -Force

Write-Host "GPU forcing complete. Restart your application."
'''
        
        script_path = Path("force_gpu.ps1")
        with open(script_path, 'w') as f:
            f.write(ps_script)
        
        print(f" Created PowerShell script: {script_path}")
        print("   Run this script as administrator to force GPU selection")
    
    def run_gpu_force_commands(self):
        """Run commands to force GPU selection"""
        print(" Running GPU force commands...")
        
        try:
            # Try to run PowerShell commands
            ps_commands = [
                'Set-ItemProperty -Path "HKCU:\\SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences" -Name "' + self.python_path + '" -Value "GpuPreference=2;" -Force',
                '$env:AmdPowerXpressRequestHighPerformance = "1"',
                '$env:AMD_OPENCL_DEVICE = "1"'
            ]
            
            for cmd in ps_commands:
                try:
                    result = subprocess.run(['powershell', '-Command', cmd], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        print(f" Executed: {cmd}")
                    else:
                        print(f"  Failed: {cmd}")
                except Exception as e:
                    print(f"  Command failed: {e}")
                    
        except Exception as e:
            print(f"  Could not run GPU force commands: {e}")
    
    def test_gpu_usage(self):
        """Test which GPU is actually being used"""
        print("\n Testing actual GPU usage...")
        
        try:
            import pyopencl as cl
            import numpy as np
            
            # Create a workload that will show up in Task Manager
            platforms = cl.get_platforms()
            
            for platform in platforms:
                devices = platform.get_devices()
                for i, device in enumerate(devices):
                    device_name = device.name.lower()
                    
                    if 'gfx1103' in device_name:  # RX 7700S
                        print(f" Testing RX 7700S (Device {i}): {device.name}")
                        
                        try:
                            context = cl.Context([device])
                            queue = cl.CommandQueue(context)
                            
                            # Create a large workload
                            size = 1024 * 1024 * 4  # 4M elements
                            a = np.random.rand(size).astype(np.float32)
                            b = np.random.rand(size).astype(np.float32)
                            
                            # Create buffers
                            a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                            b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                            c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
                            
                            # Compute-intensive kernel
                            kernel_code = """
                            __kernel void intensive_workload(__global const float* a,
                                                           __global const float* b,
                                                           __global float* c) {
                                int gid = get_global_id(0);
                                float temp = a[gid] * b[gid];
                                
                                // Multiple operations to stress GPU
                                for(int i = 0; i < 1000; i++) {
                                    temp = sqrt(temp);
                                    temp = sin(temp) + cos(temp);
                                    temp = temp * temp;
                                }
                                c[gid] = temp;
                            }
                            """
                            
                            program = cl.Program(context, kernel_code).build()
                            kernel = program.intensive_workload
                            
                            print("    Running intensive workload...")
                            print("    Check Task Manager - RX 7700S (GPU 0) should show activity")
                            
                            # Run the kernel
                            kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
                            queue.finish()
                            
                            print("    Workload complete")
                            print("    Check Task Manager to see which GPU was used")
                            
                            return True
                            
                        except Exception as e:
                            print(f"    RX 7700S test failed: {e}")
                            return False
            
            print(" RX 7700S not found for testing")
            return False
            
        except Exception as e:
            print(f" GPU test failed: {e}")
            return False

def main():
    """Main function"""
    print("=== Windows GPU Force ===")
    print("Forcing Windows to use RX 7700S at the system level")
    
    # Create GPU force controller
    gpu_force = WindowsGPUForce()
    
    # Set all configurations
    gpu_force.set_environment_variables()
    gpu_force.set_windows_graphics_settings()
    gpu_force.set_amd_radeon_settings()
    gpu_force.create_gpu_force_script()
    
    # Run GPU force commands
    gpu_force.run_gpu_force_commands()
    
    # Test GPU usage
    success = gpu_force.test_gpu_usage()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("="*60)
    print("1. Check Task Manager while the test runs")
    print("2. RX 7700S (GPU 0) should show activity, not 780M (GPU 1)")
    print("3. If still using 780M, try:")
    print("   - Restart your Python/IDE")
    print("   - Run the PowerShell script as administrator")
    print("   - Manually set Windows Graphics Settings")
    print("4. For permanent solution:")
    print("   - Win + I → System → Display → Graphics settings")
    print("   - Add python.exe → Set to 'High performance'")

if __name__ == "__main__":
    main()

