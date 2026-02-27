#!/usr/bin/env python3
"""
RX 7700S Environment Setup

Based on successful GPU selection guide, this module configures
the environment to force RX 7700S usage using HIP/ROCm variables.
"""

import os
import sys
import subprocess
from pathlib import Path

class RX7700SEnvironmentSetup:
    """Setup environment for RX 7700S exclusive usage."""
    
    def __init__(self):
        self.environment_vars = {
            # HIP/ROCm device selection (most effective for AMD)
            'HIP_VISIBLE_DEVICES': '1',           # Use GPU 1 (RX 7700S)
            'ROCR_VISIBLE_DEVICES': '1',          # ROCm device selection
            'GPU_MAX_ALLOC_PERCENT': '95',        # Allow high memory usage
            'GPU_MAX_HEAP_SIZE': '100',           # Maximum heap allocation
            'HSA_OVERRIDE_GFX_VERSION': '11.0.0', # RDNA3 architecture
            
            # Additional AMD GPU preferences
            'GPU_FORCE_64BIT_PTR': '1',
            'GPU_USE_SYNC_OBJECTS': '1',
            'GPU_SINGLE_ALLOC_PERCENT': '100',
            
            # OpenCL backup (secondary)
            'OPENCL_DEVICE_FILTER': 'discrete',
            'AMD_OPENCL_DEVICE_MASK': '2',        # Binary: 10 = GPU 1 only
            
            # DirectX/Vulkan preferences
            'DXVK_FILTER_DEVICE_NAME': 'RX 7700S',
        }
    
    def apply_environment_variables(self):
        """Apply RX 7700S environment variables."""
        print(" Configuring RX 7700S Environment Variables")
        print("=" * 50)
        
        applied_vars = []
        
        for var_name, var_value in self.environment_vars.items():
            try:
                os.environ[var_name] = var_value
                applied_vars.append((var_name, var_value))
                print(f" {var_name} = {var_value}")
            except Exception as e:
                print(f" Failed to set {var_name}: {e}")
        
        print(f"\n Applied {len(applied_vars)} environment variables")
        return applied_vars
    
    def configure_windows_graphics_settings(self):
        """Configure Windows Graphics Settings for high performance."""
        print("\n🖥️ Windows Graphics Settings Configuration")
        print("=" * 50)
        
        python_exe = sys.executable
        print(f"Python executable: {python_exe}")
        
        # PowerShell command to set graphics preference
        ps_command = f'''
        $app = "{python_exe}"
        $preference = "GpuPreference=2;"  # 2 = High Performance (discrete GPU)
        
        # Set registry entry for graphics preference
        $regPath = "HKCU:\\Software\\Microsoft\\DirectX\\UserGpuPreferences"
        
        if (!(Test-Path $regPath)) {{
            New-Item -Path $regPath -Force | Out-Null
        }}
        
        Set-ItemProperty -Path $regPath -Name $app -Value $preference
        Write-Host " Set graphics preference for Python to High Performance"
        '''
        
        try:
            result = subprocess.run([
                'powershell', '-Command', ps_command
            ], capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                print(" Windows Graphics Settings configured")
                print("   Python.exe set to High Performance (RX 7700S)")
                return True
            else:
                print(f" PowerShell command failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f" Windows Graphics Settings configuration failed: {e}")
            return False
    
    def configure_amd_software_settings(self):
        """Configure AMD Software settings via registry."""
        print("\n🔴 AMD Software Configuration")
        print("=" * 50)
        
        # Registry commands for AMD settings
        registry_commands = [
            # PowerXpress high performance
            'reg add "HKCU\\Software\\AMD\\CN" /v "PowerXpressRequestPolicy" /t REG_DWORD /d 1 /f',
            
            # Global graphics high performance
            'reg add "HKLM\\SOFTWARE\\AMD\\CN" /v "WorkloadPolicy" /t REG_DWORD /d 1 /f',
            
            # Force discrete GPU
            'reg add "HKLM\\SOFTWARE\\AMD\\CN" /v "SwitchableGraphicsGlobal" /t REG_DWORD /d 2 /f'
        ]
        
        success_count = 0
        
        for cmd in registry_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f" {cmd.split('/')[-3]} configured")
                    success_count += 1
                else:
                    print(f" Failed: {cmd}")
            except Exception as e:
                print(f" Registry command failed: {e}")
        
        if success_count > 0:
            print(f"\n AMD Software: {success_count}/{len(registry_commands)} settings applied")
            print("   Note: Restart may be required for full effect")
            return True
        else:
            print("\n AMD Software configuration failed")
            return False
    
    def verify_environment_setup(self):
        """Verify environment variables are set correctly."""
        print("\n Environment Verification")
        print("=" * 50)
        
        verification_results = {}
        
        for var_name, expected_value in self.environment_vars.items():
            actual_value = os.environ.get(var_name)
            
            if actual_value == expected_value:
                print(f" {var_name}: {actual_value}")
                verification_results[var_name] = True
            else:
                print(f" {var_name}: Expected '{expected_value}', got '{actual_value}'")
                verification_results[var_name] = False
        
        success_rate = sum(verification_results.values()) / len(verification_results) * 100
        print(f"\n Environment Setup: {success_rate:.1f}% successful")
        
        return verification_results
    
    def full_setup(self):
        """Execute complete RX 7700S environment setup."""
        print(" RX 7700S COMPLETE ENVIRONMENT SETUP")
        print("=" * 60)
        print("This will configure your system to use RX 7700S for GPU computing")
        print()
        
        results = {}
        
        # Step 1: Environment variables
        results['environment_vars'] = self.apply_environment_variables()
        
        # Step 2: Windows Graphics Settings
        results['windows_graphics'] = self.configure_windows_graphics_settings()
        
        # Step 3: AMD Software Settings
        results['amd_software'] = self.configure_amd_software_settings()
        
        # Step 4: Verification
        results['verification'] = self.verify_environment_setup()
        
        # Summary
        print("\n" + "=" * 60)
        print(" RX 7700S SETUP SUMMARY")
        print("=" * 60)
        
        env_success = len(results['environment_vars']) > 0
        win_success = results['windows_graphics']
        amd_success = results['amd_software']
        verify_success = sum(results['verification'].values()) > len(results['verification']) * 0.8
        
        print(f"Environment Variables: {'' if env_success else ''}")
        print(f"Windows Graphics:      {'' if win_success else ''}")
        print(f"AMD Software:          {'' if amd_success else ''}")
        print(f"Verification:          {'' if verify_success else ''}")
        
        overall_success = env_success and (win_success or amd_success)
        
        if overall_success:
            print("\n🎉 SUCCESS: RX 7700S environment configured!")
            print("\nNext steps:")
            print("1. Restart your application")
            print("2. Monitor Task Manager → Performance → GPU 1")
            print("3. Run: python rx7700s_brain_analyzer_test.py")
        else:
            print("\n PARTIAL SUCCESS: Some configurations failed")
            print("The environment variables should still force RX 7700S usage")
        
        return overall_success

def setup_rx7700s_environment():
    """Convenience function to setup RX 7700S environment."""
    setup = RX7700SEnvironmentSetup()
    return setup.full_setup()

if __name__ == "__main__":
    print("RX 7700S Environment Setup")
    print("This will configure your system to use the discrete GPU")
    print()
    
    setup_rx7700s_environment()
    
    input("\nPress Enter to exit...")