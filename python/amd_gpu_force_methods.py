#!/usr/bin/env python3
"""
AMD GPU Selection Methods for Windows Dual-GPU Systems
Based on AMD documentation and community solutions
"""

import os
import subprocess
import winreg
from pathlib import Path

class AMDGPUSelector:
    def __init__(self):
        self.methods = {
            'environment': self.set_environment_variables,
            'registry': self.modify_registry,
            'adrenalin': self.configure_adrenalin,
            'windows_graphics': self.set_windows_graphics
        }
    
    def set_environment_variables(self):
        """Method 1: Environment Variables (Most Reliable)"""
        env_vars = {
            # Force discrete GPU for OpenCL
            'GPU_FORCE_64BIT_PTR': '1',
            'GPU_MAX_HEAP_SIZE': '100',
            'GPU_USE_SYNC_OBJECTS': '1',
            'GPU_MAX_ALLOC_PERCENT': '100',
            
            # AMD-specific variables
            'AMD_OPENCL_DEVICE_MASK': '2',  # Use GPU 1 (discrete)
            'OPENCL_DEVICE_FILTER': 'discrete',
            'HSA_OVERRIDE_GFX_VERSION': '11.0.0',
            
            # DirectX/Vulkan preferences
            'DXVK_FILTER_DEVICE_NAME': 'RX 7700S',
            'VK_ICD_FILENAMES': r'C:\Windows\System32\amdvlk64.dll'
        }
        
        print("Setting AMD GPU environment variables...")
        for var, value in env_vars.items():
            os.environ[var] = value
            print(f"Set {var} = {value}")
        
        return True
    
    def modify_registry(self):
        """Method 2: Registry Modifications"""
        try:
            # PowerXpress settings
            key_path = r"SOFTWARE\AMD\CN"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "PowerXpressRequestPolicy", 0, winreg.REG_DWORD, 1)
            
            # DirectX GPU preferences
            dx_key = r"Software\Microsoft\DirectX\UserGpuPreferences"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, dx_key) as key:
                python_exe = str(Path(os.sys.executable).resolve())
                winreg.SetValueEx(key, python_exe, 0, winreg.REG_SZ, "GpuPreference=2;")
            
            print("Registry modifications applied successfully")
            return True
            
        except Exception as e:
            print(f"Registry modification failed: {e}")
            return False
    
    def configure_adrenalin(self):
        """Method 3: AMD Adrenalin Configuration"""
        adrenalin_commands = [
            # Set global graphics to high performance
            'reg add "HKCU\\Software\\AMD\\CN" /v "PowerXpressRequestPolicy" /t REG_DWORD /d 1 /f',
            
            # Enable compute workload
            'reg add "HKLM\\SOFTWARE\\AMD\\CN" /v "WorkloadPolicy" /t REG_DWORD /d 1 /f',
            
            # Force discrete GPU
            'reg add "HKLM\\SOFTWARE\\AMD\\CN" /v "SwitchableGraphicsGlobal" /t REG_DWORD /d 2 /f'
        ]
        
        print("Configuring AMD Adrenalin settings...")
        for cmd in adrenalin_commands:
            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"✓ {cmd}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {cmd} - {e}")
        
        return True
    
    def set_windows_graphics(self):
        """Method 4: Windows Graphics Settings"""
        print("Windows Graphics Settings Configuration:")
        print("1. Open Settings > System > Display > Graphics settings")
        print("2. Click 'Browse' and select your Python executable:")
        print(f"   {os.sys.executable}")
        print("3. Click 'Options' and select 'High performance'")
        print("4. Save settings")
        
        # Also set via PowerShell if possible
        ps_command = f'''
        $app = "{os.sys.executable}"
        Add-AppxPackage -RegisterByFamilyName -MainPackage Microsoft.Windows.ShellExperienceHost_cw5n1h2txyewy
        '''
        
        return True
    
    def apply_all_methods(self):
        """Apply all GPU selection methods"""
        print("=== AMD GPU Selection Configuration ===")
        
        results = {}
        for method_name, method_func in self.methods.items():
            print(f"\n--- {method_name.upper()} METHOD ---")
            try:
                results[method_name] = method_func()
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                results[method_name] = False
        
        print("\n=== RESULTS ===")
        for method, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{method}: {status}")
        
        print("\n=== NEXT STEPS ===")
        print("1. Restart your computer for registry changes to take effect")
        print("2. Open AMD Software and verify GPU workload is set to 'Compute'")
        print("3. Test with: python test_rx7700s_selection.py")
        
        return results

if __name__ == "__main__":
    selector = AMDGPUSelector()
    selector.apply_all_methods()