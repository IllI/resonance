#!/usr/bin/env python3
"""
Temporary GPU 0 Disabler

This script provides instructions and automation to temporarily disable GPU 0 (780M)
so that only GPU 1 (RX 7700S) is available to applications.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_admin_rights():
    """Check if running with administrator privileges."""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def disable_gpu0_powershell():
    """Use PowerShell to disable GPU 0 (requires admin)."""
    try:
        print("Attempting to disable GPU 0 using PowerShell...")
        
        # PowerShell command to disable integrated GPU
        ps_command = '''
        $gpu0 = Get-PnpDevice | Where-Object {$_.FriendlyName -like "*780M*" -or ($_.FriendlyName -like "*AMD*" -and $_.FriendlyName -like "*Radeon*" -and $_.Class -eq "Display")} | Select-Object -First 1
        if ($gpu0) {
            Write-Host "Found GPU 0: $($gpu0.FriendlyName)"
            Write-Host "Disabling GPU 0..."
            Disable-PnpDevice -InstanceId $gpu0.InstanceId -Confirm:$false
            Write-Host "GPU 0 disabled successfully"
        } else {
            Write-Host "GPU 0 (780M) not found"
        }
        '''
        
        # Execute PowerShell command
        result = subprocess.run([
            'powershell', '-Command', ps_command
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print(" GPU 0 disabled successfully")
            print(result.stdout)
            return True
        else:
            print(" Failed to disable GPU 0")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"PowerShell disable failed: {e}")
        return False

def enable_gpu0_powershell():
    """Use PowerShell to re-enable GPU 0."""
    try:
        print("Re-enabling GPU 0...")
        
        # PowerShell command to enable integrated GPU
        ps_command = '''
        $gpu0 = Get-PnpDevice | Where-Object {$_.FriendlyName -like "*780M*" -or ($_.FriendlyName -like "*AMD*" -and $_.FriendlyName -like "*Radeon*" -and $_.Class -eq "Display")} | Select-Object -First 1
        if ($gpu0) {
            Write-Host "Found GPU 0: $($gpu0.FriendlyName)"
            Write-Host "Enabling GPU 0..."
            Enable-PnpDevice -InstanceId $gpu0.InstanceId -Confirm:$false
            Write-Host "GPU 0 enabled successfully"
        } else {
            Write-Host "GPU 0 (780M) not found"
        }
        '''
        
        # Execute PowerShell command
        result = subprocess.run([
            'powershell', '-Command', ps_command
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print(" GPU 0 re-enabled successfully")
            print(result.stdout)
            return True
        else:
            print(" Failed to re-enable GPU 0")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"PowerShell enable failed: {e}")
        return False

def test_gpu_availability():
    """Test which GPUs are currently available."""
    try:
        import pyopencl as cl
        
        print("Testing GPU availability...")
        
        platforms = cl.get_platforms()
        available_gpus = []
        
        for platform in platforms:
            if 'AMD' in platform.name:
                devices = platform.get_devices(cl.device_type.GPU)
                
                for i, device in enumerate(devices):
                    device_name = device.name.strip()
                    memory_gb = device.global_mem_size // (1024**3)
                    
                    available_gpus.append({
                        'index': i,
                        'name': device_name,
                        'memory': memory_gb,
                        'device': device
                    })
                    
                    print(f"  Available GPU {i}: {device_name} ({memory_gb} GB)")
        
        return available_gpus
        
    except Exception as e:
        print(f"GPU availability test failed: {e}")
        return []

def run_brain_gui_test():
    """Run a quick brain GUI test to verify GPU 1 usage."""
    try:
        print("\nRunning brain GUI test...")
        
        # Test if we can import and initialize the brain analyzer
        sys.path.insert(0, str(Path.cwd()))
        
        from rx7700s_only_brain_analyzer import RX7700SOnlyBrainAnalyzer
        import numpy as np
        
        # Create analyzer
        analyzer = RX7700SOnlyBrainAnalyzer()
        
        # Test with small brain volume
        test_volume = np.random.rand(32, 32, 16).astype(np.float32)
        test_volume[8:24, 8:24, 4:12] = 0.8  # Brain region
        
        print("Processing test brain volume...")
        result = analyzer.process_brain_volume_gpu(test_volume)
        
        if result and result.get('success', False):
            device_used = result.get('gpu_device', 'Unknown')
            processing_time = result.get('processing_time', 0)
            
            print(f" Brain processing successful!")
            print(f"   Device used: {device_used}")
            print(f"   Processing time: {processing_time:.3f}s")
            
            # Check if it's the RX 7700S
            if 'gfx1103' in device_used or memory_gb >= 10:
                print(f" Confirmed: Using RX 7700S (GPU 1)")
                return True
            else:
                print(f" Warning: May still be using integrated GPU")
                return False
        else:
            print(f" Brain processing failed")
            return False
            
    except Exception as e:
        print(f"Brain GUI test failed: {e}")
        return False

def main():
    """Main function for temporary GPU 0 disabling."""
    print("Temporary GPU 0 Disabler")
    print("=" * 40)
    print("This will temporarily disable GPU 0 (780M) to force GPU 1 (RX 7700S) usage")
    print()
    
    # Check admin rights
    is_admin = check_admin_rights()
    print(f"Administrator rights: {' Yes' if is_admin else ' No'}")
    
    if not is_admin:
        print("\n WARNING: Administrator rights required for automatic GPU disabling")
        print("\nManual method:")
        print("1. Right-click Start → Device Manager")
        print("2. Expand 'Display adapters'")
        print("3. Right-click 'AMD Radeon 780M' → Disable device")
        print("4. Run brain GUI")
        print("5. Re-enable GPU when done")
        print()
        
        choice = input("Continue with manual method? (y/n): ").lower().strip()
        if choice != 'y':
            return
    
    # Test current GPU availability
    print("\nCurrent GPU availability:")
    available_gpus = test_gpu_availability()
    
    gpu0_present = any(gpu['memory'] < 10 for gpu in available_gpus)
    gpu1_present = any(gpu['memory'] >= 10 for gpu in available_gpus)
    
    print(f"GPU 0 (780M) present: {'Yes' if gpu0_present else 'No'}")
    print(f"GPU 1 (RX 7700S) present: {'Yes' if gpu1_present else 'No'}")
    
    if not gpu1_present:
        print(" ERROR: GPU 1 (RX 7700S) not found!")
        return
    
    if gpu0_present and is_admin:
        print("\nDisabling GPU 0 temporarily...")
        
        if disable_gpu0_powershell():
            print(" GPU 0 disabled successfully")
            
            # Wait a moment for system to update
            time.sleep(2)
            
            # Test GPU availability again
            print("\nTesting GPU availability after disable...")
            new_available_gpus = test_gpu_availability()
            
            if len(new_available_gpus) == 1 and new_available_gpus[0]['memory'] >= 10:
                print(" Only GPU 1 (RX 7700S) is now available!")
                
                # Test brain GUI
                brain_test_success = run_brain_gui_test()
                
                if brain_test_success:
                    print("\n🎉 SUCCESS: Brain GUI will now use GPU 1 exclusively!")
                    print("\nYou can now run the brain GUI:")
                    print("  python quick_start_gui.py")
                    print("\nGPU 1 should show usage in Task Manager")
                    
                    input("\nPress Enter when done testing to re-enable GPU 0...")
                    
                    # Re-enable GPU 0
                    enable_gpu0_powershell()
                    print(" GPU 0 re-enabled")
                else:
                    print(" Brain GUI test failed, re-enabling GPU 0...")
                    enable_gpu0_powershell()
            else:
                print(" GPU disable may not have worked properly")
                enable_gpu0_powershell()
        else:
            print(" Failed to disable GPU 0")
    
    elif gpu0_present:
        print("\n Manual GPU 0 disable instructions:")
        print("1. Run as Administrator or use Device Manager manually")
        print("2. Device Manager → Display adapters → AMD Radeon 780M")
        print("3. Right-click → Disable device")
        print("4. Run brain GUI")
        print("5. Check Task Manager for GPU 1 usage")
        print("6. Re-enable GPU 0 when done")
    
    else:
        print(" GPU 0 already disabled or not present")
        print("Brain GUI should use GPU 1 exclusively")

if __name__ == "__main__":
    main()