#!/usr/bin/env python3
"""
Force GPU 1 Rendering - Bypass OpenCL

Since OpenCL cannot see the RX 7700S, this script forces the system
to use GPU 1 for all graphics rendering by disabling GPU 0.
"""

import os
import sys
import subprocess
import time

def disable_gpu0_for_rendering():
    """Disable GPU 0 to force GPU 1 usage for rendering."""
    print("Force GPU 1 Rendering - Bypass OpenCL")
    print("=" * 50)
    print("Since OpenCL cannot see the RX 7700S, we'll force GPU 1 usage")
    print("by making GPU 0 unavailable for rendering.")
    print()
    
    print("MANUAL METHOD (Recommended):")
    print("1. Right-click Start → Device Manager")
    print("2. Expand 'Display adapters'")
    print("3. Right-click 'AMD Radeon 780M' → Disable device")
    print("4. Run brain GUI")
    print("5. GPU 1 (RX 7700S) will be forced to handle all rendering")
    print()
    
    choice = input("Proceed with manual method? (y/n): ").lower().strip()
    
    if choice == 'y':
        print("\nAfter disabling GPU 0 in Device Manager:")
        print("1. Run: python quick_start_gui.py")
        print("2. Load MRI data and generate 3D model")
        print("3. Check Task Manager - GPU 1 should show usage")
        print("4. The brain should render smoothly on RX 7700S")
        print()
        print("Note: This bypasses OpenCL entirely and forces")
        print("the system to use GPU 1 for all graphics operations.")
        
        input("\nPress Enter after disabling GPU 0 to test brain GUI...")
        
        # Test if brain GUI works with GPU 1
        test_brain_gui_with_gpu1()
    
    else:
        print("Manual GPU disable cancelled.")

def test_brain_gui_with_gpu1():
    """Test brain GUI with GPU 1 forced."""
    try:
        print("Testing brain GUI with GPU 1 forced...")
        
        # Since OpenCL can't see RX 7700S, we'll test the GUI directly
        print("Launching brain GUI...")
        print("Generate a 3D brain model and check Task Manager")
        print("GPU 1 should show usage for rendering operations")
        
        # Launch GUI
        subprocess.Popen([sys.executable, 'quick_start_gui.py'])
        
        print("\nBrain GUI launched!")
        print("Expected behavior:")
        print(" GPU 1 shows usage in Task Manager")
        print(" Brain model renders smoothly")
        print(" 3D rotation is responsive")
        print(" No freezing during interaction")
        
    except Exception as e:
        print(f"Error launching brain GUI: {e}")

def check_current_gpu_status():
    """Check current GPU status after disable."""
    try:
        print("Checking current GPU status...")
        
        # Check OpenCL again
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            for platform in platforms:
                if 'AMD' in platform.name:
                    devices = platform.get_devices(cl.device_type.GPU)
                    
                    if len(devices) == 0:
                        print(" No GPUs visible to OpenCL (GPU 0 successfully disabled)")
                        print("   System will use GPU 1 for all rendering")
                    else:
                        for i, device in enumerate(devices):
                            device_name = device.name.strip()
                            memory_gb = device.global_mem_size // (1024**3)
                            print(f"   Remaining GPU {i}: {device_name} ({memory_gb} GB)")
        
        except Exception as e:
            print(f"OpenCL check failed: {e}")
            print("This is expected if GPU 0 is disabled")
        
        # Check Windows GPU status
        print("\nWindows GPU Status:")
        ps_command = '''
        Get-PnpDevice | Where-Object { $_.Class -eq "Display" -and $_.FriendlyName -like "*AMD*" } | ForEach-Object {
            $status = if ($_.Status -eq "OK") { "Enabled" } else { "Disabled" }
            Write-Host "$($_.FriendlyName): $status"
        }
        '''
        
        result = subprocess.run([
            'powershell', '-Command', ps_command
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print(result.stdout)
        
    except Exception as e:
        print(f"GPU status check failed: {e}")

def main():
    """Main function."""
    disable_gpu0_for_rendering()
    
    print("\nAfter testing, remember to:")
    print("1. Re-enable AMD Radeon 780M in Device Manager")
    print("2. This restores normal dual-GPU operation")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()