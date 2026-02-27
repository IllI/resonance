#!/usr/bin/env python3
"""
Debug GPU Selection

This script will show exactly what's happening with GPU selection
and why the wrong GPU is being chosen.
"""

import sys
import os

def debug_opencl_detection():
    """Debug OpenCL GPU detection step by step."""
    print("Debug OpenCL GPU Detection")
    print("=" * 40)
    
    try:
        import pyopencl as cl
        
        platforms = cl.get_platforms()
        print(f"Found {len(platforms)} OpenCL platform(s)")
        
        for platform in platforms:
            if 'AMD' in platform.name:
                print(f"\nPlatform: {platform.name}")
                
                devices = platform.get_devices(cl.device_type.GPU)
                print(f"Found {len(devices)} GPU device(s)")
                
                for i, device in enumerate(devices):
                    device_name = device.name.strip()
                    memory_gb = device.global_mem_size // (1024**3)
                    compute_units = device.max_compute_units
                    
                    print(f"\n  GPU {i}:")
                    print(f"    Name: {device_name}")
                    print(f"    Memory: {memory_gb} GB")
                    print(f"    Compute Units: {compute_units}")
                    print(f"    Vendor: {device.vendor}")
                    
                    # Check if this matches RX 7700S characteristics
                    is_rx7700s = False
                    reasons = []
                    
                    if 'gfx1103' in device_name:
                        is_rx7700s = True
                        reasons.append("gfx1103 architecture")
                    
                    if memory_gb >= 10:
                        is_rx7700s = True
                        reasons.append(f"{memory_gb} GB memory")
                    
                    if '7700' in device_name:
                        is_rx7700s = True
                        reasons.append("7700 in name")
                    
                    if is_rx7700s:
                        print(f"     RX 7700S DETECTED: {', '.join(reasons)}")
                    else:
                        print(f"     Not RX 7700S: {device_name}")
                
                return devices
        
        return []
        
    except Exception as e:
        print(f"OpenCL detection failed: {e}")
        return []

def debug_analyzer_selection():
    """Debug what the brain analyzer actually selects."""
    print("\nDebug Brain Analyzer Selection")
    print("=" * 40)
    
    try:
        # Force reload
        if 'rx7700s_only_brain_analyzer' in sys.modules:
            del sys.modules['rx7700s_only_brain_analyzer']
        
        from rx7700s_only_brain_analyzer import RX7700SOnlyBrainAnalyzer
        
        # Create analyzer and see what it selects
        print("Creating RX7700S analyzer...")
        analyzer = RX7700SOnlyBrainAnalyzer()
        
        if hasattr(analyzer, 'rx7700s_device'):
            device = analyzer.rx7700s_device
            device_name = device.name.strip()
            memory_gb = device.global_mem_size // (1024**3)
            
            print(f"\nAnalyzer Selected:")
            print(f"  Device: {device_name}")
            print(f"  Memory: {memory_gb} GB")
            
            # Check if this is actually the RX 7700S
            if 'gfx1103' in device_name or memory_gb >= 10:
                print(f"   CORRECT: This is the RX 7700S")
                return True
            else:
                print(f"   WRONG: This is NOT the RX 7700S")
                return False
        else:
            print(" ERROR: Analyzer didn't select any device")
            return False
            
    except Exception as e:
        print(f"Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_task_manager_mapping():
    """Show how OpenCL GPUs map to Task Manager GPUs."""
    print("\nTask Manager GPU Mapping")
    print("=" * 40)
    
    devices = debug_opencl_detection()
    
    if len(devices) >= 2:
        print("\nTask Manager Mapping:")
        for i, device in enumerate(devices):
            device_name = device.name.strip()
            memory_gb = device.global_mem_size // (1024**3)
            
            if 'gfx1103' in device_name or memory_gb >= 10:
                gpu_type = "RX 7700S (DISCRETE)"
            else:
                gpu_type = "780M (INTEGRATED)"
            
            print(f"  OpenCL GPU {i} = Task Manager GPU {i} = {gpu_type}")
            print(f"    {device_name} ({memory_gb} GB)")

def main():
    """Main debug function."""
    print("GPU Selection Debug Tool")
    print("=" * 50)
    
    # Step 1: Debug OpenCL detection
    debug_opencl_detection()
    
    # Step 2: Show Task Manager mapping
    show_task_manager_mapping()
    
    # Step 3: Debug analyzer selection
    analyzer_correct = debug_analyzer_selection()
    
    print("\n" + "=" * 50)
    print("DIAGNOSIS:")
    
    if analyzer_correct:
        print(" Brain analyzer is selecting RX 7700S correctly")
        print("   The issue may be elsewhere in the brain GUI")
    else:
        print(" Brain analyzer is selecting the WRONG GPU")
        print("   This explains why Task Manager shows 780M usage")
    
    print("\nRECOMMENDATION:")
    if not analyzer_correct:
        print("The GPU selection logic needs to be fixed")
        print("The analyzer should select the GPU with gfx1103 or >=10GB memory")
    else:
        print("Check if the brain GUI is actually using the analyzer's selection")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()