#!/usr/bin/env python3
"""
Test RX 7700S Selection

Quick test to verify the brain analyzer is selecting the RX 7700S correctly.
"""

import sys
import os

def test_rx7700s_selection():
    """Test that the brain analyzer selects RX 7700S."""
    print("Testing RX 7700S Selection...")
    print("=" * 40)
    
    try:
        # Force reload modules
        modules_to_reload = [
            'rx7700s_only_brain_analyzer',
            'brain_3d_model_generator'
        ]
        
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                del sys.modules[module_name]
                print(f"Reloaded: {module_name}")
        
        # Import fresh analyzer
        from rx7700s_only_brain_analyzer import RX7700SOnlyBrainAnalyzer
        
        # Create analyzer
        analyzer = RX7700SOnlyBrainAnalyzer()
        
        # Check which GPU was selected
        if hasattr(analyzer, 'rx7700s_device'):
            device_name = analyzer.rx7700s_device.name.strip()
            memory_gb = analyzer.rx7700s_device.global_mem_size // (1024**3)
            
            print(f"\n GPU Selected: {device_name}")
            print(f"   Memory: {memory_gb} GB")
            
            # Verify it's the RX 7700S
            if 'gfx1103' in device_name or memory_gb >= 10:
                print(f" CONFIRMED: This is the RX 7700S!")
                print(f"   The brain GUI should now use GPU 0 (RX 7700S)")
                return True
            else:
                print(f" ERROR: This is NOT the RX 7700S")
                print(f"   Device: {device_name}, Memory: {memory_gb} GB")
                return False
        else:
            print(" ERROR: No GPU device selected")
            return False
            
    except Exception as e:
        print(f" ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_rx7700s_selection()
    
    if success:
        print("\n🎉 SUCCESS: RX 7700S selection is working!")
        print("\nNow run the brain GUI:")
        print("  python quick_start_gui.py")
        print("\nExpected result:")
        print("  GPU 0 (RX 7700S) should show usage in Task Manager")
        print("  GPU 1 (780M) should remain at 0%")
    else:
        print("\n FAILED: RX 7700S selection not working")
        print("The brain GUI will still use the integrated GPU")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()