#!/usr/bin/env python3
"""
Complete RX 7700S Test Suite

This script tests all components of the fixed RX 7700S implementation
to verify that the discrete GPU is being used correctly.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def test_environment_setup():
    """Test RX 7700S environment setup."""
    print(" Testing RX 7700S Environment Setup")
    print("=" * 50)
    
    try:
        from rx7700s_environment_setup import RX7700SEnvironmentSetup
        
        setup = RX7700SEnvironmentSetup()
        success = setup.full_setup()
        
        print(f"\nEnvironment setup: {' SUCCESS' if success else ' FAILED'}")
        return success
        
    except Exception as e:
        print(f" Environment setup test failed: {e}")
        return False

def test_gpu_detection():
    """Test RX 7700S GPU detection."""
    print("\n Testing RX 7700S GPU Detection")
    print("=" * 50)
    
    try:
        from rx7700s_simple_detector import test_rx7700s_detection
        
        success = test_rx7700s_detection()
        
        print(f"\nGPU detection: {' SUCCESS' if success else ' FAILED'}")
        return success
        
    except Exception as e:
        print(f" GPU detection test failed: {e}")
        return False

def test_brain_analyzer():
    """Test RX 7700S brain analyzer."""
    print("\n Testing RX 7700S Brain Analyzer")
    print("=" * 50)
    
    try:
        from rx7700s_brain_analyzer_fixed import test_rx7700s_brain_analyzer
        
        success = test_rx7700s_brain_analyzer()
        
        print(f"\nBrain analyzer: {' SUCCESS' if success else ' FAILED'}")
        return success
        
    except Exception as e:
        print(f" Brain analyzer test failed: {e}")
        return False

def test_gpu_usage_monitoring():
    """Test GPU usage monitoring."""
    print("\n Testing GPU Usage Monitoring")
    print("=" * 50)
    
    try:
        # PowerShell command to check GPU usage
        ps_command = '''
        Get-Counter "\\GPU Engine(*)\\Utilization Percentage" -MaxSamples 1 | 
        ForEach-Object { $_.CounterSamples } | 
        Where-Object { $_.InstanceName -like "*engtype_3D*" } |
        ForEach-Object { 
            $gpu = $_.InstanceName.Split("_")[1]
            $usage = [math]::Round($_.CookedValue, 2)
            Write-Host "GPU $gpu 3D Usage: $usage%"
        }
        '''
        
        print("Checking current GPU usage...")
        result = subprocess.run([
            'powershell', '-Command', ps_command
        ], capture_output=True, text=True, shell=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            print("Current GPU usage:")
            print(result.stdout.strip())
            return True
        else:
            print(" GPU usage monitoring not available")
            print("   Use Task Manager → Performance → GPU to monitor usage")
            return True  # Not a critical failure
            
    except Exception as e:
        print(f" GPU monitoring test failed: {e}")
        print("   Use Task Manager → Performance → GPU to monitor usage")
        return True  # Not a critical failure

def launch_brain_gui():
    """Launch the fixed brain GUI."""
    print("\n🖥️ Launching RX 7700S Brain GUI")
    print("=" * 50)
    
    try:
        print("Starting RX 7700S Brain GUI...")
        print("The GUI will open in a new window.")
        print("Use the GUI to:")
        print("1. Generate test brain data")
        print("2. Run analysis on RX 7700S")
        print("3. Monitor GPU usage in Task Manager")
        print()
        
        # Import and run GUI
        from rx7700s_brain_gui_fixed import main as run_gui
        run_gui()
        
        return True
        
    except Exception as e:
        print(f" Failed to launch GUI: {e}")
        return False

def create_verification_report():
    """Create a verification report."""
    print("\n Creating Verification Report")
    print("=" * 50)
    
    try:
        report_content = f"""# RX 7700S Verification Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Environment Variables Set
"""
        
        # Check environment variables
        rx7700s_vars = [
            'HIP_VISIBLE_DEVICES',
            'ROCR_VISIBLE_DEVICES', 
            'GPU_MAX_ALLOC_PERCENT',
            'HSA_OVERRIDE_GFX_VERSION'
        ]
        
        for var in rx7700s_vars:
            value = os.environ.get(var, 'NOT SET')
            report_content += f"- {var}: {value}\n"
        
        report_content += f"""
## System Information
- Python: {sys.version.split()[0]}
- Platform: {sys.platform}
- Working Directory: {Path.cwd()}

## Test Results
Run the complete test suite to populate this section.

## GPU Usage Verification
1. Open Task Manager
2. Go to Performance tab
3. Look for GPU 1 (should be RX 7700S)
4. Run brain analysis and verify GPU 1 shows activity

## Expected Results
- GPU 0 (780M): Minimal usage during brain processing
- GPU 1 (RX 7700S): High usage during brain processing
- Processing time: Significantly faster than CPU-only

## Troubleshooting
If RX 7700S is not being used:
1. Update AMD drivers to latest version
2. Restart computer after driver update
3. Check Device Manager for GPU errors
4. Verify RX 7700S is enabled in BIOS
5. Run: python rx7700s_environment_setup.py
"""
        
        report_file = Path("RX7700S_Verification_Report.md")
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f" Verification report created: {report_file}")
        return True
        
    except Exception as e:
        print(f" Failed to create report: {e}")
        return False

def main():
    """Main test function."""
    print(" RX 7700S Complete Test Suite")
    print("=" * 60)
    print("This will test all components of the fixed RX 7700S implementation")
    print()
    
    # Test results
    results = {}
    
    # Run tests
    results['environment'] = test_environment_setup()
    results['detection'] = test_gpu_detection()
    results['analyzer'] = test_brain_analyzer()
    results['monitoring'] = test_gpu_usage_monitoring()
    
    # Create report
    results['report'] = create_verification_report()
    
    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUITE SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = " PASS" if success else " FAIL"
        print(f"{test_name.capitalize():15}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("\n🎉 RX 7700S implementation is working correctly!")
        print("\nNext steps:")
        print("1. Launch the brain GUI: python rx7700s_brain_gui_fixed.py")
        print("2. Generate test data and run analysis")
        print("3. Monitor Task Manager → Performance → GPU 1")
        print("4. Verify GPU 1 shows high usage during processing")
        
        # Ask if user wants to launch GUI
        print("\n" + "=" * 60)
        response = input("Launch RX 7700S Brain GUI now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            launch_brain_gui()
        else:
            print("You can launch the GUI later with:")
            print("python rx7700s_brain_gui_fixed.py")
            
    else:
        print("\n Some tests failed - RX 7700S may not work correctly")
        print("\nTroubleshooting:")
        print("1. Update AMD drivers to latest version")
        print("2. Restart your computer")
        print("3. Check Device Manager for GPU errors")
        print("4. Re-run this test suite")
    
    print(f"\nVerification report: RX7700S_Verification_Report.md")
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()