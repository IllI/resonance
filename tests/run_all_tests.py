#!/usr/bin/env python3
"""
Master Test Runner for Real-time fMRI AI Analysis

Runs all test suites and provides comprehensive reporting for:
1. Core fMRI analysis functionality
2. GPU acceleration layer (optional)
3. AI models and real-time processing
4. Integration and system tests

This is the entry point for CI/CD pipelines and development testing.
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add python directory to path
sys.path.append(str(Path(__file__).parent.parent / "python"))

def run_test_suite(test_script: str, description: str) -> Tuple[bool, Dict]:
    """
    Run a specific test suite and return results.
    
    Args:
        test_script: Path to test script
        description: Human-readable description
        
    Returns:
        Tuple of (success, results_dict)
    """
    print(f"\n{'='*80}")
    print(f"🧪 Running: {description}")
    print(f"   Script: {test_script}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run test script directly as it works when run manually
        result = subprocess.run([
            sys.executable, test_script
        ], capture_output=True, text=True, timeout=300, cwd=Path(__file__).parent.parent)  # 5 minute timeout
        
        duration = time.time() - start_time
        
        # Parse output for test metrics
        output = result.stdout + result.stderr
        
        # Extract test counts from unittest output format
        tests_run = 0
        failures = 0
        errors = 0
        skipped = 0
        
        for line in output.split('\n'):
            # Look for "Ran X tests in Y.Zs" pattern
            if line.startswith("Ran ") and " tests in " in line:
                try:
                    tests_run = int(line.split("Ran ")[1].split(" tests")[0])
                except:
                    pass
            # Look for failure/error counts in format like "FAILED (failures=2, errors=1)"
            if line.startswith("FAILED (") or "failures=" in line or "errors=" in line:
                if "failures=" in line:
                    try:
                        failures = int(line.split("failures=")[1].split(",")[0].split(")")[0])
                    except:
                        pass
                if "errors=" in line:
                    try:
                        errors = int(line.split("errors=")[1].split(",")[0].split(")")[0])
                    except:
                        pass
            # Look for OK indicating success
            if line.strip() == "OK":
                failures = 0
                errors = 0
        
        success = result.returncode == 0 and failures == 0 and errors == 0
        
        results = {
            'success': success,
            'return_code': result.returncode,
            'duration': duration,
            'tests_run': tests_run,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        # Print summary
        if success:
            print(f"✅ {description}: PASSED")
        else:
            print(f"❌ {description}: FAILED")
        
        print(f"   Duration: {duration:.2f}s")
        print(f"   Tests: {tests_run}, Failures: {failures}, Errors: {errors}, Skipped: {skipped}")
        
        return success, results
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ {description}: TIMEOUT ({duration:.1f}s)")
        
        return False, {
            'success': False,
            'timeout': True,
            'duration': duration,
            'error': 'Test suite timed out'
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {description}: ERROR - {e}")
        
        return False, {
            'success': False,
            'exception': str(e),
            'duration': duration
        }

def check_dependencies() -> Dict[str, bool]:
    """Check availability of optional dependencies."""
    print("🔍 Checking Dependencies...")
    
    dependencies = {}
    
    # Core dependencies (should always be available)
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    try:
        import scipy  
        dependencies['scipy'] = True
    except ImportError:
        dependencies['scipy'] = False
    
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        dependencies['matplotlib'] = False
    
    # Optional ML dependencies
    try:
        import sklearn
        dependencies['scikit-learn'] = True
    except ImportError:
        dependencies['scikit-learn'] = False
    
    try:
        import torch
        dependencies['pytorch'] = True
    except ImportError:
        dependencies['pytorch'] = False
    
    # Neuroimaging dependencies
    try:
        import nibabel
        dependencies['nibabel'] = True
    except ImportError:
        dependencies['nibabel'] = False
    
    try:
        import nilearn
        dependencies['nilearn'] = True
    except ImportError:
        dependencies['nilearn'] = False
    
    # Visualization dependencies
    try:
        import plotly
        dependencies['plotly'] = True
    except ImportError:
        dependencies['plotly'] = False
    
    # GPU dependencies
    try:
        import pyopencl
        dependencies['pyopencl'] = True
    except ImportError:
        dependencies['pyopencl'] = False
    
    # Print dependency status
    for dep, available in dependencies.items():
        status = "✅" if available else "❌"
        print(f"   {status} {dep}")
    
    return dependencies

def generate_test_report(test_results: Dict, dependencies: Dict, output_file: str = None):
    """Generate comprehensive test report."""
    
    # Calculate overall statistics
    total_tests = sum(result.get('tests_run', 0) for result in test_results.values())
    total_failures = sum(result.get('failures', 0) for result in test_results.values())
    total_errors = sum(result.get('errors', 0) for result in test_results.values())
    total_skipped = sum(result.get('skipped', 0) for result in test_results.values())
    total_duration = sum(result.get('duration', 0) for result in test_results.values())
    
    successful_suites = sum(1 for result in test_results.values() if result.get('success', False))
    total_suites = len(test_results)
    
    # Create report
    report = {
        'timestamp': time.time(),
        'summary': {
            'total_test_suites': total_suites,
            'successful_suites': successful_suites,
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'total_duration': total_duration,
            'success_rate': successful_suites / total_suites if total_suites > 0 else 0
        },
        'dependencies': dependencies,
        'test_results': test_results
    }
    
    # Print summary report
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE TEST REPORT")
    print(f"{'='*80}")
    
    print(f"\n🎯 Overall Results:")
    print(f"   Test Suites: {successful_suites}/{total_suites} passed ({successful_suites/total_suites*100:.1f}%)")
    print(f"   Individual Tests: {total_tests} run, {total_failures} failed, {total_errors} errors, {total_skipped} skipped")
    print(f"   Total Duration: {total_duration:.2f}s")
    
    print(f"\n🔧 Test Suite Breakdown:")
    for suite_name, results in test_results.items():
        status = "✅" if results.get('success') else "❌"
        duration = results.get('duration', 0)
        tests = results.get('tests_run', 0)
        print(f"   {status} {suite_name}: {tests} tests in {duration:.2f}s")
    
    print(f"\n📦 Dependencies:")
    core_deps = ['numpy', 'scipy', 'matplotlib', 'plotly']
    optional_deps = ['scikit-learn', 'pytorch', 'nibabel', 'nilearn', 'pyopencl']
    
    core_available = sum(1 for dep in core_deps if dependencies.get(dep, False))
    optional_available = sum(1 for dep in optional_deps if dependencies.get(dep, False))
    
    print(f"   Core: {core_available}/{len(core_deps)} available")
    print(f"   Optional: {optional_available}/{len(optional_deps)} available")
    
    # Overall assessment
    print(f"\n🏆 Assessment:")
    if successful_suites == total_suites and total_failures == 0 and total_errors == 0:
        print("   🎉 ALL TESTS PASSED - Ready for production!")
    elif successful_suites >= total_suites * 0.8:
        print("   ✅ MOSTLY SUCCESSFUL - Core functionality verified")
    else:
        print("   ⚠️ ISSUES DETECTED - Review failed tests before deployment")
    
    # Save detailed report if requested
    if output_file:
        report_path = Path(output_file)
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_path}")
    
    return report

def main():
    """Run comprehensive test suite for the project."""
    print("🚀 Real-time fMRI AI Analysis - Comprehensive Test Suite")
    print("=" * 80)
    print("Testing modular architecture with independent components:")
    print("• Core fMRI Analysis (hardware-independent)")  
    print("• GPU Acceleration Layer (optional)")
    print("• AI Models (PyTorch optional)")
    print("• Integration Tests")
    
    # Check dependencies first
    dependencies = check_dependencies()
    
    # Define test suites
    test_suites = [
        {
            'script': 'tests/test_core_fmri_analysis.py',
            'name': 'Core fMRI Analysis',
            'description': 'Core brain network analysis (no GPU required)',
            'required': True
        },
        {
            'script': 'tests/test_gpu_acceleration.py', 
            'name': 'GPU Acceleration',
            'description': 'GPU acceleration layer with CPU fallbacks',
            'required': False
        },
        {
            'script': 'tests/test_ai_models.py',
            'name': 'AI Models',
            'description': 'AI-powered ROI detection and classification',
            'required': False
        }
    ]
    
    # Run all test suites
    test_results = {}
    overall_success = True
    
    for suite in test_suites:
        script_path = Path(suite['script'])
        
        if script_path.exists():
            success, results = run_test_suite(
                str(script_path), 
                suite['description']
            )
            
            test_results[suite['name']] = results
            
            if suite['required'] and not success:
                overall_success = False
                
        else:
            print(f"⚠️ Test suite not found: {script_path}")
            test_results[suite['name']] = {
                'success': False,
                'error': 'Test file not found'
            }
            
            if suite['required']:
                overall_success = False
    
    # Generate comprehensive report
    report = generate_test_report(
        test_results, 
        dependencies,
        output_file='reports/test_report.json'
    )
    
    # Exit with appropriate code
    if overall_success:
        print(f"\n🎉 Test suite completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Test suite failed - check results above")
        sys.exit(1)

if __name__ == "__main__":
    main() 