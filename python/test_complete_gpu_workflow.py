#!/usr/bin/env python3
"""
Complete GPU workflow test - verifies RX 7700S integration in brain GUI.
This test demonstrates the full pipeline from MRI data to GPU-accelerated brain visualization.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

def create_realistic_brain_mri():
    """Create realistic synthetic MRI brain data for testing."""
    print("Creating realistic synthetic MRI brain volume...")
    
    # Create brain-shaped volume (64x64x32 for fast testing)
    x, y, z = np.mgrid[0:64, 0:64, 0:32]
    
    # Create brain ellipsoid
    cx, cy, cz = 32, 32, 16  # Center
    rx, ry, rz = 25, 20, 12  # Radii
    
    # Distance from center (ellipsoid equation)
    dist = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2
    
    # Create brain mask
    brain_mask = dist <= 1.0
    
    # Add brain tissue intensities
    volume = np.zeros((64, 64, 32), dtype=np.float32)
    
    # Gray matter (higher intensity)
    gray_matter = (dist <= 0.9) & (dist > 0.7)
    volume[gray_matter] = 0.8 + 0.1 * np.random.random(gray_matter.sum())
    
    # White matter (medium intensity) 
    white_matter = dist <= 0.7
    volume[white_matter] = 0.6 + 0.1 * np.random.random(white_matter.sum())
    
    # Add some noise
    volume += 0.05 * np.random.random(volume.shape)
    
    # Normalize
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    print(f"Created MRI volume: {volume.shape}, range: {volume.min():.3f} to {volume.max():.3f}")
    print(f"Brain voxels: {brain_mask.sum():,} / {volume.size:,}")
    
    return volume

def test_gpu_brain_processing_direct():
    """Test GPU brain processing directly."""
    try:
        from enhanced_dual_gpu_analyzer import EnhancedDualGPUBrainAnalyzer
        
        print("\n" + "="*60)
        print("DIRECT GPU BRAIN PROCESSING TEST")
        print("="*60)
        
        # Create test data
        brain_volume = create_realistic_brain_mri()
        
        # Initialize GPU analyzer
        analyzer = EnhancedDualGPUBrainAnalyzer()
        
        # Process on RX 7700S
        print("\nProcessing brain volume on RX 7700S...")
        start_time = time.time()
        
        result = analyzer.process_brain_volume_gpu(brain_volume, use_discrete_gpu=True)
        
        processing_time = time.time() - start_time
        
        if result and 'brain_surface' in result:
            vertices = len(result['brain_surface']['vertices'])
            faces = len(result['brain_surface']['faces'])
            gpu_time = result.get('processing_time', processing_time)
            
            print(f"\nSUCCESS: GPU processing completed!")
            print(f"Processing time: {gpu_time:.3f}s")
            print(f"Generated surface: {vertices} vertices, {faces} faces")
            
            return True, vertices, faces, gpu_time
        else:
            print("FAILED: No surface generated")
            return False, 0, 0, processing_time
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0, 0

def test_gui_integration():
    """Test GUI integration with GPU processing."""
    try:
        from ultra_realistic_brain_gui import UltraRealisticBrainGUI
        import tkinter as tk
        
        print("\n" + "="*60)
        print("GUI GPU INTEGRATION TEST")
        print("="*60)
        
        # Create GUI without showing window
        root = tk.Tk()
        root.withdraw()
        
        gui = UltraRealisticBrainGUI(root)
        
        # Create test data
        brain_volume = create_realistic_brain_mri()
        gui.fmri_data = brain_volume
        
        print("Testing GUI GPU brain processing...")
        
        # Test GPU processing method
        start_time = time.time()
        result = gui._gpu_accelerated_brain_processing(brain_volume)
        processing_time = time.time() - start_time
        
        if result:
            vertices = len(result.get('vertices', []))
            faces = len(result.get('faces', []))
            gpu_time = result.get('gpu_time', processing_time)
            
            print(f"\nSUCCESS: GUI GPU integration working!")
            print(f"Processing time: {gpu_time:.3f}s")
            print(f"Generated surface: {vertices} vertices, {faces} faces")
            
            root.destroy()
            return True, vertices, faces, gpu_time
        else:
            print("FAILED: GUI GPU processing returned no results")
            root.destroy()
            return False, 0, 0, processing_time
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0, 0

def visualize_gpu_performance(direct_result, gui_result):
    """Create performance visualization."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance comparison
        methods = ['Direct GPU', 'GUI GPU']
        times = [direct_result[3], gui_result[3]]
        vertices = [direct_result[1], gui_result[1]]
        
        ax1.bar(methods, times, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('RX 7700S Processing Performance')
        ax1.grid(True, alpha=0.3)
        
        # Add time labels on bars
        for i, (method, time_val) in enumerate(zip(methods, times)):
            ax1.text(i, time_val + 0.01, f'{time_val:.3f}s', 
                    ha='center', va='bottom', fontweight='bold')
        
        ax2.bar(methods, vertices, color=['#FF6B6B', '#4ECDC4'])
        ax2.set_ylabel('Vertices Generated')
        ax2.set_title('Surface Quality Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add vertex labels on bars
        for i, (method, vert_count) in enumerate(zip(methods, vertices)):
            ax2.text(i, vert_count + 5, f'{vert_count:,}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('gpu_performance_comparison.png', dpi=150, bbox_inches='tight')
        print("\nPerformance chart saved as: gpu_performance_comparison.png")
        
    except Exception as e:
        print(f"Visualization error: {e}")

def main():
    """Main test workflow."""
    print("RX 7700S GPU BRAIN PROCESSING WORKFLOW TEST")
    print("This test verifies complete GPU integration")
    print("\n>>> WATCH TASK MANAGER - GPU 1 (RX 7700S) should activate <<<")
    
    # Test 1: Direct GPU processing
    direct_success, direct_verts, direct_faces, direct_time = test_gpu_brain_processing_direct()
    
    # Test 2: GUI integration
    gui_success, gui_verts, gui_faces, gui_time = test_gui_integration()
    
    # Results summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print(f"Direct GPU Test:     {'PASS' if direct_success else 'FAIL'}")
    if direct_success:
        print(f"  - Processing time: {direct_time:.3f}s")
        print(f"  - Surface quality: {direct_verts:,} vertices, {direct_faces:,} faces")
    
    print(f"GUI Integration:     {'PASS' if gui_success else 'FAIL'}")
    if gui_success:
        print(f"  - Processing time: {gui_time:.3f}s") 
        print(f"  - Surface quality: {gui_verts:,} vertices, {gui_faces:,} faces")
    
    if direct_success and gui_success:
        print(f"\nSUCCESS: RX 7700S GPU acceleration fully integrated!")
        print(f"   Average processing time: {(direct_time + gui_time) / 2:.3f}s")
        print(f"   Total vertices generated: {direct_verts + gui_verts:,}")
        
        # Create performance visualization
        visualize_gpu_performance((direct_success, direct_verts, direct_faces, direct_time),
                                (gui_success, gui_verts, gui_faces, gui_time))
        
        print(f"\nPerformance chart created!")
        print(f"Your brain GUI is ready for real-time visualization!")
        
    elif direct_success:
        print(f"\nWARNING: GPU works but GUI integration has issues")
        
    elif gui_success:
        print(f"\nWARNING: GUI works but direct GPU has issues")
        
    else:
        print(f"\nFAILED: GPU acceleration not working properly")
    
    print("="*60)

if __name__ == "__main__":
    main()
