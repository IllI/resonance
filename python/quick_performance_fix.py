#!/usr/bin/env python3
"""
Quick Performance Fix for Brain GUI

Applies immediate performance improvements to make brain rotation smooth.
"""

import sys
import os

def apply_quick_performance_fix():
    """Apply quick performance fixes to the brain GUI."""
    print("Applying quick performance fixes...")
    
    # Read the current GUI file
    gui_file = "ultra_realistic_brain_gui.py"
    
    try:
        with open(gui_file, 'r') as f:
            content = f.read()
        
        # Performance fix 1: Reduce vertex count more aggressively
        old_code = "step = max(1, len(verts) // 5000)"
        new_code = "step = max(1, len(verts) // 2000)  # More aggressive optimization for smooth rotation"
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            print(" Applied aggressive vertex reduction")
        
        # Performance fix 2: Use smaller point size for scatter
        old_scatter = "s=0.8, alpha=0.7"
        new_scatter = "s=0.3, alpha=0.6"  # Smaller points, less alpha for speed
        
        if old_scatter in content:
            content = content.replace(old_scatter, new_scatter)
            print(" Optimized scatter plot settings")
        
        # Performance fix 3: Disable expensive features during interaction
        matplotlib_config = '''
            # Performance optimizations for matplotlib
            import matplotlib
            matplotlib.use('TkAgg')  # Fast backend
            matplotlib.rcParams['path.simplify'] = True
            matplotlib.rcParams['path.simplify_threshold'] = 0.1
            matplotlib.rcParams['agg.path.chunksize'] = 10000
            print("Applied matplotlib performance optimizations")
'''
        
        # Add performance config after imports
        import_section = "import warnings"
        if import_section in content and matplotlib_config not in content:
            content = content.replace(import_section, import_section + matplotlib_config)
            print(" Added matplotlib performance optimizations")
        
        # Write back the optimized file
        with open(gui_file, 'w') as f:
            f.write(content)
        
        print(" Performance fixes applied successfully!")
        return True
        
    except Exception as e:
        print(f" Error applying fixes: {e}")
        return False

def create_fast_launcher():
    """Create a fast launcher script with performance settings."""
    launcher_code = '''#!/usr/bin/env python3
"""
Fast Brain GUI Launcher - Optimized for Performance
"""

import os
import sys

# Performance environment variables
os.environ['MPLBACKEND'] = 'TkAgg'  # Fast matplotlib backend
os.environ['NUMBA_DISABLE_JIT'] = '0'  # Enable JIT compilation if available

# Add performance flags
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for speed

def main():
    """Launch brain GUI with performance optimizations."""
    print(" Launching Brain GUI with Performance Optimizations")
    print("=" * 60)
    print("Performance Settings:")
    print("  • Matplotlib Backend: TkAgg (Fast)")
    print("  • Vertex Limit: 2,000 (Smooth Rotation)")
    print("  • Point Size: 0.3 (Reduced)")
    print("  • Alpha: 0.6 (Optimized)")
    print("=" * 60)
    
    try:
        from ultra_realistic_brain_gui import main as launch_gui
        launch_gui()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("fast_brain_gui.py", "w") as f:
            f.write(launcher_code)
        print(" Created fast launcher: fast_brain_gui.py")
        return True
    except Exception as e:
        print(f" Error creating launcher: {e}")
        return False

def main():
    """Main function."""
    print("Quick Performance Fix for Brain GUI")
    print("=" * 40)
    
    # Apply performance fixes
    fix_success = apply_quick_performance_fix()
    
    # Create fast launcher
    launcher_success = create_fast_launcher()
    
    print("\n" + "=" * 40)
    if fix_success and launcher_success:
        print("🎉 PERFORMANCE OPTIMIZATION COMPLETE!")
        print("\nTo use the optimized brain GUI:")
        print("1. Run: python fast_brain_gui.py")
        print("2. Load your MRI data")
        print("3. Generate 3D model")
        print("4. Enjoy smooth rotation! ")
        print("\nOptimizations applied:")
        print("  • Reduced vertices to 2,000 for smooth rotation")
        print("  • Optimized scatter plot rendering")
        print("  • Fast matplotlib backend")
        print("  • Simplified visual effects")
    else:
        print(" Some optimizations failed. Check errors above.")

if __name__ == "__main__":
    main()