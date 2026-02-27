#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Brain GUI Launcher

A simplified launcher that directly starts the brain visualization GUI
without problematic imports or Unicode characters.
"""

import sys
import os
from pathlib import Path

def main():
    """Direct launcher for the brain GUI."""
    print("Ultra-Realistic Brain 3D Visualization - Direct Launch")
    print("=" * 60)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        # Import and launch the GUI directly
        import tkinter as tk
        from tkinter import ttk
        
        # Create root window
        root = tk.Tk()
        root.title("Ultra-Realistic Brain 3D Visualization")
        root.geometry("1400x900")
        root.configure(bg='#2b2b2b')
        
        # Try to import the GUI class
        try:
            from ultra_realistic_brain_gui import UltraRealisticBrainGUI
            app = UltraRealisticBrainGUI(root)
            print("GUI initialized successfully")
        except Exception as e:
            print(f"Error importing GUI: {e}")
            # Create a simple fallback GUI
            create_fallback_gui(root)
        
        # Center the window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        # Start the GUI
        print("Starting GUI main loop...")
        root.mainloop()
        return True
        
    except Exception as e:
        print(f"Failed to launch GUI: {e}")
        print("\nTrying alternative launch method...")
        
        # Alternative launch
        try:
            os.system("python ultra_realistic_brain_gui.py")
            return True
        except Exception as e2:
            print(f"Alternative launch failed: {e2}")
            return False

def create_fallback_gui(root):
    """Create a simple fallback GUI if main GUI fails."""
    # Main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill="both", expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="Brain 3D Visualization", 
                           font=("Arial", 24, "bold"))
    title_label.pack(pady=(0, 30))
    
    # Status
    status_label = ttk.Label(main_frame, text="Fallback mode - Some features may be limited", 
                            font=("Arial", 12))
    status_label.pack(pady=(0, 20))
    
    # Load button
    load_btn = ttk.Button(main_frame, text="Load MRI Volume", 
                         command=lambda: load_mri_fallback())
    load_btn.pack(pady=10)
    
    # Visualization area
    viz_frame = ttk.LabelFrame(main_frame, text="3D Visualization", padding="10")
    viz_frame.pack(fill="both", expand=True, pady=(20, 0))
    
    viz_label = ttk.Label(viz_frame, text="3D brain visualization will appear here", 
                         font=("Arial", 14))
    viz_label.pack(expand=True)

def load_mri_fallback():
    """Fallback MRI loading function."""
    from tkinter import filedialog, messagebox
    
    file_path = filedialog.askopenfilename(
        title="Select MRI Volume",
        filetypes=[
            ("NIfTI files", "*.nii *.nii.gz"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        messagebox.showinfo("Success", f"MRI file selected:\n{Path(file_path).name}")

if __name__ == "__main__":
    success = main()
    if success:
        print("GUI launched successfully!")
    else:
        print("Failed to launch GUI. Check error messages above.")
        input("Press Enter to exit...")
