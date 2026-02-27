#!/usr/bin/env python3
"""
Fix Brain GUI Blank Screen Issue
Creates a working brain visualization that bypasses GPU issues
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

def create_simple_brain_model():
    """Create a simple but recognizable brain-shaped 3D model"""
    # Create brain-like surface using parametric equations
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    # Brain-like shape (modified ellipsoid with bumps)
    a, b, c = 2.0, 1.5, 1.2  # Brain proportions
    
    # Base ellipsoid
    x = a * np.sin(V) * np.cos(U)
    y = b * np.sin(V) * np.sin(U) 
    z = c * np.cos(V)
    
    # Add brain-like bumps and indentations
    brain_texture = 0.1 * (np.sin(5*U) * np.sin(5*V) + 0.5 * np.sin(10*U) * np.sin(3*V))
    
    # Apply texture to create brain-like surface
    x += brain_texture * np.sin(V) * np.cos(U)
    y += brain_texture * np.sin(V) * np.sin(U)
    z += brain_texture * np.cos(V)
    
    # Create hemispheres (left and right brain)
    left_hemisphere = x < 0
    right_hemisphere = x >= 0
    
    # Add central sulcus (groove between hemispheres)
    central_groove = np.abs(x) < 0.1
    z[central_groove] -= 0.2
    
    return x, y, z, left_hemisphere, right_hemisphere

def create_brain_visualization_widget(parent_frame):
    """Create a working brain visualization widget"""
    try:
        # Clear existing widgets
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('#2b2b2b')
        
        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#2b2b2b')
        
        # Generate brain model
        print("Generating brain model...")
        x, y, z, left_hem, right_hem = create_simple_brain_model()
        
        # Plot brain hemispheres with different colors
        ax.plot_surface(x, y, z, 
                       facecolors=np.where(left_hem, 'lightcoral', 'lightblue'),
                       alpha=0.8, 
                       linewidth=0,
                       antialiased=True)
        
        # Customize the plot
        ax.set_xlabel('X (Left-Right)', color='white')
        ax.set_ylabel('Y (Anterior-Posterior)', color='white') 
        ax.set_zlabel('Z (Superior-Inferior)', color='white')
        ax.set_title('3D Brain Model', color='white', fontsize=14, pad=20)
        
        # Set equal aspect ratio
        max_range = 2.5
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Style the axes
        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        
        # Create canvas and add to parent frame
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for interaction
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        
        print("✓ Brain visualization created successfully!")
        return True
        
    except Exception as e:
        print(f"Error creating brain visualization: {e}")
        
        # Create fallback text display
        label = tk.Label(parent_frame, 
                        text=f"Brain Visualization Error:\n{str(e)}\n\nTry restarting the application",
                        bg='#2b2b2b', fg='white', 
                        font=('Arial', 12),
                        justify=tk.CENTER)
        label.pack(expand=True, fill=tk.BOTH)
        return False

def test_brain_visualization():
    """Test the brain visualization in a standalone window"""
    root = tk.Tk()
    root.title("Brain Visualization Test")
    root.geometry("800x600")
    root.configure(bg='#2b2b2b')
    
    # Create frame for visualization
    viz_frame = tk.Frame(root, bg='#2b2b2b')
    viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create brain visualization
    success = create_brain_visualization_widget(viz_frame)
    
    if success:
        print("Brain visualization test successful!")
    else:
        print("Brain visualization test failed!")
    
    root.mainloop()

if __name__ == "__main__":
    test_brain_visualization()