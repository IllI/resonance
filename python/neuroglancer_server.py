#!/usr/bin/env python3
"""
Neuroglancer Server Manager
"""

import neuroglancer
import tornado.web
import os
from pathlib import Path

# Global viewer instance
_viewer = None

def start_neuroglancer_server(data_path="data/h01_sample"):
    """
    Start a local Neuroglancer server serving data from data_path.
    Returns the viewer URL.
    """
    global _viewer
    
    # Configure neuroglancer to use a specific bind address if needed
    neuroglancer.set_server_bind_address('127.0.0.1')
    
    if _viewer is None:
        _viewer = neuroglancer.Viewer()
        
    # Convert relative path to absolute
    abs_path = Path(data_path).absolute()
    image_path = f"precomputed://file://{abs_path}/image"
    seg_path = f"precomputed://file://{abs_path}/segmentation"
    
    print(f"Serving data from: {abs_path}")
    
    with _viewer.txn() as s:
        # Add Image Layer
        s.layers['image'] = neuroglancer.ImageLayer(
            source=image_path,
        )
        
        # Add Segmentation Layer
        s.layers['segmentation'] = neuroglancer.SegmentationLayer(
            source=seg_path,
        )
        
        # Set initial position (center of our mock data)
        # Mock data size is 512x512x64
        s.position = [256, 256, 32]
        s.crossSectionScale = 4
        s.projectionScale = 2048
        
    print(f"Neuroglancer running at: {_viewer}")
    return str(_viewer)

def stop_neuroglancer_server():
    """Stop the Neuroglancer server (if possible)."""
    # Neuroglancer server runs in background threads, difficult to fully stop 
    # without exiting process, but we can clear the viewer.
    global _viewer
    if _viewer is not None:
        # _viewer = None 
        pass

if __name__ == "__main__":
    # Test run
    url = start_neuroglancer_server()
    print(f"Open this URL: {url}")
    input("Press Enter to exit...")
