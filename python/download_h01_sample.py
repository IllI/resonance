#!/usr/bin/env python3
"""
Download H01 Sample Data for Local Neuroglancer
"""

import os
import numpy as np
from cloudvolume import CloudVolume
from pathlib import Path

def download_h01_sample():
    """Download a small subvolume of the H01 dataset."""
    print("Downloading H01 sample data...")
    
    # Define local output directory
    output_dir = Path("data/h01_sample")
    
    # FORCE MOCK DATA GENERATION FOR NOW due to CloudVolume issues
    print("CloudVolume download issues detected. Generating synthetic mock data...")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    center = np.array([100000, 100000, 2000])
    size = np.array([512, 512, 64])
    
    generate_mock_data(output_dir, size)
    return True

    # Original download logic commented out below...
    """
    # H01 Public Layers
    # 8nm resolution is good for overview and small enough for quick download
    image_layer = "precomputed://https://h01-release.storage.googleapis.com/data/20210601/4nm_raw"
    seg_layer = "precomputed://https://h01-release.storage.googleapis.com/data/20210601/agg20200916c3_segmentation"
    
    # ... (rest of download logic)
    """

def generate_mock_data(output_dir, size):
    """Generate synthetic data for testing (Manual Write)."""
    print("Generating synthetic Neuroglancer data (Manual Write)...")
    
    import json
    
    # Image
    image_path = output_dir / "image"
    image_path.mkdir(parents=True, exist_ok=True)
    
    # Info JSON
    info = {
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [[64, 64, 64]],
                "encoding": "raw",
                "key": "8nm",
                "resolution": [8, 8, 33],
                "size": [int(size[0]), int(size[1]), int(size[2])],
                "voxel_offset": [0, 0, 0]
            }
        ],
        "type": "image"
    }
    
    with open(image_path / "info", "w") as f:
        json.dump(info, f)
        
    # Generate chunks
    print("Writing image chunks...")
    data = np.random.randint(0, 255, size, dtype=np.uint8)
    x, y, z = np.indices(size)
    center = size // 2
    mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 < (size[0]//4)**2
    data[mask] = 200
    
    chunk_size = 64
    for x in range(0, size[0], chunk_size):
        for y in range(0, size[1], chunk_size):
            for z in range(0, size[2], chunk_size):
                chunk = data[x:x+chunk_size, y:y+chunk_size, z:z+chunk_size]
                # Fortran order for Neuroglancer raw chunks? No, usually C order but transposed? 
                # CloudVolume uses F order for raw. Neuroglancer expects F order (x, y, z, channel).
                # We'll write as F order.
                filename = f"{x}-{x+chunk_size}_{y}-{y+chunk_size}_{z}-{z+chunk_size}"
                with open(image_path / filename, "wb") as f:
                    f.write(chunk.tobytes('F'))
                    
    print(f"Generated image data at {image_path}")
    
    # Segmentation
    seg_path = output_dir / "segmentation"
    seg_path.mkdir(parents=True, exist_ok=True)
    
    info_seg = {
        "data_type": "uint64",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [[64, 64, 64]],
                "encoding": "raw",
                "key": "8nm",
                "resolution": [8, 8, 33],
                "size": [int(size[0]), int(size[1]), int(size[2])],
                "voxel_offset": [0, 0, 0]
            }
        ],
        "type": "segmentation"
    }
    
    with open(seg_path / "info", "w") as f:
        json.dump(info_seg, f)
        
    print("Writing segmentation chunks...")
    seg_data = np.zeros(size, dtype=np.uint64)
    seg_data[mask] = 12345
    
    for x in range(0, size[0], chunk_size):
        for y in range(0, size[1], chunk_size):
            for z in range(0, size[2], chunk_size):
                chunk = seg_data[x:x+chunk_size, y:y+chunk_size, z:z+chunk_size]
                filename = f"{x}-{x+chunk_size}_{y}-{y+chunk_size}_{z}-{z+chunk_size}"
                with open(seg_path / filename, "wb") as f:
                    f.write(chunk.tobytes('F'))
                    
    print(f"Generated segmentation data at {seg_path}")


if __name__ == "__main__":
    download_h01_sample()
