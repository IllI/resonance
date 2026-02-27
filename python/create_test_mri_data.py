#!/usr/bin/env python3
"""
Create test MRI data for brain GUI testing.
Generates synthetic brain volumes in NIfTI format.
"""

import numpy as np
import nibabel as nib
import os

def create_synthetic_brain_mri(size=(128, 128, 64), filename="test_brain.nii.gz"):
    """Create realistic synthetic MRI brain data."""
    print(f"Creating synthetic brain MRI: {size}")
    
    x, y, z = np.mgrid[0:size[0], 0:size[1], 0:size[2]]
    
    # Brain center and shape parameters
    cx, cy, cz = size[0]//2, size[1]//2, size[2]//2
    rx, ry, rz = size[0]//3, size[1]//3, size[2]//3
    
    # Create brain ellipsoid
    dist = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2
    
    # Initialize volume
    volume = np.zeros(size, dtype=np.float32)
    
    # Add brain tissues with realistic MRI intensities
    
    # 1. White matter (core)
    white_matter = dist <= 0.5
    volume[white_matter] = 0.7 + 0.1 * np.random.random(white_matter.sum())
    
    # 2. Gray matter (cortex)
    gray_matter = (dist <= 0.8) & (dist > 0.5)
    volume[gray_matter] = 0.5 + 0.1 * np.random.random(gray_matter.sum())
    
    # 3. CSF (cerebrospinal fluid)
    csf = (dist <= 0.9) & (dist > 0.8)
    volume[csf] = 0.2 + 0.05 * np.random.random(csf.sum())
    
    # 4. Add some anatomical structures
    
    # Ventricles (darker regions)
    ventricle_mask = (dist <= 0.3) & (np.abs(x - cx) < rx/4) & (np.abs(y - cy) < ry/4)
    volume[ventricle_mask] = 0.1
    
    # Cortical folding simulation
    noise_scale = 0.02
    cortical_noise = noise_scale * np.sin(x * 0.3) * np.cos(y * 0.25) * np.sin(z * 0.4)
    cortex_region = (dist <= 0.85) & (dist > 0.6)
    volume[cortex_region] += cortical_noise[cortex_region]
    
    # Add realistic noise
    volume += 0.02 * np.random.random(volume.shape)
    
    # Normalize to 0-1000 (typical MRI range)
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    volume = (volume * 1000).astype(np.uint16)
    
    # Create NIfTI image with proper header
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm voxel size
    
    nii_img = nib.Nifti1Image(volume, affine)
    
    # Save file
    nib.save(nii_img, filename)
    
    print(f"Saved synthetic brain MRI: {filename}")
    print(f"  Shape: {volume.shape}")
    print(f"  Data range: {volume.min()} to {volume.max()}")
    print(f"  File size: {os.path.getsize(filename) / 1024 / 1024:.1f} MB")
    
    return filename

def create_test_datasets():
    """Create multiple test datasets for different scenarios."""
    
    print("Creating test MRI datasets...")
    
    datasets = []
    
    # 1. Small dataset for quick testing
    small_file = create_synthetic_brain_mri((64, 64, 32), "test_brain_small.nii.gz")
    datasets.append(("Small Brain (Quick Test)", small_file))
    
    # 2. Medium dataset for normal testing  
    medium_file = create_synthetic_brain_mri((128, 128, 64), "test_brain_medium.nii.gz")
    datasets.append(("Medium Brain (Standard)", medium_file))
    
    # 3. Large dataset for GPU stress testing
    large_file = create_synthetic_brain_mri((256, 256, 128), "test_brain_large.nii.gz")
    datasets.append(("Large Brain (GPU Stress Test)", large_file))
    
    print(f"\nCreated {len(datasets)} test datasets:")
    for name, filename in datasets:
        print(f"  - {name}: {filename}")
    
    return datasets

if __name__ == "__main__":
    create_test_datasets()
    print("\nTest MRI data ready for brain GUI testing!")
