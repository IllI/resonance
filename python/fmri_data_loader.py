#!/usr/bin/env python3
"""
fMRI Data Loader Module

This module provides functionality for loading and preprocessing fMRI data
from various sources including OpenNeuro datasets.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
from tqdm import tqdm

class fMRIDataLoader:
    """
    A comprehensive fMRI data loader supporting multiple data sources.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the fMRI data loader.
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        print(f" Data directory: {self.data_dir.absolute()}")
    
    def load_nifti(self, filepath: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """
        Load a NIfTI file and return data array and image object.
        
        Args:
            filepath: Path to the NIfTI file
            
        Returns:
            Tuple of (data_array, nifti_image)
        """
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            print(f" Loaded NIfTI: {filepath}")
            print(f"   Shape: {data.shape}")
            print(f"   Data type: {data.dtype}")
            print(f"   Voxel size: {img.header.get_zooms()}")
            return data, img
        
        except Exception as e:
            print(f" Error loading {filepath}: {e}")
            raise
    
    def download_sample_data(self) -> Dict[str, Path]:
        """
        Download sample fMRI data for testing.
        
        Returns:
            Dictionary mapping dataset names to local paths
        """
        print(" Downloading sample fMRI data...")
        
        try:
            from nilearn import datasets
            
            # Download Haxby dataset (visual object recognition)
            print(" Fetching Haxby visual recognition dataset...")
            haxby = datasets.fetch_haxby(
                subjects=[1], 
                fetch_stimuli=True, 
                data_dir=str(self.data_dir)
            )
            
            sample_data = {
                'haxby_func': Path(haxby.func[0]),
                'haxby_mask': Path(haxby.mask),
                'haxby_session_target': Path(haxby.session_target[0])
            }
            
            # Add conditions_target if available
            if hasattr(haxby, 'conditions_target') and haxby.conditions_target:
                sample_data['haxby_conditions_target'] = Path(haxby.conditions_target[0])
            
            print(" Sample data downloaded successfully:")
            for name, path in sample_data.items():
                print(f"   {name}: {path}")
                
            return sample_data
            
        except Exception as e:
            print(f" Error downloading sample data: {e}")
            raise
    
    def inspect_dataset(self, nifti_path: str) -> Dict:
        """
        Inspect a NIfTI dataset and return metadata.
        
        Args:
            nifti_path: Path to NIfTI file
            
        Returns:
            Dictionary containing dataset metadata
        """
        data, img = self.load_nifti(nifti_path)
        
        metadata = {
            'shape': data.shape,
            'ndim': data.ndim,
            'dtype': str(data.dtype),
            'voxel_sizes': img.header.get_zooms(),
            'affine': img.affine,
            'data_min': np.min(data),
            'data_max': np.max(data),
            'data_mean': np.mean(data),
            'data_std': np.std(data),
            'non_zero_voxels': np.count_nonzero(data),
            'file_size_mb': Path(nifti_path).stat().st_size / (1024 * 1024)
        }
        
        return metadata
    
    def load_behavioral_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load behavioral/experimental data from CSV.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Pandas DataFrame with behavioral data
        """
        try:
            df = pd.read_csv(csv_path)
            print(f" Loaded behavioral data: {csv_path}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            return df
        
        except Exception as e:
            print(f" Error loading behavioral data: {e}")
            raise
    
    def prepare_for_analysis(self, func_data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prepare fMRI data for analysis by applying basic preprocessing.
        
        Args:
            func_data: 4D functional MRI data (x, y, z, time)
            mask: Optional 3D brain mask
            
        Returns:
            Preprocessed data ready for analysis
        """
        print(" Preparing data for analysis...")
        
        # Apply mask if provided
        if mask is not None:
            print("   Applying brain mask...")
            masked_data = func_data[mask > 0]
        else:
            # Reshape to 2D (voxels x time)
            original_shape = func_data.shape
            masked_data = func_data.reshape(-1, original_shape[-1])
        
        # Remove voxels with zero variance
        var_mask = np.var(masked_data, axis=1) > 0
        cleaned_data = masked_data[var_mask]
        
        print(f"   Original shape: {func_data.shape}")
        print(f"   Masked shape: {masked_data.shape}")
        print(f"   Cleaned shape: {cleaned_data.shape}")
        print(f"   Removed {np.sum(~var_mask)} zero-variance voxels")
        
        return cleaned_data
    
    def get_sample_analysis_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Get sample data ready for analysis.
        
        Returns:
            Tuple of (fmri_data, behavioral_data)
        """
        print(" Preparing sample analysis dataset...")
        
        # Download sample data
        sample_data = self.download_sample_data()
        
        # Load functional data
        func_data, func_img = self.load_nifti(str(sample_data['haxby_func']))
        
        # Load mask
        mask_data, mask_img = self.load_nifti(str(sample_data['haxby_mask']))
        
        # Load behavioral data
        behavioral_data = self.load_behavioral_data(str(sample_data['haxby_session_target']))
        
        # Prepare data for analysis
        analysis_data = self.prepare_for_analysis(func_data, mask_data)
        
        print(" Sample analysis data prepared!")
        
        return analysis_data, behavioral_data

def main():
    """Test the fMRI data loader."""
    print(" Testing fMRI Data Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = fMRIDataLoader()
    
    try:
        # Get sample data for analysis
        fmri_data, behavioral_data = loader.get_sample_analysis_data()
        
        print(f"\n Analysis Data Summary:")
        print(f"   fMRI data shape: {fmri_data.shape}")
        print(f"   Behavioral data shape: {behavioral_data.shape}")
        print(f"   Data type: {fmri_data.dtype}")
        print(f"   Memory usage: {fmri_data.nbytes / (1024**2):.1f} MB")
        
        print("\n🎉 fMRI Data Loader test completed successfully!")
        
    except Exception as e:
        print(f" Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 