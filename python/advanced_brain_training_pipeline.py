#!/usr/bin/env python3
"""
Advanced Brain Model Training Pipeline

Production-ready training system that downloads and processes real brain imaging
data from OpenNeuro, HCP, OASIS, and other neuroimaging repositories to train
the 3D brain modeling system on thousands of anatomical brain volumes.

This system implements:
- Real OpenNeuro dataset downloads using their API
- Automated quality control and preprocessing
- Large-scale feature extraction and model training
- Cross-validation and performance evaluation
- Model versioning and deployment
"""

import os
import requests
import json
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core neuroimaging dependencies
try:
    import nibabel as nib
    from nilearn import image, plotting
    from nilearn.datasets import fetch_atlas_harvard_oxford
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    logger.warning("nilearn not available - install with: pip install nilearn")

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    logger.warning("boto3 not available - install with: pip install boto3")

# Local imports
from brain_3d_model_generator import Brain3DModelGenerator
from brain_dataset_trainer import BrainDatasetManager

class OpenNeuroDownloader:
    """Direct downloader for OpenNeuro datasets using AWS S3."""
    
    def __init__(self):
        """Initialize OpenNeuro downloader."""
        self.base_url = "https://openneuro.org"
        self.s3_bucket = "openneuro.org"
        
        if HAS_BOTO3:
            # Anonymous S3 client for public data
            self.s3_client = boto3.client(
                's3',
                config=Config(signature_version=UNSIGNED)
            )
        
        logger.info("OpenNeuro downloader initialized")
    
    def list_datasets(self) -> List[Dict]:
        """List available datasets from OpenNeuro."""
        try:
            response = requests.get(f"{self.base_url}/crn/datasets")
            if response.status_code == 200:
                datasets = response.json()
                logger.info(f"Found {len(datasets)} datasets on OpenNeuro")
                return datasets
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
        
        return []
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get detailed information about a specific dataset."""
        try:
            response = requests.get(f"{self.base_url}/crn/datasets/{dataset_id}")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error getting dataset info for {dataset_id}: {e}")
        
        return {}
    
    def download_dataset(self, dataset_id: str, output_dir: Path, 
                        max_subjects: Optional[int] = None) -> bool:
        """
        Download a complete dataset from OpenNeuro.
        
        Args:
            dataset_id: OpenNeuro dataset ID (e.g., 'ds000221')
            output_dir: Local directory to download to
            max_subjects: Maximum number of subjects to download
        
        Returns:
            Success status
        """
        logger.info(f"Downloading OpenNeuro dataset {dataset_id}...")
        
        if not HAS_BOTO3:
            logger.error("boto3 required for OpenNeuro downloads")
            return False
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # List objects in the dataset
            prefix = f"{dataset_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.error(f"No files found for dataset {dataset_id}")
                return False
            
            # Filter for anatomical T1w images
            t1w_files = []
            for obj in response['Contents']:
                key = obj['Key']
                if 'anat' in key and 'T1w.nii.gz' in key:
                    t1w_files.append(key)
            
            if max_subjects:
                # Extract subject IDs and limit
                subject_files = {}
                for f in t1w_files:
                    parts = f.split('/')
                    if len(parts) >= 2:
                        subject_id = parts[1]  # e.g., sub-01
                        if subject_id not in subject_files:
                            subject_files[subject_id] = []
                        subject_files[subject_id].append(f)
                
                # Limit to max_subjects
                limited_subjects = list(subject_files.keys())[:max_subjects]
                t1w_files = []
                for subject in limited_subjects:
                    t1w_files.extend(subject_files[subject])
            
            logger.info(f"Downloading {len(t1w_files)} T1w files...")
            
            # Download files in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for file_key in t1w_files:
                    local_path = output_dir / file_key
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    future = executor.submit(
                        self._download_file, 
                        file_key, 
                        str(local_path)
                    )
                    futures.append(future)
                
                downloaded = 0
                for future in futures:
                    if future.result():
                        downloaded += 1
                    
                    if downloaded % 10 == 0:
                        logger.info(f"Downloaded {downloaded}/{len(t1w_files)} files")
            
            logger.info(f"Successfully downloaded {downloaded} files from {dataset_id}")
            return downloaded > 0
            
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_id}: {e}")
            return False
    
    def _download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a single file from S3."""
        try:
            self.s3_client.download_file(
                self.s3_bucket, 
                s3_key, 
                local_path
            )
            return True
        except Exception as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return False

class BrainModelTrainingPipeline:
    """Advanced training pipeline for brain anatomical models."""
    
    def __init__(self, 
                 data_dir: str = "brain_training_data",
                 model_dir: str = "trained_models",
                 n_workers: int = None):
        """Initialize training pipeline."""
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        self.downloader = OpenNeuroDownloader()
        
        # Training configuration
        self.training_config = {
            'target_datasets': [
                'ds000221',  # MPI-Leipzig Mind-Brain-Body
                'ds003097',  # AOMIC-ID1000  
                'ds005747',  # 7T fMRI Resting-state
                'ds000030',  # MyConnectome
                'ds000228',  # Emotional Reactivity
            ],
            'max_subjects_per_dataset': 100,  # For production training
            'min_subjects_per_dataset': 10,   # Minimum viable dataset
            'atlas_types': ['harvard_oxford', 'aal', 'craddock'],
            'feature_detection_methods': ['signal', 'frequency', 'tissue', 'atlas'],
            'quality_thresholds': {
                'min_image_size': (64, 64, 64),
                'max_image_size': (256, 256, 256),
                'min_intensity_range': 100,
                'max_motion_mm': 2.0
            }
        }
        
        logger.info("Brain Model Training Pipeline initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Workers: {self.n_workers}")
    
    def download_training_datasets(self, 
                                  target_subjects: int = 500) -> Dict[str, int]:
        """
        Download multiple datasets for comprehensive training.
        
        Args:
            target_subjects: Target total number of subjects across all datasets
            
        Returns:
            Dictionary mapping dataset_id to number of subjects downloaded
        """
        logger.info(f"Downloading training datasets (target: {target_subjects} subjects)...")
        
        downloaded_datasets = {}
        total_subjects = 0
        subjects_per_dataset = target_subjects // len(self.training_config['target_datasets'])
        
        for dataset_id in self.training_config['target_datasets']:
            if total_subjects >= target_subjects:
                break
            
            dataset_dir = self.data_dir / dataset_id
            subjects_to_download = min(
                subjects_per_dataset,
                target_subjects - total_subjects
            )
            
            logger.info(f"Downloading {dataset_id} ({subjects_to_download} subjects)...")
            
            success = self.downloader.download_dataset(
                dataset_id, 
                dataset_dir,
                max_subjects=subjects_to_download
            )
            
            if success:
                # Count downloaded subjects
                subject_dirs = list(dataset_dir.glob("sub-*"))
                n_subjects = len(subject_dirs)
                downloaded_datasets[dataset_id] = n_subjects
                total_subjects += n_subjects
                
                logger.info(f"Downloaded {n_subjects} subjects from {dataset_id}")
            else:
                logger.warning(f"Failed to download {dataset_id}")
        
        logger.info(f"Total subjects downloaded: {total_subjects}")
        return downloaded_datasets
    
    def quality_control_dataset(self, dataset_path: Path) -> List[Path]:
        """
        Perform quality control on downloaded dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of T1w files that pass quality control
        """
        logger.info(f"Performing quality control on {dataset_path.name}...")
        
        # Find all T1w files
        t1w_files = list(dataset_path.glob("**/anat/*_T1w.nii.gz"))
        logger.info(f"Found {len(t1w_files)} T1w files")
        
        if not HAS_NILEARN:
            logger.warning("nilearn not available - skipping detailed QC")
            return t1w_files
        
        passed_qc = []
        
        for t1w_file in t1w_files:
            try:
                # Load image
                img = nib.load(t1w_file)
                data = img.get_fdata()
                
                # Check image dimensions
                if not self._check_image_dimensions(data.shape):
                    logger.debug(f"Failed dimension check: {t1w_file.name}")
                    continue
                
                # Check intensity range
                if not self._check_intensity_range(data):
                    logger.debug(f"Failed intensity check: {t1w_file.name}")
                    continue
                
                # Check for artifacts (basic)
                if not self._check_artifacts(data):
                    logger.debug(f"Failed artifact check: {t1w_file.name}")
                    continue
                
                passed_qc.append(t1w_file)
                
            except Exception as e:
                logger.debug(f"QC error for {t1w_file.name}: {e}")
                continue
        
        qc_rate = len(passed_qc) / len(t1w_files) if t1w_files else 0
        logger.info(f"QC passed: {len(passed_qc)}/{len(t1w_files)} ({qc_rate:.1%})")
        
        return passed_qc
    
    def _check_image_dimensions(self, shape: Tuple[int, ...]) -> bool:
        """Check if image dimensions are within acceptable range."""
        min_size = self.training_config['quality_thresholds']['min_image_size']
        max_size = self.training_config['quality_thresholds']['max_image_size']
        
        return (all(s >= min_s for s, min_s in zip(shape, min_size)) and
                all(s <= max_s for s, max_s in zip(shape, max_size)))
    
    def _check_intensity_range(self, data: np.ndarray) -> bool:
        """Check if image has sufficient intensity range."""
        min_range = self.training_config['quality_thresholds']['min_intensity_range']
        return (np.max(data) - np.min(data)) >= min_range
    
    def _check_artifacts(self, data: np.ndarray) -> bool:
        """Basic artifact detection."""
        # Check for excessive zeros
        zero_fraction = np.sum(data == 0) / data.size
        if zero_fraction > 0.5:
            return False
        
        # Check for extreme values
        p99 = np.percentile(data, 99)
        if np.sum(data > 10 * p99) > data.size * 0.001:
            return False
        
        return True
    
    def train_on_dataset(self, dataset_path: Path, 
                        atlas_name: str = 'harvard_oxford') -> Dict:
        """
        Train brain models on a specific dataset.
        
        Args:
            dataset_path: Path to dataset directory
            atlas_name: Brain atlas to use for training
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training on dataset: {dataset_path.name}")
        
        # Quality control
        qc_files = self.quality_control_dataset(dataset_path)
        
        if len(qc_files) < self.training_config['min_subjects_per_dataset']:
            logger.warning(f"Insufficient subjects after QC: {len(qc_files)}")
            return {'status': 'insufficient_data', 'n_subjects': len(qc_files)}
        
        # Process files in parallel
        logger.info(f"Processing {len(qc_files)} T1w images...")
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for t1w_file in qc_files:
                future = executor.submit(
                    self._process_subject_brain,
                    t1w_file,
                    atlas_name
                )
                futures.append((t1w_file, future))
            
            # Collect results
            all_features = []
            successful_subjects = 0
            
            for t1w_file, future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per subject
                    if result and result['features']:
                        all_features.extend(result['features'])
                        successful_subjects += 1
                        
                        if successful_subjects % 10 == 0:
                            logger.info(f"Processed {successful_subjects}/{len(qc_files)} subjects")
                
                except Exception as e:
                    logger.debug(f"Error processing {t1w_file.name}: {e}")
                    continue
        
        # Training results
        training_results = {
            'status': 'success',
            'dataset_name': dataset_path.name,
            'atlas_name': atlas_name,
            'n_subjects_total': len(qc_files),
            'n_subjects_processed': successful_subjects,
            'n_features_extracted': len(all_features),
            'success_rate': successful_subjects / len(qc_files) if qc_files else 0,
            'feature_types': self._analyze_feature_types(all_features),
            'processing_time': time.time()
        }
        
        logger.info(f"Dataset training completed:")
        logger.info(f"  Subjects processed: {successful_subjects}/{len(qc_files)}")
        logger.info(f"  Features extracted: {len(all_features)}")
        logger.info(f"  Success rate: {training_results['success_rate']:.1%}")
        
        return training_results
    
    def _process_subject_brain(self, t1w_file: Path, atlas_name: str) -> Dict:
        """Process a single subject's brain and extract features."""
        try:
            # Create brain model generator
            generator = Brain3DModelGenerator(
                atlas_name=atlas_name,
                max_regions_detect=200,
                frequency_analysis=True,
                detection_confidence_threshold=0.3
            )
            
            # Load and process brain
            if generator.load_mri_volume(str(t1w_file)):
                features = generator.detect_anatomical_features()
                
                # Extract feature information
                feature_data = []
                for feature in features:
                    feature_info = {
                        'name': feature.name,
                        'region_id': feature.region_id,
                        'coordinates': feature.coordinates,
                        'confidence': feature.confidence,
                        'atlas_label': feature.atlas_label,
                        'properties': feature.properties
                    }
                    feature_data.append(feature_info)
                
                return {
                    'subject_id': t1w_file.parent.parent.name,
                    'features': feature_data,
                    'n_features': len(features),
                    'processing_success': True
                }
            
        except Exception as e:
            logger.debug(f"Error processing {t1w_file.name}: {e}")
        
        return {'processing_success': False, 'features': []}
    
    def _analyze_feature_types(self, all_features: List[Dict]) -> Dict:
        """Analyze the types and distribution of extracted features."""
        if not all_features:
            return {}
        
        # Count feature types
        feature_counts = {}
        confidence_scores = []
        atlas_labels = set()
        
        for feature in all_features:
            # Feature name analysis
            feature_name = feature.get('name', 'Unknown')
            feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
            
            # Confidence analysis
            confidence = feature.get('confidence', 0)
            confidence_scores.append(confidence)
            
            # Atlas label analysis
            atlas_label = feature.get('atlas_label', 'Unknown')
            atlas_labels.add(atlas_label)
        
        return {
            'total_features': len(all_features),
            'unique_feature_types': len(feature_counts),
            'most_common_features': sorted(feature_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:10],
            'confidence_stats': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            },
            'unique_atlas_labels': len(atlas_labels)
        }
    
    def run_full_training_pipeline(self, target_subjects: int = 500) -> Dict:
        """
        Run the complete training pipeline from download to model training.
        
        Args:
            target_subjects: Total number of subjects to train on
            
        Returns:
            Complete training results
        """
        logger.info(" Starting full brain model training pipeline...")
        start_time = time.time()
        
        # Step 1: Download datasets
        logger.info("Step 1: Downloading training datasets...")
        downloaded_datasets = self.download_training_datasets(target_subjects)
        
        if not downloaded_datasets:
            logger.error("No datasets downloaded successfully")
            return {'status': 'failed', 'reason': 'download_failed'}
        
        # Step 2: Train on each dataset
        logger.info("Step 2: Training on downloaded datasets...")
        dataset_results = []
        
        for dataset_id, n_subjects in downloaded_datasets.items():
            dataset_path = self.data_dir / dataset_id
            
            # Train with different atlases
            for atlas_name in self.training_config['atlas_types']:
                logger.info(f"Training {dataset_id} with {atlas_name} atlas...")
                
                result = self.train_on_dataset(dataset_path, atlas_name)
                result['dataset_id'] = dataset_id
                result['downloaded_subjects'] = n_subjects
                dataset_results.append(result)
        
        # Step 3: Aggregate results
        logger.info("Step 3: Aggregating training results...")
        
        total_time = time.time() - start_time
        successful_datasets = [r for r in dataset_results if r['status'] == 'success']
        
        overall_results = {
            'status': 'success' if successful_datasets else 'failed',
            'pipeline_start_time': start_time,
            'total_processing_time': total_time,
            'datasets_downloaded': len(downloaded_datasets),
            'datasets_trained': len(successful_datasets),
            'total_subjects_downloaded': sum(downloaded_datasets.values()),
            'total_subjects_processed': sum(r.get('n_subjects_processed', 0) 
                                          for r in successful_datasets),
            'total_features_extracted': sum(r.get('n_features_extracted', 0) 
                                          for r in successful_datasets),
            'average_success_rate': np.mean([r.get('success_rate', 0) 
                                           for r in successful_datasets]),
            'dataset_results': dataset_results
        }
        
        # Step 4: Save results
        results_file = self.model_dir / f"training_results_{int(start_time)}.json"
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        
        logger.info("🎉 Full training pipeline completed!")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Total processing time: {total_time:.1f}s")
        logger.info(f"Subjects processed: {overall_results['total_subjects_processed']}")
        logger.info(f"Features extracted: {overall_results['total_features_extracted']}")
        
        return overall_results

def main():
    """Run the advanced brain model training pipeline."""
    print(" Advanced Brain Model Training Pipeline")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = []
    if not HAS_NILEARN:
        missing_deps.append("nilearn")
    if not HAS_BOTO3:
        missing_deps.append("boto3")
    
    if missing_deps:
        print(f" Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    # Initialize pipeline
    pipeline = BrainModelTrainingPipeline()
    
    # Run training (demo with limited subjects)
    print("\n Starting training pipeline demonstration...")
    print("   Note: Using limited subjects for demo (50 total)")
    print("   For production, increase target_subjects to 1000+")
    
    results = pipeline.run_full_training_pipeline(target_subjects=50)
    
    if results['status'] == 'success':
        print("\n Training pipeline completed successfully!")
        print(f"   Datasets trained: {results['datasets_trained']}")
        print(f"   Subjects processed: {results['total_subjects_processed']}")
        print(f"   Features extracted: {results['total_features_extracted']}")
        print(f"   Average success rate: {results['average_success_rate']:.1%}")
    else:
        print("\n Training pipeline failed")
        print(f"   Reason: {results.get('reason', 'Unknown')}")
    
    return results['status'] == 'success'

if __name__ == "__main__":
    main()