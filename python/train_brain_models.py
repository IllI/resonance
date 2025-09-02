#!/usr/bin/env python3
"""
Brain 3D Model Training Orchestrator

This is the main training script that orchestrates the complete training process
for the 3D brain modeling system using OpenNeuro datasets.

Usage:
    python train_brain_models.py [--config config.json] [--quick-demo]

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from brain_training_pipeline import BrainTrainingPipeline, TrainingConfig
    from openneuro_data_manager import OpenNeuroDataManager
    from brain_3d_model_generator import Brain3DModelGenerator
    from brain_atlas_manager import BrainAtlasManager
    from brain_3d_visualizer import Brain3DVisualizer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure all required Python files are in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BrainModelTrainer:
    """Main orchestrator for brain model training"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the brain model trainer.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.pipeline = None
        self.results = None
        
        logger.info("Brain Model Trainer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> TrainingConfig:
        """
        Load training configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Training configuration
        """
        if config_path and Path(config_path).exists():
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Convert dict to TrainingConfig
            return TrainingConfig(**config_dict)
        else:
            logger.info("Using default configuration")
            return TrainingConfig(
                max_datasets=3,
                max_subjects_per_dataset=20,
                max_volumes_per_subject=1,
                target_shape=(128, 128, 128),
                normalize_data=True,
                augment_data=True,
                batch_size=4,
                max_epochs=10,
                learning_rate=0.001,
                output_dir="./training_results",
                save_models=True,
                generate_reports=True,
                create_visualizations=True,
                max_workers=4
            )
    
    def create_quick_demo_config(self) -> TrainingConfig:
        """
        Create a quick demo configuration for testing.
        
        Returns:
            Demo training configuration
        """
        return TrainingConfig(
            max_datasets=2,
            max_subjects_per_dataset=5,
            max_volumes_per_subject=1,
            target_shape=(64, 64, 64),  # Smaller for faster processing
            normalize_data=True,
            augment_data=False,  # Skip augmentation for demo
            batch_size=2,
            max_epochs=3,
            learning_rate=0.001,
            output_dir="./demo_training_results",
            save_models=True,
            generate_reports=True,
            create_visualizations=True,
            max_workers=2
        )
    
    def validate_environment(self) -> bool:
        """
        Validate that the environment is ready for training.
        
        Returns:
            True if environment is valid
        """
        logger.info("Validating training environment...")
        
        # Check required modules
        required_modules = [
            'numpy', 'scipy', 'matplotlib', 'seaborn'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            logger.warning(f"Missing optional modules: {missing_modules}")
            logger.warning("Some functionality may be limited")
        
        # Check disk space
        output_dir = Path(self.config.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ready: {output_dir}")
        except Exception as e:
            logger.error(f"Cannot create output directory: {e}")
            return False
        
        # Check memory requirements (rough estimate)
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        required_memory_gb = self.config.memory_limit_gb
        
        if available_memory_gb < required_memory_gb:
            logger.warning(f"Available memory ({available_memory_gb:.1f}GB) is less than recommended ({required_memory_gb}GB)")
            logger.warning("Consider reducing batch size or target shape")
        
        logger.info("Environment validation completed")
        return True
    
    def run_training(self, quick_demo: bool = False) -> Dict[str, Any]:
        """
        Run the complete training process.
        
        Args:
            quick_demo: Whether to run a quick demo
            
        Returns:
            Training results summary
        """
        logger.info("Starting brain model training process...")
        
        try:
            # Use demo config if requested
            if quick_demo:
                self.config = self.create_quick_demo_config()
                logger.info("Using quick demo configuration")
            
            # Validate environment
            if not self.validate_environment():
                raise RuntimeError("Environment validation failed")
            
            # Initialize training pipeline
            self.pipeline = BrainTrainingPipeline(self.config)
            
            # Run training
            logger.info("Executing training pipeline...")
            self.results = self.pipeline.run_training_pipeline()
            
            # Create summary
            summary = {
                "status": "success",
                "datasets_used": self.results.datasets_used,
                "total_volumes": self.results.total_volumes,
                "training_duration_minutes": self.results.training_duration / 60,
                "average_accuracy": sum(m.accuracy for m in self.results.metrics) / len(self.results.metrics),
                "model_files": self.results.model_paths,
                "report_path": self.results.report_path,
                "visualization_count": len(self.results.visualization_paths),
                "output_directory": str(Path(self.config.output_dir).absolute())
            }
            
            logger.info("Training completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "output_directory": str(Path(self.config.output_dir).absolute())
            }
    
    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        config_dict = {
            "max_datasets": self.config.max_datasets,
            "max_subjects_per_dataset": self.config.max_subjects_per_dataset,
            "max_volumes_per_subject": self.config.max_volumes_per_subject,
            "train_test_split": self.config.train_test_split,
            "target_shape": self.config.target_shape,
            "normalize_data": self.config.normalize_data,
            "augment_data": self.config.augment_data,
            "batch_size": self.config.batch_size,
            "max_epochs": self.config.max_epochs,
            "learning_rate": self.config.learning_rate,
            "validation_split": self.config.validation_split,
            "output_dir": self.config.output_dir,
            "save_models": self.config.save_models,
            "generate_reports": self.config.generate_reports,
            "create_visualizations": self.config.create_visualizations,
            "max_workers": self.config.max_workers,
            "use_gpu": self.config.use_gpu,
            "memory_limit_gb": self.config.memory_limit_gb
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """
        Print a formatted summary of training results.
        
        Args:
            summary: Training results summary
        """
        print("\n" + "=" * 60)
        print("BRAIN 3D MODEL TRAINING SUMMARY")
        print("=" * 60)
        
        if summary["status"] == "success":
            print(f"✅ Status: {summary['status'].upper()}")
            print(f"📊 Datasets Used: {len(summary['datasets_used'])}")
            for dataset in summary['datasets_used']:
                print(f"   - {dataset}")
            print(f"🧠 Total Brain Volumes: {summary['total_volumes']}")
            print(f"⏱️  Training Duration: {summary['training_duration_minutes']:.1f} minutes")
            print(f"🎯 Average Accuracy: {summary['average_accuracy']:.3f}")
            print(f"📈 Visualizations Created: {summary['visualization_count']}")
            
            print("\n📁 Generated Files:")
            print(f"   📋 Training Report: {summary['report_path']}")
            print("   🤖 Model Files:")
            for model_name, model_path in summary['model_files'].items():
                print(f"      - {model_name}: {Path(model_path).name}")
            
            print(f"\n📂 All outputs saved to: {summary['output_directory']}")
            
        else:
            print(f"❌ Status: {summary['status'].upper()}")
            print(f"💥 Error: {summary['error']}")
            print(f"📂 Check logs in: {summary['output_directory']}")
        
        print("=" * 60)

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "max_datasets": 3,
        "max_subjects_per_dataset": 15,
        "max_volumes_per_subject": 1,
        "train_test_split": 0.8,
        "target_shape": [128, 128, 128],
        "normalize_data": True,
        "augment_data": True,
        "batch_size": 4,
        "max_epochs": 10,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "output_dir": "./training_results",
        "save_models": True,
        "generate_reports": True,
        "create_visualizations": True,
        "max_workers": 4,
        "use_gpu": False,
        "memory_limit_gb": 8.0
    }
    
    with open("training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration saved to 'training_config.json'")
    print("Edit this file to customize training parameters.")

def main():
    """Main entry point for the training script"""
    parser = argparse.ArgumentParser(
        description="Train 3D brain models using OpenNeuro datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_brain_models.py --quick-demo
  python train_brain_models.py --config my_config.json
  python train_brain_models.py --create-config
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--quick-demo", 
        action="store_true", 
        help="Run a quick demo with reduced dataset size"
    )
    
    parser.add_argument(
        "--create-config", 
        action="store_true", 
        help="Create a sample configuration file and exit"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return 0
    
    # Print header
    print("🧠 Brain 3D Model Training System")
    print("🔬 Training on OpenNeuro Brain Volume Datasets")
    print("🤖 AI-Powered Anatomical Feature Detection")
    print()
    
    try:
        # Initialize trainer
        trainer = BrainModelTrainer(config_path=args.config)
        
        # Save current config for reference
        config_save_path = Path(trainer.config.output_dir) / "used_config.json"
        config_save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_config(str(config_save_path))
        
        # Run training
        summary = trainer.run_training(quick_demo=args.quick_demo)
        
        # Print results
        trainer.print_summary(summary)
        
        # Return appropriate exit code
        return 0 if summary["status"] == "success" else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())