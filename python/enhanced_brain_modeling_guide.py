#!/usr/bin/env python3
"""
Enhanced 3D Brain Modeling System - Usage Guide

This guide demonstrates how to use the enhanced 3D brain modeling system that can
derive anatomically accurate 3D brain models from any MRI volume data using
frequency-based signal processing and advanced feature detection AI.

Features Implemented:
✅ Frequency-based 3D brain volume processing preserving signal characteristics
✅ Multi-modal anatomical feature detection (signal, frequency, tissue, atlas)
✅ Integration with brain atlases from OpenNeuro for maximum region detection
✅ Signal-based tissue differentiation (gray matter, white matter, CSF)
✅ 3D interface system for loading and visualizing labeled brain models
✅ Integration with existing neural network training pipeline
✅ Comprehensive testing and validation system
"""

from pathlib import Path
import numpy as np

# Import the enhanced brain modeling components
try:
    from brain_3d_model_generator import Brain3DModelGenerator
    from brain_3d_interface import Brain3DInterface
    from frequency_brain_processor import FrequencyBrainProcessor
    print("✅ All enhanced brain modeling components imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all components are available in the python/ directory")

def example_1_basic_usage():
    """Example 1: Basic 3D brain model generation from MRI volume."""
    print("\n" + "="*60)
    print("📚 EXAMPLE 1: Basic 3D Brain Model Generation")
    print("="*60)
    
    # Create the enhanced brain model generator
    generator = Brain3DModelGenerator(
        atlas_name='harvard_oxford',           # Brain atlas for anatomical labeling
        detection_confidence_threshold=0.6,   # Confidence threshold for features
        frequency_analysis=True,               # Enable frequency-based analysis
        max_regions_detect=100                 # Maximum regions to detect
    )
    
    # For this example, we'll create realistic synthetic data
    # In practice, you would load real MRI data using:
    # generator.load_mri_volume("path/to/your/mri_volume.nii.gz")
    
    print("🧠 Creating realistic synthetic brain volume...")
    volume_shape = (91, 109, 91)  # Standard MNI dimensions
    
    # Create realistic brain structure
    mock_volume = np.random.randn(*volume_shape) * 0.1 + 0.3
    
    # Add brain structures with different signal characteristics
    # Gray matter (cortical regions)
    mock_volume[20:70, 20:90, 20:70] += 0.3
    # White matter (central core)
    mock_volume[35:55, 35:75, 35:55] += 0.5
    # CSF (ventricles - low signal)
    mock_volume[43:48, 50:60, 43:48] = 0.1
    
    generator.brain_volume = mock_volume
    generator._preprocess_volume()  # Apply frequency analysis and tissue maps
    
    print("🔍 Detecting anatomical features...")
    features = generator.detect_anatomical_features()
    
    print("🧠 Generating 3D brain model...")
    brain_model = generator.generate_3d_model()
    
    print("💾 Saving brain model...")
    output_path = Path("models/example_brain_model.npz")
    generator.save_model(output_path)
    
    print(f"✅ Basic example completed!")
    print(f"   Features detected: {len(features)}")
    print(f"   Model saved to: {output_path}")
    
    return generator, brain_model

def example_2_frequency_analysis():
    """Example 2: Advanced frequency-based brain analysis."""
    print("\n" + "="*60)
    print("📚 EXAMPLE 2: Frequency-Based Brain Analysis")
    print("="*60)
    
    # Create frequency processor
    processor = FrequencyBrainProcessor()
    
    # Create mock brain volume with frequency characteristics
    volume_shape = (64, 64, 64)  # Smaller for faster processing
    volume = np.random.randn(*volume_shape) * 0.1 + 0.5
    
    print("🌊 Applying frequency-based processing...")
    
    # Step 1: Advanced noise reduction
    denoised_volume = processor.advanced_noise_reduction(volume)
    
    # Step 2: Signal-aware normalization
    normalized_volume = processor.signal_aware_normalization(denoised_volume)
    
    # Step 3: Create frequency domain representation
    frequency_volume = processor.create_frequency_volume(normalized_volume)
    
    # Step 4: Generate tissue probability maps
    tissue_maps = processor.generate_tissue_probability_maps(normalized_volume)
    
    print(f"✅ Frequency analysis completed!")
    print(f"   Frequency bands: {list(frequency_volume.keys())}")
    print(f"   Tissue types: {list(tissue_maps.keys())}")
    
    # Step 5: Detect frequency-based features
    freq_features = processor.detect_frequency_based_features(frequency_volume)
    
    print(f"   Frequency features: {len(freq_features)}")
    
    return frequency_volume, tissue_maps, freq_features

def example_3_3d_interface():
    """Example 3: Using the 3D interface system."""
    print("\n" + "="*60)
    print("📚 EXAMPLE 3: 3D Interface System")
    print("="*60)
    
    # Create the 3D interface
    interface = Brain3DInterface()
    
    # Create a brain model from synthetic MRI data
    print("🧠 Creating brain model through interface...")
    
    # For demonstration, we'll use the generator directly
    # In practice, you could load from files or create from MRI
    generator = Brain3DModelGenerator(max_regions_detect=30)
    
    # Create and process synthetic data
    volume = np.random.randn(64, 64, 64) * 0.1 + 0.5
    # Add brain structures
    volume[15:50, 15:50, 15:50] += 0.3  # Gray matter
    volume[25:40, 25:40, 25:40] += 0.5  # White matter
    
    generator.brain_volume = volume
    generator._preprocess_volume()
    features = generator.detect_anatomical_features()
    brain_model = generator.generate_3d_model()
    
    # Save and load through interface
    model_path = "models/interface_test_model.npz"
    generator.save_model(model_path)
    
    # Load the model through the interface
    success = interface.load_brain_model(model_path, "Demo_Model")
    
    if success:
        print("✅ Model loaded successfully through interface")
        
        # Create visualization
        print("🎨 Creating 3D visualization...")
        fig = interface.visualize_brain_model(
            show_features=True,
            confidence_threshold=0.5
        )
        
        if fig:
            # Save visualization to HTML
            viz_path = Path("visualizations/brain_model_3d.html")
            viz_path.parent.mkdir(exist_ok=True)
            fig.write_html(str(viz_path))
            print(f"   3D visualization saved to: {viz_path}")
        
        # Create analysis dashboard
        print("📊 Creating analysis dashboard...")
        dashboard = interface.create_model_dashboard()
        
        if dashboard:
            dashboard_path = Path("visualizations/brain_dashboard.html")
            dashboard.write_html(str(dashboard_path))
            print(f"   Dashboard saved to: {dashboard_path}")
        
        # Export analysis report
        print("📄 Exporting analysis report...")
        report_success = interface.export_model_report(
            output_path="reports/brain_analysis_report.json"
        )
        
        if report_success:
            print("   Analysis report exported successfully")
    
    return interface

def example_4_integration_with_neural_network():
    """Example 4: Integration with existing neural network training pipeline."""
    print("\n" + "="*60)
    print("📚 EXAMPLE 4: Neural Network Integration")
    print("="*60)
    
    print("🤖 Demonstrating integration with neural network training...")
    
    # Create brain model generator
    generator = Brain3DModelGenerator(
        max_regions_detect=50,
        frequency_analysis=True
    )
    
    # Generate multiple brain models for training
    brain_models = []
    feature_vectors = []
    
    for i in range(5):  # Create 5 example models
        print(f"   Generating brain model {i+1}/5...")
        
        # Create varied synthetic brain data
        volume = np.random.randn(64, 64, 64) * 0.1 + 0.4 + i * 0.1
        
        # Add varied brain structures
        offset = i * 2
        volume[15+offset:45+offset, 15+offset:45+offset, 15+offset:45+offset] += 0.3
        volume[25+offset:35+offset, 25+offset:35+offset, 25+offset:35+offset] += 0.4
        
        generator.brain_volume = volume
        generator._preprocess_volume()
        features = generator.detect_anatomical_features()
        brain_model = generator.generate_3d_model()
        
        brain_models.append(brain_model)
        
        # Extract feature vector for neural network training
        feature_vector = []
        for feature in features:
            feature_vector.extend([
                feature.confidence,
                feature.coordinates[0], feature.coordinates[1], feature.coordinates[2],
                1 if feature.properties.get('tissue_type') == 'white_matter' else 0,
                1 if feature.properties.get('tissue_type') == 'gray_matter' else 0
            ])
        
        # Pad or truncate to fixed size for neural network
        target_size = 300  # Fixed feature vector size
        if len(feature_vector) > target_size:
            feature_vector = feature_vector[:target_size]
        else:
            feature_vector.extend([0] * (target_size - len(feature_vector)))
        
        feature_vectors.append(feature_vector)
    
    # Convert to numpy arrays for neural network training
    feature_matrix = np.array(feature_vectors)
    
    print(f"✅ Generated training data:")
    print(f"   Brain models: {len(brain_models)}")
    print(f"   Feature matrix shape: {feature_matrix.shape}")
    print(f"   Ready for neural network training!")
    
    # Save training data
    training_data_path = Path("training_data/brain_features.npz")
    training_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(training_data_path, 
             features=feature_matrix,
             labels=np.arange(len(brain_models)),  # Mock labels
             model_info=[model['metadata'] for model in brain_models])
    
    print(f"   Training data saved to: {training_data_path}")
    
    return feature_matrix, brain_models

def main():
    """Run all examples to demonstrate the enhanced 3D brain modeling system."""
    print("🧠 Enhanced 3D Brain Modeling System - Comprehensive Usage Guide")
    print("=" * 80)
    
    print("""
This enhanced system provides:

✅ Frequency-based 3D brain volume processing preserving signal characteristics
✅ Advanced feature detection AI using signal analysis and brain atlas integration
✅ Signal-based tissue differentiation (gray matter, white matter, CSF)
✅ 3D interface system for loading labeled brain models with anatomical features
✅ Integration with existing neural network training pipeline

The system can take any MRI volume and automatically detect as many brain regions 
as possible using multiple detection methods:
- Signal-based ROI detection
- Frequency-domain analysis
- Tissue probability mapping
- Atlas-guided detection
- Spatial analysis fallback
""")
    
    try:
        # Run all examples
        print("\n🚀 Running comprehensive examples...")
        
        # Example 1: Basic usage
        generator, brain_model = example_1_basic_usage()
        
        # Example 2: Frequency analysis
        freq_volume, tissue_maps, freq_features = example_2_frequency_analysis()
        
        # Example 3: 3D interface
        interface = example_3_3d_interface()
        
        # Example 4: Neural network integration
        feature_matrix, brain_models = example_4_integration_with_neural_network()
        
        # Summary
        print("\n" + "="*80)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("""
📁 Generated Files:
   • models/example_brain_model.npz - Basic brain model
   • models/interface_test_model.npz - Interface demo model
   • visualizations/brain_model_3d.html - 3D brain visualization
   • visualizations/brain_dashboard.html - Analysis dashboard
   • reports/brain_analysis_report.json - Detailed analysis report
   • training_data/brain_features.npz - Neural network training data

🛠️ Usage in Your Projects:
   1. Load MRI volume: generator.load_mri_volume("your_mri.nii.gz")
   2. Detect features: features = generator.detect_anatomical_features()
   3. Generate model: model = generator.generate_3d_model()
   4. Visualize: interface.visualize_brain_model()
   5. Export results: interface.export_model_report()

🔬 Key Advantages:
   • Works with any MRI volume input
   • Preserves signal characteristics (no 2D conversion needed)
   • Detects maximum number of brain regions using multiple methods
   • Provides physically accurate 3D models with anatomical labeling
   • Integrates with existing neural network training pipelines
   • Supports interactive 3D visualization and analysis
""")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()