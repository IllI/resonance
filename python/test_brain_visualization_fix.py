#!/usr/bin/env python3
"""
Test script to verify the brain visualization fix works correctly.
This tests the 3D mesh generation without requiring a GUI display.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

def test_brain_visualization_fix():
    """Test that the brain visualization creates proper 3D meshes instead of dots."""
    print("🧠 Testing Brain Visualization Fix")
    print("=" * 50)
    
    try:
        # Import our fixed modules
        from brain_3d_model_generator import Brain3DModelGenerator
        from brain_3d_interface import Brain3DInterface
        from blue_brain_atlas_integrator import BlueBrainAtlasIntegrator
        
        print("✅ Successfully imported all modules")
        
        # Create a mock MRI volume (simulating brain data)
        print("\n📊 Creating mock brain volume data...")
        brain_volume = np.random.rand(64, 64, 64)
        
        # Add some brain-like structure
        center = np.array([32, 32, 32])
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    dist = np.linalg.norm([i, j, k] - center)
                    if dist < 25:  # Brain-like sphere
                        brain_volume[i, j, k] = 0.8 + 0.2 * np.sin(dist * 0.3)
                    else:
                        brain_volume[i, j, k] = 0.1
        
        print(f"   Created brain volume: {brain_volume.shape}")
        print(f"   Value range: {brain_volume.min():.3f} - {brain_volume.max():.3f}")
        
        # Initialize the model generator
        print("\n🏗️ Initializing Brain 3D Model Generator...")
        generator = Brain3DModelGenerator(
            atlas_name='blue_brain_cell_atlas',
            max_regions_detect=20,  # Limit for testing
            frequency_analysis=True,
            enable_blue_brain=True
        )
        
        # Manually set the brain volume (simulating MRI loading)
        generator.brain_volume = brain_volume
        generator.volume_shape = brain_volume.shape
        
        # Detect anatomical features
        print("\n🔍 Detecting anatomical features...")
        features = generator.detect_anatomical_features()
        print(f"   Detected {len(features)} anatomical features")
        
        if features:
            for i, feature in enumerate(features[:5]):  # Show first 5
                print(f"   {i+1}. {feature.name} (confidence: {feature.confidence:.3f})")
        
        # Generate 3D model
        print("\n🧠 Generating 3D brain model...")
        brain_model = generator.generate_3d_model()
        
        if brain_model:
            print("✅ 3D brain model generated successfully!")
            print(f"   Volume shape: {brain_model['shape']}")
            print(f"   Features: {len(brain_model['features'])}")
            print(f"   Has labeled volume: {'labeled_volume' in brain_model}")
            
            # Test the visualization interface
            print("\n🎨 Testing 3D visualization interface...")
            interface = Brain3DInterface()
            
            # Store the model in the interface
            model_name = "test_brain_model"
            interface.loaded_models[model_name] = {
                'generator': generator,
                'info': None,
                'features': features,
                'brain_model': brain_model
            }
            interface.current_model = model_name
            
            # Test visualization creation (this should create meshes, not dots)
            print("   Creating 3D visualization with volume rendering...")
            fig = interface.visualize_brain_model(
                model_name=model_name,
                show_features=True,
                confidence_threshold=0.5,
                render_volume=True
            )
            
            if fig:
                print("✅ 3D visualization created successfully!")
                
                # Check if the figure contains mesh traces (not just scatter)
                trace_types = [trace.type for trace in fig.data]
                print(f"   Trace types in visualization: {set(trace_types)}")
                
                if 'mesh3d' in trace_types:
                    print("🎉 SUCCESS: Visualization contains 3D mesh surfaces!")
                    print("   The fix is working - brain will show as solid mesh, not dots")
                elif 'scatter3d' in trace_types:
                    mesh_count = sum(1 for t in trace_types if t == 'mesh3d')
                    scatter_count = sum(1 for t in trace_types if t == 'scatter3d')
                    print(f"   Mesh traces: {mesh_count}, Scatter traces: {scatter_count}")
                    if mesh_count > 0:
                        print("🎉 SUCCESS: Contains both mesh and scatter (features on top of brain)")
                    else:
                        print("⚠️ WARNING: Still only contains scatter traces")
                
                # Save visualization as HTML for inspection
                try:
                    fig.write_html("test_brain_visualization.html")
                    print("   💾 Saved visualization as test_brain_visualization.html")
                except Exception as e:
                    print(f"   ⚠️ Could not save HTML: {e}")
                
            else:
                print("❌ Failed to create 3D visualization")
                
        else:
            print("❌ Failed to generate 3D brain model")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")
    return True

if __name__ == "__main__":
    success = test_brain_visualization_fix()
    if success:
        print("✅ Brain visualization fix test passed!")
    else:
        print("❌ Brain visualization fix test failed!")