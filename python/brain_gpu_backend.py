#!/usr/bin/env python3
"""
Python wrapper for Brain GPU Backend C++ library
Provides high-performance AMD GPU processing with OpenGL/OpenCL interop
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path

class BrainGPUBackend:
    def __init__(self):
        self.dll = None
        self.initialized = False
        self._load_library()
    
    def _load_library(self):
        """Load the C++ backend library"""
        # Try different possible locations
        possible_paths = [
            Path(__file__).parent / "brain_gpu_backend.dll",
            Path(__file__).parent.parent / "cpp" / "brain_gpu_backend" / "build" / "bin" / "Release" / "brain_gpu_backend.dll",
            Path(__file__).parent.parent / "cpp" / "brain_gpu_backend" / "build" / "bin" / "brain_gpu_backend.dll",
            "brain_gpu_backend.dll"  # Current directory
        ]
        
        for path in possible_paths:
            try:
                if isinstance(path, Path):
                    path_str = str(path.resolve())
                else:
                    path_str = path
                
                if os.path.exists(path_str):
                    print(f"Attempting to load: {path_str}")
                    self.dll = ctypes.CDLL(path_str)
                    self._setup_function_signatures()
                    print(f"✓ Successfully loaded Brain GPU Backend from: {path_str}")
                    return
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
        
        print("  Brain GPU Backend C++ library not found")
        print("   Falling back to Python-only mode")
        print("   To build the C++ backend:")
        print("   1. Open Visual Studio Developer Command Prompt")
        print("   2. cd cpp/brain_gpu_backend")
        print("   3. run build_msvc.bat")
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures"""
        if not self.dll:
            return
        
        # init_device(bool enable_gl_interop) -> bool
        self.dll.init_device.argtypes = [ctypes.c_bool]
        self.dll.init_device.restype = ctypes.c_bool
        
        # get_device_info() -> const char*
        self.dll.get_device_info.argtypes = []
        self.dll.get_device_info.restype = ctypes.c_char_p
        
        # process_mesh(float* vertices, int vertex_count, float* normals, int* indices, int index_count) -> bool
        self.dll.process_mesh.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # vertices
            ctypes.c_int,                    # vertex_count
            ctypes.POINTER(ctypes.c_float),  # normals
            ctypes.POINTER(ctypes.c_int),    # indices
            ctypes.c_int                     # index_count
        ]
        self.dll.process_mesh.restype = ctypes.c_bool
        
        # render_frame(float* vertices, int vertex_count, int* indices, int index_count) -> bool
        self.dll.render_frame.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # vertices
            ctypes.c_int,                    # vertex_count
            ctypes.POINTER(ctypes.c_int),    # indices
            ctypes.c_int                     # index_count
        ]
        self.dll.render_frame.restype = ctypes.c_bool
        
        # cleanup_device() -> void
        self.dll.cleanup_device.argtypes = []
        self.dll.cleanup_device.restype = None
    
    def initialize(self, enable_gl_interop=True):
        """Initialize the GPU backend"""
        if not self.dll:
            print("C++ backend not available, using Python fallback")
            return False
        
        try:
            success = self.dll.init_device(enable_gl_interop)
            if success:
                self.initialized = True
                print("✓ Brain GPU Backend initialized successfully")
                print(self.get_device_info())
            else:
                print("✗ Failed to initialize Brain GPU Backend")
            return success
        except Exception as e:
            print(f"Error initializing backend: {e}")
            return False
    
    def get_device_info(self):
        """Get information about the selected GPU device"""
        if not self.dll or not self.initialized:
            return "Backend not available or not initialized"
        
        try:
            info_ptr = self.dll.get_device_info()
            return info_ptr.decode('utf-8') if info_ptr else "No device info available"
        except Exception as e:
            return f"Error getting device info: {e}"
    
    def process_brain_mesh(self, vertices, normals, indices):
        """
        Process brain mesh data using AMD GPU
        
        Args:
            vertices: numpy array of shape (N, 3) with vertex positions
            normals: numpy array of shape (N, 3) with vertex normals  
            indices: numpy array of shape (M,) with triangle indices
            
        Returns:
            tuple: (processed_vertices, processed_normals, success)
        """
        if not self.dll or not self.initialized:
            print("Using Python fallback for mesh processing")
            return self._python_fallback_process_mesh(vertices, normals, indices)
        
        try:
            # Ensure arrays are contiguous and correct type
            vertices = np.ascontiguousarray(vertices, dtype=np.float32)
            normals = np.ascontiguousarray(normals, dtype=np.float32)
            indices = np.ascontiguousarray(indices, dtype=np.int32)
            
            # Flatten if needed
            if vertices.ndim == 2:
                vertices = vertices.flatten()
            if normals.ndim == 2:
                normals = normals.flatten()
            
            vertex_count = len(vertices) // 3
            index_count = len(indices)
            
            # Create ctypes pointers
            vertices_ptr = vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            normals_ptr = normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            
            # Call C++ function
            success = self.dll.process_mesh(vertices_ptr, vertex_count, normals_ptr, indices_ptr, index_count)
            
            if success:
                # Reshape back to (N, 3)
                processed_vertices = vertices.reshape(-1, 3)
                processed_normals = normals.reshape(-1, 3)
                return processed_vertices, processed_normals, True
            else:
                print("C++ mesh processing failed, using Python fallback")
                return self._python_fallback_process_mesh(vertices.reshape(-1, 3), normals.reshape(-1, 3), indices)
                
        except Exception as e:
            print(f"Error in C++ mesh processing: {e}")
            return self._python_fallback_process_mesh(vertices, normals, indices)
    
    def render_frame(self, vertices, indices):
        """
        Render frame using OpenGL (if GL interop is enabled)
        
        Args:
            vertices: numpy array of shape (N, 3) with vertex positions
            indices: numpy array of shape (M,) with triangle indices
            
        Returns:
            bool: Success status
        """
        if not self.dll or not self.initialized:
            return False
        
        try:
            # Ensure arrays are contiguous and correct type
            vertices = np.ascontiguousarray(vertices, dtype=np.float32)
            indices = np.ascontiguousarray(indices, dtype=np.int32)
            
            # Flatten if needed
            if vertices.ndim == 2:
                vertices = vertices.flatten()
            
            vertex_count = len(vertices) // 3
            index_count = len(indices)
            
            # Create ctypes pointers
            vertices_ptr = vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            
            # Call C++ function
            return self.dll.render_frame(vertices_ptr, vertex_count, indices_ptr, index_count)
            
        except Exception as e:
            print(f"Error in C++ rendering: {e}")
            return False
    
    def _python_fallback_process_mesh(self, vertices, normals, indices):
        """Python fallback for mesh processing when C++ backend is not available"""
        try:
            # Simple mesh optimization - reduce vertex count if too high
            if len(vertices) > 8000:  # 8K vertex limit for smooth performance
                # Simple decimation - take every nth vertex
                step = len(vertices) // 8000
                vertices = vertices[::step]
                normals = normals[::step]
                
                # Update indices (simple approach)
                max_vertex = len(vertices) - 1
                indices = indices[indices <= max_vertex]
            
            # Normalize normals
            norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            norm_lengths[norm_lengths == 0] = 1  # Avoid division by zero
            normals = normals / norm_lengths
            
            print(f"Python fallback processed mesh: {len(vertices)} vertices, {len(indices)} indices")
            return vertices, normals, True
            
        except Exception as e:
            print(f"Error in Python fallback: {e}")
            return vertices, normals, False
    
    def cleanup(self):
        """Cleanup resources"""
        if self.dll and self.initialized:
            try:
                self.dll.cleanup_device()
                self.initialized = False
                print("Brain GPU Backend cleaned up")
            except Exception as e:
                print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

# Global instance
_backend_instance = None

def get_backend():
    """Get global backend instance"""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = BrainGPUBackend()
    return _backend_instance

def initialize_backend(enable_gl_interop=True):
    """Initialize the global backend instance"""
    backend = get_backend()
    return backend.initialize(enable_gl_interop)

def process_brain_mesh(vertices, normals, indices):
    """Process brain mesh using the global backend"""
    backend = get_backend()
    return backend.process_brain_mesh(vertices, normals, indices)

def get_device_info():
    """Get device info from the global backend"""
    backend = get_backend()
    return backend.get_device_info()

# Test function
def test_backend():
    """Test the backend functionality"""
    print("=== Brain GPU Backend Test ===")
    
    backend = get_backend()
    
    if backend.initialize():
        print("\n" + backend.get_device_info())
        
        # Test mesh processing
        print("\nTesting mesh processing...")
        vertices = np.random.rand(1000, 3).astype(np.float32)
        normals = np.random.rand(1000, 3).astype(np.float32)
        indices = np.arange(3000, dtype=np.int32)
        
        processed_vertices, processed_normals, success = backend.process_brain_mesh(vertices, normals, indices)
        
        if success:
            print(f"✓ Mesh processing successful")
            print(f"  Input vertices: {vertices.shape}")
            print(f"  Output vertices: {processed_vertices.shape}")
        else:
            print("✗ Mesh processing failed")
    else:
        print("✗ Backend initialization failed")

if __name__ == "__main__":
    test_backend()