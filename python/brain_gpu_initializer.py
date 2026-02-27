#!/usr/bin/env python3
"""
Brain GPU Initializer
Simple initialization for brain analysis applications to use RX 7700S
"""

import pyopencl as cl
import os
import sys

class BrainGPUInitializer:
    """Simple GPU initializer for brain analysis applications"""
    
    def __init__(self):
        self.context = None
        self.queue = None
        self.device = None
        self.initialize_rx7700s()
    
    def initialize_rx7700s(self):
        """Initialize RX 7700S GPU for brain analysis"""
        print(" Initializing Brain Analysis GPU...")
        
        # Set environment variables
        os.environ['AmdPowerXpressRequestHighPerformance'] = '1'
        os.environ['AMD_OPENCL_DEVICE'] = '1'
        os.environ['OPENCL_DEVICE'] = '1'
        
        # Find RX 7700S
        platforms = cl.get_platforms()
        rx7700s_device = None
        
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                if 'gfx1103' in device.name.lower():  # RX 7700S
                    rx7700s_device = device
                    break
            if rx7700s_device:
                break
        
        if not rx7700s_device:
            raise RuntimeError("RX 7700S GPU not found!")
        
        # Create context with RX 7700S only
        self.context = cl.Context([rx7700s_device])
        self.queue = cl.CommandQueue(self.context)
        self.device = rx7700s_device
        
        print(f" Brain Analysis GPU Ready: {self.device.name}")
        print(f"   Memory: {self.device.global_mem_size // (1024**3)} GB")
        print(f"   Compute Units: {self.device.max_compute_units}")
    
    def get_context(self):
        """Get OpenCL context for brain analysis"""
        return self.context
    
    def get_queue(self):
        """Get OpenCL command queue for brain analysis"""
        return self.queue
    
    def get_device(self):
        """Get OpenCL device for brain analysis"""
        return self.device
    
    def get_device_info(self):
        """Get device information"""
        return {
            'name': self.device.name,
            'memory_gb': self.device.global_mem_size // (1024**3),
            'compute_units': self.device.max_compute_units,
            'max_frequency': self.device.max_clock_frequency
        }

# Global instance for easy access
brain_gpu = None

def initialize_brain_gpu():
    """Initialize brain GPU (call this once at startup)"""
    global brain_gpu
    if brain_gpu is None:
        brain_gpu = BrainGPUInitializer()
    return brain_gpu

def get_brain_context():
    """Get brain analysis OpenCL context"""
    global brain_gpu
    if brain_gpu is None:
        brain_gpu = initialize_brain_gpu()
    return brain_gpu.get_context()

def get_brain_queue():
    """Get brain analysis OpenCL command queue"""
    global brain_gpu
    if brain_gpu is None:
        brain_gpu = initialize_brain_gpu()
    return brain_gpu.get_queue()

def get_brain_device():
    """Get brain analysis OpenCL device"""
    global brain_gpu
    if brain_gpu is None:
        brain_gpu = initialize_brain_gpu()
    return brain_gpu.get_device()

def main():
    """Test the brain GPU initializer"""
    print("=== Brain GPU Initializer Test ===")
    
    # Initialize brain GPU
    gpu = initialize_brain_gpu()
    
    # Get device info
    info = gpu.get_device_info()
    print(f"\n Brain Analysis GPU Info:")
    print(f"   Name: {info['name']}")
    print(f"   Memory: {info['memory_gb']} GB")
    print(f"   Compute Units: {info['compute_units']}")
    print(f"   Max Frequency: {info['max_frequency']} MHz")
    
    # Test context creation
    context = get_brain_context()
    queue = get_brain_queue()
    device = get_brain_device()
    
    print(f"\n Brain GPU Initialization Complete!")
    print(f"   Ready for brain analysis computations")

if __name__ == "__main__":
    main()

