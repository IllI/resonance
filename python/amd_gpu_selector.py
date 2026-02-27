#!/usr/bin/env python3
"""
AMD GPU Selector - Force specific GPU usage for OpenCL operations
"""

import pyopencl as cl
import os
import sys
from typing import Optional, Dict, List, Tuple

class AMDGPUSelector:
    """Utility class to select and force specific AMD GPU usage"""
    
    def __init__(self):
        self.platforms = cl.get_platforms()
        self.devices = self._enumerate_devices()
        self.target_gpu = None
        
    def _enumerate_devices(self) -> List[Dict]:
        """Enumerate all available OpenCL devices"""
        devices = []
        for i, platform in enumerate(self.platforms):
            platform_devices = platform.get_devices()
            for j, device in enumerate(platform_devices):
                device_info = {
                    'platform_id': i,
                    'device_id': j,
                    'name': device.name,
                    'type': cl.device_type.to_string(device.type),
                    'memory': device.global_mem_size // (1024**3),  # GB
                    'compute_units': device.max_compute_units,
                    'max_freq': device.max_clock_frequency,
                    'device': device,
                    'platform': platform,
                    'is_discrete': self._is_discrete_gpu(device.name),
                    'is_target': self._is_target_gpu(device.name)
                }
                devices.append(device_info)
        return devices
    
    def _is_discrete_gpu(self, device_name: str) -> bool:
        """Check if device is a discrete GPU (not integrated)"""
        name_lower = device_name.lower()
        # Common patterns for discrete GPUs
        discrete_patterns = ['rx', 'radeon rx', 'radeon pro', 'firepro', 'radeon vii']
        integrated_patterns = ['780m', '680m', '660m', 'vega', 'integrated']
        
        for pattern in integrated_patterns:
            if pattern in name_lower:
                return False
        
        for pattern in discrete_patterns:
            if pattern in name_lower:
                return True
                
        return False
    
    def _is_target_gpu(self, device_name: str) -> bool:
        """Check if device is the target GPU (RX 7700S)"""
        name_lower = device_name.lower()
        # Check for explicit RX 7700S naming or AMD GPU codenames
        # gfx1103 is typically RX 7700S, gfx1102 is typically 780M
        return ('rx' in name_lower and '7700' in name_lower) or 'gfx1103' in name_lower
    
    def list_devices(self) -> None:
        """List all available devices with their properties"""
        print("=== Available OpenCL Devices ===")
        for i, device in enumerate(self.devices):
            status = " TARGET" if device['is_target'] else " DISCRETE" if device['is_discrete'] else " INTEGRATED"
            print(f"\n{i}: {device['name']} {status}")
            print(f"   Platform: {device['platform_id']}, Device: {device['device_id']}")
            print(f"   Type: {device['type']}")
            print(f"   Memory: {device['memory']} GB")
            print(f"   Compute Units: {device['compute_units']}")
            print(f"   Max Frequency: {device['max_freq']} MHz")
    
    def find_target_gpu(self) -> Optional[Dict]:
        """Find the target GPU (RX 7700S)"""
        for device in self.devices:
            if device['is_target']:
                return device
        return None
    
    def find_best_discrete_gpu(self) -> Optional[Dict]:
        """Find the best discrete GPU (highest memory and compute units)"""
        discrete_gpus = [d for d in self.devices if d['is_discrete']]
        if not discrete_gpus:
            return None
        
        # Sort by memory first, then compute units
        discrete_gpus.sort(key=lambda x: (x['memory'], x['compute_units']), reverse=True)
        return discrete_gpus[0]
    
    def create_context_for_device(self, device_info: Dict) -> Tuple[cl.Context, cl.CommandQueue]:
        """Create OpenCL context and command queue for specific device"""
        try:
            context = cl.Context([device_info['device']])
            queue = cl.CommandQueue(context)
            return context, queue
        except Exception as e:
            raise RuntimeError(f"Failed to create context for {device_info['name']}: {e}")
    
    def force_target_gpu(self) -> Tuple[cl.Context, cl.CommandQueue, Dict]:
        """Force usage of target GPU (RX 7700S)"""
        target_gpu = self.find_target_gpu()
        if not target_gpu:
            raise RuntimeError("Target GPU (RX 7700S) not found!")
        
        print(f" Forcing usage of target GPU: {target_gpu['name']}")
        context, queue = self.create_context_for_device(target_gpu)
        return context, queue, target_gpu
    
    def force_best_discrete_gpu(self) -> Tuple[cl.Context, cl.CommandQueue, Dict]:
        """Force usage of best discrete GPU"""
        best_gpu = self.find_best_discrete_gpu()
        if not best_gpu:
            raise RuntimeError("No discrete GPU found!")
        
        print(f" Forcing usage of best discrete GPU: {best_gpu['name']}")
        context, queue = self.create_context_for_device(best_gpu)
        return context, queue, best_gpu
    
    def set_environment_variables(self) -> None:
        """Set environment variables to influence GPU selection"""
        target_gpu = self.find_target_gpu()
        if target_gpu:
            # Try different environment variable approaches
            os.environ['AMD_OPENCL_DEVICE'] = str(target_gpu['device_id'])
            os.environ['OPENCL_DEVICE'] = str(target_gpu['device_id'])
            print(f" Set environment variables for device {target_gpu['device_id']}")
        else:
            print("  Target GPU not found, cannot set environment variables")
    
    def get_gpu_recommendations(self) -> Dict:
        """Get recommendations for GPU selection"""
        target_gpu = self.find_target_gpu()
        best_discrete = self.find_best_discrete_gpu()
        
        recommendations = {
            'target_found': target_gpu is not None,
            'target_gpu': target_gpu,
            'best_discrete': best_discrete,
            'system_config_needed': target_gpu is None or not target_gpu['is_discrete'],
            'programmatic_selection': True
        }
        
        return recommendations

def main():
    """Main function to demonstrate GPU selection"""
    print("=== AMD GPU Selector ===")
    
    selector = AMDGPUSelector()
    
    # List all devices
    selector.list_devices()
    
    # Get recommendations
    recommendations = selector.get_gpu_recommendations()
    
    print(f"\n=== Recommendations ===")
    if recommendations['target_found']:
        print(f" Target GPU (RX 7700S) found!")
        print(f"   Device: {recommendations['target_gpu']['name']}")
    else:
        print(f" Target GPU (RX 7700S) not found")
        if recommendations['best_discrete']:
            print(f"   Best alternative: {recommendations['best_discrete']['name']}")
    
    if recommendations['system_config_needed']:
        print(f"\n  System-level configuration recommended:")
        print(f"   1. Windows Graphics Settings (Win + I → System → Display → Graphics)")
        print(f"   2. AMD Radeon Settings (Right-click desktop → AMD Radeon Settings)")
        print(f"   3. Set python.exe to 'High Performance'")
    
    # Test programmatic selection
    print(f"\n=== Testing Programmatic Selection ===")
    try:
        if recommendations['target_found']:
            context, queue, device = selector.force_target_gpu()
            print(f" Successfully created context for {device['name']}")
        else:
            context, queue, device = selector.force_best_discrete_gpu()
            print(f" Successfully created context for {device['name']}")
    except Exception as e:
        print(f" Failed to create context: {e}")

if __name__ == "__main__":
    main()
