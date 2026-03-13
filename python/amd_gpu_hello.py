import os
import sys

# Proven environment variables for RX 7700S
os.environ['HIP_VISIBLE_DEVICES'] = '1'
os.environ['ROCR_VISIBLE_DEVICES'] = '1'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

def check_gpu():
    print("--- AMD GPU Verification (Minimal) ---")
    
    # 1. Check for PyOpenCL (standard for AMD on Windows)
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        print(f"Found {len(platforms)} OpenCL platforms.")
        for p in platforms:
            devices = p.get_devices()
            print(f"Platform: {p.name}")
            for d in devices:
                mem_gb = d.global_mem_size / (1024**3)
                print(f"  - Device: {d.name} ({mem_gb:.2f} GB)")
                if mem_gb > 8: # Likely the RX 7700S (12GB)
                    print("    [TARGET DETECTED] This is likely your RX 7700S.")
    except ImportError:
        print("PyOpenCL not found. Installing it might be the next step.")

    # 2. Check for PyTorch (to see if it has HIP support)
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        gpu_avail = torch.cuda.is_available() # Returns true if ROCm/HIP is setup correctly for Torch
        print(f"Torch CUDA/HIP Available: {gpu_avail}")
        if gpu_avail:
            print(f"Current Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not found.")

if __name__ == "__main__":
    check_gpu()
    print("\nIf you see the RX 7700S listed with ~12GB, we are ready to 'Scale' up.")
