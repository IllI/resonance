import torch
import time
import numpy as np

def test_cuda_rhythm(num_streams=4, duration=10, busy_wait_ms=10):
    print(f"Testing CUDA rhythm with {num_streams} concurrent streams...")
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    results = []
    
    start_time = time.time()
    
    # Simple workload: matrix multiplication
    size = 2048
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    print("Running sampling loop...")
    while time.time() - start_time < duration:
        loop_start = time.time()
        
        # Launch kernels in parallel streams
        for s in streams:
            with torch.cuda.stream(s):
                torch.mm(a, b)
        
        # Synchronize all
        torch.cuda.synchronize()
        
        loop_end = time.time()
        results.append({
            'host_time': loop_end,
            'delta': loop_end - loop_start
        })
        time.sleep(0.001) # Small sleep to avoid total CPU pegging
        
    deltas = [r['delta'] for r in results]
    print(f"Mean kernel time: {np.mean(deltas):.6f}s")
    print(f"Max kernel time: {np.max(deltas):.6f}s")
    
    # Save for analysis
    np.save('cuda_rhythm_test.npy', results)
    print("Results saved to cuda_rhythm_test.npy")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_cuda_rhythm()
    else:
        print("CUDA not available.")
