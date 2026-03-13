import os
import time
import numpy as np

# Inverted selection for RX 7700S targeting
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCR_VISIBLE_DEVICES'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

def stress_test():
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        # Find the AMD platform and the 12GB device
        target_device = None
        for p in platforms:
            if 'AMD' in p.name:
                for d in p.get_devices():
                    if d.global_mem_size > 8 * 1024**3:
                        target_device = d
                        break
        
        if not target_device:
            print("Could not find the target 12GB device.")
            return

        print(f"Stressing device: {target_device.name}")
        print("Run Task Manager and look for a spike in 'GPU 0' (RX 7700S) Compute graph.")
        print("Stress test running for 15 seconds... (Ctrl+C to stop)")
        
        ctx = cl.Context([target_device])
        queue = cl.CommandQueue(ctx)
        
        # Create a large buffer and do constant math
        size = 10**7
        a_np = np.random.rand(size).astype(np.float32)
        a_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
        
        kernel_code = """
        __kernel void stress(__global float* a) {
            int gid = get_global_id(0);
            float val = a[gid];
            for(int i=0; i<5000; i++) {
                val = native_sin(val + (float)i);
                val = native_cos(val * 0.5f);
            }
            a[gid] = val;
        }
        """
        prg = cl.Program(ctx, kernel_code).build()
        knl = prg.stress
        
        start_time = time.time()
        while time.time() - start_time < 15:
            knl(queue, (size,), None, a_g)
            queue.finish()
            
        print("\nStress test complete.")
        
    except Exception as e:
        print(f"Error during stress test: {e}")

if __name__ == "__main__":
    stress_test()
