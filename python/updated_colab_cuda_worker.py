# UPDATED CUDA PARALLELIZATION LOGIC for workers.py
# Replace the gpu_worker function in your Colab notebook with this:

def gpu_worker(
    gpu_id,
    device_worker_index,
    stream_global_index,
    workers_per_gpu,
    queue,
    stop_event,
    threads_per_block=256,
    blocks=256,
    samples_per_thread=32,
    report_samples=8192,
):
    gpu_name = 'unknown'
    stream_id = f'gpu{int(gpu_id)}_worker{int(device_worker_index)}'
    try:
        cp.cuda.Device(gpu_id).use()
        gpu_name = get_gpu_name(gpu_id)
        
        # --- NEW: USE EXPLICIT CUDA STREAM FOR TRUE PARALLELISM ---
        # This prevents the GigaThread scheduler from serializing multiple workers
        # on the same GPU via the default stream (Stream.null).
        worker_stream = cp.cuda.Stream(non_blocking=True)
        
        record_kernel = cp.RawKernel(cuda_code, 'record_time_jitter')
        total_threads = threads_per_block * blocks
        buffer_size = total_threads * samples_per_thread
        sample_limit = min(buffer_size, report_samples)
        d_times = cp.empty(buffer_size, dtype=cp.uint64)
        packet_index = 0

        while not stop_event.is_set():
            host_time_start = time.time()
            
            # Record using the worker's private stream
            record_kernel((blocks,), (threads_per_block,), (d_times, samples_per_thread), stream=worker_stream)
            
            # Synchronize ONLY this stream to prevent serializing with other local workers
            # This is critical for maintaining high local entropy.
            worker_stream.synchronize()
            
            host_time_end = time.time()
            clock_data = cp.asnumpy(d_times[:sample_limit])
            
            queue.put({
                'schema_version': STREAM_DATASET_SCHEMA_VERSION,
                'record_type': 'gpu_clock_packet',
                'gpu_id': int(gpu_id),
                'gpu_name': gpu_name,
                'pid': int(os.getpid()),
                'stream_id': stream_id,
                'device_worker_index': int(device_worker_index),
                'stream_global_index': int(stream_global_index),
                'workers_per_gpu': max(int(workers_per_gpu), 1),
                'packet_index': int(packet_index),
                'host_time_start': float(host_time_start),
                'host_time_end': float(host_time_end),
                'host_time_mid': float((host_time_start + host_time_end) / 2.0),
                'elapsed_host_seconds': float(host_time_end - host_time_start),
                'threads_per_block': int(threads_per_block),
                'blocks': int(blocks),
                'samples_per_thread': int(samples_per_thread),
                'total_threads': int(total_threads),
                'buffer_size': int(buffer_size),
                'sample_count': int(len(clock_data)),
                'clock_data': clock_data,
            })
            packet_index += 1
    except Exception as exc:
        queue.put({
            'schema_version': STREAM_DATASET_SCHEMA_VERSION,
            'record_type': 'worker_error',
            'gpu_id': int(gpu_id),
            'gpu_name': gpu_name,
            'pid': int(os.getpid()),
            'stream_id': stream_id,
            'device_worker_index': int(device_worker_index),
            'stream_global_index': int(stream_global_index),
            'host_time': float(time.time()),
            'error': repr(exc),
        })
        print(f'[GPU {gpu_id} worker {device_worker_index}] Error: {exc}')
