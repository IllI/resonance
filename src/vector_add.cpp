#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

const char* vectorAddKernel = R"(
__kernel void vector_add(__global const float* A,
                        __global const float* B,
                        __global float* C,
                        const unsigned int n)
{
    int i = get_global_id(0);
    if(i < n) {
        C[i] = A[i] + B[i];
    }
}
)";

int main() {
    const size_t VECTOR_SIZE = 1000000;
    const size_t BYTES = VECTOR_SIZE * sizeof(float);

    try {
        // Initialize data
        std::vector<float> h_a(VECTOR_SIZE);
        std::vector<float> h_b(VECTOR_SIZE);
        std::vector<float> h_c(VECTOR_SIZE);
        std::vector<float> h_c_verify(VECTOR_SIZE);

        // Fill vectors with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for(size_t i = 0; i < VECTOR_SIZE; i++) {
            h_a[i] = dis(gen);
            h_b[i] = dis(gen);
            h_c_verify[i] = h_a[i] + h_b[i]; // CPU verification
        }

        std::cout << "Vector addition of " << VECTOR_SIZE << " elements\n";
        std::cout << "Vector size: " << BYTES / (1024*1024) << " MB\n\n";

        // Get platforms and devices
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        std::vector<cl::Device> gpuDevices;
        for (auto& platform : platforms) {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            for (auto& device : devices) {
                gpuDevices.push_back(device);
            }
        }

        if (gpuDevices.empty()) {
            std::cout << "No GPU devices found!\n";
            return 1;
        }

        std::cout << "Found " << gpuDevices.size() << " GPU device(s):\n";
        for (size_t i = 0; i < gpuDevices.size(); i++) {
            std::cout << "  GPU " << i << ": " << gpuDevices[i].getInfo<CL_DEVICE_NAME>() << "\n";
        }
        std::cout << "\n";

        // Test each GPU individually
        for (size_t gpu_idx = 0; gpu_idx < gpuDevices.size(); gpu_idx++) {
            auto& device = gpuDevices[gpu_idx];
            std::cout << "Testing GPU " << gpu_idx << ": " << device.getInfo<CL_DEVICE_NAME>() << "\n";

            // Create context and command queue for this device
            cl::Context context(device);
            cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

            // Create memory buffers
            cl::Buffer d_a(context, CL_MEM_READ_ONLY, BYTES);
            cl::Buffer d_b(context, CL_MEM_READ_ONLY, BYTES);
            cl::Buffer d_c(context, CL_MEM_WRITE_ONLY, BYTES);

            // Copy data to device
            auto start_copy = std::chrono::high_resolution_clock::now();
            queue.enqueueWriteBuffer(d_a, CL_FALSE, 0, BYTES, h_a.data());
            queue.enqueueWriteBuffer(d_b, CL_FALSE, 0, BYTES, h_b.data());
            queue.finish();
            auto end_copy = std::chrono::high_resolution_clock::now();

            // Create program and kernel
            cl::Program::Sources sources;
            sources.push_back({vectorAddKernel, strlen(vectorAddKernel)});
            cl::Program program(context, sources);

            try {
                program.build({device});
            } catch (cl::Error& e) {
                std::cout << "Build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
                throw;
            } catch (std::exception& e) {
                std::cout << "Build error: " << e.what() << "\n";
                throw;
            }

            cl::Kernel kernel(program, "vector_add");

            // Set kernel arguments
            kernel.setArg(0, d_a);
            kernel.setArg(1, d_b);
            kernel.setArg(2, d_c);
            kernel.setArg(3, static_cast<unsigned int>(VECTOR_SIZE));

            // Execute kernel
            cl::NDRange global(VECTOR_SIZE);
            cl::NDRange local = cl::NullRange; // Let OpenCL choose

            auto start_kernel = std::chrono::high_resolution_clock::now();
            cl::Event event;
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &event);
            queue.finish();
            auto end_kernel = std::chrono::high_resolution_clock::now();

            // Copy result back
            auto start_read = std::chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_c, CL_TRUE, 0, BYTES, h_c.data());
            auto end_read = std::chrono::high_resolution_clock::now();

            // Verify results
            bool correct = true;
            for (size_t i = 0; i < VECTOR_SIZE && i < 100; i++) {
                if (std::abs(h_c[i] - h_c_verify[i]) > 1e-5) {
                    correct = false;
                    std::cout << "  Error at index " << i << ": expected " << h_c_verify[i] 
                              << ", got " << h_c[i] << "\n";
                    break;
                }
            }

            // Calculate timings
            auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(end_copy - start_copy).count();
            auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(end_kernel - start_kernel).count();
            auto read_time = std::chrono::duration_cast<std::chrono::microseconds>(end_read - start_read).count();

            // Get profiling info
            cl_ulong prof_start, prof_end;
            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &prof_start);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &prof_end);
            double kernel_time_ms = (prof_end - prof_start) / 1000000.0;

            std::cout << "  Results: " << (correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n";
            std::cout << "  Copy to device: " << copy_time / 1000.0 << " ms\n";
            std::cout << "  Kernel execution: " << kernel_time_ms << " ms\n";
            std::cout << "  Copy from device: " << read_time / 1000.0 << " ms\n";
            std::cout << "  Total time: " << (copy_time + kernel_time + read_time) / 1000.0 << " ms\n";
            
            // Calculate performance
            double gflops = (VECTOR_SIZE / (kernel_time_ms / 1000.0)) / 1e9;
            std::cout << "  Performance: " << gflops << " GFLOPS\n\n";
        }

        // Test parallel execution if we have multiple GPUs
        if (gpuDevices.size() > 1) {
            std::cout << "Testing parallel execution across " << gpuDevices.size() << " GPUs...\n";
            
            size_t chunk_size = VECTOR_SIZE / gpuDevices.size();
            std::vector<cl::Context> contexts;
            std::vector<cl::CommandQueue> queues;
            std::vector<cl::Buffer> buffers_a, buffers_b, buffers_c;
            std::vector<cl::Program> programs;
            std::vector<cl::Kernel> kernels;

            // Set up contexts and buffers for each GPU
            for (size_t i = 0; i < gpuDevices.size(); i++) {
                contexts.emplace_back(gpuDevices[i]);
                queues.emplace_back(contexts[i], gpuDevices[i], CL_QUEUE_PROFILING_ENABLE);
                
                size_t current_chunk = (i == gpuDevices.size() - 1) ? 
                    VECTOR_SIZE - i * chunk_size : chunk_size;
                size_t chunk_bytes = current_chunk * sizeof(float);

                buffers_a.emplace_back(contexts[i], CL_MEM_READ_ONLY, chunk_bytes);
                buffers_b.emplace_back(contexts[i], CL_MEM_READ_ONLY, chunk_bytes);
                buffers_c.emplace_back(contexts[i], CL_MEM_WRITE_ONLY, chunk_bytes);

                // Create program and kernel
                cl::Program::Sources sources;
                sources.push_back({vectorAddKernel, strlen(vectorAddKernel)});
                programs.emplace_back(contexts[i], sources);
                programs[i].build({gpuDevices[i]});
                kernels.emplace_back(programs[i], "vector_add");
            }

            auto start_parallel = std::chrono::high_resolution_clock::now();

            // Launch on all GPUs
            std::vector<cl::Event> events;
            for (size_t i = 0; i < gpuDevices.size(); i++) {
                size_t offset = i * chunk_size;
                size_t current_chunk = (i == gpuDevices.size() - 1) ? 
                    VECTOR_SIZE - offset : chunk_size;
                size_t chunk_bytes = current_chunk * sizeof(float);

                // Copy data
                queues[i].enqueueWriteBuffer(buffers_a[i], CL_FALSE, 0, chunk_bytes, &h_a[offset]);
                queues[i].enqueueWriteBuffer(buffers_b[i], CL_FALSE, 0, chunk_bytes, &h_b[offset]);

                // Set kernel arguments
                kernels[i].setArg(0, buffers_a[i]);
                kernels[i].setArg(1, buffers_b[i]);
                kernels[i].setArg(2, buffers_c[i]);
                kernels[i].setArg(3, static_cast<unsigned int>(current_chunk));

                // Launch kernel
                cl::Event event;
                queues[i].enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(current_chunk), 
                                             cl::NullRange, nullptr, &event);
                events.push_back(event);

                // Read result back
                queues[i].enqueueReadBuffer(buffers_c[i], CL_FALSE, 0, chunk_bytes, &h_c[offset]);
            }

            // Wait for all to complete
            for (auto& queue : queues) {
                queue.finish();
            }

            auto end_parallel = std::chrono::high_resolution_clock::now();
            auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - start_parallel).count();

            // Verify parallel results
            bool parallel_correct = true;
            for (size_t i = 0; i < VECTOR_SIZE && i < 100; i++) {
                if (std::abs(h_c[i] - h_c_verify[i]) > 1e-5) {
                    parallel_correct = false;
                    break;
                }
            }

            std::cout << "  Parallel results: " << (parallel_correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n";
            std::cout << "  Parallel execution time: " << parallel_time / 1000.0 << " ms\n";
            
            double parallel_gflops = (VECTOR_SIZE / (parallel_time / 1000000.0)) / 1e9;
            std::cout << "  Parallel performance: " << parallel_gflops << " GFLOPS\n";
        }

    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (Error code: " << e.err() << ")\n";
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred\n";
        return 1;
    }

    return 0;
} 