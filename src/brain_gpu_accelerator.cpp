#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <cstdlib>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

// Brain GPU Accelerator
// High-performance OpenCL kernels for fMRI brain analysis
// Designed to integrate with Python neuroimaging pipeline

class BrainGPUAccelerator {
private:
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Context context;
    std::vector<cl::CommandQueue> queues;
    std::map<std::string, cl::Program> programs;
    std::map<std::string, cl::Kernel> kernels;
    
    bool initialized;
    int primary_gpu_index;
    int secondary_gpu_index;

public:
    BrainGPUAccelerator() : initialized(false), primary_gpu_index(-1), secondary_gpu_index(-1) {}
    
    // Initialize dual-GPU OpenCL environment
    bool initialize() {
        try {
            std::cout << "🚀 Initializing Brain GPU Accelerator..." << std::endl;
            
            // Get platforms and devices
            cl::Platform::get(&platforms);
            if (platforms.empty()) {
                std::cerr << "❌ No OpenCL platforms found" << std::endl;
                return false;
            }
            
            // Find GPU devices
            std::vector<cl::Device> all_devices;
            for (auto& platform : platforms) {
                std::vector<cl::Device> platform_devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &platform_devices);
                all_devices.insert(all_devices.end(), platform_devices.begin(), platform_devices.end());
            }
            
            if (all_devices.empty()) {
                std::cerr << "❌ No GPU devices found" << std::endl;
                return false;
            }
            
            devices = all_devices;
            primary_gpu_index = 0;
            if (devices.size() > 1) {
                secondary_gpu_index = 1;
            }
            
            // Create context
            context = cl::Context(devices);
            
            // Create command queues for each device
            for (size_t i = 0; i < devices.size(); ++i) {
                cl::CommandQueue queue(context, devices[i], CL_QUEUE_PROFILING_ENABLE);
                queues.push_back(queue);
            }
            
            // Load and compile kernels
            if (!loadKernels()) {
                return false;
            }
            
            initialized = true;
            std::cout << "✅ Brain GPU Accelerator initialized with " << devices.size() << " GPU(s)" << std::endl;
            
            return true;
            
        } catch (cl::Error& e) {
            std::cerr << "❌ OpenCL Error: " << e.what() << " (Code: " << e.err() << ")" << std::endl;
            return false;
        }
    }
    
    // Load OpenCL kernels for brain analysis
    bool loadKernels() {
        std::cout << "📥 Loading brain analysis kernels..." << std::endl;
        
        // Correlation matrix kernel
        std::string correlation_kernel = R"(
        __kernel void compute_correlation_matrix(
            __global const float* time_series,
            __global float* correlation_matrix,
            const int n_regions,
            const int n_timepoints
        ) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            
            if (i >= n_regions || j >= n_regions) return;
            
            // Only compute upper triangle (matrix is symmetric)
            if (i > j) return;
            
            // Calculate Pearson correlation between regions i and j
            float mean_i = 0.0f, mean_j = 0.0f;
            
            // Calculate means
            for (int t = 0; t < n_timepoints; t++) {
                mean_i += time_series[i * n_timepoints + t];
                mean_j += time_series[j * n_timepoints + t];
            }
            mean_i /= n_timepoints;
            mean_j /= n_timepoints;
            
            // Calculate correlation
            float numerator = 0.0f;
            float sum_sq_i = 0.0f, sum_sq_j = 0.0f;
            
            for (int t = 0; t < n_timepoints; t++) {
                float diff_i = time_series[i * n_timepoints + t] - mean_i;
                float diff_j = time_series[j * n_timepoints + t] - mean_j;
                
                numerator += diff_i * diff_j;
                sum_sq_i += diff_i * diff_i;
                sum_sq_j += diff_j * diff_j;
            }
            
            float denominator = sqrt(sum_sq_i * sum_sq_j);
            float correlation = (denominator > 0.0f) ? (numerator / denominator) : 0.0f;
            
            // Store in both positions (symmetric matrix)
            correlation_matrix[i * n_regions + j] = correlation;
            if (i != j) {
                correlation_matrix[j * n_regions + i] = correlation;
            }
        }
        )";
        
        // Preprocessing kernel
        std::string preprocessing_kernel = R"(
        __kernel void preprocess_timeseries(
            __global const float* input_data,
            __global float* output_data,
            const int n_regions,
            const int n_timepoints,
            const float filter_low,
            const float filter_high
        ) {
            int region = get_global_id(0);
            
            if (region >= n_regions) return;
            
            // Detrending: remove linear trend
            float sum_t = 0.0f, sum_y = 0.0f, sum_ty = 0.0f, sum_tt = 0.0f;
            
            for (int t = 0; t < n_timepoints; t++) {
                float y = input_data[region * n_timepoints + t];
                sum_t += t;
                sum_y += y;
                sum_ty += t * y;
                sum_tt += t * t;
            }
            
            float slope = (n_timepoints * sum_ty - sum_t * sum_y) / 
                         (n_timepoints * sum_tt - sum_t * sum_t);
            float intercept = (sum_y - slope * sum_t) / n_timepoints;
            
            // Apply detrending and basic filtering
            for (int t = 0; t < n_timepoints; t++) {
                float detrended = input_data[region * n_timepoints + t] - (slope * t + intercept);
                output_data[region * n_timepoints + t] = detrended;
            }
        }
        )";
        
        // Network metrics kernel
        std::string network_metrics_kernel = R"(
        __kernel void compute_node_metrics(
            __global const float* adjacency_matrix,
            __global float* node_strength,
            __global float* clustering_coeff,
            const int n_nodes,
            const float threshold
        ) {
            int node = get_global_id(0);
            
            if (node >= n_nodes) return;
            
            // Compute node strength (degree)
            float strength = 0.0f;
            int degree = 0;
            
            for (int j = 0; j < n_nodes; j++) {
                if (j != node) {
                    float weight = adjacency_matrix[node * n_nodes + j];
                    if (fabs(weight) > threshold) {
                        strength += fabs(weight);
                        degree++;
                    }
                }
            }
            
            node_strength[node] = strength;
            
            // Compute clustering coefficient
            if (degree < 2) {
                clustering_coeff[node] = 0.0f;
                return;
            }
            
            int triangles = 0;
            int possible_triangles = 0;
            
            for (int j = 0; j < n_nodes; j++) {
                if (j == node) continue;
                
                float weight_ij = adjacency_matrix[node * n_nodes + j];
                if (fabs(weight_ij) <= threshold) continue;
                
                for (int k = j + 1; k < n_nodes; k++) {
                    if (k == node) continue;
                    
                    float weight_ik = adjacency_matrix[node * n_nodes + k];
                    float weight_jk = adjacency_matrix[j * n_nodes + k];
                    
                    if (fabs(weight_ik) > threshold) {
                        possible_triangles++;
                        if (fabs(weight_jk) > threshold) {
                            triangles++;
                        }
                    }
                }
            }
            
            clustering_coeff[node] = (possible_triangles > 0) ? 
                                   ((float)triangles / possible_triangles) : 0.0f;
        }
        )";
        
        try {
            // Compile kernels
            cl::Program correlation_prog(context, correlation_kernel);
            correlation_prog.build(devices);
            programs["correlation"] = correlation_prog;
            kernels["compute_correlation"] = cl::Kernel(correlation_prog, "compute_correlation_matrix");
            
            cl::Program preprocessing_prog(context, preprocessing_kernel);
            preprocessing_prog.build(devices);
            programs["preprocessing"] = preprocessing_prog;
            kernels["preprocess_timeseries"] = cl::Kernel(preprocessing_prog, "preprocess_timeseries");
            
            cl::Program network_prog(context, network_metrics_kernel);
            network_prog.build(devices);
            programs["network"] = network_prog;
            kernels["compute_node_metrics"] = cl::Kernel(network_prog, "compute_node_metrics");
            
            std::cout << "✅ Brain analysis kernels loaded successfully" << std::endl;
            return true;
            
        } catch (cl::Error& e) {
            std::cerr << "❌ Kernel compilation error: " << e.what() << std::endl;
            return false;
        }
    }
    
    // GPU-accelerated correlation matrix computation
    bool computeCorrelationMatrix(
        const std::vector<float>& time_series,
        std::vector<float>& correlation_matrix,
        int n_regions,
        int n_timepoints
    ) {
        if (!initialized) {
            std::cerr << "❌ GPU Accelerator not initialized" << std::endl;
            return false;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Create buffers
            cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  time_series.size() * sizeof(float), (void*)time_series.data());
            
            cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY,
                                   correlation_matrix.size() * sizeof(float));
            
            // Set kernel arguments
            auto& kernel = kernels["compute_correlation"];
            kernel.setArg(0, input_buffer);
            kernel.setArg(1, output_buffer);
            kernel.setArg(2, n_regions);
            kernel.setArg(3, n_timepoints);
            
            // Execute kernel
            cl::NDRange global_size(n_regions, n_regions);
            cl::NDRange local_size(16, 16);  // Adjust based on GPU capabilities
            
            queues[primary_gpu_index].enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size);
            
            // Read results
            queues[primary_gpu_index].enqueueReadBuffer(output_buffer, CL_TRUE, 0,
                                                      correlation_matrix.size() * sizeof(float),
                                                      correlation_matrix.data());
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::cout << "✅ GPU correlation matrix computed in " << duration.count() / 1000.0f << " ms" << std::endl;
            return true;
            
        } catch (cl::Error& e) {
            std::cerr << "❌ GPU correlation error: " << e.what() << std::endl;
            return false;
        }
    }
    
    // GPU-accelerated time series preprocessing
    bool preprocessTimeSeries(
        const std::vector<float>& input_data,
        std::vector<float>& output_data,
        int n_regions,
        int n_timepoints,
        float filter_low = 0.01f,
        float filter_high = 0.1f
    ) {
        if (!initialized) {
            std::cerr << "❌ GPU Accelerator not initialized" << std::endl;
            return false;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Create buffers
            cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  input_data.size() * sizeof(float), (void*)input_data.data());
            
            cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY,
                                   output_data.size() * sizeof(float));
            
            // Set kernel arguments
            auto& kernel = kernels["preprocess_timeseries"];
            kernel.setArg(0, input_buffer);
            kernel.setArg(1, output_buffer);
            kernel.setArg(2, n_regions);
            kernel.setArg(3, n_timepoints);
            kernel.setArg(4, filter_low);
            kernel.setArg(5, filter_high);
            
            // Execute kernel
            cl::NDRange global_size(n_regions);
            cl::NDRange local_size(256);  // Adjust based on GPU capabilities
            
            queues[primary_gpu_index].enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size);
            
            // Read results
            queues[primary_gpu_index].enqueueReadBuffer(output_buffer, CL_TRUE, 0,
                                                      output_data.size() * sizeof(float),
                                                      output_data.data());
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::cout << "✅ GPU preprocessing completed in " << duration.count() / 1000.0f << " ms" << std::endl;
            return true;
            
        } catch (cl::Error& e) {
            std::cerr << "❌ GPU preprocessing error: " << e.what() << std::endl;
            return false;
        }
    }
    
    // GPU-accelerated network metrics computation
    bool computeNetworkMetrics(
        const std::vector<float>& adjacency_matrix,
        std::vector<float>& node_strength,
        std::vector<float>& clustering_coeff,
        int n_nodes,
        float threshold = 0.3f
    ) {
        if (!initialized) {
            std::cerr << "❌ GPU Accelerator not initialized" << std::endl;
            return false;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Create buffers
            cl::Buffer adj_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                adjacency_matrix.size() * sizeof(float), 
                                (void*)adjacency_matrix.data());
            
            cl::Buffer strength_buffer(context, CL_MEM_WRITE_ONLY,
                                     node_strength.size() * sizeof(float));
                                     
            cl::Buffer clustering_buffer(context, CL_MEM_WRITE_ONLY,
                                       clustering_coeff.size() * sizeof(float));
            
            // Set kernel arguments
            auto& kernel = kernels["compute_node_metrics"];
            kernel.setArg(0, adj_buffer);
            kernel.setArg(1, strength_buffer);
            kernel.setArg(2, clustering_buffer);
            kernel.setArg(3, n_nodes);
            kernel.setArg(4, threshold);
            
            // Execute kernel
            cl::NDRange global_size(n_nodes);
            cl::NDRange local_size(256);
            
            queues[primary_gpu_index].enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size);
            
            // Read results
            queues[primary_gpu_index].enqueueReadBuffer(strength_buffer, CL_TRUE, 0,
                                                      node_strength.size() * sizeof(float),
                                                      node_strength.data());
                                                      
            queues[primary_gpu_index].enqueueReadBuffer(clustering_buffer, CL_TRUE, 0,
                                                      clustering_coeff.size() * sizeof(float),
                                                      clustering_coeff.data());
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::cout << "✅ GPU network metrics computed in " << duration.count() / 1000.0f << " ms" << std::endl;
            return true;
            
        } catch (cl::Error& e) {
            std::cerr << "❌ GPU network metrics error: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Get performance information
    void getPerformanceInfo() const {
        if (!initialized) return;
        
        std::cout << "🖥️ Brain GPU Accelerator Performance Info:" << std::endl;
        
        for (size_t i = 0; i < devices.size(); ++i) {
            std::string device_name = devices[i].getInfo<CL_DEVICE_NAME>();
            cl_uint compute_units = devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            cl_ulong global_mem = devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
            cl_uint max_freq = devices[i].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
            
            std::cout << "   GPU " << i << ": " << device_name << std::endl;
            std::cout << "     Compute Units: " << compute_units << std::endl;
            std::cout << "     Global Memory: " << global_mem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "     Max Frequency: " << max_freq << " MHz" << std::endl;
            
            if (i == primary_gpu_index) {
                std::cout << "     Status: PRIMARY GPU" << std::endl;
            } else if (i == secondary_gpu_index) {
                std::cout << "     Status: SECONDARY GPU" << std::endl;
            }
        }
    }
    
    ~BrainGPUAccelerator() {
        // Cleanup handled by OpenCL C++ wrapper destructors
    }
};

// C-style interface for Python integration
extern "C" {
    static BrainGPUAccelerator* gpu_accelerator = nullptr;
    
    // Initialize GPU accelerator
    __declspec(dllexport) bool init_brain_gpu_accelerator() {
        if (gpu_accelerator == nullptr) {
            gpu_accelerator = new BrainGPUAccelerator();
        }
        return gpu_accelerator->initialize();
    }
    
    // Compute correlation matrix
    __declspec(dllexport) bool gpu_compute_correlation_matrix(
        const float* time_series,
        float* correlation_matrix,
        int n_regions,
        int n_timepoints
    ) {
        if (gpu_accelerator == nullptr) return false;
        
        std::vector<float> ts_vector(time_series, time_series + n_regions * n_timepoints);
        std::vector<float> corr_vector(n_regions * n_regions);
        
        bool result = gpu_accelerator->computeCorrelationMatrix(ts_vector, corr_vector, n_regions, n_timepoints);
        
        if (result) {
            std::copy(corr_vector.begin(), corr_vector.end(), correlation_matrix);
        }
        
        return result;
    }
    
    // Preprocess time series
    __declspec(dllexport) bool gpu_preprocess_timeseries(
        const float* input_data,
        float* output_data,
        int n_regions,
        int n_timepoints,
        float filter_low,
        float filter_high
    ) {
        if (gpu_accelerator == nullptr) return false;
        
        std::vector<float> input_vector(input_data, input_data + n_regions * n_timepoints);
        std::vector<float> output_vector(n_regions * n_timepoints);
        
        bool result = gpu_accelerator->preprocessTimeSeries(input_vector, output_vector, 
                                                           n_regions, n_timepoints, 
                                                           filter_low, filter_high);
        
        if (result) {
            std::copy(output_vector.begin(), output_vector.end(), output_data);
        }
        
        return result;
    }
    
    // Compute network metrics
    __declspec(dllexport) bool gpu_compute_network_metrics(
        const float* adjacency_matrix,
        float* node_strength,
        float* clustering_coeff,
        int n_nodes,
        float threshold
    ) {
        if (gpu_accelerator == nullptr) return false;
        
        std::vector<float> adj_vector(adjacency_matrix, adjacency_matrix + n_nodes * n_nodes);
        std::vector<float> strength_vector(n_nodes);
        std::vector<float> clustering_vector(n_nodes);
        
        bool result = gpu_accelerator->computeNetworkMetrics(adj_vector, strength_vector, 
                                                           clustering_vector, n_nodes, threshold);
        
        if (result) {
            std::copy(strength_vector.begin(), strength_vector.end(), node_strength);
            std::copy(clustering_vector.begin(), clustering_vector.end(), clustering_coeff);
        }
        
        return result;
    }
    
    // Get performance info
    __declspec(dllexport) void gpu_get_performance_info() {
        if (gpu_accelerator != nullptr) {
            gpu_accelerator->getPerformanceInfo();
        }
    }
    
    // Cleanup
    __declspec(dllexport) void cleanup_brain_gpu_accelerator() {
        if (gpu_accelerator != nullptr) {
            delete gpu_accelerator;
            gpu_accelerator = nullptr;
        }
    }
}

// Test function for standalone execution
int main() {
    std::cout << "🧠 Brain GPU Accelerator Test" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Initialize
    if (!init_brain_gpu_accelerator()) {
        std::cerr << "❌ Failed to initialize GPU accelerator" << std::endl;
        return 1;
    }
    
    // Show performance info
    gpu_get_performance_info();
    
    // Test with mock data
    const int n_regions = 100;
    const int n_timepoints = 200;
    
    // Create mock time series data
    std::vector<float> time_series(n_regions * n_timepoints);
    std::vector<float> correlation_matrix(n_regions * n_regions);
    
    // Fill with random data
    for (int i = 0; i < n_regions * n_timepoints; ++i) {
        time_series[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Test correlation computation
    std::cout << "\n🔗 Testing GPU correlation matrix computation..." << std::endl;
    bool success = gpu_compute_correlation_matrix(
        time_series.data(),
        correlation_matrix.data(),
        n_regions,
        n_timepoints
    );
    
    if (success) {
        std::cout << "✅ Correlation matrix computation successful" << std::endl;
        std::cout << "   Matrix range: [" << *std::min_element(correlation_matrix.begin(), correlation_matrix.end())
                  << ", " << *std::max_element(correlation_matrix.begin(), correlation_matrix.end()) << "]" << std::endl;
    }
    
    // Cleanup
    cleanup_brain_gpu_accelerator();
    
    std::cout << "\n🎉 Brain GPU Accelerator test completed!" << std::endl;
    
    return 0;
} 