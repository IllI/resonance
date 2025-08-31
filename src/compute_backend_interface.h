/**
 * @file compute_backend_interface.h
 * @brief Abstract interface for fMRI computation backends
 * 
 * This header defines a clean interface that separates GPU acceleration
 * from the core fMRI analysis algorithms. Different backend implementations
 * can provide CPU, single GPU, or multi-GPU acceleration while maintaining
 * the same API.
 * 
 * Key Design Principles:
 * 1. Hardware-agnostic interface
 * 2. Easy to swap backends at runtime  
 * 3. Graceful fallback to CPU if GPU fails
 * 4. Minimal dependencies for core functionality
 */

#ifndef COMPUTE_BACKEND_INTERFACE_H
#define COMPUTE_BACKEND_INTERFACE_H

#include <vector>
#include <string>
#include <memory>
#include <map>

namespace fmri_compute {

/**
 * @brief Performance metrics for backend operations
 */
struct PerformanceMetrics {
    double computation_time_ms;
    double memory_usage_mb;
    double gpu_utilization_percent;
    std::string backend_type;
    std::map<std::string, double> detailed_timings;
};

/**
 * @brief Abstract base class for all computation backends
 * 
 * This interface defines the core operations needed for fMRI analysis.
 * Implementations can use CPU, GPU, or cloud computing while providing
 * the same API to the fMRI analysis layer.
 */
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    
    /**
     * @brief Initialize the backend with specified configuration
     * @param config Configuration parameters (device IDs, memory limits, etc.)
     * @return true if initialization successful, false otherwise
     */
    virtual bool initialize(const std::map<std::string, std::string>& config = {}) = 0;
    
    /**
     * @brief Check if backend is ready for computation
     * @return true if backend is available and functional
     */
    virtual bool is_ready() const = 0;
    
    /**
     * @brief Get human-readable description of this backend
     * @return Backend description (e.g., "CPU", "Single GPU - AMD RX 7700S", "Dual GPU")
     */
    virtual std::string get_description() const = 0;
    
    /**
     * @brief Get estimated performance rating (higher is better)
     * @return Performance score (1.0 = baseline CPU, higher values indicate acceleration)
     */
    virtual double get_performance_rating() const = 0;
    
    // =========================================================================
    // Core fMRI Analysis Operations
    // =========================================================================
    
    /**
     * @brief Compute correlation matrix from fMRI time series
     * @param time_series Input data [regions x timepoints]
     * @param n_regions Number of brain regions
     * @param n_timepoints Number of time points
     * @param correlation_matrix Output correlation matrix [regions x regions]
     * @return Performance metrics for this operation
     */
    virtual PerformanceMetrics compute_correlation_matrix(
        const std::vector<float>& time_series,
        int n_regions,
        int n_timepoints,
        std::vector<float>& correlation_matrix
    ) = 0;
    
    /**
     * @brief Preprocess fMRI time series (detrending, filtering, standardization)
     * @param time_series Input/output data [regions x timepoints]
     * @param n_regions Number of brain regions
     * @param n_timepoints Number of time points
     * @param filter_low Low-pass filter cutoff (Hz)
     * @param filter_high High-pass filter cutoff (Hz)
     * @return Performance metrics for this operation
     */
    virtual PerformanceMetrics preprocess_time_series(
        std::vector<float>& time_series,
        int n_regions,
        int n_timepoints,
        float filter_low = 0.01f,
        float filter_high = 0.08f
    ) = 0;
    
    /**
     * @brief Compute network metrics from connectivity matrix
     * @param correlation_matrix Input correlation matrix [regions x regions]
     * @param n_regions Number of brain regions
     * @param node_strength Output node strength values [regions]
     * @param clustering_coeffs Output clustering coefficients [regions]
     * @param path_lengths Output shortest path lengths [regions x regions]
     * @return Performance metrics for this operation
     */
    virtual PerformanceMetrics compute_network_metrics(
        const std::vector<float>& correlation_matrix,
        int n_regions,
        std::vector<float>& node_strength,
        std::vector<float>& clustering_coeffs,
        std::vector<float>& path_lengths
    ) = 0;
    
    /**
     * @brief Detect significant changes in dynamic connectivity
     * @param dynamic_matrices Series of connectivity matrices over time
     * @param n_regions Number of brain regions
     * @param n_timepoints Number of time windows
     * @param threshold Change detection threshold
     * @param change_points Output indices of significant changes
     * @return Performance metrics for this operation
     */
    virtual PerformanceMetrics detect_connectivity_changes(
        const std::vector<std::vector<float>>& dynamic_matrices,
        int n_regions,
        int n_timepoints,
        float threshold,
        std::vector<int>& change_points
    ) = 0;
    
    // =========================================================================
    // Real-time Processing Support
    // =========================================================================
    
    /**
     * @brief Process single fMRI volume in real-time
     * @param volume_data New fMRI volume [regions]
     * @param time_series_buffer Sliding window buffer [regions x window_size]
     * @param window_size Size of sliding window
     * @param results Output analysis results
     * @return Performance metrics for this operation
     */
    virtual PerformanceMetrics process_realtime_volume(
        const std::vector<float>& volume_data,
        std::vector<std::vector<float>>& time_series_buffer,
        int window_size,
        std::map<std::string, std::vector<float>>& results
    ) = 0;
    
    // =========================================================================
    // Backend Management
    // =========================================================================
    
    /**
     * @brief Clean up resources and shutdown backend
     */
    virtual void shutdown() = 0;
    
    /**
     * @brief Get current performance statistics
     * @return Current backend performance metrics
     */
    virtual PerformanceMetrics get_performance_stats() const = 0;
};

// =============================================================================
// Backend Factory and Management
// =============================================================================

/**
 * @brief Available backend types
 */
enum class BackendType {
    CPU,              // Pure CPU implementation (always available)
    SINGLE_GPU,       // Single GPU acceleration (CUDA/OpenCL)  
    DUAL_AMD_GPU,     // Our specific dual AMD GPU optimization
    CLOUD_GPU,        // Cloud-based GPU compute
    AUTO              // Automatically select best available backend
};

/**
 * @brief Factory class for creating computation backends
 */
class BackendFactory {
public:
    /**
     * @brief Create backend of specified type
     * @param type Backend type to create
     * @param config Optional configuration parameters
     * @return Unique pointer to backend instance, nullptr if creation failed
     */
    static std::unique_ptr<ComputeBackend> create_backend(
        BackendType type,
        const std::map<std::string, std::string>& config = {}
    );
    
    /**
     * @brief Get list of available backends on current system
     * @return Vector of available backend types
     */
    static std::vector<BackendType> get_available_backends();
    
    /**
     * @brief Automatically select best available backend
     * @return Backend type with highest performance rating
     */
    static BackendType select_optimal_backend();
    
    /**
     * @brief Get human-readable name for backend type
     * @param type Backend type
     * @return Descriptive name
     */
    static std::string get_backend_name(BackendType type);
};

// =============================================================================
// Concrete Backend Implementations (Forward Declarations)
// =============================================================================

/**
 * @brief CPU-only backend implementation (always available)
 * 
 * This backend provides the baseline implementation using standard
 * CPU computation with NumPy/BLAS optimization where possible.
 * Serves as the fallback when no GPU acceleration is available.
 */
class CPUBackend : public ComputeBackend {
public:
    bool initialize(const std::map<std::string, std::string>& config = {}) override;
    bool is_ready() const override;
    std::string get_description() const override;
    double get_performance_rating() const override;
    
    PerformanceMetrics compute_correlation_matrix(
        const std::vector<float>& time_series,
        int n_regions, int n_timepoints,
        std::vector<float>& correlation_matrix
    ) override;
    
    PerformanceMetrics preprocess_time_series(
        std::vector<float>& time_series,
        int n_regions, int n_timepoints,
        float filter_low, float filter_high
    ) override;
    
    PerformanceMetrics compute_network_metrics(
        const std::vector<float>& correlation_matrix, int n_regions,
        std::vector<float>& node_strength,
        std::vector<float>& clustering_coeffs,
        std::vector<float>& path_lengths
    ) override;
    
    PerformanceMetrics detect_connectivity_changes(
        const std::vector<std::vector<float>>& dynamic_matrices,
        int n_regions, int n_timepoints, float threshold,
        std::vector<int>& change_points
    ) override;
    
    PerformanceMetrics process_realtime_volume(
        const std::vector<float>& volume_data,
        std::vector<std::vector<float>>& time_series_buffer,
        int window_size,
        std::map<std::string, std::vector<float>>& results
    ) override;
    
    void shutdown() override;
    PerformanceMetrics get_performance_stats() const override;
    
private:
    bool initialized = false;
    PerformanceMetrics cumulative_stats;
};

/**
 * @brief Single GPU backend implementation
 * 
 * Provides GPU acceleration using OpenCL or CUDA for compatible hardware.
 * Automatically detects and uses the most powerful GPU in the system.
 */
class SingleGPUBackend : public ComputeBackend {
    // Implementation in single_gpu_backend.cpp
};

/**
 * @brief Dual AMD GPU backend implementation
 * 
 * Our specific optimization for AMD Radeon 780M + RX 7700S hardware.
 * Implements load balancing and parallel execution across both GPUs.
 * 
 * This is where all our Phase 3 dual-GPU work is encapsulated.
 */
class DualAMDGPUBackend : public ComputeBackend {
    // Implementation in dual_amd_gpu_backend.cpp
    // Wraps our existing dual_gpu_brain_accelerator.cpp code
};

} // namespace fmri_compute

#endif // COMPUTE_BACKEND_INTERFACE_H 