#include <iostream>
#include <vector>
#include <string>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

int main() {
    try {
        // Get all platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        std::cout << "Found " << platforms.size() << " OpenCL platform(s):\n\n";

        for (size_t i = 0; i < platforms.size(); i++) {
            cl::Platform platform = platforms[i];
            
            std::cout << "Platform " << i << ":\n";
            std::cout << "  Name: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
            std::cout << "  Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
            std::cout << "  Version: " << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
            std::cout << "  Profile: " << platform.getInfo<CL_PLATFORM_PROFILE>() << "\n";

            // Get devices for this platform
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            std::cout << "  Devices (" << devices.size() << "):\n";

            for (size_t j = 0; j < devices.size(); j++) {
                cl::Device device = devices[j];
                
                std::cout << "    Device " << j << ":\n";
                std::cout << "      Name: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
                std::cout << "      Type: ";
                
                auto deviceType = device.getInfo<CL_DEVICE_TYPE>();
                if (deviceType & CL_DEVICE_TYPE_CPU) std::cout << "CPU ";
                if (deviceType & CL_DEVICE_TYPE_GPU) std::cout << "GPU ";
                if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "ACCELERATOR ";
                std::cout << "\n";
                
                std::cout << "      Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
                std::cout << "      OpenCL Version: " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
                std::cout << "      Driver Version: " << device.getInfo<CL_DRIVER_VERSION>() << "\n";
                std::cout << "      Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
                std::cout << "      Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
                std::cout << "      Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024) << " MB\n";
                std::cout << "      Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB\n";
                std::cout << "      Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz\n";
                
                // Check if it supports double precision
                auto extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
                bool supportsDouble = extensions.find("cl_khr_fp64") != std::string::npos;
                std::cout << "      Double Precision: " << (supportsDouble ? "Yes" : "No") << "\n";
                
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        // Test creating a context with GPU devices
        std::vector<cl::Device> gpuDevices;
        for (auto& platform : platforms) {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            for (auto& device : devices) {
                gpuDevices.push_back(device);
            }
        }

        if (gpuDevices.size() > 0) {
            std::cout << "Testing OpenCL context creation with " << gpuDevices.size() << " GPU device(s)...\n";
            
            try {
                cl::Context context(gpuDevices);
                std::cout << "✓ Successfully created OpenCL context with GPU devices!\n";
                
                // Create command queues for each GPU
                for (size_t i = 0; i < gpuDevices.size(); i++) {
                    cl::CommandQueue queue(context, gpuDevices[i]);
                    std::cout << "✓ Created command queue for GPU " << i << ": " 
                              << gpuDevices[i].getInfo<CL_DEVICE_NAME>() << "\n";
                }
                    } catch (cl::Error& e) {
            std::cout << "✗ Failed to create OpenCL context: " << e.what() << " (Error code: " << e.err() << ")\n";
        } catch (std::exception& e) {
            std::cout << "✗ Failed to create OpenCL context: " << e.what() << "\n";
        }
        } else {
            std::cout << "No GPU devices found!\n";
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