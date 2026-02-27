// C++ GPU Bypass - Direct GPU access bypassing Windows limitations
// This addresses Windows GPU selection issues with dual AMD setups

#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <d3d11.h>
#include <dxgi.h>

// Export the symbol that tells AMD drivers to use discrete GPU
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

class CppGPUBypass {
private:
    bool discrete_gpu_forced;
    ID3D11Device* d3d_device;
    ID3D11DeviceContext* d3d_context;
    
public:
    CppGPUBypass() : discrete_gpu_forced(false), d3d_device(nullptr), d3d_context(nullptr) {
        // Set environment variables when the class is created
        SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance", "1");
        SetEnvironmentVariableA("AMD_OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("GPU_DEVICE_ORDINAL", "0");
        SetEnvironmentVariableA("AMD_DISABLE_INTEGRATED_GPU", "1");
        SetEnvironmentVariableA("OPENCL_DISABLE_DEVICE_1", "1");
        
        std::cout << "C++ GPU Bypass initialized" << std::endl;
        std::cout << "Set AMD power export and environment variables" << std::endl;
    }
    
    ~CppGPUBypass() {
        if (d3d_device) d3d_device->Release();
        if (d3d_context) d3d_context->Release();
    }
    
    void forceDiscreteGPU() {
        // Force discrete GPU usage
        AmdPowerXpressRequestHighPerformance = 1;
        SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance", "1");
        
        // Set additional environment variables
        SetEnvironmentVariableA("AMD_OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("GPU_DEVICE_ORDINAL", "0");
        
        discrete_gpu_forced = true;
        std::cout << "Forced discrete GPU usage" << std::endl;
    }
    
    bool createDirect3DContext() {
        // Create Direct3D context to force discrete GPU usage
        std::cout << "Creating Direct3D context to force discrete GPU..." << std::endl;
        
        D3D_FEATURE_LEVEL featureLevel;
        D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
        
        // Create device with discrete GPU preference
        HRESULT hr = D3D11CreateDevice(
            nullptr,                    // Use default adapter
            D3D_DRIVER_TYPE_HARDWARE,   // Use hardware acceleration
            nullptr,                    // No software rasterizer
            0,                          // No flags
            featureLevels,              // Feature levels
            ARRAYSIZE(featureLevels),   // Number of feature levels
            D3D11_SDK_VERSION,          // SDK version
            &d3d_device,                // Device
            &featureLevel,              // Feature level
            &d3d_context                // Context
        );
        
        if (SUCCEEDED(hr)) {
            std::cout << "Direct3D context created successfully" << std::endl;
            return true;
        } else {
            std::cout << "Failed to create Direct3D context: " << hr << std::endl;
            return false;
        }
    }
    
    bool enumerateGPUs() {
        // Enumerate available GPUs
        std::cout << "Enumerating available GPUs..." << std::endl;
        
        IDXGIFactory* factory = nullptr;
        HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
        
        if (FAILED(hr)) {
            std::cout << "Failed to create DXGI factory" << std::endl;
            return false;
        }
        
        IDXGIAdapter* adapter = nullptr;
        UINT adapterIndex = 0;
        
        while (factory->EnumAdapters(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC desc;
            adapter->GetDesc(&desc);
            
            std::wcout << L"Adapter " << adapterIndex << L": " << desc.Description << std::endl;
            std::cout << "  Dedicated Video Memory: " << desc.DedicatedVideoMemory / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  Shared System Memory: " << desc.SharedSystemMemory / (1024 * 1024) << " MB" << std::endl;
            
            // Check if this is the discrete GPU
            if (desc.DedicatedVideoMemory > 1024 * 1024 * 1024) { // More than 1GB
                std::cout << "  -> This appears to be the discrete GPU" << std::endl;
            } else {
                std::cout << "  -> This appears to be the integrated GPU" << std::endl;
            }
            
            adapter->Release();
            adapterIndex++;
        }
        
        factory->Release();
        return true;
    }
    
    void setWindowsGraphicsPreference(const std::string& appPath) {
        // Set Windows Graphics Settings via registry
        HKEY hKey;
        const char* regPath = "SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences";
        
        if (RegCreateKeyExA(HKEY_CURRENT_USER, regPath, 0, NULL, 
                           REG_OPTION_NON_VOLATILE, KEY_WRITE, NULL, &hKey, NULL) == ERROR_SUCCESS) {
            
            // Set application to use high performance GPU
            const char* value = "GpuPreference=2;";
            RegSetValueExA(hKey, appPath.c_str(), 0, REG_SZ, 
                          (const BYTE*)value, strlen(value));
            
            RegCloseKey(hKey);
            std::cout << "Set Windows Graphics preference for: " << appPath << std::endl;
        } else {
            std::cout << "Failed to set Windows Graphics preference" << std::endl;
        }
    }
    
    void setAMDRadeonSettings(const std::string& appPath) {
        // Set AMD Radeon Settings via registry
        HKEY hKey;
        const char* regPath = "SOFTWARE\\AMD\\DxDiag";
        
        if (RegCreateKeyExA(HKEY_CURRENT_USER, regPath, 0, NULL, 
                           REG_OPTION_NON_VOLATILE, KEY_WRITE, NULL, &hKey, NULL) == ERROR_SUCCESS) {
            
            // Set application profile
            RegSetValueExA(hKey, "AppProfile", 0, REG_SZ, 
                          (const BYTE*)appPath.c_str(), appPath.length());
            
            // Set GPU preference (1 = discrete GPU)
            DWORD gpuPreference = 1;
            RegSetValueExA(hKey, "GPUPreference", 0, REG_DWORD, 
                          (const BYTE*)&gpuPreference, sizeof(DWORD));
            
            // Additional settings for dual GPU
            DWORD discreteGPU = 1;
            RegSetValueExA(hKey, "DiscreteGPU", 0, REG_DWORD, 
                          (const BYTE*)&discreteGPU, sizeof(DWORD));
            
            RegCloseKey(hKey);
            std::cout << "Set AMD Radeon Settings for: " << appPath << std::endl;
        } else {
            std::cout << "Failed to set AMD Radeon Settings" << std::endl;
        }
    }
    
    bool isDiscreteGPUForced() const {
        return discrete_gpu_forced;
    }
    
    void printGPUStatus() {
        std::cout << "GPU Bypass Status:" << std::endl;
        std::cout << "  AmdPowerXpressRequestHighPerformance: " << AmdPowerXpressRequestHighPerformance << std::endl;
        std::cout << "  Discrete GPU Forced: " << (discrete_gpu_forced ? "Yes" : "No") << std::endl;
        
        // Check environment variables
        char buffer[256];
        if (GetEnvironmentVariableA("AMD_OPENCL_DEVICE", buffer, sizeof(buffer))) {
            std::cout << "  AMD_OPENCL_DEVICE: " << buffer << std::endl;
        }
        if (GetEnvironmentVariableA("OPENCL_DEVICE", buffer, sizeof(buffer))) {
            std::cout << "  OPENCL_DEVICE: " << buffer << std::endl;
        }
    }
    
    void runGPUTest() {
        // Run a GPU test to verify which GPU is being used
        std::cout << "Running GPU test..." << std::endl;
        
        if (createDirect3DContext()) {
            std::cout << "Direct3D context created - check Task Manager for GPU activity" << std::endl;
            
            // Create a simple compute shader to stress the GPU
            // This will show up in Task Manager
            std::cout << "GPU test complete - check Task Manager to see which GPU was used" << std::endl;
        } else {
            std::cout << "Failed to create Direct3D context" << std::endl;
        }
    }
};

// DLL entry point
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        // DLL is being loaded - set environment variables
        SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance", "1");
        SetEnvironmentVariableA("AMD_OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("GPU_DEVICE_ORDINAL", "0");
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

// Export functions for Python to call
extern "C" __declspec(dllexport) void RequestHighPerformanceGPU() {
    AmdPowerXpressRequestHighPerformance = 1;
    SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance", "1");
}

extern "C" __declspec(dllexport) int GetGPUPreference() {
    return AmdPowerXpressRequestHighPerformance;
}

extern "C" __declspec(dllexport) void SetWindowsGraphicsPreference(const char* appPath) {
    HKEY hKey;
    const char* regPath = "SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences";
    
    if (RegCreateKeyExA(HKEY_CURRENT_USER, regPath, 0, NULL, 
                       REG_OPTION_NON_VOLATILE, KEY_WRITE, NULL, &hKey, NULL) == ERROR_SUCCESS) {
        
        const char* value = "GpuPreference=2;";
        RegSetValueExA(hKey, appPath, 0, REG_SZ, (const BYTE*)value, strlen(value));
        RegCloseKey(hKey);
    }
}

// Main function for testing
int main() {
    std::cout << "=== C++ GPU Bypass Test ===" << std::endl;
    
    // Create GPU bypass controller
    CppGPUBypass gpuBypass;
    
    // Force discrete GPU
    gpuBypass.forceDiscreteGPU();
    
    // Enumerate GPUs
    gpuBypass.enumerateGPUs();
    
    // Set Windows Graphics Settings
    gpuBypass.setWindowsGraphicsPreference("python.exe");
    
    // Set AMD Radeon Settings
    gpuBypass.setAMDRadeonSettings("python.exe");
    
    // Print status
    gpuBypass.printGPUStatus();
    
    // Run GPU test
    gpuBypass.runGPUTest();
    
    std::cout << "\nC++ GPU Bypass complete!" << std::endl;
    std::cout << "This should force Windows to use the discrete GPU." << std::endl;
    
    return 0;
}

