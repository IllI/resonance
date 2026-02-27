// C++ GPU Control - Direct GPU selection without Windows interference
// This addresses the issue where Windows overrides programmatic GPU selection

#include <windows.h>
#include <iostream>
#include <vector>
#include <string>

// Export the symbol that tells AMD drivers to use discrete GPU
// This is the key method from AMD's switchable graphics technology
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

// Additional exports for different scenarios
extern "C" __declspec(dllexport) int NvOptimusEnablement = 1;  // For NVIDIA compatibility
extern "C" __declspec(dllexport) int DXVK_HUD = 0;  // Disable DXVK HUD

class CppGPUController {
private:
    bool discrete_gpu_forced;
    
public:
    CppGPUController() : discrete_gpu_forced(false) {
        // Set environment variables when the class is created
        SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance", "1");
        SetEnvironmentVariableA("AMD_OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("OPENCL_DEVICE", "0");
        SetEnvironmentVariableA("GPU_DEVICE_ORDINAL", "0");
        SetEnvironmentVariableA("AMD_DISABLE_INTEGRATED_GPU", "1");
        SetEnvironmentVariableA("OPENCL_DISABLE_DEVICE_1", "1");
        
        std::cout << "C++ GPU Controller initialized" << std::endl;
        std::cout << "Set AMD power export and environment variables" << std::endl;
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
        std::cout << "GPU Control Status:" << std::endl;
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
    std::cout << "=== C++ GPU Control Test ===" << std::endl;
    
    // Create GPU controller
    CppGPUController gpuController;
    
    // Force discrete GPU
    gpuController.forceDiscreteGPU();
    
    // Set Windows Graphics Settings
    gpuController.setWindowsGraphicsPreference("python.exe");
    
    // Set AMD Radeon Settings
    gpuController.setAMDRadeonSettings("python.exe");
    
    // Print status
    gpuController.printGPUStatus();
    
    std::cout << "\nC++ GPU Control complete!" << std::endl;
    std::cout << "This should force Windows to use the discrete GPU." << std::endl;
    
    return 0;
}

