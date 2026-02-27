
#include <windows.h>

// Export the symbol that tells AMD drivers to use discrete GPU
// This is the key method from AMD's switchable graphics technology
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

// Additional exports for different scenarios
extern "C" __declspec(dllexport) int NvOptimusEnablement = 1;  // For NVIDIA compatibility
extern "C" __declspec(dllexport) int DXVK_HUD = 0;  // Disable DXVK HUD

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

// Function to explicitly request high performance GPU
extern "C" __declspec(dllexport) void RequestHighPerformanceGPU() {
    AmdPowerXpressRequestHighPerformance = 1;
    SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance", "1");
}

// Function to get current GPU preference
extern "C" __declspec(dllexport) int GetGPUPreference() {
    return AmdPowerXpressRequestHighPerformance;
}
