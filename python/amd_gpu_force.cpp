
#include <windows.h>

// Export the symbol that tells AMD drivers to use discrete GPU
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    return TRUE;
}

