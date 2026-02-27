
# Manual GPU Setup Guide (No Compilation Required)

## The Problem
Windows overrides programmatic GPU selection, causing applications to use the integrated GPU (780M) instead of the discrete GPU (RX 7700S).

## Solution 1: Windows Graphics Settings (Most Reliable)
1. Press Win + I
2. Go to System → Display
3. Scroll down and click "Graphics settings"
4. Click "Browse" and find: C:\Users\cityz\AppData\Local\Programs\Python\Python310\python.exe
5. Set to "High performance"
6. Click "Save"
7. Restart Python/IDE

## Solution 2: AMD Radeon Settings
1. Right-click desktop → AMD Radeon Settings
2. Go to System → Switchable Graphics
3. Find Python in the list (or add it)
4. Set to "High Performance"
5. Restart Python/IDE

## Solution 3: PowerShell Script (Run as Administrator)
1. Right-click PowerShell and "Run as Administrator"
2. Navigate to your project directory
3. Run: .\gpu_force_script.ps1
4. Restart Python/IDE

## Solution 4: Registry (Advanced)
Run this PowerShell command as Administrator:
Set-ItemProperty -Path "HKCU:\SOFTWARE\Microsoft\DirectX\UserGpuPreferences" -Name "C:\Users\cityz\AppData\Local\Programs\Python\Python310\python.exe" -Value "GpuPreference=2;" -Force

## Verification
After setup, run: python no_compile_gpu_solution.py
Check Task Manager - GPU 0 (RX 7700S) should show activity, not GPU 1 (780M)

## If All Else Fails
Consider using a different approach:
1. Use CUDA instead of OpenCL (if available)
2. Use a different Python environment
3. Use a different machine with better GPU support
4. Consider cloud computing with proper GPU support
