
# Manual GPU Setup Guide

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

## Solution 3: Registry (Advanced)
Run this PowerShell command as Administrator:
Set-ItemProperty -Path "HKCU:\SOFTWARE\Microsoft\DirectX\UserGpuPreferences" -Name "C:\Users\cityz\AppData\Local\Programs\Python\Python310\python.exe" -Value "GpuPreference=2;" -Force

## Solution 4: C++ DLL (Most Reliable)
Compile the amd_gpu_force_dll.cpp and load it in your Python application.

## Verification
After setup, run: python windows_gpu_override_solution.py
Check Task Manager - GPU 0 (RX 7700S) should show activity, not GPU 1 (780M)

## If All Else Fails
Consider rewriting the application in C++ with direct GPU control.
