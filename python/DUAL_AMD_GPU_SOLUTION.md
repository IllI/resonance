
# Comprehensive Dual AMD GPU Solution Guide

## The Problem
Windows defaults to integrated GPU (780M) instead of discrete GPU (RX 7700S) in dual AMD GPU setups.

## Solution 1: Windows Graphics Settings (Most Reliable)
1. Press Win + I
2. Go to System → Display
3. Scroll down and click "Graphics settings"
4. Click "Browse" and find: C:\Users\cityz\IllI\newer_all\venv_fmri\Scripts\python.exe
5. Set to "High performance"
6. Click "Save"
7. Restart Python/IDE

## Solution 2: AMD Radeon Settings
1. Right-click desktop → AMD Radeon Settings
2. Go to Gaming tab
3. Click "Add" to include your application
4. Set Graphics Profile to "High Performance"
5. Restart Python/IDE

## Solution 3: Check Display Connections
- Ensure your primary monitor is connected to RX 7700S
- Windows prioritizes the GPU connected to the primary display
- If monitor is connected to integrated GPU, apps may default to it

## Solution 4: Update Drivers
- Visit AMD's website for latest RX 7700S drivers
- Ensure integrated GPU drivers are also up to date
- Outdated drivers can cause GPU selection issues

## Solution 5: Disable Integrated GPU (If Necessary)
- Press Win + X and select Device Manager
- Expand Display adapters
- Right-click on integrated GPU (780M) and select Disable device
- This forces the system to use the discrete GPU exclusively
- Be cautious as this may affect other functionalities

## Solution 6: PowerShell Script (Run as Administrator)
1. Right-click PowerShell and "Run as Administrator"
2. Navigate to your project directory
3. Run: .\dual_amd_gpu_script.ps1
4. Restart Python/IDE

## Verification
After setup, run: python dual_amd_gpu_solution.py
Check Task Manager - GPU 0 (RX 7700S) should show activity, not GPU 1 (780M)

## If All Else Fails
Consider these alternatives:
1. Use CUDA instead of OpenCL (if available)
2. Use a different Python environment
3. Use cloud computing with proper GPU support
4. Consider using a different machine with better GPU support
