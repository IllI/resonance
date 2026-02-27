
# Dual AMD GPU Setup Script
# Based on web research for dual AMD GPU configurations

Write-Host "=== Dual AMD GPU Setup Script ===" -ForegroundColor Green
Write-Host "Configuring system for dual AMD GPU setup" -ForegroundColor Yellow

# Set environment variables
$env:AmdPowerXpressRequestHighPerformance = "1"
$env:AMD_OPENCL_DEVICE = "0"
$env:OPENCL_DEVICE = "0"
$env:GPU_DEVICE_ORDINAL = "0"
$env:AMD_DISABLE_INTEGRATED_GPU = "1"
$env:OPENCL_DISABLE_DEVICE_1 = "1"

Write-Host "Set environment variables" -ForegroundColor Green

# Set Windows Graphics Settings
$pythonPath = "C:\Users\cityz\IllI\newer_all\venv_fmri\Scripts\python.exe"
$regPath = "HKCU:\SOFTWARE\Microsoft\DirectX\UserGpuPreferences"

# Create the registry key if it doesn't exist
if (!(Test-Path $regPath)) {
    New-Item -Path $regPath -Force | Out-Null
}

# Set Python to use high performance GPU
Set-ItemProperty -Path $regPath -Name $pythonPath -Value "GpuPreference=2;" -Force
Write-Host "Set Windows Graphics preference for Python" -ForegroundColor Green

# Set for common Python variants
$pythonDir = Split-Path $pythonPath
$pythonVariants = @("python.exe", "python3.exe", "pythonw.exe", "python3w.exe")

foreach ($variant in $pythonVariants) {
    $fullPath = Join-Path $pythonDir $variant
    if (Test-Path $fullPath) {
        Set-ItemProperty -Path $regPath -Name $fullPath -Value "GpuPreference=2;" -Force
        Write-Host "Set preference for $variant" -ForegroundColor Green
    }
}

# Set AMD Radeon Settings
$amdRegPath = "HKCU:\SOFTWARE\AMD\DxDiag"

if (!(Test-Path $amdRegPath)) {
    New-Item -Path $amdRegPath -Force | Out-Null
}

Set-ItemProperty -Path $amdRegPath -Name "AppProfile" -Value $pythonPath -Force
Set-ItemProperty -Path $amdRegPath -Name "GPUPreference" -Value 1 -Type DWord -Force
Set-ItemProperty -Path $amdRegPath -Name "DiscreteGPU" -Value 1 -Type DWord -Force
Set-ItemProperty -Path $amdRegPath -Name "ForceDiscreteGPU" -Value 1 -Type DWord -Force
Write-Host "Set AMD Radeon Settings for high performance" -ForegroundColor Green

Write-Host "`n=== Configuration Complete ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your Python/IDE" -ForegroundColor White
Write-Host "2. Run: python dual_amd_gpu_solution.py" -ForegroundColor White
Write-Host "3. Check Task Manager while it runs" -ForegroundColor White
Write-Host "4. RX 7700S (GPU 0) should show activity, not 780M (GPU 1)" -ForegroundColor White

Write-Host "`nPress any key to continue..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
