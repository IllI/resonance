
# Force GPU Selection PowerShell Script
Write-Host "Forcing RX 7700S GPU usage..."

# Set environment variables
$env:AmdPowerXpressRequestHighPerformance = "1"
$env:AMD_OPENCL_DEVICE = "1"
$env:OPENCL_DEVICE = "1"

# Set Windows Graphics Settings via PowerShell
$pythonPath = "' + self.python_path + '"
$regPath = "HKCU:\SOFTWARE\Microsoft\DirectX\UserGpuPreferences"
Set-ItemProperty -Path $regPath -Name $pythonPath -Value "GpuPreference=2;" -Force

Write-Host "GPU forcing complete. Restart your application."
