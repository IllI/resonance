# Force GPU Selection - Run as Administrator
# This script forces Windows to use the discrete GPU for Python applications

Write-Host "=== Force GPU Selection (Administrator) ===" -ForegroundColor Green
Write-Host "This script will force Windows to use RX 7700S for Python applications" -ForegroundColor Yellow

# Get Python executable path
$pythonPath = (Get-Command python).Source
Write-Host "Python path: $pythonPath" -ForegroundColor Cyan

# Set environment variables
Write-Host "Setting environment variables..." -ForegroundColor Yellow
$env:AmdPowerXpressRequestHighPerformance = "1"
$env:AMD_OPENCL_DEVICE = "0"  # Try device 0 (RX 7700S)
$env:OPENCL_DEVICE = "0"
$env:GPU_DEVICE_ORDINAL = "0"

# Set Windows Graphics Settings
Write-Host "Setting Windows Graphics Settings..." -ForegroundColor Yellow
$regPath = "HKCU:\SOFTWARE\Microsoft\DirectX\UserGpuPreferences"

# Create the registry key if it doesn't exist
if (!(Test-Path $regPath)) {
    New-Item -Path $regPath -Force | Out-Null
}

# Set Python to use high performance GPU
Set-ItemProperty -Path $regPath -Name $pythonPath -Value "GpuPreference=2;" -Force
Write-Host "Set Windows Graphics preference for Python" -ForegroundColor Green

# Set for common Python executables
$pythonVariants = @(
    "python.exe",
    "python3.exe",
    "pythonw.exe",
    "python3w.exe"
)

foreach ($variant in $pythonVariants) {
    $fullPath = Join-Path (Split-Path $pythonPath) $variant
    if (Test-Path $fullPath) {
        Set-ItemProperty -Path $regPath -Name $fullPath -Value "GpuPreference=2;" -Force
        Write-Host "Set preference for $variant" -ForegroundColor Green
    }
}

# Set AMD Radeon Settings
Write-Host "Setting AMD Radeon Settings..." -ForegroundColor Yellow
$amdRegPath = "HKCU:\SOFTWARE\AMD\DxDiag"

if (!(Test-Path $amdRegPath)) {
    New-Item -Path $amdRegPath -Force | Out-Null
}

Set-ItemProperty -Path $amdRegPath -Name "AppProfile" -Value $pythonPath -Force
Set-ItemProperty -Path $amdRegPath -Name "GPUPreference" -Value 1 -Type DWord -Force
Write-Host "Set AMD Radeon Settings for high performance" -ForegroundColor Green

# Set system-wide environment variables
Write-Host "Setting system-wide environment variables..." -ForegroundColor Yellow
[Environment]::SetEnvironmentVariable("AmdPowerXpressRequestHighPerformance", "1", "User")
[Environment]::SetEnvironmentVariable("AMD_OPENCL_DEVICE", "0", "User")
[Environment]::SetEnvironmentVariable("OPENCL_DEVICE", "0", "User")

Write-Host "Set system-wide environment variables" -ForegroundColor Green

# Create a test script
$testScript = @"
import pyopencl as cl
import numpy as np
import time

print("Testing GPU usage...")
platforms = cl.get_platforms()
for platform in platforms:
    devices = platform.get_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device.name}")
        
        if 'gfx1103' in device.name.lower():  # RX 7700S
            print(f"Testing RX 7700S (Device {i})...")
            try:
                context = cl.Context([device])
                queue = cl.CommandQueue(context)
                
                size = 1024 * 1024 * 4
                a = np.random.rand(size).astype(np.float32)
                b = np.random.rand(size).astype(np.float32)
                
                a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
                
                kernel_code = '''
                __kernel void test_kernel(__global const float* a,
                                        __global const float* b,
                                        __global float* c) {
                    int gid = get_global_id(0);
                    float temp = a[gid] * b[gid];
                    
                    for(int i = 0; i < 10000; i++) {
                        temp = sqrt(temp);
                        temp = sin(temp) + cos(temp);
                        temp = temp * temp;
                    }
                    c[gid] = temp;
                }
                '''
                
                program = cl.Program(context, kernel_code).build()
                kernel = program.test_kernel
                
                print("Running intensive workload...")
                print("Check Task Manager - RX 7700S (GPU 0) should show activity")
                
                for i in range(5):
                    print(f"Iteration {i+1}/5...")
                    kernel(queue, (size,), None, a_buffer, b_buffer, c_buffer)
                    queue.finish()
                    time.sleep(1)
                
                print("Test complete - check Task Manager results")
                break
                
            except Exception as e:
                print(f"Error: {e}")
"@

$testScript | Out-File -FilePath "gpu_test.py" -Encoding UTF8
Write-Host "Created gpu_test.py for testing" -ForegroundColor Green

Write-Host "`n=== Configuration Complete ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your Python/IDE" -ForegroundColor White
Write-Host "2. Run: python gpu_test.py" -ForegroundColor White
Write-Host "3. Check Task Manager while it runs" -ForegroundColor White
Write-Host "4. RX 7700S (GPU 0) should show activity, not 780M (GPU 1)" -ForegroundColor White

Write-Host "`nPress any key to continue..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

