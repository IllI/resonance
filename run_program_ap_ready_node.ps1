# Program AP ready-node launcher.
# TRC/Egress discipline:
# - Use only an already READY TRC-covered TPU VM.
# - Upload only program_ap_tunnel_eigen_recovery.py plus explicit taichi_*.png inputs.
# - Do not download Europe result folders; inspect summaries remotely.

$Project = "time-emission"
$Zone = if ($env:PROGRAM_AP_ZONE) { $env:PROGRAM_AP_ZONE } else { "europe-west4-a" }
$NodeId = if ($env:PROGRAM_AP_NODE) { $env:PROGRAM_AP_NODE } else { "program-ap-eu-node-v1" }
$ProgramMode = if ($env:PROGRAM_AP_MODE) { $env:PROGRAM_AP_MODE } else { "--ap-s" }
$ProgramExtraArgs = if ($env:PROGRAM_AP_EXTRA_ARGS) { $env:PROGRAM_AP_EXTRA_ARGS } else { "" }
$SrcDir = "$PSScriptRoot\emergent_quantum_geometries"
$YinyangReferenceDir = "$PSScriptRoot\tpu_previews\yinyang_reference_v2"
$RemoteDir = "/home/cityz/program_ap"
$RemoteOutDir = "$RemoteDir/program_ap_results"
$LogFile = "prog_ap_run.log"

$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"

$AllowedZones = @("us-east1-d", "europe-west4-a")
if ($Zone -notin $AllowedZones) {
    Write-Host "[ERROR] Zone $Zone is not in the current AP v6e spot allow-list."
    exit 1
}

$AllowedModes = @("--ap-focused", "--ap-lite", "--ap-r", "--ap-s", "--ap-s-mech", "--ap-s-fidelity", "--ap-t", "--ap-u", "--aq-0", "--aq-0b", "--aq-1a-tf", "--aq-1a-tf-s29", "--aq-1b", "--aq-1c", "--aq-seq-0", "--aq-img-0", "--aq-img-1", "--aq-img-1-lite", "--aq-fft-0", "--aq-fft-1", "--aq-fft-2", "--aq-fft-3", "--aq-fft-4", "--aq-fft-5", "--aq-fft-6", "--aq-fft-7", "--aq-yinyang-0", "--aq-yinyang-1", "--aq-yinyang-2", "--aq-yinyang-3", "--aq-yinyang-4", "--aq-dna-0", "--aq-hybrid-0", "--as-0", "--as-0b")
if ($ProgramMode -notin $AllowedModes) {
    Write-Host "[ERROR] PROGRAM_AP_MODE must be one of: $($AllowedModes -join ', ')"
    exit 1
}
if (-not [string]::IsNullOrWhiteSpace($ProgramExtraArgs) -and $ProgramExtraArgs -match '[;&|`<>$]') {
    Write-Host "[ERROR] PROGRAM_AP_EXTRA_ARGS contains shell control characters."
    exit 1
}

Write-Host "[CHECK] Verifying ready TPU VM: $NodeId in $Zone"
$NodeState = (gcloud compute tpus tpu-vm describe $NodeId --project=$Project --zone=$Zone --format="value(state)" 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($NodeState)) {
    Write-Host "[ERROR] TPU VM not found. Queue a TRC v6e spot node first."
    exit 1
}
$NodeState = $NodeState.Trim()
if ($NodeState -ne "READY") {
    Write-Host "[ERROR] TPU VM state is $NodeState, expected READY."
    exit 1
}

Write-Host "[SSH] Preparing remote workspace..."
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command="mkdir -p $RemoteDir/emergent_quantum_geometries $RemoteOutDir"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] SSH workspace setup failed."
    exit 1
}

Write-Host "[SSH] Verifying TPU-backed JAX runtime..."
$RuntimeCheck = "python3 -c 'import jax; print(jax.__version__, jax.default_backend(), jax.devices()); assert jax.default_backend() == ""tpu""'"
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command=$RuntimeCheck
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] TPU JAX runtime check failed."
    exit 1
}

Write-Host "[SCP] Uploading Program AP source only..."
"y" | gcloud compute tpus tpu-vm scp "$SrcDir\program_ap_tunnel_eigen_recovery.py" "$NodeId`:$RemoteDir/emergent_quantum_geometries/" --project=$Project --zone=$Zone --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Source upload failed."
    exit 1
}

if (Test-Path $YinyangReferenceDir) {
    $ReferencePngs = Get-ChildItem -Path $YinyangReferenceDir -Filter "taichi_*.png" -File
    if ($ReferencePngs.Count -gt 0) {
        Write-Host "[SCP] Uploading explicit yin-yang reference PNG inputs..."
        "y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command="mkdir -p $RemoteDir/yinyang_reference_v2"
        foreach ($png in $ReferencePngs) {
            "y" | gcloud compute tpus tpu-vm scp $png.FullName "$NodeId`:$RemoteDir/yinyang_reference_v2/" --project=$Project --zone=$Zone --quiet
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[ERROR] Reference PNG upload failed: $($png.Name)"
                exit 1
            }
        }
    }
}

Write-Host "[RUN] Launching Program AP on TPU with mode $ProgramMode..."
$GitCommit = (git -C $PSScriptRoot rev-parse --short HEAD 2>$null)
if ([string]::IsNullOrWhiteSpace($GitCommit)) {
    $GitCommit = "unknown"
}
$RemoteTimeout = if ($env:PROGRAM_AP_TIMEOUT_SECONDS) { $env:PROGRAM_AP_TIMEOUT_SECONDS } elseif ($ProgramMode -like "*-lite") { "900" } else { "" }
$PythonCmd = "python3 -u $RemoteDir/emergent_quantum_geometries/program_ap_tunnel_eigen_recovery.py --require-tpu $ProgramMode --out-dir $RemoteOutDir --git-commit $GitCommit $ProgramExtraArgs"
if (-not [string]::IsNullOrWhiteSpace($RemoteTimeout)) {
    Write-Host "[RUN] Applying remote timeout guard: ${RemoteTimeout}s"
    $PythonCmd = "timeout ${RemoteTimeout}s $PythonCmd"
}
$RunCmd = "cd $RemoteDir; rm -f $LogFile; PROGRAM_AP_GIT_COMMIT=$GitCommit nohup $PythonCmd > $RemoteDir/$LogFile 2>&1 & echo PROGRAM_AP_PID=`$!"
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command=$RunCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Program launch failed."
    exit 1
}

Write-Host "[MONITOR] Waiting for initial compile output..."
Start-Sleep -Seconds 45
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command="tail -n 80 $RemoteDir/$LogFile 2>/dev/null || true"

Write-Host "[DONE] Program AP launched. No result files downloaded."
