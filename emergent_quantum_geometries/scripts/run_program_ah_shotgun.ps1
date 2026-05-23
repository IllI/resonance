# run_program_ah_shotgun.ps1  --  Program AH TPU Orchestrator
# Project: time-emission
# Target: emergent_quantum_geometries/program_ah_tpu.py

param(
    [switch]$SmokeTest
)

$Project   = "time-emission"
$SrcDir    = Split-Path $PSScriptRoot -Parent
$RemoteDir = "/home/cityz/program_ah"
$LogFile   = "prog_ah_run.log"

$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}

# Auto-confirm gcloud prompts
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"

# Target config (v6e-8 in us-east1-d)
$QRName  = "chronos-v6e-ah"
$NodeId  = "chronos-v6e-ah-node"
$Zone    = "europe-west4-a"
$Type    = "v6e-8"
$Runtime = "v2-alpha-tpuv6e"

function Get-QueuedResourceState {
    $descJson = gcloud compute tpus queued-resources describe $QRName `
        --project=$Project --zone=$Zone --format=json 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($descJson)) { return "MISSING" }
    try {
        $desc = $descJson | ConvertFrom-Json
        if ($desc.state -and $desc.state.state) { return [string]$desc.state.state }
        if ($desc.state) { return [string]$desc.state }
    } catch { return "UNKNOWN" }
    return "UNKNOWN"
}

function Cleanup-Tpu {
    Write-Host "[CLEANUP] Cleaning up existing TPU resources..."
    gcloud compute tpus tpu-vm delete $NodeId --project=$Project --zone=$Zone --quiet 2>$null
    gcloud compute tpus queued-resources delete $QRName --project=$Project --zone=$Zone --quiet 2>$null
    Start-Sleep -Seconds 5
}

# 1. Check existing state
$state = Get-QueuedResourceState
if ($state -eq "ACTIVE") {
    $nodeState = (gcloud compute tpus tpu-vm describe $NodeId --project=$Project --zone=$Zone --format="value(state)" 2>$null)
    if ($nodeState) { $nodeState = $nodeState.Trim() }
    if ($nodeState -eq "PREEMPTED" -or $nodeState -eq "TERMINATED") {
        Write-Host "[SETUP] QR is ACTIVE but Node is $nodeState. Re-creating."
        Cleanup-Tpu
    } else {
        Write-Host "[SETUP] Found active TPU VM. Reusing existing resource."
    }
} elseif ($state -match "PROVISIONING|ACCEPTED|WAITING_FOR_RESOURCES") {
    Write-Host "[SETUP] TPU VM already exists and is allocating: $state. Reusing allocation."
} else {
    if ($state -ne "MISSING") {
        Cleanup-Tpu
    }
}

# 2. Allocate TPU VM using Queued Resource API
$currentState = Get-QueuedResourceState
if ($currentState -eq "MISSING") {
    Write-Host "[ALLOCATE] Requesting fresh TPU VM ($Type in $Zone)..."
    gcloud compute tpus queued-resources create $QRName `
        --node-id=$NodeId `
        --project=$Project `
        --zone=$Zone `
        --accelerator-type=$Type `
        --runtime-version=$Runtime `
        --spot `
        --quiet

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Allocation command failed!"
        exit 1
    }
} else {
    Write-Host "[SETUP] Queued resource already exists in state: $currentState. Polling."
}

if ((Get-QueuedResourceState) -ne "ACTIVE") {
    Write-Host "[POLL] Waiting for TPU to become ACTIVE..."
    $timeout = (Get-Date).AddMinutes(20)
    while ((Get-Date) -lt $timeout) {
        $state = Get-QueuedResourceState
        Write-Host "  State: $state"
        if ($state -eq "ACTIVE") { break }
        if ($state -match "FAILED|SUSPENDED|MISSING") {
            Write-Host "[ERROR] TPU allocation failed or suspended!"
            exit 1
        }
        Start-Sleep -Seconds 20
    }
    if ((Get-QueuedResourceState) -ne "ACTIVE") {
        Write-Host "[ERROR] Timeout waiting for ACTIVE state!"
        exit 1
    }
}

Write-Host "[WIN] TPU VM is active!"

# 3. Wait for SSH to become ready
Write-Host "[SSH] Probing SSH daemon..."
$sshReady = $false
$sshTimeout = (Get-Date).AddMinutes(5)
while (-not $sshReady -and (Get-Date) -lt $sshTimeout) {
    "y" | gcloud compute tpus tpu-vm ssh $NodeId `
        --project=$Project --zone=$Zone `
        --command="echo SSH_READY" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $sshReady = $true }
    else { Start-Sleep -Seconds 10 }
}
if (-not $sshReady) {
    Write-Host "[ERROR] SSH failed to connect within 5 minutes!"
    exit 1
}
Write-Host "[SSH] Connection established."

# 4. Install Dependencies
Write-Host "[DEPS] Installing JAX TPU backend..."
"y" | gcloud compute tpus tpu-vm ssh $NodeId `
    --project=$Project --zone=$Zone `
    --command="mkdir -p $RemoteDir/emergent_quantum_geometries && pip install -q -U 'jax[tpu]' scipy numpy -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && python3 -c 'import jax; print(jax.default_backend(), jax.devices())'"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dependency installation failed!"
    exit 1
}

# 5. SCP Files
Write-Host "[SCP] Uploading code files..."
$localPath = Join-Path $SrcDir "*.py"
$remotePath = "$NodeId`:$RemoteDir`/emergent_quantum_geometries`/"
"y" | gcloud compute tpus tpu-vm scp $localPath $remotePath --project=$Project --zone=$Zone 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] SCP failed!"
    exit 1
}
Write-Host "[SCP] Upload complete."

# 6. Execute Experiment
Write-Host "[EXECUTE] Launching Flagship Latent Semantic Normal Modes (Program AH) experiment..."
$argsStr = "--require-tpu --out-dir $RemoteDir/program_ah_results"
if ($SmokeTest) {
    $argsStr += " --smoke-test"
}
$RunCmd = "export OMP_NUM_THREADS=64; nohup python3 -u $RemoteDir/emergent_quantum_geometries/program_ah_tpu.py $argsStr > $RemoteDir/$LogFile 2>&1 &"
"y" | gcloud compute tpus tpu-vm ssh $NodeId `
    --project=$Project --zone=$Zone `
    --command=$RunCmd

# 7. Monitor execution logs
Write-Host "[MONITOR] Live stream initialized (refreshing every 30s):"
$localOutDir = Join-Path $SrcDir "program_ah_results"
New-Item -ItemType Directory -Force -Path $localOutDir | Out-Null

$StartTime = Get-Date
$RunTimeout = $StartTime.AddHours(2)
while ((Get-Date) -lt $RunTimeout) {
    Start-Sleep -Seconds 30
    
    # Sync intermediate results and log files in flight
    $remoteResult = "$NodeId`:$RemoteDir`/program_ah_results`/program_ah_summary.json"
    $remoteLog    = "$NodeId`:$RemoteDir`/$LogFile"
    "y" | gcloud compute tpus tpu-vm scp $remoteResult "$localOutDir\program_ah_summary.json" --project=$Project --zone=$Zone 2>&1 | Out-Null
    "y" | gcloud compute tpus tpu-vm scp $remoteLog "$localOutDir\program_ah_run.log" --project=$Project --zone=$Zone 2>&1 | Out-Null
    
    # Check preemption
    $nodeState = (gcloud compute tpus tpu-vm describe $NodeId --project=$Project --zone=$Zone --format="value(state)" 2>$null)
    if ($nodeState -and $nodeState.Trim() -match "PREEMPTED|TERMINATED") {
        Write-Host "[ERROR] Node was PREEMPTED by Google mid-execution!"
        exit 1
    }

    # Tail the remote log file
    $tail = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
        --project=$Project --zone=$Zone `
        --command="tail -n 8 $RemoteDir/$LogFile 2>/dev/null" 2>$null
    Write-Host "[$([math]::Round(((Get-Date)-$StartTime).TotalMinutes,1))m] -------------------"
    Write-Host $tail
    
    # Check if complete
    $doneRaw = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
        --project=$Project --zone=$Zone `
        --command="grep -c 'Abl-Phase' $RemoteDir/$LogFile 2>/dev/null || echo 0" 2>$null
    $doneVal = if ($doneRaw -is [array]) { $doneRaw[-1] } else { $doneRaw }
    # Look for the final table header to confirm success
    if ($doneVal -ne $null -and [int]($doneVal.Trim()) -gt 0) {
        Write-Host "[SUCCESS] Experiment completed successfully!"
        break
    }
    
    # Check if process is still alive
    $isRunning = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
        --project=$Project --zone=$Zone `
        --command="pgrep -f program_ah_tpu.py >/dev/null && echo 1 || echo 0" 2>$null
    $isRunningVal = if ($isRunning -is [array]) { $isRunning[-1] } else { $isRunning }
    if ($isRunningVal -ne $null -and [int]($isRunningVal.Trim()) -eq 0) {
        Write-Host "[ERROR] Python script crashed silently!"
        $errorLog = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
            --project=$Project --zone=$Zone `
            --command="tail -n 50 $RemoteDir/$LogFile" 2>$null
        Write-Host $errorLog
        exit 1
    }
}

# 8. Download Results
Write-Host "[DOWNLOAD] Fetching results to local workspace..."
$remoteResult = "$NodeId`:$RemoteDir`/program_ah_results`/program_ah_summary.json"
$remoteLog    = "$NodeId`:$RemoteDir`/$LogFile"

$localOutDir = Join-Path $SrcDir "program_ah_results"
New-Item -ItemType Directory -Force -Path $localOutDir | Out-Null
"y" | gcloud compute tpus tpu-vm scp $remoteResult "$localOutDir\program_ah_summary.json" --project=$Project --zone=$Zone
"y" | gcloud compute tpus tpu-vm scp $remoteLog "$localOutDir\program_ah_run.log" --project=$Project --zone=$Zone

Write-Host "[COMPLETE] Results downloaded successfully! Cleaning up TPU resources..."
Cleanup-Tpu
Write-Host "[DONE]"
