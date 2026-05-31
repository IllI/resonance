# run_program_am_shotgun.ps1  --  Program AM TPU Orchestrator
# Project: time-emission
# Target: emergent_quantum_geometries/program_am_semantic_topology.py

$Project   = "time-emission"
$SrcDir    = "$PSScriptRoot\emergent_quantum_geometries"
$RemoteDir = "/home/cityz/program_am"
$LogFile   = "prog_am_run.log"
$RemoteOutDir = "$RemoteDir/program_am_results"

$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}

# Auto-confirm gcloud prompts
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"

# Target config. TRC-covered quota includes 64 spot Cloud TPU v6e chips in
# europe-west4-a and us-east1-d; v6e-8 consumes 8 of those spot chips.
$QRName  = "program-am-qr-v6"
$NodeId  = "program-am-node-v6"
$Zone    = "us-east1-d"
$Type    = "v6e-8"
$Runtime = "v2-alpha-tpuv6e"
$TargetZones = @("us-east1-d", "europe-west4-a")

$AllowedTrcV6eSpotZones = @("us-east1-d", "europe-west4-a")
if ($Type -notmatch "^v6e-" -or $Runtime -ne "v2-alpha-tpuv6e") {
    Write-Host "[ERROR] Program AM is configured for a non-TRC v6e runtime/type. See TRC_TPU_RULES.md."
    exit 1
}
foreach ($targetZone in $TargetZones) {
    if ($targetZone -notin $AllowedTrcV6eSpotZones) {
        Write-Host "[ERROR] Zone $targetZone is not TRC-covered for spot v6e Program AM. See TRC_TPU_RULES.md."
        exit 1
    }
}

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

# 1-2. Queue TPU VMs in parallel using only TRC-covered spot v6e zones.
$allocated = $false
$WinningZone = $null

foreach ($candidateZone in $TargetZones) {
    $Zone = $candidateZone
    $state = Get-QueuedResourceState
    if ($state -eq "ACTIVE") {
        $nodeState = (gcloud compute tpus tpu-vm describe $NodeId --project=$Project --zone=$Zone --format="value(state)" 2>$null)
        if ($nodeState) { $nodeState = $nodeState.Trim() }
        if ($nodeState -ne "PREEMPTED" -and $nodeState -ne "TERMINATED") {
            Write-Host "[SETUP] Found active TPU VM in $Zone. Reusing existing resource."
            $allocated = $true
            $WinningZone = $Zone
            break
        }
    }
}

if (-not $allocated) {
    foreach ($candidateZone in $TargetZones) {
        $Zone = $candidateZone
        Write-Host "[SETUP] Queueing TRC spot v6e zone: $Zone"
        Cleanup-Tpu

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
            Write-Host "[WARN] Allocation command failed in $Zone."
            Cleanup-Tpu
        }
    }

    Write-Host "[POLL] Waiting for first TRC zone to become ACTIVE..."
    $timeout = (Get-Date).AddMinutes(20)
    while ((Get-Date) -lt $timeout) {
        foreach ($candidateZone in $TargetZones) {
            $Zone = $candidateZone
            $state = Get-QueuedResourceState
            Write-Host "  $Zone State: $state"
            if ($state -eq "ACTIVE") {
                $nodeState = (gcloud compute tpus tpu-vm describe $NodeId --project=$Project --zone=$Zone --format="value(state)" 2>$null)
                if ($nodeState) { $nodeState = $nodeState.Trim() }
                if ($nodeState -ne "PREEMPTED" -and $nodeState -ne "TERMINATED") {
                    $allocated = $true
                    $WinningZone = $Zone
                    break
                }
            }
        }
        if ($allocated) {
            break
        }
        Start-Sleep -Seconds 20
    }
}

if ($allocated) {
    foreach ($candidateZone in $TargetZones) {
        if ($candidateZone -ne $WinningZone) {
            $Zone = $candidateZone
            Write-Host "[CLEANUP] Deleting losing queued request in $Zone."
            Cleanup-Tpu
        }
    }
    $Zone = $WinningZone
}

if (-not $allocated) {
    foreach ($candidateZone in $TargetZones) {
        $Zone = $candidateZone
        Cleanup-Tpu
    }
    Write-Host "[ERROR] Could not allocate a TRC-covered spot v6e TPU in any configured zone."
    exit 1
}

Write-Host "[WIN] TPU VM is active in $Zone!"

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
    --command="mkdir -p $RemoteDir/emergent_quantum_geometries $RemoteOutDir && pip install -q -U 'jax[tpu]' scipy numpy -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && python3 -c 'import jax; print(jax.default_backend(), jax.devices())'"
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
Write-Host "[EXECUTE] Launching Program AM Semantic Topology experiment..."
$RunCmd = "export OMP_NUM_THREADS=64; nohup python3 -u $RemoteDir/emergent_quantum_geometries/program_am_semantic_topology.py --require-tpu --out-dir $RemoteOutDir > $RemoteDir/$LogFile 2>&1 &"
"y" | gcloud compute tpus tpu-vm ssh $NodeId `
    --project=$Project --zone=$Zone `
    --command=$RunCmd

# 7. Monitor execution logs
Write-Host "[MONITOR] Live stream initialized (refreshing every 30s):"
$StartTime = Get-Date
$RunTimeout = $StartTime.AddHours(2)
while ((Get-Date) -lt $RunTimeout) {
    Start-Sleep -Seconds 30
    
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
    
    # Check if complete. The old check looked for "Gates", which appears in
    # intermediate table headers and can stop monitoring too early.
    $doneRaw = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
        --project=$Project --zone=$Zone `
        --command="test -s $RemoteOutDir/program_am_summary.json && grep -q '^={20,}' $RemoteDir/$LogFile && echo 1 || echo 0" 2>$null
    $doneVal = if ($doneRaw -is [array]) { $doneRaw[-1] } else { $doneRaw }
    if ($doneVal -ne $null -and [int]($doneVal.Trim()) -gt 0) {
        Write-Host "[SUCCESS] Experiment completed successfully!"
        break
    }
    
    # Check if process is still alive
    $isRunning = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
        --project=$Project --zone=$Zone `
        --command="pgrep -f '[p]rogram_am_semantic_topology.py' >/dev/null && echo 1 || echo 0" 2>$null
    if ([string]::IsNullOrWhiteSpace($isRunning)) {
        Write-Host "[WARNING] SSH failed. Checking QR State..."
        $state = Get-QueuedResourceState
        if ($state -ne "ACTIVE") {
            Write-Host "[ERROR] TPU Node preempted!"
            exit 1
        }
    } else {
        $isRunningVal = if ($isRunning -is [array]) { $isRunning[-1] } else { $isRunning }
        if ($isRunningVal -ne $null -and [int]($isRunningVal.Trim()) -eq 0) {
            Write-Host "[ERROR] Python script crashed silently!"
            exit 1
        }
    }
}

# 8. Download Results
Write-Host "[DOWNLOAD] Fetching results to local workspace..."
$remoteSummary   = "$NodeId`:$RemoteOutDir`/program_am_summary.json"
$remoteLog       = "$NodeId`:$RemoteDir`/$LogFile"

New-Item -ItemType Directory -Force -Path "$SrcDir\program_am_results" | Out-Null
"y" | gcloud compute tpus tpu-vm scp $remoteSummary "$SrcDir\program_am_results\program_am_summary.json" --project=$Project --zone=$Zone
"y" | gcloud compute tpus tpu-vm scp $remoteLog "$SrcDir\program_am_results\program_am_run.log" --project=$Project --zone=$Zone

Write-Host "[COMPLETE] Results downloaded successfully! Cleaning up TPU resources..."
Cleanup-Tpu
Write-Host "[DONE]"
