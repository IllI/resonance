# run_program_aj_shotgun.ps1  --  Program AJ TPU Orchestrator
# Project: time-emission
# Target: emergent_quantum_geometries/program_aj_predictive_horizon.py

$Project   = "time-emission"
$SrcDir    = "$PSScriptRoot"
$RemoteDir = "/home/cityz/program_aj"
$LogFile   = "prog_aj_run.log"

$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}

# Auto-confirm gcloud prompts
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"

# Target config (v6e-8 in us-east1-d)
$QRName  = "program-aj-qr-v4"
$NodeId  = "program-aj-node-v4"
$Zone    = "us-central2-b"
$Type    = "v4-8"
$Runtime = "tpu-vm-v4-base"

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

$success = $false
while (-not $success) {
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
    } elseif ($state -match "WAITING_FOR_RESOURCES|PROVISIONING|ACCEPTED|CREATING_NODES") {
        Write-Host "[SETUP] Found QR in $state state. Resuming poll."
    } else {
        Cleanup-Tpu
    }

    # 2. Allocate TPU VM using Queued Resource API
    $qrState = Get-QueuedResourceState
    if ($qrState -match "MISSING|FAILED|SUSPENDED") {
        Write-Host "[ALLOCATE] Requesting fresh TPU VM ($Type in $Zone)..."
        gcloud compute tpus queued-resources create $QRName `
            --node-id=$NodeId `
            --project=$Project `
            --zone=$Zone `
            --accelerator-type=$Type `
            --runtime-version=$Runtime `
            --quiet

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Allocation command failed! Retrying in 30s..."
            Start-Sleep -Seconds 30
            continue
        }

        Write-Host "[POLL] Waiting for TPU to become ACTIVE..."
        $timeout = (Get-Date).AddMinutes(60)
        while ((Get-Date) -lt $timeout) {
            $state = Get-QueuedResourceState
            Write-Host "  State: $state"
            if ($state -eq "ACTIVE") { break }
            if ($state -match "FAILED|SUSPENDED|MISSING") {
                Write-Host "[ERROR] TPU allocation failed or suspended!"
                break
            }
            Start-Sleep -Seconds 20
        }
        if ((Get-QueuedResourceState) -ne "ACTIVE") {
            Write-Host "[ERROR] Timeout waiting for ACTIVE state! Cleaning up..."
            Cleanup-Tpu
            continue
        }
    } else {
        Write-Host "[POLL] Waiting for existing TPU to become ACTIVE..."
        $timeout = (Get-Date).AddMinutes(60)
        while ((Get-Date) -lt $timeout) {
            $state = Get-QueuedResourceState
            Write-Host "  State: $state"
            if ($state -eq "ACTIVE") { break }
            if ($state -match "FAILED|SUSPENDED|MISSING") {
                Write-Host "[ERROR] TPU allocation failed or suspended!"
                break
            }
            Start-Sleep -Seconds 20
        }
        if ((Get-QueuedResourceState) -ne "ACTIVE") {
            Write-Host "[ERROR] Timeout waiting for ACTIVE state! Cleaning up..."
            Cleanup-Tpu
            continue
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
        Write-Host "[ERROR] SSH failed to connect within 5 minutes! Re-creating..."
        Cleanup-Tpu
        continue
    }
    Write-Host "[SSH] Connection established."

    # 4. Install Dependencies
    Write-Host "[DEPS] Installing JAX TPU backend..."
    "y" | gcloud compute tpus tpu-vm ssh $NodeId `
        --project=$Project --zone=$Zone `
        --command="mkdir -p $RemoteDir/emergent_quantum_geometries/program_aj_results && pip install -q -U 'jax[tpu]' scipy numpy -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    
    # 5. SCP Files (Using tar to avoid wildcard issues on Windows)
    Write-Host "[SCP] Uploading code files..."
    cd $SrcDir
    tar.exe -czf "payload_aj.tar.gz" *.py
    "y" | gcloud compute tpus tpu-vm scp payload_aj.tar.gz "$NodeId`:$RemoteDir`/emergent_quantum_geometries`/" --project=$Project --zone=$Zone 2>&1 | Out-Null
    "y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --command="cd $RemoteDir/emergent_quantum_geometries && tar -xzf payload_aj.tar.gz && rm payload_aj.tar.gz" 2>&1 | Out-Null
    Remove-Item -Force "payload_aj.tar.gz" -ErrorAction SilentlyContinue
    Write-Host "[SCP] Upload complete."

    Write-Host "[RESTORE] Uploading previously computed checkpoints..."
    if (Test-Path "$SrcDir\program_aj_results") {
        tar.exe -czf "results_aj.tar.gz" program_aj_results
        "y" | gcloud compute tpus tpu-vm scp results_aj.tar.gz "$NodeId`:$RemoteDir`/emergent_quantum_geometries`/" --project=$Project --zone=$Zone 2>&1 | Out-Null
        "y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --command="cd $RemoteDir/emergent_quantum_geometries && tar -xzf results_aj.tar.gz && rm results_aj.tar.gz" 2>&1 | Out-Null
        Remove-Item -Force "results_aj.tar.gz" -ErrorAction SilentlyContinue
    }

    # 6. Execute Experiment
    Write-Host "[EXECUTE] Launching Forecast Horizon & Stability Phase Diagram experiment..."
    $RunCmd = "export OMP_NUM_THREADS=64; cd $RemoteDir/emergent_quantum_geometries; nohup python3 -u program_aj_predictive_horizon.py --require-tpu > $LogFile 2>&1 &"
    "y" | gcloud compute tpus tpu-vm ssh $NodeId `
        --project=$Project --zone=$Zone `
        --command=$RunCmd

    # 7. Monitor execution logs & Atomic Sync
    Write-Host "[MONITOR] Live stream initialized (refreshing every 60s):"
    $StartTime = Get-Date
    $RunTimeout = $StartTime.AddHours(2)
    
    while ((Get-Date) -lt $RunTimeout) {
        Start-Sleep -Seconds 60
        
        # Check preemption
        $nodeState = (gcloud compute tpus tpu-vm describe $NodeId --project=$Project --zone=$Zone --format="value(state)" 2>$null)
        if ($nodeState -and $nodeState.Trim() -match "PREEMPTED|TERMINATED") {
            Write-Host "[ERROR] Node was PREEMPTED by Google mid-execution! Restarting loop..."
            break
        }

        # Tail the remote log file
        $tail = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
            --project=$Project --zone=$Zone `
            --command="tail -n 8 $RemoteDir/emergent_quantum_geometries/$LogFile 2>/dev/null" 2>$null
        Write-Host "[$([math]::Round(((Get-Date)-$StartTime).TotalMinutes,1))m] -------------------"
        Write-Host $tail
        
        # ATOMIC SYNC
        Write-Host "[SYNC] Pulling latest checkpoints and trajectories..."
        New-Item -ItemType Directory -Force -Path "$SrcDir\program_aj_results" | Out-Null
        "y" | gcloud compute tpus tpu-vm scp --recurse "$NodeId`:$RemoteDir`/emergent_quantum_geometries`/program_aj_results`/*" "$SrcDir\program_aj_results\" --project=$Project --zone=$Zone --quiet 2>$null
        
        # Check if complete
        $doneRaw = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
            --project=$Project --zone=$Zone `
            --command="grep -c 'Summary master printout' $RemoteDir/emergent_quantum_geometries/$LogFile 2>/dev/null || echo 0" 2>$null
        $doneVal = if ($doneRaw -is [array]) { $doneRaw[-1] } else { $doneRaw }
        if ($doneVal -ne $null -and [int]($doneVal.Trim()) -gt 0) {
            Write-Host "[SUCCESS] Experiment completed successfully!"
            $success = $true
            break
        }
        
        # Check Heartbeat Watchdog
        $heartbeat = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
            --project=$Project --zone=$Zone `
            --command="cat $RemoteDir/emergent_quantum_geometries/heartbeat.txt 2>/dev/null || echo 0" 2>$null
        $hbVal = if ($heartbeat -is [array]) { $heartbeat[-1] } else { $heartbeat }
        
        if ($hbVal -ne $null -and [double]($hbVal.Trim()) -gt 0) {
            $currentTime = [double]("y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --command="date +%s" 2>$null | Select-Object -Last 1)
            $diff = $currentTime - [double]($hbVal.Trim())
            if ($diff -gt 180) {
                Write-Host "[ERROR] Heartbeat stale by over 3 minutes! Python script likely crashed silently."
                $errorLog = "y" | gcloud compute tpus tpu-vm ssh $NodeId `
                    --project=$Project --zone=$Zone `
                    --command="tail -n 50 $RemoteDir/emergent_quantum_geometries/$LogFile" 2>$null
                Write-Host $errorLog
                break
            }
        }
    }
}

Write-Host "[COMPLETE] Results completely synced! Cleaning up TPU resources..."
Cleanup-Tpu
Write-Host "[DONE]"
