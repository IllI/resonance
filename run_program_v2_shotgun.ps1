<#
.SYNOPSIS
  run_program_v2_shotgun.ps1  --  Program V2 TPU launcher
  Queued Resource API only. Correct delete ritual. gcloud native SSH auto-accept.
  Project: time-emission
  Target: program_v2_tpu.py (N=12 by default for 3x4 grid)
#>

param(
    [int]$Lx             = 3,
    [int]$Ly             = 4,
    [string]$Models      = "GridXXZ_hetero",
    [string]$Controllers = "free static dd vector_adaptive",
    [string]$TMax        = "8.0",
    [string]$NSteps      = "40",
    [string]$B           = "3",
    [string]$NTraj       = "100",
    [string]$Seed        = "42",
    [int]   $QueueMaxMin = 30,
    [int]   $MaxMin      = 240,
    [string]$OutDir      = "program_v2_results"
)

$Project   = "time-emission"
$SrcDir    = "$PSScriptRoot\emergent_quantum_geometries"
$RemoteDir = "/home/cityz/program_v2"
$LogFile   = "prog_v2_run.log"

$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}

# Auto-confirm all gcloud interactive prompts
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"

# ── Bible-approved zones and configs (from TRC rules) ─────────────────────────
$Candidates = @(
    @{ QRName="prog-v-v4od";  NodeId="prog-v-v4od-node";  Zone="us-central2-b";  Type="v4-8";         Runtime="tpu-vm-v4-base";      Flag="" },
    @{ QRName="prog-v-v6e1";  NodeId="prog-v-v6e1-node";  Zone="us-east1-d";     Type="v6e-8";        Runtime="v2-alpha-tpuv6e";     Flag="--spot" },
    @{ QRName="prog-v-v6e2";  NodeId="prog-v-v6e2-node";  Zone="europe-west4-a"; Type="v6e-8";        Runtime="v2-alpha-tpuv6e";     Flag="--spot" },
    @{ QRName="prog-v-v5e1";  NodeId="prog-v-v5e1-node";  Zone="europe-west4-b"; Type="v5litepod-4";  Runtime="v2-alpha-tpuv5-lite"; Flag="--spot" },
    @{ QRName="prog-v-v5e2";  NodeId="prog-v-v5e2-node";  Zone="us-central1-a";  Type="v5litepod-4";  Runtime="v2-alpha-tpuv5-lite"; Flag="--spot" },
    @{ QRName="prog-v-v4s";   NodeId="prog-v-v4s-node";   Zone="us-central2-b";  Type="v4-8";         Runtime="tpu-vm-v4-base";      Flag="--spot" }
)

$AllSanctionedZones = @("us-central2-b","europe-west4-a","us-east1-d","us-central1-a","europe-west4-b")

function Get-QueuedResourceState {
    param([string]$Name, [string]$Zone)
    $descJson = gcloud compute tpus queued-resources describe $Name `
        --project=$Project --zone=$Zone --format=json 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($descJson)) { return "MISSING" }
    try {
        $desc = $descJson | ConvertFrom-Json
        if ($desc.state -and $desc.state.state) { return [string]$desc.state.state }
        if ($desc.state) { return [string]$desc.state }
    } catch { return "UNKNOWN" }
    return "UNKNOWN"
}

function Delete-All {
    param([array]$candidates)
    Write-Host "`n[RITUAL] Running Bible delete ritual..."
    foreach ($c in $candidates) {
        gcloud compute tpus tpu-vm delete $c.NodeId --project=$Project --zone=$($c.Zone) --quiet 2>$null
        gcloud compute tpus queued-resources delete $c.QRName --project=$Project --zone=$($c.Zone) --quiet 2>$null
    }
    Start-Sleep -Seconds 10
    Write-Host "[RITUAL] Verifying all resources deleted..."
    foreach ($z in $AllSanctionedZones) {
        $qrs = gcloud compute tpus queued-resources list --project=$Project --zone=$z --format="value(name)" 2>$null
        if ($qrs) { Write-Host "[WARN] Zone ${z}: still has queued resources: $qrs" }
        else { Write-Host "[OK]   Zone ${z}: no queued resources." }
    }
    Write-Host "[RITUAL] Delete ritual complete.`n"
}

$curProject = (gcloud config list project --format="value(core.project)" 2>$null).Trim()
if ($curProject -ne $Project) {
    Write-Host "[SETUP] Setting project to $Project..."
    gcloud config set project $Project
}
Write-Host "[SETUP] Project: $Project"

$Winner = $null
foreach ($c in $Candidates) {
    $state = Get-QueuedResourceState $c.QRName $c.Zone
    if ($state -eq "ACTIVE") {
        $nodeState = (gcloud compute tpus tpu-vm describe $c.NodeId --project=$Project --zone=$($c.Zone) --format="value(state)" 2>$null)
        if ($nodeState) { $nodeState = $nodeState.Trim() }
        if ($nodeState -eq "PREEMPTED" -or $nodeState -eq "TERMINATED") {
            Write-Host "[SETUP] Candidate $($c.QRName) is ACTIVE but node is $nodeState. Cleaning it up."
            gcloud compute tpus tpu-vm delete $c.NodeId --project=$Project --zone=$($c.Zone) --quiet 2>$null
            gcloud compute tpus queued-resources delete $c.QRName --project=$Project --zone=$($c.Zone) --quiet 2>$null
        } else {
            Write-Host "[SETUP] Found already ACTIVE candidate: $($c.QRName) in $($c.Zone). Reusing it."
            $Winner = $c
            break
        }
    }
}

if (-not $Winner) {
    Delete-All $Candidates

    Write-Host "[SHOTGUN] Creating queued resources in all zones..."
    foreach ($c in $Candidates) {
        $createArgs = @("compute", "tpus", "queued-resources", "create", $c.QRName,
            "--node-id=$($c.NodeId)", "--project=$Project", "--zone=$($c.Zone)",
            "--accelerator-type=$($c.Type)", "--runtime-version=$($c.Runtime)", "--quiet")
        if ($c.Flag) { $createArgs += $c.Flag }
        Write-Host "  Creating: $($c.QRName) in $($c.Zone)..."
        & gcloud @createArgs 2>&1 | ForEach-Object { Write-Host "    $_" }
        if ($LASTEXITCODE -eq 0) { Write-Host "  Queued: $($c.QRName)" }
        else { Write-Host "  [WARN] Create failed: $($c.QRName)" }
    }

    Write-Host "[POLL] Waiting for first zone to become ACTIVE (up to $QueueMaxMin min)..."
    $Deadline  = (Get-Date).AddMinutes($QueueMaxMin)

    while (-not $Winner -and (Get-Date) -lt $Deadline) {
        Start-Sleep -Seconds 20
        foreach ($c in $Candidates) {
            $state = Get-QueuedResourceState $c.QRName $c.Zone
            Write-Host "  $($c.Zone): $state"
            if ($state -eq "ACTIVE") {
                $Winner = $c
                break
            }
            if ($state -match "FAILED|SUSPENDED|MISSING") {
                Write-Host "  [RETRY] $($c.QRName) is $state; re-queueing..."
                gcloud compute tpus tpu-vm delete $c.NodeId --project=$Project --zone=$($c.Zone) --quiet 2>$null
                gcloud compute tpus queued-resources delete $c.QRName --project=$Project --zone=$($c.Zone) --quiet 2>$null
                Start-Sleep -Seconds 5
                $createArgs = @("compute", "tpus", "queued-resources", "create", $c.QRName,
                    "--node-id=$($c.NodeId)", "--project=$Project", "--zone=$($c.Zone)",
                    "--accelerator-type=$($c.Type)", "--runtime-version=$($c.Runtime)", "--quiet")
                if ($c.Flag) { $createArgs += $c.Flag }
                & gcloud @createArgs 2>&1 | ForEach-Object { Write-Host "    $_" }
            }
        }
    }
}

if (-not $Winner) {
    Write-Host "[FAIL] No zone became ACTIVE within $QueueMaxMin minutes."
    Delete-All $Candidates; exit 1
}
Write-Host "[WIN] $($Winner.Zone)  node=$($Winner.NodeId)"

foreach ($c in $Candidates) {
    if ($c.QRName -ne $Winner.QRName) {
        Write-Host "[CANCEL] Initiating deletion of loser $($c.QRName) asynchronously..."
        Start-Job -ScriptBlock {
            param($NodeId, $QRName, $Proj, $Zone)
            gcloud compute tpus tpu-vm delete $NodeId --project=$Proj --zone=$Zone --quiet 2>$null
            gcloud compute tpus queued-resources delete $QRName --project=$Proj --zone=$Zone --quiet 2>$null
        } -ArgumentList $c.NodeId, $c.QRName, $Project, $($c.Zone) >$null
    }
}

$ExtIP = gcloud compute tpus tpu-vm describe $($Winner.NodeId) `
    --project=$Project --zone=$($Winner.Zone) `
    --format="value(networkEndpoints[0].accessConfig.externalIp)" 2>$null
Write-Host "[SSH] External IP: $ExtIP"

Write-Host "[SETUP] Accepting host key / waiting for SSH daemon..."
$sshReady = $false
$sshDeadline = (Get-Date).AddMinutes(5)
while (-not $sshReady -and (Get-Date) -lt $sshDeadline) {
    $stateNow = Get-QueuedResourceState $Winner.QRName $Winner.Zone
    if ($stateNow -ne "ACTIVE") {
        Write-Host "[ERROR] Winner became $stateNow before SSH was ready. Keeping TPU alive."
        exit 1
    }
    "y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
        --project=$Project --zone=$($Winner.Zone) `
        --command="echo SSH_READY" 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -eq 0) { $sshReady = $true }
    else { Start-Sleep -Seconds 10 }
}
if (-not $sshReady) {
    Write-Host "[ERROR] SSH did not become ready within 5 minutes. Keeping TPU alive."
    exit 1
}

Write-Host "[SETUP] Installing JAX/TPU deps..."
"y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
    --project=$Project --zone=$($Winner.Zone) `
    --command="mkdir -p $RemoteDir && pip install -q -U 'jax[tpu]' scipy numpy -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && python3 -c 'import jax, jax.numpy as jnp; print(jax.default_backend()); print(jax.devices()); res=jnp.dot(jnp.ones((1000, 1000)), jnp.ones((1000, 1000))); print(res.shape)' && echo DEPS_OK"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dependency install failed. Keeping TPU alive for inspection."
    exit 1
}

Write-Host "[SCP] Uploading program files..."
foreach ($file in @("program_v2_tpu.py", "program_v_tpu.py", "program_t_tpu.py", "program_l_tpu.py", "program_s1_tpu.py")) {
    $localPath  = Join-Path $SrcDir $file
    $remoteDest = $Winner.NodeId + ':' + $RemoteDir + '/' + $file
    Write-Host "  $file -> $remoteDest"
    "y" | gcloud compute tpus tpu-vm scp $localPath $remoteDest --project=$Project --zone=$($Winner.Zone) 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] SCP failed for $file. Keeping TPU alive for inspection."
        exit 1
    }
}
Write-Host "[SCP] Upload complete."

$N = $Lx * $Ly
$RunCmd = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nohup python3 -u $RemoteDir/program_v2_tpu.py" +
          " --Lx $Lx" +
          " --Ly $Ly" +
          " --T-max $TMax" +
          " --n-steps $NSteps" +
          " --B $B" +
          " --n-trajectories $NTraj" +
          " --seed $Seed" +
          " --models $Models" +
          " --out-dir $RemoteDir/$OutDir" +
          " --require-tpu" +
          " --controllers $Controllers" +
          " > $RemoteDir/$LogFile 2>&1 &"

Write-Host "[RUN] Launching Program V2 (nohup background)..."
"y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
    --project=$Project --zone=$($Winner.Zone) `
    --command=$RunCmd
Write-Host "[RUN] Job launched. Polling log..."

$StartTime = Get-Date
$Timeout   = $StartTime.AddMinutes($MaxMin)

while ((Get-Date) -lt $Timeout) {
    Start-Sleep -Seconds 60
    
    $nodeState = (gcloud compute tpus tpu-vm describe $($Winner.NodeId) --project=$Project --zone=$($Winner.Zone) --format="value(state)" 2>$null)
    if ($nodeState) { $nodeState = $nodeState.Trim() }
    if ($nodeState -eq "PREEMPTED" -or $nodeState -eq "TERMINATED") {
        Write-Host "[ERROR] Node $($Winner.NodeId) was $nodeState mid-execution!"
        exit 1
    }

    $tail = "y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
                --project=$Project --zone=$($Winner.Zone) `
                --command="tail -5 $RemoteDir/$LogFile 2>/dev/null" 2>$null
    Write-Host "[$([math]::Round(((Get-Date)-$StartTime).TotalMinutes,0))m] $tail"
    
    $doneRaw = "y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
                --project=$Project --zone=$($Winner.Zone) `
                --command="grep -c 'Saved ->' $RemoteDir/$LogFile 2>/dev/null || echo 0" 2>$null
    $doneVal = if ($doneRaw -is [array]) { $doneRaw[-1] } else { $doneRaw }
    if ($doneVal -ne $null -and [int]($doneVal.Trim()) -gt 0) {
        Write-Host "[DONE] Experiment complete."
        break
    }

    $isRunning = "y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
                --project=$Project --zone=$($Winner.Zone) `
                --command="pgrep -f program_v2_tpu.py >/dev/null && echo 1 || echo 0" 2>$null
    $isRunningVal = if ($isRunning -is [array]) { $isRunning[-1] } else { $isRunning }
    if ($isRunningVal -ne $null -and [int]($isRunningVal.Trim()) -eq 0) {
        Write-Host "[WARN] Process not found. Checking grace period..."
        Start-Sleep -Seconds 10
        $isRunning2 = "y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
                    --project=$Project --zone=$($Winner.Zone) `
                    --command="pgrep -f program_v2_tpu.py >/dev/null && echo 1 || echo 0" 2>$null
        $isRunningVal2 = if ($isRunning2 -is [array]) { $isRunning2[-1] } else { $isRunning2 }
        
        $doneRaw2 = "y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
                    --project=$Project --zone=$($Winner.Zone) `
                    --command="grep -c 'Saved ->' $RemoteDir/$LogFile 2>/dev/null || echo 0" 2>$null
        $doneVal2 = if ($doneRaw2 -is [array]) { $doneRaw2[-1] } else { $doneRaw2 }
        
        if ([int]($doneVal2.Trim()) -gt 0) {
            Write-Host "[DONE] Experiment complete (caught during grace period)."
            break
        } elseif ([int]($isRunningVal2.Trim()) -eq 0) {
            Write-Host "[ERROR] Python script crashed silently! 'Saved ->' missing and process is dead."
            $errorLog = "y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
                        --project=$Project --zone=$($Winner.Zone) `
                        --command="tail -n 50 $RemoteDir/$LogFile" 2>$null
            $grepError = "y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
                        --project=$Project --zone=$($Winner.Zone) `
                        --command="grep -i -E 'error|traceback' $RemoteDir/$LogFile 2>/dev/null" 2>$null
            Write-Host "--- Highlighted Errors / Tracebacks ---"
            Write-Host $grepError
            Write-Host "--- Last 50 lines of log ---"
            Write-Host $errorLog
            Write-Host "----------------------------"
            Write-Host "Keeping TPU VM $($Winner.NodeId) alive for debugging. Exiting script."
            exit 1
        }
    }
}

$remoteResult = $Winner.NodeId + ':' + $RemoteDir + '/' + $OutDir + "/program_v2_N${N}_results.json"
$srcLog     = $Winner.NodeId + ':' + $RemoteDir + '/' + $LogFile
New-Item -ItemType Directory -Force -Path "$SrcDir\$OutDir" | Out-Null
"y" | gcloud compute tpus tpu-vm scp $remoteResult "$SrcDir\$OutDir\program_v2_N${N}_results.json" `
    --project=$Project --zone=$($Winner.Zone)
"y" | gcloud compute tpus tpu-vm scp $srcLog "$SrcDir\$OutDir\$LogFile" `
    --project=$Project --zone=$($Winner.Zone) 2>$null

Write-Host "[COMPLETE] Program V2 shotgun done. Results downloaded."
Write-Host "[CLEANUP] Running explicit delete ritual for completion..."
Delete-All $Candidates
