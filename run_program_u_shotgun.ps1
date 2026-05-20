<#
.SYNOPSIS
  run_program_u_shotgun.ps1  --  Program U TPU launcher (Bible-compliant)
  Queued Resource API only. Correct delete ritual. gcloud native SSH auto-accept.
  Project: time-emission
  Target: program_u_tpu.py (N=10 by default)
#>

param(
    [string]$N           = "10",
    [string]$Models      = "XXZ W3 OAT Ising",
    [string]$Controllers = "static dd random_sparse sparse_predictive",
    [string]$TMax        = "6.0",
    [string]$NSteps      = "30",
    [string]$B           = "3",
    [string]$Seed        = "42",
    [int]   $QueueMaxMin = 30,
    [int]   $MaxMin      = 240,
    [string]$OutDir      = "program_u_results"
)

$Project   = "time-emission"
$SrcDir    = "$PSScriptRoot\emergent_quantum_geometries"
$RemoteDir = "/home/cityz/program_u"
$LogFile   = "prog_u_run.log"

$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}

# Auto-confirm all gcloud interactive prompts
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"

# ── Bible-approved zones and configs (from TRC rules) ─────────────────────────
$Candidates = @(
    @{ QRName="prog-u-v4od";  NodeId="prog-u-v4od-node";  Zone="us-central2-b";  Type="v4-8";         Runtime="tpu-vm-v4-base";      Flag="" },
    @{ QRName="prog-u-v6e1";  NodeId="prog-u-v6e1-node";  Zone="us-east1-d";     Type="v6e-8";        Runtime="v2-alpha-tpuv6e";     Flag="--spot" },
    @{ QRName="prog-u-v6e2";  NodeId="prog-u-v6e2-node";  Zone="europe-west4-a"; Type="v6e-8";        Runtime="v2-alpha-tpuv6e";     Flag="--spot" },
    @{ QRName="prog-u-v5e1";  NodeId="prog-u-v5e1-node";  Zone="europe-west4-b"; Type="v5litepod-4";  Runtime="v2-alpha-tpuv5-lite"; Flag="--spot" },
    @{ QRName="prog-u-v5e2";  NodeId="prog-u-v5e2-node";  Zone="us-central1-a";  Type="v5litepod-4";  Runtime="v2-alpha-tpuv5-lite"; Flag="--spot" },
    @{ QRName="prog-u-v4s";   NodeId="prog-u-v4s-node";   Zone="us-central2-b";  Type="v4-8";         Runtime="tpu-vm-v4-base";      Flag="--spot" }
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
$Winner    = $null
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

if (-not $Winner) {
    Write-Host "[FAIL] No zone became ACTIVE within $QueueMaxMin minutes."
    Delete-All $Candidates; exit 1
}
Write-Host "[WIN] $($Winner.Zone)  node=$($Winner.NodeId)"

foreach ($c in $Candidates) {
    if ($c.QRName -ne $Winner.QRName) {
        Write-Host "[CANCEL] Deleting loser $($c.QRName)..."
        gcloud compute tpus tpu-vm delete $c.NodeId --project=$Project --zone=$($c.Zone) --quiet 2>$null
        gcloud compute tpus queued-resources delete $c.QRName --project=$Project --zone=$($c.Zone) --quiet 2>$null
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
        Write-Host "[ERROR] Winner became $stateNow before SSH was ready."
        Delete-All @($Winner); exit 1
    }
    "y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
        --project=$Project --zone=$($Winner.Zone) `
        --command="echo SSH_READY" 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -eq 0) { $sshReady = $true }
    else { Start-Sleep -Seconds 10 }
}
if (-not $sshReady) {
    Write-Host "[ERROR] SSH did not become ready within 5 minutes."
    Delete-All @($Winner); exit 1
}

Write-Host "[SETUP] Installing JAX/TPU deps..."
"y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
    --project=$Project --zone=$($Winner.Zone) `
    --command="mkdir -p $RemoteDir && pip install -q -U 'jax[tpu]' scipy numpy -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && python3 -c 'import jax; print(jax.default_backend()); print(jax.devices())' && echo DEPS_OK"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dependency install failed."
    Delete-All @($Winner); exit 1
}

Write-Host "[SCP] Uploading program files..."
foreach ($file in @("program_u_tpu.py", "program_l_tpu.py", "program_s1_tpu.py")) {
    $localPath  = Join-Path $SrcDir $file
    $remoteDest = $Winner.NodeId + ':' + $RemoteDir + '/' + $file
    Write-Host "  $file -> $remoteDest"
    "y" | gcloud compute tpus tpu-vm scp $localPath $remoteDest --project=$Project --zone=$($Winner.Zone) 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] SCP failed for $file"
        Delete-All @($Winner); exit 1
    }
}
Write-Host "[SCP] Upload complete."

$RunCmd = "nohup python3 $RemoteDir/program_u_tpu.py" +
          " --N $N" +
          " --T-max $TMax" +
          " --n-steps $NSteps" +
          " --B $B" +
          " --seed $Seed" +
          " --models $Models" +
          " --out-dir $RemoteDir/$OutDir" +
          " --require-tpu" +
          " --controllers $Controllers" +
          " > $RemoteDir/$LogFile 2>&1 &"

Write-Host "[RUN] Launching Program U (nohup background)..."
"y" | gcloud compute tpus tpu-vm ssh $($Winner.NodeId) `
    --project=$Project --zone=$($Winner.Zone) `
    --command=$RunCmd
Write-Host "[RUN] Job launched. Polling log..."

$StartTime = Get-Date
$Timeout   = $StartTime.AddMinutes($MaxMin)

while ((Get-Date) -lt $Timeout) {
    Start-Sleep -Seconds 60
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
}

$remoteResult = $Winner.NodeId + ':' + $RemoteDir + '/' + $OutDir + "/program_u_N${N}_results.json"
$srcLog     = $Winner.NodeId + ':' + $RemoteDir + '/' + $LogFile
New-Item -ItemType Directory -Force -Path "$SrcDir\$OutDir" | Out-Null
"y" | gcloud compute tpus tpu-vm scp $remoteResult "$SrcDir\$OutDir\program_u_N${N}_results.json" `
    --project=$Project --zone=$($Winner.Zone)
"y" | gcloud compute tpus tpu-vm scp $srcLog "$SrcDir\$OutDir\$LogFile" `
    --project=$Project --zone=$($Winner.Zone) 2>$null

Delete-All @($Winner)
Write-Host "[COMPLETE] Program U shotgun done."
