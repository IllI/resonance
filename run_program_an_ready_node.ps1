# Program AN ready-node launcher.
# Deployment bib:
# - TRC-covered v6e spot zones only.
# - Prefer us-east1-d to avoid EU-to-US egress categories.
# - Upload only required source; download only summary JSON + short log tail.
# - Never recursive SCP result directories.
# - This reuses an already READY TPU VM and does not delete TPU resources.

$Project = "time-emission"
$Zone = if ($env:PROGRAM_AN_ZONE) { $env:PROGRAM_AN_ZONE } else { "us-east1-d" }
$NodeId = if ($env:PROGRAM_AN_NODE) { $env:PROGRAM_AN_NODE } else { "program-an-node-v1" }
$SrcDir = "$PSScriptRoot\emergent_quantum_geometries"
$RemoteDir = "/home/cityz/program_an"
$RemoteOutDir = "$RemoteDir/program_an_results"
$LogFile = "prog_an_run.log"
$TailFile = "prog_an_run_tail.log"

$CloudSdkBin = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $CloudSdkBin) {
    $env:Path = "$CloudSdkBin;$env:Path"
}
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = "1"

$AllowedZones = @("us-east1-d", "europe-west4-a")
if ($Zone -notin $AllowedZones) {
    Write-Host "[ERROR] Zone $Zone is not in the current TRC v6e spot allow-list."
    exit 1
}

Write-Host "[CHECK] Verifying ready TPU VM: $NodeId in $Zone"
$NodeState = (gcloud compute tpus tpu-vm describe $NodeId --project=$Project --zone=$Zone --format="value(state)" 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($NodeState)) {
    Write-Host "[ERROR] TPU VM not found. Start/queue a TRC v6e spot node first."
    exit 1
}
$NodeState = $NodeState.Trim()
if ($NodeState -ne "READY") {
    Write-Host "[ERROR] TPU VM state is $NodeState, expected READY."
    exit 1
}

Write-Host "[SSH] Preparing remote workspace..."
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --command="mkdir -p $RemoteDir/emergent_quantum_geometries $RemoteOutDir"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] SSH workspace setup failed."
    exit 1
}

Write-Host "[SSH] Verifying TPU Python runtime..."
$RuntimeCmd = "python3 -c 'import jax, numpy; assert any(getattr(d, ""platform"", """") == ""tpu"" for d in jax.devices()); print(jax.__version__, jax.default_backend())' 2>/dev/null || (python3 -m pip install -q -U pip && python3 -m pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && python3 -c 'import jax, numpy; assert any(getattr(d, ""platform"", """") == ""tpu"" for d in jax.devices()); print(jax.__version__, jax.default_backend())')"
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --command=$RuntimeCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] TPU Python runtime check failed."
    exit 1
}

Write-Host "[SCP] Uploading required source only..."
"y" | gcloud compute tpus tpu-vm scp "$SrcDir\program_an_kinematic_filaments.py" "$NodeId`:$RemoteDir/emergent_quantum_geometries/" --project=$Project --zone=$Zone
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Source upload failed."
    exit 1
}

Write-Host "[RUN] Launching Program AN on existing TPU node..."
$RunCmd = "cd $RemoteDir; nohup python3 -u $RemoteDir/emergent_quantum_geometries/program_an_kinematic_filaments.py --require-tpu --include-controls --run-heldout-transport --heldout-include-controls --run-semantic-recovery --out-dir $RemoteOutDir > $RemoteDir/$LogFile 2>&1 & echo PROGRAM_AN_PID=`$!"
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --command=$RunCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Program launch failed."
    exit 1
}

Write-Host "[MONITOR] Waiting for initial output..."
Start-Sleep -Seconds 30
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --command="tail -n 80 $RemoteDir/$LogFile 2>/dev/null || true; tail -n 200 $RemoteDir/$LogFile > $RemoteDir/$TailFile 2>/dev/null || true"

Write-Host "[DOWNLOAD] Fetching allow-listed tiny files only..."
New-Item -ItemType Directory -Force -Path "$SrcDir\program_an_results" | Out-Null
"y" | gcloud compute tpus tpu-vm scp "$NodeId`:$RemoteOutDir/program_an_summary.json" "$SrcDir\program_an_results\program_an_summary.json" --project=$Project --zone=$Zone 2>$null
"y" | gcloud compute tpus tpu-vm scp "$NodeId`:$RemoteDir/$TailFile" "$SrcDir\program_an_results\program_an_run_tail.log" --project=$Project --zone=$Zone 2>$null

Write-Host "[DONE] Program AN launched. TPU node was not deleted."
