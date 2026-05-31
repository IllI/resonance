# Program AO ready-node launcher.
# TRC/Egress discipline:
# - Use only an already READY TRC-covered TPU VM.
# - Upload only program_ao_folded_field_recovery.py.
# - Do not download results from Europe zones by default; inspect remotely.

$Project = "time-emission"
$Zone = if ($env:PROGRAM_AO_ZONE) { $env:PROGRAM_AO_ZONE } else { "europe-west4-a" }
$NodeId = if ($env:PROGRAM_AO_NODE) { $env:PROGRAM_AO_NODE } else { "program-ao-node-v1" }
$SrcDir = "$PSScriptRoot\emergent_quantum_geometries"
$RemoteDir = "/home/cityz/program_ao"
$RemoteOutDir = "$RemoteDir/program_ao_results"
$LogFile = "prog_ao_run.log"

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

Write-Host "[SSH] Verifying TPU Python runtime..."
$RuntimeCheck = 'python3 -c "import jax, numpy; assert any(getattr(d, ''platform'', '''') == ''tpu'' for d in jax.devices()); print(jax.__version__, jax.default_backend())"'
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command=$RuntimeCheck
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] TPU Python runtime check failed. Install jax[tpu] before launch."
    exit 1
}

Write-Host "[SCP] Uploading Program AO source only..."
"y" | gcloud compute tpus tpu-vm scp "$SrcDir\program_ao_folded_field_recovery.py" "$NodeId`:$RemoteDir/emergent_quantum_geometries/" --project=$Project --zone=$Zone --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Source upload failed."
    exit 1
}

Write-Host "[RUN] Launching Program AO on TPU..."
$RunCmd = "cd $RemoteDir; nohup python3 -u $RemoteDir/emergent_quantum_geometries/program_ao_folded_field_recovery.py --require-tpu --ao-focused --out-dir $RemoteOutDir > $RemoteDir/$LogFile 2>&1 & echo PROGRAM_AO_PID=`$!"
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command=$RunCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Program launch failed."
    exit 1
}

Write-Host "[MONITOR] Waiting for initial compile output..."
Start-Sleep -Seconds 45
"y" | gcloud compute tpus tpu-vm ssh $NodeId --project=$Project --zone=$Zone --quiet --command="tail -n 80 $RemoteDir/$LogFile 2>/dev/null || true"

Write-Host "[DONE] Program AO launched. No result files downloaded."
