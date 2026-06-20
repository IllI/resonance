param(
  [string]$Project = "time-emission",
  [string]$Type = "v6e-8",
  [string]$Runtime = "v2-alpha-tpuv6e"
)

$ErrorActionPreference = "Stop"
$Eqg = (Resolve-Path "$PSScriptRoot\..").Path
$Gcloud = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$Log = Join-Path $PSScriptRoot "queue.log"
$Candidates = @(
  @{ Zone="us-east1-d"; QR="aq-dlinoss-mera-us-v1"; Node="aq-dlinoss-mera-us-node-v1" },
  @{ Zone="europe-west4-a"; QR="aq-dlinoss-mera-eu-v1"; Node="aq-dlinoss-mera-eu-node-v1" }
)

function Log([string]$Message) { Add-Content -LiteralPath $Log -Value "$(Get-Date -Format o) $Message" }
function State($Candidate, [string]$Kind) {
  $old=$ErrorActionPreference; $ErrorActionPreference="Continue"
  try {
    if ($Kind -eq "queue") {
      $v=& $Gcloud compute tpus queued-resources describe $($Candidate.QR) --project=$Project --zone=$($Candidate.Zone) --format="value(state.state)" 2>$null
    } else {
      $v=& $Gcloud compute tpus tpu-vm describe $($Candidate.Node) --project=$Project --zone=$($Candidate.Zone) --format="value(state)" 2>$null
    }
  } finally { $ErrorActionPreference=$old }
  if ([string]::IsNullOrWhiteSpace($v)) { return "MISSING" }
  return $v.Trim()
}
function Ensure($Candidate) {
  $state=State $Candidate "queue"
  if ($state -match "ACCEPTED|WAITING_FOR_RESOURCES|PROVISIONING|CREATING_NODES|ACTIVE") { return }
  if ($state -ne "MISSING") {
    & $Gcloud compute tpus queued-resources delete $($Candidate.QR) --project=$Project --zone=$($Candidate.Zone) --force --quiet | Out-Null
  }
  Log "create zone=$($Candidate.Zone) queue=$($Candidate.QR)"
  & $Gcloud compute tpus queued-resources create $($Candidate.QR) --node-id=$($Candidate.Node) --project=$Project --zone=$($Candidate.Zone) --accelerator-type=$Type --runtime-version=$Runtime --spot --quiet
}
function Ssh($Candidate, [string]$Command) {
  "y" | & $Gcloud compute tpus tpu-vm ssh $($Candidate.Node) --project=$Project --zone=$($Candidate.Zone) --quiet --command=$Command
  if ($LASTEXITCODE -ne 0) { throw "SSH failed" }
}
function Scp($Candidate, [string]$Local, [string]$Remote) {
  "y" | & $Gcloud compute tpus tpu-vm scp $Local "$($Candidate.Node):$Remote" --project=$Project --zone=$($Candidate.Zone) --quiet
  if ($LASTEXITCODE -ne 0) { throw "SCP failed" }
}

foreach ($candidate in $Candidates) { Ensure $candidate }
$Winner=$null
$Deadline=(Get-Date).AddHours(3)
while (-not $Winner -and (Get-Date) -lt $Deadline) {
  foreach ($candidate in $Candidates) {
    $queue=State $candidate "queue"; $node=State $candidate "node"
    Log "poll zone=$($candidate.Zone) queue=$queue node=$node"
    if ($queue -eq "ACTIVE" -and $node -eq "READY") { $Winner=$candidate; break }
  }
  if (-not $Winner) { Start-Sleep -Seconds 30 }
}
if (-not $Winner) { throw "No TRC-covered v6e-8 became ready" }

Push-Location $Eqg
try {
  $Payload="payload_dlinoss_mera_capacity.tar.gz"
  $PayloadItems=@(
    "aq_dlinoss_mera_spectral_capacity_test_0/models.py",
    "aq_dlinoss_mera_spectral_capacity_test_0/run_capacity.py",
    "aq_dlinoss_mera_spectral_capacity_test_0/run_tpu.sh",
    "aq_dlinoss_mera_spectral_capacity_test_0/README.md"
  )
  tar.exe -czf $Payload @PayloadItems
  Ssh $Winner "mkdir -p /home/cityz/eqg/aq_trappist1e_dlinoss_stellar_state_residual_0/results/shards /home/cityz/eqg/aq_dlinoss_mera_spectral_capacity_test_0/results"
  Scp $Winner $Payload "/home/cityz/eqg/$Payload"
  Scp $Winner "aq_trappist1e_dlinoss_stellar_state_residual_0/results/shards/dlinoss_equal_supervision_capacity_test_0.npz" "/home/cityz/eqg/aq_trappist1e_dlinoss_stellar_state_residual_0/results/shards/dlinoss_equal_supervision_capacity_test_0.npz"
  Ssh $Winner "cd /home/cityz/eqg && tar -xzf $Payload && rm -f $Payload && chmod +x aq_dlinoss_mera_spectral_capacity_test_0/run_tpu.sh && (python3 -c 'import jax; print(jax.__version__)' || python3 -m pip install -q -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html)"
  Ssh $Winner "cd /home/cityz/eqg && pkill -f '[r]un_capacity.py' || true; nohup bash aq_dlinoss_mera_spectral_capacity_test_0/run_tpu.sh > aq_dlinoss_mera_spectral_capacity_test_0/results/run.log 2>&1 < /dev/null &"
  Log "launched winner=$($Winner.Zone) node=$($Winner.Node)"
} finally {
  Remove-Item -LiteralPath (Join-Path $Eqg "payload_dlinoss_mera_capacity.tar.gz") -Force -ErrorAction SilentlyContinue
  Pop-Location
}

foreach ($candidate in $Candidates) {
  if ($candidate.Node -eq $Winner.Node) { continue }
  Start-Process -FilePath $Gcloud -ArgumentList @("compute","tpus","tpu-vm","delete",$candidate.Node,"--project=$Project","--zone=$($candidate.Zone)","--quiet") -WindowStyle Hidden | Out-Null
  Start-Process -FilePath $Gcloud -ArgumentList @("compute","tpus","queued-resources","delete",$candidate.QR,"--project=$Project","--zone=$($candidate.Zone)","--force","--quiet") -WindowStyle Hidden | Out-Null
}
Log "winner retained for result inspection; only compact JSON/log outputs may be downloaded"
