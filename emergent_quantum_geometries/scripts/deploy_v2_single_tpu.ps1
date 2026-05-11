<#
.SYNOPSIS
Deploys the V2 Emergent Quantum Geometries experiment to a single TPU (Bob).
#>

$ErrorActionPreference = "Continue"
$WorkDir = "c:\Users\cityz\IllI\newer_all\emergent_quantum_geometries"
$Project = "time-emission"
$Zone = "europe-west4-a"
$Node = "chronos-bob-node"

Write-Host "=== Deploying V2 Experiment to $Node ==="

Write-Host "1. Installing remote dependencies (JAX, Flax, NetKet)..."
$setupCmd = "pip install -q 'jax[tpu]' flax optax netket qutip pandas && python3 -c 'import jax; print(jax.devices())'"
Write-Output "y" | gcloud compute tpus tpu-vm ssh $Node --project=$Project --zone=$Zone --command="$setupCmd" 2>&1

Write-Host "2. Uploading V2 workspace..."
# We create the directory on the remote first
Write-Output "y" | gcloud compute tpus tpu-vm ssh $Node --project=$Project --zone=$Zone --command="mkdir -p /home/cityz/emergent_quantum_geometries/1_ingestion /home/cityz/emergent_quantum_geometries/2_state_mapping /home/cityz/emergent_quantum_geometries/3_projection /home/cityz/emergent_quantum_geometries/4_observation /home/cityz/emergent_quantum_geometries/5_classification" 2>&1 | Out-Null

# Then upload files individually
$files = @(
    "run_v2_experiment_updated.py",
    "fetch_datasets.py",
    "1_ingestion\experimental_data_ingester.py",
    "2_state_mapping\dlinoss_quantum_damper.py",
    "3_projection\heisenberg_space.py",
    "3_projection\bitwistor_space.py",
    "3_projection\syk_space.py",
    "3_projection\mbl_space.py",
    "3_projection\lindblad_space.py",
    "4_observation\otoc_metrics.py",
    "4_observation\or_kink_detection.py",
    "5_classification\nqs_meta_learner.py",
    "5_classification\framework_residuals.py"
)

foreach ($f in $files) {
    $localPath = "$WorkDir\$f"
    $remotePath = "/home/cityz/emergent_quantum_geometries/$($f.Replace('\','/'))"
    Write-Host "  Uploading $f..."
    Write-Output "y" | gcloud compute tpus tpu-vm scp $localPath "${Node}:$remotePath" --project=$Project --zone=$Zone 2>&1 | Out-Null
}

Write-Host "3. Generating Datasets on TPU..."
Write-Output "y" | gcloud compute tpus tpu-vm ssh $Node --project=$Project --zone=$Zone --command="cd /home/cityz/emergent_quantum_geometries && python3 fetch_datasets.py --generate-syk" 2>&1

Write-Host "4. Running V2.1 Master Script on TPU..."
Write-Output "y" | gcloud compute tpus tpu-vm ssh $Node --project=$Project --zone=$Zone --command="cd /home/cityz/emergent_quantum_geometries && python3 run_v2_experiment_updated.py --source SYK_ED --data_dir ./data --both-classifiers" 2>&1

Write-Host "=== Deployment Complete ==="
