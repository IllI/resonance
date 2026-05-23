# Program L TRC TPU Attempt Handoff

Date: 2026-05-15
Project: `time-emission`
Grant boundary: Google TRC TPU allocation only. Do not create TPUs outside the sanctioned TRC zones in `[TRC] You have access to free Cloud TPUs.html` and `trc_tpu_bible.md.resolved`.

## What Program L Is Trying To Test

Program L asks whether an adaptive controller can stabilize recoverable quantum transport under noise using only observable dynamics.

The controller is not supposed to see the Hamiltonian or fidelity directly. It should act from a 12-dimensional observable vector:

- `ptm_sv1`, `ptm_sv2`, `ptm_sv3`, `ptm_H`
- `EE`, `MI`, `OS`, `H_T`
- `D_eff`, `front_v`, `noise_slope`, `basis_invar`

The intended scientific claim is narrow:

> Recoverable transport can be predicted and stabilized from observable dynamics without Hamiltonian access, across multiple independently chosen representations.

This is not a claim about teleportation, nonlocal communication, or an ontologically real hidden geometry.

## Intended Run Command

The requested shotgun command was:

```powershell
.\run_program_l_shotgun.ps1 -N 10 -B 3 -Models "XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 OAT Ising TiltedIsing"
```

The launcher races TRC-approved zones and should keep only the first active queued resource:

- `prog-l-v4`: `us-central2-b`, `v4-8`, on-demand, no spot flag
- `prog-l-v6e1`: `europe-west4-a`, `v6e-8`, spot
- `prog-l-v6e2`: `us-east1-d`, `v6e-8`, spot

The local `gcloud` version recognizes `--spot` for queued resources. It does not show `--best-effort` in `gcloud compute tpus queued-resources create --help`.

## What Actually Happened

One shotgun attempt was launched through the installed Google Cloud SDK:

```powershell
$env:Path = "C:\Users\cityz\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin;" + $env:Path
.\run_program_l_shotgun.ps1 -N 10 -B 3 -Models "XXZ DisorderedXXZ_W1 DisorderedXXZ_W3 OAT Ising TiltedIsing"
```

Observed behavior:

- The script confirmed project `time-emission`.
- Pre-flight delete reported no queued resources in `us-central2-b` or `europe-west4-a`.
- Pre-flight delete still saw `prog-l-v6e2` in `us-east1-d`.
- `prog-l-v4` was created in `us-central2-b`.
- `prog-l-v6e1` was created in `europe-west4-a`.
- `prog-l-v6e2` creation failed with `ALREADY_EXISTS` because the queued resource already existed.
- The poll loop did not detect any resource becoming `ACTIVE` within 10 minutes.
- The script exited with failure before SSH, dependency install, upload, execution, download, or local analysis.

No `program_l_N10_results.json` was produced by this attempt.

## Resource State Seen Afterward

The TRC-approved zone scan reported:

```text
us-central2-b   prog-l-v4    WAITING_FOR_RESOURCES
europe-west4-a  prog-l-v6e1  SUSPENDED
us-east1-d      prog-l-v6e2  SUSPENDED
us-central1-a   no rows
europe-west4-b  no rows
```

All three named resources are in sanctioned TRC zones according to the local TRC rules. The TRC bible still says to delete queued resources before ending a session, including suspended resources, so verify and clean them up before closing the terminal unless you intentionally want them left queued.

Safe verification command:

```powershell
$g="C:\Users\cityz\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$zones=@("us-central2-b","europe-west4-a","us-east1-d","us-central1-a","europe-west4-b")
foreach ($z in $zones) { & $g compute tpus queued-resources list --project=time-emission --zone=$z }
```

Delete only the Program L queued resources:

```powershell
gcloud compute tpus queued-resources delete prog-l-v4 --project=time-emission --zone=us-central2-b --quiet
gcloud compute tpus queued-resources delete prog-l-v6e1 --project=time-emission --zone=europe-west4-a --quiet
gcloud compute tpus queued-resources delete prog-l-v6e2 --project=time-emission --zone=us-east1-d --quiet
```

## Compute-Path Fix Applied

The first attempted `program_l_tpu.py` did not follow the earlier TPU/JAX practice. That has now been partially repaired.

What was wrong before the fix:

- `program_l_tpu.py` imports `numpy as np` and `scipy.linalg as la`.
- It does not import `jax` or `jax.numpy`.
- It does not assert `jax.devices()` or `jax.default_backend()`.
- It uses dense full-state and density-matrix style operations such as `np.linalg.eigh`, `la.expm`, `np.kron`, `np.linalg.svd`, and Python loops.
- `simulate_trajectory()` evolves a dense state path with NumPy operations.

That means a TPU VM may have mostly behaved like a remote CPU host for this script. This likely explains why the last run felt much slower than the prior tensor-network/JAX TPU workflow.

Applied changes:

- `program_l_tpu.py` now has `--backend auto|numpy|jax` and `--require-tpu`.
- `--require-tpu` fails fast unless JAX sees a real TPU device.
- The Program L launcher now runs the experiment with `--backend jax --require-tpu`.
- The launcher installs JAX for TPU using the libtpu release index and prints `jax.default_backend()` plus `jax.devices()` before launching.
- Clifford pulses in the JAX path now use tensor reshaping/contraction rather than full dense Kronecker matrices.
- Hamiltonian-aware greedy evolution and boundary contractions can run through JAX device arrays while keeping the pre-registered stochastic noise path.
- The unnecessary agnostic-controller full density matrix allocation was removed.

Earlier working TPU-oriented files use JAX explicitly:

- `teleport_sim_tpu.py` imports `jax`, uses `jax.jit`, `jax.vmap`, and prints `jax.devices()`.
- `tpu_dlinoss_training_gen.py` imports `jax`, `jax.numpy as jnp`, `jit`, `vmap`, and prints devices.
- Program K deployment notes include TPU warm-up behavior and Windows/plink handling.

## Likely Root Causes Of The Failed Attempt

- Capacity did not reach `ACTIVE` within the script's 10-minute polling window.
- A stale `prog-l-v6e2` resource already existed in `us-east1-d`, so one candidate did not start cleanly.
- The launcher previously polled `state.state`, but the patched launcher now parses queued-resource JSON and handles both observed state shapes.
- Before the patch, this `program_l_tpu.py` version likely would not have used TPU acceleration effectively because it was NumPy/SciPy-first.
- `run_program_l_shotgun.ps1` previously used an OpenSSH-style `--ssh-flag="-o StrictHostKeyChecking=no"` path; the patched launcher removed it.
- Program K's launcher includes a 3-minute SSH daemon warm-up after `ACTIVE`; Program L now does the same.

## Recommended Next Fix Before Any More TRC Runs

Do not launch the full N=10/B=3 Program L run again until a small TPU smoke job passes.

Recommended order:

1. Run a small smoke job first: `-N 6 -B 1 -Models "XXZ Ising"` before any N=10 sweep.
2. Confirm the remote log prints TPU devices, not CPU only.
3. Confirm the result JSON records `"backend": "jax"` and `"has_tpu": true`.
4. Only after the smoke job produces a result JSON and analysis output, rerun the requested N=10/B=3 sweep.
5. If N=10 is still slow, the next optimization is a deeper rewrite away from dense eigensystem evolution toward a true MPS/tensor-network time evolution path.

## Minimal Safe Smoke Command

After the JAX/TPU path is repaired:

```powershell
.\run_program_l_shotgun.ps1 -N 6 -B 1 -Models "XXZ Ising" -MaxMin 60 -OutDir "program_l_smoke"
```

Expected smoke-test intent:

- XXZ should show agnostic improvement over static.
- Ising should remain a null control.
- Results should land under `emergent_quantum_geometries\program_l_smoke`.
- The launcher must delete queued resources and verify approved zones are empty when done.

## Bottom Line

This attempt did not produce scientific Program L data. It did confirm the local `gcloud` session works and that queued resources can be created in TRC-approved zones. The Program L path has now been moved back toward JAX/TPU execution, but the next TRC use should be a smoke job that proves the remote log sees TPU devices before spending queue time on the full N=10/B=3 sweep.
