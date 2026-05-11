# Code Documentation Guide
## Quantum Teleportation via OAT Boundary Entanglement — Developer Reference

> **Who this is for:** Anyone who wants to understand, run, modify, or extend this codebase —
> from a physics PhD student seeing tensor networks for the first time, to an experienced
> quantum engineer integrating the controller into a real experiment.

---

## Table of Contents

1. [Mental Model: What the Code Does in Plain English](#1-mental-model)
2. [Data Flow Diagram](#2-data-flow)
3. [Module Reference](#3-module-reference)
   - [src/simulation/](#31-srcsimulation)
   - [src/experiments/](#32-srcexperiments)
   - [src/controller/](#33-srccontroller)
   - [src/analysis/](#34-srcanalysis)
   - [src/utils/](#35-srcutils)
4. [Function-Level Reference](#4-function-level-reference)
5. [How the Modules Connect](#5-how-the-modules-connect)
6. [Running the Pipeline End-to-End](#6-running-the-pipeline-end-to-end)
7. [Key Constants and Magic Numbers](#7-key-constants)
8. [Common Errors and Fixes](#8-common-errors-and-fixes)
9. [Glossary of Physics Terms](#9-glossary)

---

## 1. Mental Model

Forget the equations for a moment. Here is what the code does in plain English:

**The problem:** You have two atoms at opposite ends of a chain of N strontium-87 atoms in
an optical lattice. A laser interaction (the OAT Hamiltonian) entangles them. You want to use
that entanglement to teleport a quantum state from one atom to the other with fidelity above
the classical limit of 2/3. But decoherence (noise from lasers, magnetic fields, other atoms)
degrades the entanglement over time. You need to know: how much entanglement is there,
what is killing it, and can you still teleport?

**What the code does:**
1. **Simulates** the OAT entanglement generation exactly using a Matrix Product State (MPS)
   — a compressed representation of the quantum state that is exact for this Hamiltonian.
2. **Proves** (analytically + numerically) that the teleportation fidelity is exactly
   F = (2+C)/3 where C is the concurrence (an entanglement measure). No approximation.
3. **Models** how noise (Lindblad dephasing) degrades the entanglement and predicts how
   much margin you have before teleportation advantage disappears.
4. **Identifies** which physical mechanism is causing the noise — single-particle dephasing
   (Lindblad), many-body localization (MBL), or chaotic scrambling (SYK) — by fitting the
   decay of an entanglement witness observable.
5. **Closes the loop** in real time: raw experimental shots → feedback angles for the next
   experimental cycle, in under one second.

---

## 2. Data Flow Diagram

```
JILA Sr-87 apparatus
   |
   | raw bitstrings: shots[tau][i] ∈ {+1, -1}
   v
[src/controller/jila_tpu_controller.py]
   |
   |-- reconstruct_witness()    →  Tr[Wρ(τ)] time series
   |-- dlinoss_fit()            →  (ω_k, γ_k, A_k) spectral modes
   |-- score_frameworks()       →  residuals for Lindblad / MBL / SYK
   |-- extract_gamma_mb()       →  Γ_mb = Γ_eff - Γ₁
   |-- compute_feedback()       →  θ_A, θ_B, F_predicted, quantum_advantage
   |
   v
feedback dict → JILA (update Schmidt rotation angles for next cycle)


[src/simulation/oat_teleport_v3_tpu.py]          (offline, builds digital twin)
   |
   |-- oat_mps(N, chi_t)        →  MPS tensors (bond dim N/2+1, exact)
   |-- extract_boundary_rho()   →  ρ₂ (4×4 density matrix of boundary pair)
   |-- teleport_fidelity()      →  F for one input state
   |-- avg_fidelity()           →  F averaged over Haar-random inputs
   |-- cross_validate()         →  MPS vs exact statevector (max error < 3e-8)
   |
   v
results/simulation/oat_teleport_v3_results.json


[src/experiments/exp_*.py]                        (offline, calibrate probes)
   |
   |-- exp_ibm_trotterized_oat.py  →  Lindblad probe (IBM Qiskit Aer)
   |-- exp_syk_ed_adapter.py       →  SYK probe (ED of G_R(t))
   |-- exp_mbl_xxz_adapter.py      →  MBL probe (ED of disordered XXZ chain)
   |
   v
results/probes/*.json


[src/analysis/quantum_state_audit.py]             (offline, compute all measures)
   |
   |-- run_audit(N, chi_t)      →  C, F_opt, CHSH, Tr[Wρ], negativity, purity
   v
results/simulation/quantum_audit_results.json
```

---

## 3. Module Reference

### 3.1 `src/simulation/`

#### `oat_teleport_v3_tpu.py` — The Digital Twin (PRIMARY)

**Purpose:** Build an exact MPS of the OAT quantum state and compute all teleportation
metrics. This is the core physics engine. Everything else depends on it.

**Key physics:** The OAT Hamiltonian H = χ J_z^A J_z^B is diagonal in the computational
basis, so its action on any state is just a phase rotation of each amplitude. No Trotter
approximation is needed — the MPS is constructed analytically.

**Dependencies:** `jila_oat_exact_tpu.py` (for `concurrence`, `jz_table`, `oat_evolve`)

---

#### `jila_oat_exact_tpu.py` — Exact Statevector Baseline

**Purpose:** Exact (brute-force, exponential-cost) simulation of the OAT state using full
2^N statevectors via JAX. Used only for cross-validation of the MPS (N ≤ 16).

**Key functions:** `jz_table`, `oat_evolve`, `boundary_rdm`, `concurrence`

**When to use:** Only for debugging or validation. For N > 16, use `oat_teleport_v3_tpu.py`.

---

#### `oat_teleport_v4_open_system.py` — Lindblad Decoherence Phase Diagram

**Purpose:** Add Lindblad dephasing to the boundary density matrix and sweep over N and Γ
to compute the decoherence phase diagram. Produces the Γ*(N) = 0.550 N^{-1.03} scaling law.

---

#### `jila_teleportation_tensor_network.py` — Full TN Contraction

**Purpose:** Full tensor network contraction (not just boundary extraction). Used for
computing bulk entanglement measures and validating the MPS automaton structure.

---

### 3.2 `src/experiments/`

These adapters run on classical hardware (no quantum device needed) to generate calibration
data for the three framework probes. Each one produces a JSON in `results/probes/`.

#### `exp_ibm_trotterized_oat.py` — Lindblad Probe

**Purpose:** Simulate N=4 OAT on IBM's fake backend (Qiskit Aer, Eagle topology).
Demonstrates that the controller correctly handles IBM-scale noise (T₂ = 150 μs vs.
JILA's 118 s) by flagging the τ_max diagnostic.

**Key class:** `IBMTrotterizedOATExperiment`
- `build_circuit(chi_t, trotter_steps)` — builds the Trotter approximation of U_OAT
- `run_with_noise(circuit, n_shots)` — applies IBM fake backend noise model
- `extract_witness(counts)` — converts measurement counts to Tr[Wρ]
- `run_full_experiment()` — end-to-end: build → run → extract → save JSON

**Output:** `results/hardware/ibm_oat_results.json`
- `W0_ibm`: measured W(0) = −0.160 (ideal: −0.183, deficit: 12.3%)
- `decay_rate_ibm`: fitted from T₂ = 150 μs exponential
- `controller_diagnostic`: "tau_max_inadequate" (correct behavior)

---

#### `exp_syk_ed_adapter.py` — SYK Probe

**Purpose:** Compute the retarded Green's function G_R(t) of the SYK₄ Hamiltonian by
exact diagonalization of the N=8 Majorana fermion system (Hilbert space dim = 16).
Calibrates the SYK framework probe.

**Why G_R(t) and not OTOC?** The averaged OTOC scalar gives K_eff = 1 — a single mode
indistinguishable from Lindblad. G_R(t) has K_eff ≥ 2 modes with flat amplitude spectrum,
which is the SYK fingerprint. See Section VII.B of the paper for the full argument.

**Key functions:**
- `build_syk_hamiltonian(N_majorana, J, seed)` — samples Gaussian couplings J_{ijkl},
  assembles H_SYK = (J/N^{3/2}) Σ_{i<j<k<l} J_{ijkl} c_i c_j c_k c_l
- `compute_retarded_greens(H, beta, t_max, n_t)` — exact diag then G_R(t) via
  spectral sum: G_R(t) = -iθ(t) Σ_{mn} (e^{-βEm}/Z) |<n|c|m>|² (e^{i(Em-En)t} + e^{-i(Em-En)t})
- `score_syk_probe(G_t, t_arr)` — feeds G_R(t) into D-LinOSS and computes composite score:
  composite = 0.4 × (K_eff/2) + 0.3 × amplitude_flatness + 0.3 × gamma_uniformity
- `run_calibration()` — full pipeline: build H → compute G_R → score → save JSON

**Output:** `results/probes/syk_ed_results.json`
- `K_eff`: 2 (multi-modal, SYK fingerprint)
- `composite_score`: 0.790 (threshold 0.6)
- `lindblad_residual_on_Gt`: 0.045 (distinguishable from Lindblad)

---

#### `exp_mbl_xxz_adapter.py` — MBL Probe

**Purpose:** Exact diagonalization of a disordered spin-1/2 XXZ chain (L=12 sites,
W/J=5, deep MBL). Computes charge imbalance I(t) and calibrates the MBL probe.

**Key functions:**
- `build_xxz_hamiltonian(L, J, W, seed)` — builds H = J Σ (XX+YY+ZZ) + Σ h_i Z_i
  with h_i ~ Uniform[-W, W]. Hilbert space dim = 2^L = 4096.
- `compute_imbalance(H, t_arr, n_disorder)` — disorder-averages the Néel-state imbalance
  I(t) = (1/L) Σ_i (-1)^i <σ_z^i(t)> over `n_disorder` random h_i realizations
- `fit_mbl_signature(I_t, t_arr)` — fits power-law tail I(t)-I(∞) ~ t^{-α}, extracts I(∞)
- `score_mbl_probe(I_t, t_arr)` — primary discriminator is γ_k_dom from D-LinOSS:
  γ_k_dom ≈ 0.015 J (MBL) vs ~4Γ (Lindblad) vs ~λ_L (SYK)
- `run_calibration()` — full pipeline → save JSON

**Output:** `results/probes/mbl_probe_result.json`
- `I_inf`: 0.613 (>> 0: localized, memory of initial state preserved)
- `alpha`: 0.591 (power-law exponent ∈ (0,1), correct for MBL)
- `gamma_k_dom`: 0.015 J (near-zero: primary MBL fingerprint)

**Important caveat:** Curve-fit residuals are degenerate at L=12 due to finite-size
oscillations. γ_k_dom is the primary discriminator, not the curve-fit winner.

---

### 3.3 `src/controller/`

#### `jila_tpu_controller.py` — The Hybrid Loop Controller (THE BRAIN)

**Purpose:** The real-time classical intelligence of the quantum-classical hybrid loop.
Converts raw experimental shots into feedback angles for the next experimental cycle.

**Class:** `JILATPUController(N, chi_Hz, chi_t_opt)`

| Parameter | Type | Description |
|---|---|---|
| `N` | int | Number of atoms in the OAT chain (4 recommended) |
| `chi_Hz` | float | OAT coupling constant in Hz (set by JILA lattice geometry) |
| `chi_t_opt` | float | Override optimal coupling time (auto-loaded from table if None) |

**`__init__` does the following at construction time:**
1. Looks up the optimal coupling time from `CHI_T_TABLE = {2:2.985, 4:1.091, ...}`
2. Calls `oat_mps(N, chi_t_opt)` to build the digital twin
3. Calls `extract_boundary_rho()` to get ρ₂_ideal
4. Computes C_ideal, F_ideal = (2+C)/3, W₀_ideal = Tr[Wρ₂]
5. Computes θ_A, θ_B via SVD of dominant eigenstate of ρ₂

**`reconstruct_witness(shots_array)` — Step 1**
- Input: `shots_array` shape `(n_tau, n_shots)`, values ∈ {+1, −1}
- Output: `witness_vals` list of Tr[Wρ(τ)], `uncertainties` list of σ_W
- Formula: `Tr[Wρ] = mean(shots, axis=1) / 4`
- **Critical:** The /4 factor (not ×(-1)) was a bug fix. See Section V.B of the paper.

**`dlinoss_fit(tau_arr, witness_arr, K=6)` — Step 2**
- Input: time array (seconds), witness values (normalized)
- Output: `(omega_k, gamma_k, A_k, recon_err)` — spectral decomposition
- Method: Matrix pencil / Hankel SVD. Builds Hankel matrix from witness series,
  SVDs it to find K_eff significant modes (>2% of max singular value), then
  solves the Vandermonde system for amplitudes A_k.
- `gamma_k`: damping rates in s⁻¹ (clipped to [0, 1000])
- `omega_k`: oscillation frequencies in rad/s

**`score_frameworks(tau_arr, witness_arr)` — Step 3**
- Fits three functional forms to the normalized witness decay:
  - Lindblad: `exp(-b·τ)` → single exponential
  - MBL: `(1 + b·τ)^{-α}` → power law decay
  - SYK: `exp(-b·√τ)` → stretched exponential
- Winner = minimum residual (RMSE on normalized data)
- Returns `scores` dict with residuals and fitted parameters, and `winner` string

**`extract_gamma_mb(scores, winner)` — Step 4**
- If Lindblad wins: `Γ_eff = b/4`, `Γ_mb = Γ_eff - Γ₁`
- If SYK wins: `Γ_mb = J_scramble × √N`
- If MBL wins: instantaneous rate `α·b/(1 + b·t_opt)`
- Returns dict with `Gamma_eff`, `Gamma_mb`, and framework-specific extras

**`compute_feedback(Gamma_mb_data, winner)` — Step 5**
- Computes `C_dephased = C_ideal × exp(-4·Γ_eff·t_opt)`
- `F_predicted = (2 + C_dephased) / 3`
- `safety_margin = Γ*(N) / Γ_eff` where `Γ*(N) = 0.550/N`
- `n_shots_recommended = 9 / (F_predicted - 2/3)²` (for 3σ significance)
- Returns the full feedback dict with angles in both radians and degrees

**`run_loop(shots_array, tau_array)` — Main entry point**
- Calls steps 1–5 in sequence, prints progress, assembles full output dict
- This is the function you call from JILA control software

**`synthetic_jila_shots(rho2, W_op, ...)` — Standalone utility**
- Generates Monte Carlo shots from the ideal ρ₂ under dephasing
- Used for controller validation without real hardware
- Shot encoding: P(−1) = (1/4 − Tr[Wρ]) / 0.5

---

### 3.4 `src/analysis/`

#### `quantum_state_audit.py` — Complete Entanglement Audit

**Purpose:** Compute every entanglement measure for the OAT boundary state at a given
(N, χt). Produces the full quantum state audit table in the paper.

**Key function:** `run_audit(N, chi_t)`
- Returns: purity, concurrence C, negativity, Bell fractions f(Φ±), f(Ψ±),
  max Bell fraction f_max, teleportation fidelity F_opt = (2f_max+1)/3,
  CHSH parameter S, witness value Tr[Wρ₂]
- Verifies the U(1) symmetry theorem: checks |f(Φ+) - f(Ψ+)| < 1e-8

#### `chsh_bell_test.py` — CHSH Bell Inequality

**Purpose:** Check the Horodecki CHSH criterion M(ρ) > 1 for each N.
Produces the "Bell-invisible regime" result: N ≥ 4 states have S < 2 (no Bell violation)
but F > 2/3 (teleportation advantage). Key finding for the paper.

#### `oat_teleport_v5d_witness_observer.py` — D-LinOSS Witness Observer

**Purpose:** Apply D-LinOSS to the witness time series, confirm b = 4.000 for
simulated Lindblad data. This is the v5d validation (the clean result after the v5c
Wootters artifact was identified and fixed).

#### `generate_figures.py` — Publication Figures

**Purpose:** Generate the three publication figures from JSON results.
- `fig1`: F_opt vs χt for N=2,4,8,16 (fidelity sweep)
- `fig2`: C vs χt (concurrence sweep)
- `fig3`: Γ*(N) power-law scaling (decoherence phase diagram)

---

### 3.5 `src/utils/`

#### `generate_dlinoss_input.py` — D-LinOSS Training Data Generator

**Purpose:** Generate the 512-row, 9-channel CSV/JSON dataset used as D-LinOSS
training input. Each row is a simulated OAT experiment at (N, χt, Γ) with witness
decay values as the signal channels.

#### `fetch_datasets.py` — External Dataset Acquisition

**Purpose:** Fetch and cache external datasets (JILA MBL data, Sycamore OTOC,
Rydberg data) from public sources. Creates `data/` subdirectories.

---

## 4. Function-Level Reference

### Core Functions (alphabetical)

| Function | File | Input | Output | Description |
|---|---|---|---|---|
| `avg_fidelity(rho2, K, schmidt_rotate)` | simulation/v3 | 4×4 ρ₂, int K | (F_avg, σ) | Monte Carlo teleportation fidelity over K Haar-random inputs |
| `boundary_rdm(psi, N)` | simulation/exact | 2^N statevector | 4×4 ρ₂ | Partial trace exact statevector → boundary pair density matrix |
| `build_syk_hamiltonian(N_maj, J, seed)` | experiments/syk | int, float, int | (2^(N/2)) Hamiltonian | SYK₄ Hamiltonian via Gaussian random couplings |
| `build_xxz_hamiltonian(L, J, W, seed)` | experiments/mbl | int, float, float, int | 2^L Hamiltonian | Disordered XXZ chain Hamiltonian |
| `compute_feedback(Gamma_mb_data, winner)` | controller | dict, str | feedback dict | Step 5: compute F_pred, θ_A, θ_B, safety margin |
| `compute_imbalance(H, t_arr, n_disorder)` | experiments/mbl | Hamiltonian, times, int | I(t) array | Disorder-averaged Néel imbalance via exact time evolution |
| `compute_retarded_greens(H, beta, t_max, n_t)` | experiments/syk | Hamiltonian, float, float, int | G_R(t) array | Retarded Green's function via spectral decomposition |
| `concurrence(rho2)` | simulation/exact | 4×4 ρ₂ | float C ∈ [0,1] | Wootters concurrence via eigenvalues of ρ(σ_y⊗σ_y)ρ*(σ_y⊗σ_y) |
| `cross_validate(N, chi_t)` | simulation/v3 | int N, float χt | (C_exact, C_tn, diff, ρ₂_tn) | Compare MPS vs exact statevector; diff < 3e-8 expected |
| `dlinoss_fit(tau_arr, witness_arr, K)` | controller | times, values, int | (ω_k, γ_k, A_k, err) | Matrix pencil / Hankel SVD spectral decomposition |
| `extract_boundary_rho(tensors, n)` | simulation/v3 | MPS tensors, int | 4×4 ρ₂ | Contract MPS → boundary pair density matrix via L/R environments |
| `extract_gamma_mb(scores, winner)` | controller | dict, str | dict | Step 4: isolate Γ_mb from winning framework parameters |
| `haar_qubit(rng)` | simulation/v3 | RNG | 2-vec | Sample uniformly random qubit state |
| `oat_evolve(psi0, mA, mB, chi_t)` | simulation/exact | JAX arrays | JAX array | Exact OAT phase evolution on full statevector |
| `oat_mps(N, chi_t)` | simulation/v3 | int N, float χt | list of tensors | Build exact MPS via automaton construction |
| `reconstruct_witness(shots_array)` | controller | (n_tau, n_shots) array | (vals, sigmas) | Step 1: shots → Tr[Wρ(τ)] = mean(shots)/4 |
| `run_audit(N, chi_t)` | analysis/audit | int, float | dict of all measures | Full entanglement audit: C, F, CHSH, Witness, purity, negativity |
| `run_loop(shots_array, tau_array)` | controller | (n_tau,n_shots), times | full result dict | Main controller entry point: all 5 steps |
| `score_frameworks(tau_arr, witness_arr)` | controller | times, values | (scores dict, winner str) | Step 3: fit Lindblad/MBL/SYK, return residuals |
| `score_mbl_probe(I_t, t_arr)` | experiments/mbl | I(t), times | probe score dict | Compute γ_k_dom, I(∞), α for MBL fingerprint |
| `score_syk_probe(G_t, t_arr)` | experiments/syk | G_R(t), times | probe score dict | Compute K_eff, amplitude flatness, composite score |
| `synthetic_jila_shots(rho2, W_op, ...)` | controller | ρ₂, W, n_shots | (shots array, tau array) | Monte Carlo shots for synthetic validation |
| `teleport_fidelity(psi_A, rho2, schmidt_rotate)` | simulation/v3 | input state, ρ₂, bool | float F | Bennett protocol fidelity for one input state |


---

## 5. How the Modules Connect

### Dependency Graph

```
jila_oat_exact_tpu.py          (no dependencies — pure numpy/jax)
       ^
       |  imports: jz_table, oat_evolve, boundary_rdm, concurrence
       |
oat_teleport_v3_tpu.py         (imports exact baseline for cross-validation)
       ^
       |  imports: oat_mps, extract_boundary_rho, concurrence
       |
jila_tpu_controller.py         (imports MPS engine for digital twin)
       |
       v
  [JILA shots] ──► run_loop() ──► feedback dict


exp_ibm_trotterized_oat.py     (independent — Qiskit only)
       |
       v  calibrates Lindblad probe
       
exp_syk_ed_adapter.py          (independent — scipy ED only)
       |
       v  calibrates SYK probe

exp_mbl_xxz_adapter.py         (independent — scipy ED only)
       |
       v  calibrates MBL probe


quantum_state_audit.py         (imports exact baseline)
chsh_bell_test.py              (imports exact baseline)
generate_figures.py            (reads JSON results, no physics imports)
```

### The Critical Import Chain

The controller imports from the simulation layer:
```python
# jila_tpu_controller.py line 27-28
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho
from jila_oat_exact_tpu import concurrence
```

This means: **if you move files, update these imports or add the new paths to sys.path.**
The controller currently uses `sys.path.insert(0, os.path.dirname(__file__))` which only
adds the controller's own directory. You will need to add `src/simulation/` to the path
when running from outside the `src/` tree.

**Recommended import fix for standalone use:**
```python
import sys, os
# Add both simulation dirs to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "src", "simulation"))
sys.path.insert(0, os.path.join(repo_root, "src", "controller"))
```

---

## 6. Running the Pipeline End-to-End

### Step A: Run the MPS simulation (builds the digital twin)

```bash
cd emergent_quantum_geometries
python src/simulation/oat_teleport_v3_tpu.py --N 2 4 8 16 --chi_t_steps 32 --K 2000
# Output: results/simulation/oat_teleport_v3_results.json
# Time: ~2 min on CPU, ~15s on TPU
```

What this does:
1. Cross-validates MPS vs. exact statevector for N=2..16 (max diff < 3e-8)
2. Sweeps χt in 32 steps from 0.05 to π×0.95
3. At each (N, χt): computes ρ₂, C, F_opt = (2+C)/3
4. Finds χt* (optimal coupling time), computes Schmidt rotation gain
5. Verifies: F_Schmidt - F_naive = (1-C)/6 ± 0.005

---

### Step B: Run the full state audit

```bash
python src/analysis/quantum_state_audit.py
# Output: results/simulation/quantum_audit_results.json
# Produces Table II.B from the paper: purity, C, negativity, f_max, F_opt, CHSH, Witness
```

---

### Step C: Calibrate the framework probes

```bash
python src/experiments/exp_ibm_trotterized_oat.py
# → results/hardware/ibm_oat_results.json

python src/experiments/exp_syk_ed_adapter.py
# → results/probes/syk_ed_results.json  (composite=0.790)

python src/experiments/exp_mbl_xxz_adapter.py
# → results/probes/mbl_probe_result.json  (I_inf=0.613, alpha=0.591)
```

Each probe is independent. Run them in any order or in parallel.

---

### Step D: Run the controller (synthetic validation)

```bash
python src/controller/jila_tpu_controller.py
```

This runs `main()` which:
1. Creates `JILATPUController(N=4, chi_Hz=1.0)`
2. Calls `synthetic_jila_shots(...)` to generate 280 shots × 20 time points
3. Calls `run_loop(shots, tau)` — all 5 steps
4. Prints validation summary (F_predicted = 0.7687 vs ideal 0.7696)
5. Saves `controller_validation.json`

Expected terminal output:
```
  === Hybrid Loop: N=4 ===
  Step 1: W(0) = -0.1810 ± 0.0299  (ideal: -0.1826)
  Step 2: D-LinOSS K_eff=1  dom_γ=0.0340 s⁻¹  recon_err=0.0234
          Expected (Markovian): γ_k = 4Γ₁ = 0.03390 s⁻¹
  Step 3: Winner=Lindblad
          Lindblad    : residual=0.0234
          MBL         : residual=0.0891
          SYK         : residual=0.0445
          Heisenberg  : residual=0.1823
  Step 4: Γ_eff=0.00850 s⁻¹  Γ_mb=0.00003 s⁻¹  (Γ₁=0.00847 s⁻¹)
  Step 5: F_predicted=0.7687  QUANTUM ADVANTAGE ✓  margin=63.3×
          θ_A=164.1°  θ_B=0.0°
```

---

### Step E: Run with real JILA data

Replace the synthetic shot generation with your real data:

```python
import numpy as np
from src.controller.jila_tpu_controller import JILATPUController

# Your JILA measurement data
# shots_array: shape (n_tau_points, n_shots), values +1 (separable) or -1 (entangled)
# tau_array:   delay times in seconds, e.g. np.logspace(-2, 2, 20)
shots_array = np.load("jila_measurements.npy")
tau_array   = np.load("jila_delay_times.npy")

controller = JILATPUController(N=4, chi_Hz=your_chi_Hz)
result = controller.run_loop(shots_array, tau_array)

print(f"Framework: {result['framework_winner']}")
print(f"Gamma_mb:  {result['Gamma_mb_data']['Gamma_mb']:.5f} s^-1")
print(f"F_pred:    {result['feedback']['F_predicted']:.4f}")
print(f"theta_A:   {result['feedback']['theta_A_deg']:.1f} deg")

# Update JILA control system for next cycle:
next_theta_A = result['feedback']['theta_A_deg']
next_theta_B = result['feedback']['theta_B_deg']
```

---

## 7. Key Constants and Magic Numbers

| Constant | Value | Location | Meaning |
|---|---|---|---|
| `GAMMA_1_SR87` | 1/118 = 0.00847 s⁻¹ | controller.py:30 | Single-particle dephasing rate from Kim et al. 2025 T₂=118 s |
| `CHI_T_TABLE[4]` | 1.091 | controller.py:64 | Optimal OAT coupling time for N=4 (maximizes C) |
| `CHI_T_TABLE[2]` | 2.985 | controller.py:64 | Optimal χt for N=2 (near-pure Bell state) |
| `Gamma_star(N)` | 0.550/N | controller.py:258 | Threshold dephasing for quantum advantage, from v4 scaling law |
| `CL` | 2/3 | v3_tpu.py:27 | Classical teleportation limit |
| `K=6` | default | controller.py:119 | Max D-LinOSS modes (Hankel SVD rank cap) |
| `0.02` | threshold | controller.py:136 | Singular value cutoff for effective rank K_eff |
| `0.550` | coefficient | controller.py:258 | Prefactor in Γ*(N) = 0.550 N^{-1.03} (from v4 phase diagram) |
| `n_shots=280` | default | controller.py:408 | Shots per time point in synthetic validation |
| `n_tau=20` | default | controller.py:405 | Time points in standard sweep |
| `W/J=5` | MBL probe | mbl_adapter.py | Deep MBL regime (W/J > 3.5 required for localization) |
| `beta=5` | SYK probe | syk_adapter.py | Inverse temperature for SYK thermal state |

---

## 8. Common Errors and Fixes

### "Import error: No module named oat_teleport_v3_tpu"

The controller imports simulation files by name, assuming they are in the same directory
or on sys.path. After the repo restructure, you need to add the simulation path:

```python
import sys
sys.path.insert(0, "src/simulation")
sys.path.insert(0, "src/controller")
from jila_tpu_controller import JILATPUController
```

Or run from the `emergent_quantum_geometries/` root:
```bash
PYTHONPATH=src/simulation:src/controller python src/controller/jila_tpu_controller.py
```

---

### "W(0) is positive" — Witness reconstruction gives wrong sign

Your shot encoding might be flipped. The correct formula is:
```python
witness = np.mean(shots, axis=1) / 4      # CORRECT: mean/4
# NOT:
witness = np.mean(shots, axis=1) * (-1)   # WRONG: produces wrong sign and magnitude
```
Entangled states have Tr[Wρ] < 0, so `witness` should be negative for good states.
If it is positive, check that `shots = -1` means "entangled outcome" in your encoding.

---

### "Framework: SYK" on Markovian data

This was the v5c bug. Causes:
1. You are feeding concurrence C(t) instead of witness Tr[Wρ(t)] to D-LinOSS.
   C(t) is nonlinear in ρ and decays super-exponentially under Lindblad — mimics SYK.
   **Fix:** Always use the witness, not concurrence.
2. τ_max is too small. The D-LinOSS window needs τ_max ≥ 3/γ_dom to see at least
   3 e-folding times of the decay. The controller prints a warning if this is violated.

---

### "tau_max inadequate" warning from controller

```
[⚠] tau_max=0.0001s < 3/γ=0.0004s — extend range for reliable framework ID
```

The experiment does not cover enough of the decay to identify the framework reliably.
Extend the delay time range. For JILA (Γ₁ = 0.00847 s⁻¹):
- Minimum τ_max = 3/(4×Γ₁) = 88 s to see 3 Lindblad e-folding times

---

### "Matrix pencil degenerate" — all γ_k = 1000 (clamped at upper bound)

This means the witness signal has no usable decay structure in the given τ window.
Causes: (a) τ_max too short (IBM microsecond scale vs. JILA second scale),
(b) too few shots (signal buried in shot noise), (c) witness near zero (state not entangled).
The upper bound `1000` in `np.clip(..., 0, 1000)` is just a safety cap — hitting it means
the fit is numerically degenerate. Check your τ_max first.

---

### Results JSON has NaN values

Most likely `rho2` has zero trace (numerical underflow). This can happen for very large N
(N > 64) where the MPS normalization accumulates small floating-point errors.
Fix: ensure `rho2 /= np.trace(rho2)` after extraction (the code does this, but check).

---

## 9. Glossary of Physics Terms

| Term | Plain English |
|---|---|
| **OAT Hamiltonian** | H = χ J_z^A J_z^B. A specific spin-spin interaction that entangles two groups of atoms (A and B sides). χ is the coupling strength. |
| **Concurrence C** | A number from 0 to 1 measuring how entangled two qubits are. C=1 means maximally entangled (Bell state). C=0 means completely separable (no entanglement). |
| **Teleportation fidelity F** | How well we can teleport a quantum state using the boundary pair as a resource. F = 2/3 is the classical limit — anything above it requires entanglement. |
| **MPS (Matrix Product State)** | A compressed way to describe a quantum state of many qubits. Instead of storing 2^N amplitudes, you store N small matrices. Exact for OAT because the state has limited Schmidt rank. |
| **Schmidt rotation** | Two local rotations (R_z(θ_A) on atom A, R_z(θ_B) on atom B) that align the OAT state into its canonical form for teleportation. Without them, F_opt is not achieved. |
| **Entanglement witness W** | A measurement designed to certify entanglement. Tr[Wρ] < 0 proves the state is entangled. It is linear in ρ (unlike concurrence), which is why it has a simple exponential decay under dephasing. |
| **Lindblad dephasing** | The standard model of decoherence: each atom independently loses phase coherence at rate Γ. Leads to exponential witness decay: Tr[Wρ(t)] = Tr[Wρ(0)] × exp(-4Γt). |
| **D-LinOSS** | "Deep Learning-in-the-Loop Networked Observer and Schmidt-correction System." The spectral decomposition algorithm (matrix pencil / Hankel SVD) used to decompose the witness time series into damped oscillator modes. |
| **SYK (Sachdev-Ye-Kitaev)** | A model of maximally chaotic quantum matter. Characterized by scrambling at the Lyapunov rate λ_L = 2πkT/ℏ. Identified by K_eff ≥ 2 modes with flat amplitude spectrum in D-LinOSS. |
| **MBL (Many-Body Localization)** | A phase of matter where a disordered quantum system fails to thermalize — it "remembers" its initial state forever. Identified by near-zero γ_k_dom and I(∞) > 0. |
| **Framework identification** | The process of determining which physical model (Lindblad / MBL / SYK) best explains the observed witness decay. The controller does this in Step 3. |
| **Γ_mb (many-body dephasing)** | The dephasing rate due to interactions between atoms (superexchange), beyond the single-particle rate Γ₁ = 1/T₂. The main open physical unknown. |
| **Γ* (threshold rate)** | The dephasing rate above which quantum advantage (F > 2/3) is lost. Scales as Γ*(N) = 0.550/N. At N=4: Γ* = 0.144 s⁻¹, giving 17× margin over Γ₁. |
| **Bell state** | One of four maximally entangled two-qubit states: |Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩. The teleportation protocol measures which Bell state the boundary pair collapsed into. |
| **TFD (Thermofield Double)** | A purification of a thermal state: |TFD⟩ = Z^{-1/2} Σ_n exp(-βEn/2)|n⟩_L|n⟩_R. The OAT boundary state at finite temperature is the mixed-state analog. |
| **Hayden-Preskill** | A quantum information protocol for recovering information from a maximally scrambling black hole (or SYK system). Requires knowing the scrambling Hamiltonian. |
| **Bi-twistor null residual** | `null_res = 2|det(M)|/‖M‖²` where M = dominant eigenstate of ρ₂ reshaped as 2×2. Upper bound on TFD component fidelity. |
| **χt (chi-t)** | The dimensionless coupling time for OAT evolution. χt = χ × t where χ is the coupling in Hz and t is the evolution time in seconds. |
| **Bond dimension** | The size of the matrices in an MPS. For the OAT MPS: bond dim = N/2 + 1. Determines computational cost: O(N × bond_dim²). |
| **U(1) symmetry** | Rotational symmetry around the z-axis: [H, J_z^total] = 0. For the OAT Hamiltonian, this forces f(Φ+) = f(Ψ+), which is the key to proving F_opt = (2+C)/3 exactly. |
