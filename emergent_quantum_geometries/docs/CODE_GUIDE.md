# Code Documentation Guide
## Quantum Teleportation via OAT Boundary Entanglement - Developer Reference

> **Who this is for:** Anyone who wants to understand, run, modify, or extend this codebase - from a physics PhD student seeing tensor networks for the first time, to an experienced quantum engineer integrating the controller into a real experiment.

---

## Table of Contents

1. [Mental Model: What the Code Does in Plain English](#1-mental-model)
2. [Data Flow Diagram](#2-data-flow)
3. [Module Reference](#3-module-reference)
4. [Function-Level Reference](#4-function-level-reference)
5. [How the Modules Connect](#5-how-the-modules-connect)
6. [Running the Pipeline End-to-End](#6-running-the-pipeline-end-to-end)
7. [Key Constants and Magic Numbers](#7-key-constants)
8. [Common Errors and Fixes](#8-common-errors-and-fixes)
9. [Glossary of Physics Terms](#9-glossary)

---

## 1. Mental Model

Forget the equations for a moment. Here is what the code does in plain English:

**The problem:** You have two atoms at opposite ends of a chain of $N$ strontium-87 atoms in an optical lattice. A laser interaction (the OAT Hamiltonian) entangles them. You want to use that entanglement to teleport a quantum state from one atom to the other with fidelity above the classical limit of 2/3. But decoherence (noise from lasers, magnetic fields, other atoms) degrades the entanglement over time. You need to know: how much entanglement is there, what is killing it, and can you still teleport?

**What the code does:**
1. **Simulates** the OAT entanglement generation exactly using a Matrix Product State (MPS) - a compressed representation of the quantum state that is exact for this Hamiltonian.
2. **Proves** (analytically + numerically) that the teleportation fidelity is exactly $F = (2+C)/3$ where $C$ is the concurrence (an entanglement measure). No approximation.
3. **Models** how noise (Lindblad dephasing) degrades the entanglement and predicts how much margin you have before teleportation advantage disappears.
4. **Identifies** which physical mechanism is causing the noise - single-particle dephasing (Lindblad), many-body localization (MBL), or chaotic scrambling (SYK) - by fitting the decay of an entanglement witness observable.
5. **Closes the loop** in real time: raw experimental shots $\to$ feedback angles for the next experimental cycle, in under one second.

---

## 2. Data Flow Diagram

```mermaid
graph TD
    A[JILA Sr-87 Apparatus] -- "raw bitstrings (shots)" --> B[TPU Controller]
    B -- "Step 1: Invert shots" --> C[Witness Tr[Wρ]]
    C -- "Step 2: D-LinOSS Fit" --> D[Spectral Modes (ω_k, γ_k)]
    D -- "Step 3: Framework ID" --> E[Lindblad / SYK / MBL]
    E -- "Step 4: Rate Extract" --> F[Gamma_mb]
    F -- "Step 5: Compute Feedback" --> G[Angles θ_A, θ_B]
    G -- "Feedback stream" --> A
```

---

## 3. Module Reference

### `src/simulation/`
- **`oat_teleport_v3_tpu.py`**: The core MPS engine. Builds the exact automaton state for OAT.
- **`jila_oat_exact_tpu.py`**: Exact statevector baseline for cross-validation ($N \le 16$).
- **`oat_teleport_v4_open_system.py`**: Lindblad master equation solver for the decoherence phase diagram.

### `src/controller/`
- **`jila_tpu_controller.py`**: The real-time hybrid loop logic. Implements the 5-step feedback protocol.

### `src/experiments/`
- **`exp_syk_ed_adapter.py`**: Exact diagonalization of SYK Hamiltonian to calibrate the chaos probe.
- **`exp_mbl_xxz_adapter.py`**: ED of disordered XXZ chains to calibrate the localization probe.
- **`exp_ibm_trotterized_oat.py`**: Qiskit circuit for Trotterized OAT on noisy IBM hardware.

---

## 7. Key Constants and Magic Numbers

| Constant | Value | Location | Meaning |
|---|---|---|---|
| `GAMMA_1_SR87` | 1/118 s$^{-1}$ | `controller.py` | Single-particle dephasing rate ($T_2 = 118$ s) |
| `CHI_T_TABLE[4]` | 1.091 | `controller.py` | Optimal OAT coupling time for $N=4$ |
| `Gamma_star(N)` | $0.550/N$ | `controller.py` | Threshold for quantum advantage |
| `CL` | 2/3 | `v3_tpu.py` | Classical teleportation limit |

---

## 9. Glossary of Physics Terms

| Term | Plain English |
|---|---|
| **OAT Hamiltonian** | $H = \chi J_z^A J_z^B$. A collective interaction that entangles boundary atoms. |
| **Concurrence $C$** | A measure of two-qubit entanglement (0 to 1). |
| **Witness $W$** | An observable whose expectation value $\text{Tr}[W \rho] < 0$ certifies entanglement. |
| **$\chi t$** | Dimensionless coupling time. |
| **Schmidt Rotation** | Local $z$-rotations applied to boundary atoms to optimize teleportation. |
| **SYK** | Sachdev-Ye-Kitaev model, a maximally chaotic system. |
| **MBL** | Many-Body Localization, where information is trapped by disorder. |
| **Lindblad** | Standard model of memoryless, independent noise (Markovian dephasing). |
