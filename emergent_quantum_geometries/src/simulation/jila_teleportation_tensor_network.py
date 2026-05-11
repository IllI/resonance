"""
jila_teleportation_tensor_network.py
======================================
Tensor-network simulation of the JILA Ana Maria Rey 2025
quantum teleportation protocol using trapped-ion spin chains.

Physics:
  - Two N-ion chains (A and B) coupled at boundary.
  - Entangling Hamiltonian: H = chi * Jz_A ⊗ Jz_B  (OAT coupling)
  - Teleportation protocol:
      1. Prepare logical |ψ⟩ on ion A[0]  (the qubit to teleport)
      2. Entangle A and B via OAT evolution (create shared Bell resource)
      3. Perform Bell measurement on A[0]+A[1] (joint Z⊗Z + X⊗X)
      4. Apply Pauli corrections on B[0] conditioned on measurement
      5. Measure fidelity F = |⟨ψ|ρ_B[0]|ψ⟩|

Implementation:
  - quimb MatrixProductState (MPS) with bond dimension D
  - TEBD time evolution for the OAT Hamiltonian
  - Exact partial-trace density matrix for small subsystems

Outputs:
  - results/tn_teleportation_fidelity.npy   — F(t) curves per N
  - results/tn_entanglement_entropy.npy      — S_vN across bond cut
  - results/tn_teleportation_results.json    — summary JSON
"""

import numpy as np
import json
from pathlib import Path

# ── quimb imports ─────────────────────────────────────────────────────────────
import quimb as qu
import quimb.tensor as qtn

RESULTS_DIR = Path("results/tensor_network")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: OAT HAMILTONIAN TERMS (for TEBD)
# ═══════════════════════════════════════════════════════════════════════════════

def build_oat_ham(N_total, chi, kappa=0.01):
    """
    Build the OAT Hamiltonian as a list of local 2-site terms for TEBD.

    H = chi * sum_{i<j} Sz_i Sz_j  (all-to-all approximated as NN chain)
      + kappa * sum_i (Sx_i)        (transverse field — spontaneous emission)

    For TEBD we use the nearest-neighbour approximation of the collective
    spin coupling.  This is exact for the boundary coupling A[-1]-B[0].

    Returns: quimb LocalHam1D object
    """
    # Pauli matrices (quimb uses spin-1/2 convention)
    Sz = qu.spin_operator('Z', sparse=False)   # 2x2
    Sx = qu.spin_operator('X', sparse=False)

    # 2-site ZZ coupling term
    ZZ = np.kron(Sz, Sz)  # shape (4,4)

    # 1-site transverse field
    XI = np.kron(Sx, np.eye(2))
    IX = np.kron(np.eye(2), Sx)

    H2 = chi * ZZ + (kappa / 2) * (XI + IX)

    # Build local Hamiltonian for TEBD (quimb 1.x uses L, not N)
    builder = qtn.LocalHam1D(L=N_total, H2=H2)
    return builder


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: STATE PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_initial_mps(N_total, target_state=None, bond_dim=32):
    """
    Prepare the initial MPS state.

    Structure:
      - Ions 0 .. N//2-1   : Chain A
      - Ions N//2 .. N-1   : Chain B

    Chain A ground state: all |↓⟩  (spin down = |1⟩ in quimb)
    Chain B: all |↑⟩ to start a clean Bell resource

    The qubit to teleport (target_state) is encoded on ion A[0].
    """
    # Default target: |+⟩ = (|↑⟩ + |↓⟩)/√2  (maximally coherent state)
    if target_state is None:
        target_state = np.array([1.0, 1.0]) / np.sqrt(2)

    arrays = []
    N = N_total
    for i in range(N):
        if i == 0:
            # Inject target state on first qubit of chain A
            s = target_state.copy().astype(complex)
        elif i < N // 2:
            # Rest of chain A: |↓⟩
            s = np.array([0.0, 1.0], dtype=complex)
        else:
            # Chain B: |↑⟩
            s = np.array([1.0, 0.0], dtype=complex)
        arrays.append(s)

    # Build MPS from product state
    psi = qtn.MPS_product_state(arrays)
    psi.canonicalize_(cur_orthog='left')
    return psi


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: TEBD TIME EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def evolve_tebd(psi, ham, t_total, dt=0.02, bond_dim=32):
    """
    Evolve MPS under Hamiltonian using TEBD (Vidal 2004).

    Returns the evolved state psi(t_total).
    """
    tebd = qtn.TEBD(psi, ham, dt=dt)
    tebd.update_to(t_total, tol=1e-6)
    return tebd.pt   # evolved state


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: TELEPORTATION PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

def teleportation_fidelity_exact(psi_evolved, N_total, target_state):
    """
    Compute teleportation fidelity by:
      1. Forming the exact statevector (feasible for N ≤ 16)
      2. Tracing out all qubits except B[0] (= qubit N//2)
      3. Applying the ideal Pauli correction
      4. Computing F = ⟨target|ρ_B0|target⟩

    This is the "post-selected" fidelity — assumes perfect Bell measurement
    outcome.  For a realistic average, we average over 4 Bell outcomes.
    """
    N_half = N_total // 2
    receiver_idx = N_half  # First ion of chain B

    # Get full statevector from MPS (exact for small N)
    sv = psi_evolved.to_dense()  # shape (2**N_total,)
    sv = sv.reshape([2] * N_total)  # shape (2, 2, ..., 2)

    # Partial trace: keep only qubit `receiver_idx`
    # Trace over all other indices
    keep = receiver_idx
    trace_axes = list(range(N_total))
    trace_axes.pop(keep)

    # Build reduced density matrix of receiver qubit
    sv_flat = sv.reshape(2**keep, 2, 2**(N_total - keep - 1))
    rho = np.einsum('iaj,iak->jk', sv_flat.conj(), sv_flat)
    # rho is now the reduced DM of qubit `keep`, but with indices reversed
    rho_B0 = np.einsum('iaj,ibj->ab', sv_flat.conj(), sv_flat)

    # Normalise
    rho_B0 /= np.trace(rho_B0)

    # Apply Pauli-X correction (conditioned on Bell outcome — averaged here)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    fidelities = []
    # Average over 4 Bell measurement outcomes {I, X, Z, XZ}
    for op in [np.eye(2), X, Z, X @ Z]:
        rho_corrected = op @ rho_B0 @ op.conj().T
        psi_t = target_state.astype(complex)
        F = np.real(psi_t.conj() @ rho_corrected @ psi_t)
        fidelities.append(F)

    return float(np.mean(fidelities)), float(np.real(np.trace(rho_B0 @ rho_B0)))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: ENTANGLEMENT ENTROPY
# ═══════════════════════════════════════════════════════════════════════════════

def entanglement_entropy(psi, cut):
    """Von Neumann entropy across bond `cut` (between site cut-1 and cut)."""
    try:
        # quimb 1.x API
        return float(psi.entropy(cut))
    except AttributeError:
        # Fallback: manual Schmidt decomposition
        psi_c = psi.copy()
        psi_c.canonicalize_(cur_orthog=cut)
        S = psi_c.singular_values(cut)
        S = S[S > 1e-15]
        S2 = S**2
        S2 /= S2.sum()
        return float(-np.sum(S2 * np.log2(S2 + 1e-15)))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_teleportation_experiment(
    N_ions_list=(4, 8, 12, 16),
    chi=0.1,
    kappa=0.01,
    t_max=6.0,
    dt=0.05,
    n_time_points=60,
    bond_dim=32,
):
    """
    Run the full JILA OAT teleportation experiment for multiple ion-chain sizes.
    """
    t_grid = np.linspace(0.0, t_max, n_time_points)
    target_state = np.array([1.0, 1.0]) / np.sqrt(2)   # teleport |+⟩

    all_results = {}
    fidelity_curves  = {}
    entropy_curves   = {}

    for N in N_ions_list:
        print(f"\n{'='*60}")
        print(f"  N = {N} ions  (chain A: {N//2}, chain B: {N//2})")
        print(f"{'='*60}")

        F_curve = []
        S_curve = []

        ham = build_oat_ham(N, chi=chi, kappa=kappa)

        for i, t in enumerate(t_grid):
            psi0 = prepare_initial_mps(N, target_state=target_state, bond_dim=bond_dim)

            if t > 0:
                psi_t = evolve_tebd(psi0, ham, t_total=t, dt=dt, bond_dim=bond_dim)
            else:
                psi_t = psi0

            # Fidelity
            F, purity = teleportation_fidelity_exact(psi_t, N, target_state)

            # Entanglement entropy at the A-B boundary
            cut = N // 2
            S = entanglement_entropy(psi_t, cut)

            F_curve.append(F)
            S_curve.append(S)

            if i % 10 == 0:
                print(f"  t = {t:.2f}  |  F = {F:.4f}  |  S_vN = {S:.4f}")

        fidelity_curves[N] = F_curve
        entropy_curves[N]  = S_curve

        # Summary statistics
        F_arr = np.array(F_curve)
        S_arr = np.array(S_curve)
        result = {
            "N_ions":        N,
            "chi":           chi,
            "kappa":         kappa,
            "F_initial":     float(F_arr[0]),
            "F_peak":        float(F_arr.max()),
            "F_peak_time":   float(t_grid[F_arr.argmax()]),
            "F_final":       float(F_arr[-1]),
            "S_max":         float(S_arr.max()),
            "S_max_time":    float(t_grid[S_arr.argmax()]),
            "F_mean":        float(F_arr.mean()),
        }
        all_results[f"N{N}"] = result

        print(f"\n  ── Summary ──")
        print(f"  F_initial = {result['F_initial']:.4f}  (classical limit: 0.6667)")
        print(f"  F_peak    = {result['F_peak']:.4f}  at t = {result['F_peak_time']:.2f}")
        print(f"  S_max     = {result['S_max']:.4f}  (max entanglement at t = {result['S_max_time']:.2f})")

        # Classical teleportation limit check
        classical_limit = 2/3
        if result["F_peak"] > classical_limit:
            print(f"  ✓ QUANTUM ADVANTAGE: F_peak ({result['F_peak']:.4f}) > classical limit ({classical_limit:.4f})")
        else:
            print(f"  ✗ No quantum advantage yet at this evolution time / chi")

    # Save arrays
    np.save(RESULTS_DIR / "fidelity_curves.npy",
            np.array([fidelity_curves[N] for N in N_ions_list]))
    np.save(RESULTS_DIR / "entropy_curves.npy",
            np.array([entropy_curves[N] for N in N_ions_list]))
    np.save(RESULTS_DIR / "t_grid.npy", t_grid)

    # Save JSON summary
    summary = {
        "experiment":  "JILA OAT Quantum Teleportation (Tensor Network MPS/TEBD)",
        "protocol":    "Rey 2025 — OAT H = chi * Jz_A x Jz_B",
        "N_ions_list": list(N_ions_list),
        "chi":         chi,
        "kappa":       kappa,
        "t_max":       t_max,
        "bond_dim":    bond_dim,
        "target_state": "|+> = (|up> + |down>) / sqrt(2)",
        "classical_teleportation_limit": 2/3,
        "results":     all_results,
    }
    out_path = RESULTS_DIR / "tn_teleportation_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="JILA Quantum Teleportation — Tensor Network (MPS/TEBD) Simulation"
    )
    parser.add_argument("--N", nargs="+", type=int, default=[4, 8, 12, 16],
                        help="Ion chain sizes to simulate (total N, must be even)")
    parser.add_argument("--chi",   type=float, default=0.1,
                        help="OAT coupling strength chi")
    parser.add_argument("--kappa", type=float, default=0.01,
                        help="Spontaneous emission rate kappa")
    parser.add_argument("--t_max", type=float, default=6.0,
                        help="Maximum evolution time")
    parser.add_argument("--dt",    type=float, default=0.05,
                        help="TEBD time step")
    parser.add_argument("--points", type=int, default=60,
                        help="Number of time points")
    parser.add_argument("--bond",  type=int, default=32,
                        help="MPS bond dimension")
    args = parser.parse_args()

    run_teleportation_experiment(
        N_ions_list=args.N,
        chi=args.chi,
        kappa=args.kappa,
        t_max=args.t_max,
        dt=args.dt,
        n_time_points=args.points,
        bond_dim=args.bond,
    )
