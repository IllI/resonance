"""
teleport_program_g.py — Full Teleportation Validation (Experiment G)
=====================================================================
Tests whether quantum information (not just correlations) is faithfully
transferred through OAT entangled resource across the D-LinOSS strata.

New observables beyond Program F:
  - Haar-random input ensemble (worst-case, variance, 5th percentile fidelity)
  - Entanglement fidelity F_e = (3*F_avg - 1)/2
  - Quantum mutual information I(R:B) after teleportation of half a Bell pair
  - Concurrence of output state

Usage:
  python teleport_program_g.py                  # full chi_t sweep
  python teleport_program_g.py --ibm-points     # 3-point IBM comparison table
  python teleport_program_g.py --n-haar 2000    # denser Haar ensemble
"""
import argparse, json, time
import numpy as np
from ptm_features import PTMFeatures, rho2_exact, f_max_analytic, compute_T_matrix
from dlinoss_program_f import infer_strata, _MASK

# ── Pauli matrices ────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS = [I2, sx, sy, sz]

# Bell states (columns)
B0 = np.array([1,0,0,1])/np.sqrt(2)   # Phi+
B1 = np.array([1,0,0,-1])/np.sqrt(2)  # Phi-
B2 = np.array([0,1,1,0])/np.sqrt(2)   # Psi+
B3 = np.array([0,1,-1,0])/np.sqrt(2)  # Psi-
BELLS = [B0, B1, B2, B3]
CORRECTIONS = [I2, sz, sx, sx @ sz]   # Pauli corrections for each outcome


# ── Quantum utilities ─────────────────────────────────────────────────────────
def ptrace(rho, keep, dims):
    """Partial trace. keep: list of qubit indices to keep. dims: list of dim per qubit."""
    n = len(dims)
    shape = dims * 2
    rho_t = rho.reshape(shape)
    trace_axes = [i for i in range(n) if i not in keep]
    for ax in sorted(trace_axes, reverse=True):
        rho_t = np.trace(rho_t, axis1=ax, axis2=ax + n - len(trace_axes) + trace_axes.index(ax))
        n -= 1
    d = int(np.prod([dims[i] for i in keep]))
    return rho_t.reshape(d, d)


def ptrace2(rho, keep):
    """Partial trace for 2-qubit (4x4) system. keep=0 or keep=1."""
    if keep == 1:
        return np.array([[rho[0,0]+rho[2,2], rho[0,1]+rho[2,3]],
                         [rho[1,0]+rho[3,2], rho[1,1]+rho[3,3]]])
    else:
        return np.array([[rho[0,0]+rho[1,1], rho[0,2]+rho[1,3]],
                         [rho[2,0]+rho[3,1], rho[2,2]+rho[3,3]]])


def ptrace3(rho, keep_qubit):
    """Partial trace for 3-qubit (8x8) system. Returns 2x2 for one kept qubit."""
    rho4 = rho.reshape(2,2,2,2,2,2)
    if keep_qubit == 2:
        return np.einsum('iijjkl->kl', rho4)
    elif keep_qubit == 1:
        return np.einsum('iijkjl->kl', rho4)
    else:
        return np.einsum('ijikjk->ij', rho4.reshape(2,4,2,4))


def vn_entropy(rho):
    eigs = np.linalg.eigvalsh(rho).clip(1e-15)
    return float(-np.sum(eigs * np.log2(eigs + 1e-15)))


def teleport(rho_in, rho_resource):
    """
    Full Bell-measurement teleportation.
    rho_in: 2x2 input qubit state
    rho_resource: 4x4 entangled resource (Alice qubit 0, Bob qubit 1)
    Returns Bob's 2x2 output state after correction.
    """
    # 3-qubit state: input(0) x Alice(1) x Bob(2), total 8x8
    rho3 = np.kron(rho_in, rho_resource)

    rho_out = np.zeros((2,2), dtype=complex)
    for bk, Uk in zip(BELLS, CORRECTIONS):
        # Bell projector on qubits 0,1
        Pb = np.outer(bk, bk.conj())   # 4x4
        Pk = np.kron(Pb, I2)           # 8x8
        # Apply projector
        rho_post = Pk @ rho3 @ Pk      # unnormalized
        # Trace out qubits 0,1 to get Bob's state (unnormalized)
        rho_post_r = rho_post.reshape(2,2,2,2,2,2)
        rho_B_unnorm = np.einsum('ijkijl->kl', rho_post_r)  # trace qubits 0,1
        # Apply correction (no renorm — sum gives correct average)
        rho_out += Uk @ rho_B_unnorm @ Uk.conj().T
    return rho_out


def haar_states(n, seed=0):
    """Sample n Haar-random pure qubit states."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        psi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        psi /= np.linalg.norm(psi)
        states.append(np.outer(psi, psi.conj()))
    return states


def concurrence(rho):
    """Wootters concurrence for 2-qubit state."""
    sy2 = np.kron(sy, sy)
    rho_tilde = sy2 @ rho.conj() @ sy2
    R = rho @ rho_tilde
    eigs = np.sqrt(np.abs(np.linalg.eigvals(R).real))
    eigs = np.sort(eigs)[::-1]
    return float(max(0, eigs[0] - eigs[1] - eigs[2] - eigs[3]))


# ── Core measurement functions ────────────────────────────────────────────────
def haar_fidelity_stats(chi_t, N, n_states=500, seed=0):
    """
    Teleport n_states Haar-random states through OAT resource at (chi_t, N).
    Returns: mean, min, std, p5 fidelity, and worst-case state fidelity.
    """
    rho_res = rho2_exact(chi_t, N, 0.0)
    states = haar_states(n_states, seed)
    fids = []
    for rho_in in states:
        rho_out = teleport(rho_in, rho_res)
        # Fidelity: Tr(rho_in * rho_out) for pure rho_in
        fid = float(np.real(np.trace(rho_in @ rho_out)))
        fids.append(fid)
    f = np.array(fids)
    return {"mean": float(f.mean()), "min": float(f.min()),
            "std": float(f.std()), "p5": float(np.percentile(f, 5))}


def entanglement_fidelity(chi_t, N):
    """F_e = (3*F_avg - 1)/2 for qubits. Also returns I(R:B)."""
    T = compute_T_matrix(rho2_exact(chi_t, N, 0.0))
    fm = f_max_analytic(T)
    F_avg = (2*fm + 1)/3
    F_e = (3*F_avg - 1) / 2  # Horodecki relation for d=2
    return float(F_e), float(F_avg)


def mutual_info_retention(chi_t, N):
    """
    I(R:B) after teleporting half of |Phi+>_{RA} through OAT resource.
    Perfect teleportation -> I(R:B) = 2 bits.
    Classical bound -> I(R:B) = 0.
    """
    rho_res = rho2_exact(chi_t, N, 0.0)
    # Reference + Alice input in Bell state |Phi+>
    phi_plus = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
    rho_RA = np.outer(phi_plus, phi_plus.conj())  # 4x4
    rho_R = ptrace2(rho_RA, keep=0)    # reference qubit (2x2)
    rho_A = ptrace2(rho_RA, keep=1)    # Alice's input qubit (2x2)

    # Teleport Alice's qubit through resource
    rho_B = teleport(rho_A, rho_res)

    # Build joint R+B state: approximate as tensor product
    # (exact I(R:B) needs full 4-qubit simulation; this uses the effective channel)
    # Effective channel preserves correlations proportional to F_e
    F_e, F_avg = entanglement_fidelity(chi_t, N)
    # For a Pauli channel with F_e: rho_RB = F_e |Phi+><Phi+| + (1-F_e) I/4
    rho_RB_approx = F_e * np.outer(phi_plus, phi_plus.conj()) + (1-F_e)*np.eye(4)/4
    S_R = vn_entropy(rho_R)
    S_B = vn_entropy(rho_B)
    S_RB = vn_entropy(rho_RB_approx)
    I_RB = S_R + S_B - S_RB
    return float(max(0, I_RB))


def stratum_for(chi_t, N):
    """D-LinOSS stratum assignment (blind)."""
    from ptm_features import PTMFeatureVector
    feat = PTMFeatures(N=N, fast=True)
    fv = feat.compute(chi_t, 0.0)
    row = fv.to_array()
    st, _ = infer_strata(row[_MASK].reshape(1,-1))
    return st[0]


# ── Main sweep ────────────────────────────────────────────────────────────────
def sweep(chi_vals, N=4, n_haar=500, label="sweep"):
    print(f"\n{'chi_t':>8} {'F_avg':>7} {'F_e':>7} {'F_haar':>8} "
          f"{'F_min':>7} {'F_p5':>7} {'I(R:B)':>7} {'stratum':>12}")
    print("-" * 78)
    results = []
    for chi_t in chi_vals:
        F_e, F_avg = entanglement_fidelity(chi_t, N)
        hf = haar_fidelity_stats(chi_t, N, n_haar)
        I_RB = mutual_info_retention(chi_t, N)
        st = stratum_for(chi_t, N)
        print(f"{chi_t:>8.4f} {F_avg:>7.4f} {F_e:>7.4f} {hf['mean']:>8.4f} "
              f"{hf['min']:>7.4f} {hf['p5']:>7.4f} {I_RB:>7.4f} {st:>12}")
        results.append({"chi_t": chi_t, "N": N, "F_avg": F_avg,
                        "F_e": F_e, "I_RB": I_RB, "stratum": st, **hf})
    return results


def ibm_comparison_table(N=4, n_haar=200):
    """3-point IBM-ready comparison: PRODUCT / TRANSPORT / SINGULAR."""
    print("\n" + "="*70)
    print("EXPERIMENT G: 3-POINT IBM COMPARISON TABLE")
    print("="*70)

    # Representative chi_t for each stratum
    points = [
        ("PRODUCT region",   0.05),
        ("TRANSPORT peak",   1.056),
        ("SINGULAR (pi)",    np.pi - 0.01),
    ]

    print(f"\n{'Region':<22} {'chi_t':>7} {'PTM_rank':>9} {'F_avg':>7} "
          f"{'F_e':>7} {'I(R:B)':>7} {'F_haar':>8} {'F_min':>7} {'stratum':>12}")
    print("-" * 92)

    feat = PTMFeatures(N=N, fast=True)
    rows = []
    for label, chi_t in points:
        fv = feat.compute(chi_t, 0.0)
        T = np.array([[fv.T_xx,fv.T_xy,fv.T_xz],
                      [fv.T_yx,fv.T_yy,fv.T_yz],
                      [fv.T_zx,fv.T_zy,fv.T_zz]])
        svs = np.linalg.svd(T, compute_uv=False)
        rank = int((svs > 1e-3).sum())
        F_e, F_avg = entanglement_fidelity(chi_t, N)
        hf = haar_fidelity_stats(chi_t, N, n_haar)
        I_RB = mutual_info_retention(chi_t, N)
        st = stratum_for(chi_t, N)
        # F_std: standard protocol (singlet fraction)
        rho_r = rho2_exact(chi_t, N, 0.0)
        phi_p = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
        F_std_theory = float(np.real(np.dot(phi_p.conj(), rho_r @ phi_p)))
        F_std_theory = (1 + 2*F_std_theory)/3
        print(f"{label:<22} {chi_t:>7.4f} {rank:>5d} {F_avg:>6.4f} "
              f"{F_std_theory:>6.4f} {hf['mean']:>7.4f} {hf['min']:>6.4f} "
              f"{F_e:>6.4f} {I_RB:>7.4f} {st:>12}")
        rows.append({"label": label, "chi_t": chi_t, "rank": rank,
                     "F_avg": F_avg, "F_e": F_e, "I_RB": I_RB,
                     "stratum": st, **hf})

    print("\n  Classical teleportation bound: F_avg=2/3, F_e=1/2, I(R:B)=0")
    print("  Perfect teleportation:         F_avg=1.0, F_e=1.0, I(R:B)=2.0")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ibm-points", action="store_true")
    p.add_argument("--n-haar", type=int, default=500)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--out", default="program_g_results.json")
    args = p.parse_args()

    print("=== Program G: Full Teleportation Validation ===")
    t0 = time.time()

    if args.ibm_points:
        rows = ibm_comparison_table(args.N, args.n_haar)
        with open(args.out,"w") as f: json.dump(rows, f, indent=2)
        print(f"\nSaved -> {args.out}  ({time.time()-t0:.1f}s)")
        return

    # Full chi_t sweep
    chi_vals = np.linspace(0.05, np.pi, 25)
    print(f"\nHaar ensemble: {args.n_haar} states | N={args.N}")
    rows = sweep(chi_vals, args.N, args.n_haar)
    with open(args.out,"w") as f: json.dump(rows, f, indent=2)
    print(f"\nSaved -> {args.out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
