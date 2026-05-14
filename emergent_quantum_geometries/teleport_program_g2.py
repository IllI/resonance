"""
teleport_program_g2.py — PTM-Aligned Teleportation (Basis Recovery from T-SVD)
================================================================================
Core claim: Teleportation advantage is carried by dominant singular directions
of the correlation tensor T, not the laboratory Z basis.

T = U S V^T  =>  Alice prerotation = su2(U^T),  Bob correction basis = su2(V)

Compares three protocols for each (chi_t, N):
  F_std : standard Bell measurement, no pre-rotation     (what IBM runs naively)
  F_rot : PTM-aligned Bell measurement via T-SVD         (analytic protocol)
  F_opt : fully optimized SU(2)x SU(2) upper bound       (from paper Table III.C)

If F_rot ≈ F_opt: teleportation advantage is analytically recoverable from
PTM geometry alone -- tightly unifying PTM theory, D-LinOSS, and teleportation.

Usage:
  python teleport_program_g2.py              # N=4 sweep
  python teleport_program_g2.py --N 8        # N=8
  python teleport_program_g2.py --table      # paper comparison table
"""
import argparse, json, time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from ptm_features import PTMFeatures, rho2_exact, compute_T_matrix, f_max_analytic
from dlinoss_program_f import infer_strata, _MASK

# ── Pauli setup ───────────────────────────────────────────────────────────────
I2=np.eye(2,dtype=complex)
sx=np.array([[0,1],[1,0]],dtype=complex)
sy=np.array([[0,-1j],[1j,0]],dtype=complex)
sz=np.array([[1,0],[0,-1]],dtype=complex)
B0=np.array([1,0,0,1])/np.sqrt(2)
B1=np.array([1,0,0,-1])/np.sqrt(2)
B2=np.array([0,1,1,0])/np.sqrt(2)
B3=np.array([0,1,-1,0])/np.sqrt(2)
BELLS=[B0,B1,B2,B3]
CORR_STD=[I2,sz,sx,sx@sz]


def su2_from_so3(R):
    """
    Map SO(3) rotation matrix to SU(2) via Cayley-Klein parameters.
    U sigma_i U† = R_ij sigma_j
    Uses the trace formula: cos(theta/2) = sqrt((1+Tr(R))/4) etc.
    """
    # Use scipy for robust conversion
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    qx,qy,qz,qw = r.as_quat()  # x,y,z,w convention
    return np.array([[ qw-1j*qz, -qy-1j*qx],
                     [ qy-1j*qx,  qw+1j*qz]], dtype=complex)


def teleport_aligned(rho_in, rho_resource, Ua, Ub):
    """
    Teleportation with optimal pre-rotations on both Alice and Bob resource qubits.
    Ua: Alice SU(2) pre-rotation on her resource qubit
    Ub: Bob SU(2) pre-rotation on his resource qubit
    Uses STANDARD Pauli corrections after Bell measurement.
    Achieves f_max = (1 + Tr(R_A^T T R_B))/4 by Horodecki theorem.
    """
    Ua_full = np.kron(Ua, I2)   # Alice resource qubit (qubit 0 of resource)
    Ub_full = np.kron(I2, Ub)   # Bob resource qubit (qubit 1 of resource)
    rho_res_rot = Ub_full @ (Ua_full @ rho_resource @ Ua_full.conj().T) @ Ub_full.conj().T
    rho3 = np.kron(rho_in, rho_res_rot)
    out = np.zeros((2,2), dtype=complex)
    for bk, Uk in zip(BELLS, CORR_STD):
        Pb = np.outer(bk, bk.conj())
        Pk = np.kron(Pb, I2)
        rp = Pk @ rho3 @ Pk
        rpr = rp.reshape(2,2,2,2,2,2)
        rB = np.einsum('ijkijl->kl', rpr)
        out += Uk @ rB @ Uk.conj().T
    return out


def teleport_standard(rho_in, rho_resource):
    return teleport_aligned(rho_in, rho_resource, I2, I2)


def haar_mean_fidelity(rho_resource, Ua, Ub, n=300, seed=0):
    rng = np.random.default_rng(seed)
    fids = []
    for _ in range(n):
        psi = rng.standard_normal(2) + 1j*rng.standard_normal(2)
        psi /= np.linalg.norm(psi)
        ri = np.outer(psi, psi.conj())
        ro = teleport_aligned(ri, rho_resource, Ua, Ub)
        fids.append(float(np.real(np.trace(ri @ ro))))
    return float(np.mean(fids)), float(np.std(fids))


def ptm_aligned_protocol(rho_resource):
    """
    Construct Alice/Bob rotations from T-matrix SVD.
    T = U S V^T
    Alice prerotation: R_A = U^T (aligns Alice's frame to T singular basis)
    Bob corrections: standard Pauli * V^T (aligns Bob's frame)
    """
    T = compute_T_matrix(rho_resource)
    U, s, Vt = np.linalg.svd(T)
    V = Vt.T

    # Ensure proper SO(3): fix determinant signs
    if np.linalg.det(U) < 0: U[:, 0] *= -1
    if np.linalg.det(V) < 0: V[:, 0] *= -1

    # Ua: Alice pre-rotates her resource to align T to diagonal basis
    # Ub: Bob pre-rotates his resource to align Bob's frame
    Ua = su2_from_so3(U)
    Ub = su2_from_so3(V)
    return Ua, Ub, s


def f_opt_numerical(rho, n_restarts=8):
    """
    Direct numerical optimization of singlet fraction over SU(2)xSU(2).
    Uses Euler angle parameterization for each SU(2).
    """
    phi_p = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)

    def su2_euler(params):
        a,b,c = params
        ca,sa = np.cos(a/2), np.sin(a/2)
        cb,sb = np.cos(b/2), np.sin(b/2)
        cc,sc = np.cos(c/2), np.sin(c/2)
        return np.array([[ np.exp(-1j*(a+c)/2)*cb, -np.exp(-1j*(a-c)/2)*sb],
                         [ np.exp(1j*(a-c)/2)*sb,   np.exp(1j*(a+c)/2)*cb]])

    def neg_singlet(params):
        Ua = su2_euler(params[:3])
        Ub = su2_euler(params[3:])
        U = np.kron(Ua, Ub)
        r = U @ rho @ U.conj().T
        return -float(np.real(phi_p.conj() @ r @ phi_p))

    best = 1.0
    for _ in range(n_restarts):
        x0 = np.random.uniform(0, 2*np.pi, 6)
        res = minimize(neg_singlet, x0, method='Nelder-Mead',
                       options={'xatol':1e-8,'fatol':1e-8,'maxiter':10000})
        if res.fun < best: best = res.fun
    return -best  # f_max


# ── Main comparison sweep ─────────────────────────────────────────────────────
def sweep(N=4, n_chi=25, n_haar=200):
    print(f"\n=== PTM-Aligned Teleportation Sweep (N={N}) ===")
    print(f"{'chi_t':>8} {'F_std':>7} {'F_rot':>7} {'F_opt':>7} "
          f"{'F_rot-F_std':>12} {'F_opt-F_rot':>12} {'stratum':>12}")
    print("-"*80)

    results = []
    for chi_t in np.linspace(0.05, np.pi, n_chi):
        rho = rho2_exact(chi_t, N, 0.0)

        # Standard protocol
        F_std, _ = haar_mean_fidelity(rho, I2, I2, n_haar)

        # PTM-aligned protocol
        Ua, Ub_corr, svs = ptm_aligned_protocol(rho)
        F_rot, _ = haar_mean_fidelity(rho, Ua, Ub_corr, n_haar)

        # Numerical f_max upper bound
        f_max = f_opt_numerical(rho, n_restarts=6)
        F_opt = (2*f_max + 1)/3

        # Stratum
        T = compute_T_matrix(rho)
        svs_T = np.sort(np.linalg.svd(T, compute_uv=False))[::-1]
        row = np.zeros(34); row[3:12]=T.flatten(); row[12:15]=svs_T
        row[30]=int((svs_T>1e-3).sum()); row[33]=float(max(0,1-np.linalg.norm(T,'fro')/2))
        st, _ = infer_strata(row[_MASK].reshape(1,-1))

        gap_rot = F_rot - F_std
        gap_opt = F_opt - F_rot
        print(f"{chi_t:>8.4f} {F_std:>7.4f} {F_rot:>7.4f} {F_opt:>7.4f} "
              f"{gap_rot:>12.4f} {gap_opt:>12.4f} {st[0]:>12}")
        results.append({"chi_t":chi_t,"N":N,"F_std":F_std,"F_rot":F_rot,
                         "F_opt":F_opt,"stratum":st[0],"svs":svs.tolist()})
    return results


def paper_comparison_table():
    """Reproduce paper Table III.C: N=4,8,16 at chi_t*."""
    print("\n=== Paper Comparison Table: F_std vs F_rot vs F_opt at chi_t* ===")
    print(f"{'N':>4} {'chi_t*':>8} {'F_std':>7} {'F_rot':>7} "
          f"{'F_opt_num':>10} {'F_opt_paper':>12} {'F_rot≈F_opt?':>13}")
    print("-"*70)

    # Paper values from Table III.C
    paper_vals = {4: 0.719, 8: 0.686, 16: 0.674}

    for N in [4, 8, 16]:
        # Find chi_t* (maximize F_rot)
        best_F = 0; best_chi = 0
        for chi_t in np.linspace(0.05, 3.0, 60):
            rho = rho2_exact(chi_t, N, 0.0)
            Ua, Ub_corr, _ = ptm_aligned_protocol(rho)
            F, _ = haar_mean_fidelity(rho, Ua, Ub_corr, n=100)
            if F > best_F: best_F = F; best_chi = chi_t

        rho_star = rho2_exact(best_chi, N, 0.0)
        F_std, _ = haar_mean_fidelity(rho_star, I2, CORR_STD, n=200)
        Ua, Ub_corr, _ = ptm_aligned_protocol(rho_star)
        F_rot, _ = haar_mean_fidelity(rho_star, Ua, Ub_corr, n=200)
        f_max = f_opt_numerical(rho_star, n_restarts=12)
        F_opt_num = (2*f_max+1)/3
        F_paper = paper_vals.get(N, float('nan'))
        match = "YES" if abs(F_rot - F_opt_num) < 0.01 else "NO"
        print(f"{N:>4} {best_chi:>8.4f} {F_std:>7.4f} {F_rot:>7.4f} "
              f"{F_opt_num:>10.4f} {F_paper:>12.4f} {match:>13}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--n-chi", type=int, default=20)
    p.add_argument("--n-haar", type=int, default=150)
    p.add_argument("--table", action="store_true")
    p.add_argument("--out", default="program_g2_results.json")
    args = p.parse_args()

    t0 = time.time()
    if args.table:
        paper_comparison_table()
    else:
        results = sweep(args.N, args.n_chi, args.n_haar)
        with open(args.out,"w") as f: json.dump(results,f,indent=2)
    print(f"\nDone  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
