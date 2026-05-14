"""
program_h3_tpu.py -- Program H3: What Survives After Concurrence Dies?
=======================================================================
Core question: is N_c(ΔF) >> N_c(C)?
If yes: transport-supporting operator geometry survives beyond pairwise entanglement.

Observables measured per N:
  ΔF     -- teleportation enhancement (F_sch - F_std)
  C      -- Wootters pairwise concurrence
  Neg    -- entanglement negativity (partial transpose cross-check)
  rank_T -- PTM rank (operator support persistence)
  H_T    -- spectral entropy of T singular values (transport richness)
  sv1,2,3-- singular value spectrum of T matrix
  F_std  -- standard protocol fidelity
  F_sch  -- Schmidt-rotated fidelity
  F_opt  -- analytic SVD upper bound
  F_null -- fidelity at chi_t=pi (Theorem 4 check)

Crossover analysis:
  N_c(C)  -- smallest N where C < thresh_C (default 0.01)
  N_c(dF) -- smallest N where ΔF < thresh_dF (default 0.01)
  ratio   -- N_c(dF) / N_c(C) -- if >>1, geometry survives entanglement

Model: OAT closed-form rho2, N swept finely (4,6,8,...,256).
       Fine N spacing near crossover to resolve N_c precisely.

TPU: upload and run -- nohup python3 program_h3_tpu.py --N-max 256 > h3.log 2>&1 &
"""

import json, time, math, cmath, zipfile, argparse
import numpy as np

try:
    import jax
    USE_JAX = True
    print(f"[DEVICE] {jax.devices()}  backend={jax.default_backend()}", flush=True)
except ImportError:
    USE_JAX = False
    print("[DEVICE] JAX not found -- numpy fallback", flush=True)

# ---- Pauli matrices -------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
P0 = np.array([[1,0],[0,0]], dtype=complex)
P1 = np.array([[0,0],[0,1]], dtype=complex)
H2g = np.array([[1,1],[1,-1]], dtype=complex)/math.sqrt(2)

# ---- Teleportation circuit -----------------------------------------------
CNOT_01 = np.kron(P0,np.kron(I2,I2)) + np.kron(P1,np.kron(X,I2))
CIRC    = np.kron(H2g,np.kron(I2,I2)) @ CNOT_01
CIRC_D  = CIRC.conj().T
PROJ    = {(m1,m2): np.kron(P0 if m1==0 else P1, np.kron(P0 if m2==0 else P1, I2))
           for m1 in range(2) for m2 in range(2)}
UCORR   = {(0,0):I2, (0,1):X, (1,0):Z, (1,1):X@Z}

def teleport_fid(psi_A, rho2):
    rho_t = np.kron(np.outer(psi_A, psi_A.conj()), rho2)
    rb    = CIRC @ rho_t @ CIRC_D
    out   = np.zeros((2,2), dtype=complex)
    for (m1,m2), Pi in PROJ.items():
        rp   = Pi @ rb @ Pi
        prob = float(np.trace(rp).real)
        if prob < 1e-12: continue
        r    = (rp/prob).reshape(2,2,2,2,2,2)
        rq   = np.einsum('ijkijl->kl', r)
        out += prob * (UCORR[(m1,m2)] @ rq @ UCORR[(m1,m2)].conj().T)
    return float(np.clip((psi_A.conj() @ out @ psi_A).real, 0, 1))

def haar_qubit(rng):
    u1,u2 = rng.uniform(0,1,2)
    th = math.acos(1 - 2*u1); ph = 2*math.pi*u2
    return np.array([math.cos(th/2), math.sin(th/2)*cmath.exp(1j*ph)], dtype=complex)

def avg_fid_haar(rho2, K=300, seed=42):
    rng = np.random.default_rng(seed)
    return float(np.mean([teleport_fid(haar_qubit(rng), rho2) for _ in range(K)]))

# ---- Observables ---------------------------------------------------------

def concurrence(rho):
    """Wootters concurrence for 2-qubit density matrix."""
    sy2 = np.kron(Y, Y)
    R   = rho @ (sy2 @ rho.conj() @ sy2)
    eigs = np.sqrt(np.clip(np.linalg.eigvals(R).real, 0, None))
    eigs = np.sort(eigs)[::-1]
    return float(max(0.0, eigs[0] - eigs[1] - eigs[2] - eigs[3]))

def negativity(rho):
    """Entanglement negativity: sum of negative eigenvalues of partial transpose."""
    # Partial transpose on second qubit: rho^{T_B}
    # rho indexed as (i,j,k,l) -> rho_T2(i,l,k,j) = rho(i,j,k,l)
    rho4 = rho.reshape(2,2,2,2)
    rho_pt = rho4.transpose(0,3,2,1).reshape(4,4)
    eigs = np.linalg.eigvalsh(rho_pt)
    return float(np.sum(np.abs(eigs[eigs < 0])))

def compute_T(rho):
    T = np.zeros((3,3))
    for i,si in enumerate([X,Y,Z]):
        for j,sj in enumerate([X,Y,Z]):
            T[i,j] = float(np.real(np.trace(np.kron(si,sj) @ rho)))
    return T

def f_opt_svd(rho):
    svs = np.linalg.svd(compute_T(rho), compute_uv=False)
    return float((1 + svs.sum())/4), svs

def spectral_entropy_T(svs):
    """Shannon entropy of normalized squared singular values of T matrix."""
    s2 = np.array(svs)**2
    s2 = s2[s2 > 1e-10]
    if len(s2) == 0: return 0.0
    p = s2 / s2.sum()
    return float(-np.sum(p * np.log(p)))

def schmidt_rotate(rho2):
    """Extract Schmidt rotation from dominant eigenvector SVD."""
    _, vecs = np.linalg.eigh(rho2)
    M       = vecs[:, -1].reshape(2, 2)
    U, sig, Vh = np.linalg.svd(M)
    U_A = U.conj().T; U_B = Vh.conj()
    Ul  = np.kron(U_A, U_B)
    return Ul @ rho2 @ Ul.conj().T, float(2*abs(sig[0])*abs(sig[1]))

# ---- OAT closed-form rho2 ------------------------------------------------
def rho2_oat(chi_t, N):
    rho = np.zeros((4,4), dtype=complex)
    ep  = max(0, N//2 - 1)
    IDX = [(0,0),(0,1),(1,0),(1,1)]
    for a,(iL,iR) in enumerate(IDX):
        for b,(jL,jR) in enumerate(IDX):
            ph  = -1j*chi_t*((iL-.5)*(iR-.5) - (jL-.5)*(jR-.5))
            cR  = math.cos((iR-jR)*chi_t/2)**ep if ep > 0 else 1.0
            cL  = math.cos((iL-jL)*chi_t/2)**ep if ep > 0 else 1.0
            rho[a,b] = 0.25*cmath.exp(ph)*cR*cL
    return rho

def analyze_rho2(rho2, K=300):
    """Full observable suite for a 2-qubit density matrix."""
    C        = concurrence(rho2)
    Neg      = negativity(rho2)
    T        = compute_T(rho2)
    fm, svs  = f_opt_svd(rho2)
    F_std    = float((1 + T[0,0]/3) / 2)
    F_opt    = float((2*fm + 1) / 3)
    H_T      = spectral_entropy_T(svs)
    rank_T   = int((np.array(svs) > 1e-3).sum())

    rho_rot, C_sch = schmidt_rotate(rho2)
    F_sch    = avg_fid_haar(rho_rot, K=K)
    dF       = F_sch - F_std

    # Transport membership: rank=3 AND F_sch > F_std + threshold
    transport_member = int(rank_T == 3 and dF > 0.005)

    return {
        "C": C, "Neg": Neg, "C_sch_pair": C_sch,
        "F_std": F_std, "F_sch": F_sch, "F_opt": F_opt,
        "dF": dF, "rank_T": rank_T, "H_T": H_T,
        "sv1": float(svs[0]), "sv2": float(svs[1]), "sv3": float(svs[2]),
        "transport_member": transport_member
    }

# ---- Adaptive chi_t sweep per N ------------------------------------------
def sweep_oat(N, n_chi=100, K=300):
    """Find optimal chi_t maximizing dF, with fine adaptive grid.

    Grid: denser near chi_t* ~ 0.5/N (expected from H1 scaling).
    Also evaluates chi_t = pi for Theorem 4 check.
    """
    # Adaptive range: sweep broadly then refine
    chi_lo = max(0.0005, 0.1/N)
    chi_hi = min(math.pi - 0.01, max(8/N, 0.8))
    chi_vals = np.concatenate([
        np.geomspace(chi_lo, min(chi_hi, 2/N + 0.01), n_chi//2),  # log-spaced near 0
        np.linspace(min(chi_hi, 2/N + 0.01), chi_hi, n_chi//2),   # linear in bulk
        [math.pi - 0.001],                                          # Theorem 4 point
    ])
    chi_vals = np.unique(np.sort(chi_vals))

    best = None
    sweep_data = []
    for chi_t in chi_vals:
        rho = rho2_oat(chi_t, N)
        r   = analyze_rho2(rho, K=K)
        r["chi_t"] = float(chi_t)
        sweep_data.append(r)
        if best is None or r["dF"] > best["dF"]:
            best = dict(r); best["chi_star"] = float(chi_t)

    # Theorem 4: F at chi_t = pi
    r_null = analyze_rho2(rho2_oat(math.pi - 0.001, N), K=10)
    best["F_null"] = r_null["F_std"]   # analytic: should be 0.5 exactly

    return best, sweep_data

# ---- Crossover analysis --------------------------------------------------
def find_crossover(Ns, vals, thresh, label):
    """Find smallest N where val drops below thresh (interpolate if needed)."""
    for i, (N, v) in enumerate(zip(Ns, vals)):
        if v < thresh:
            if i == 0:
                return N, "below_threshold_at_N_min"
            # Linear interpolate in log-N
            N_prev, v_prev = Ns[i-1], vals[i-1]
            t = (thresh - v_prev) / (v - v_prev)
            Nc = math.exp(math.log(N_prev) + t*(math.log(N) - math.log(N_prev)))
            return Nc, f"interpolated_between_{N_prev}_and_{N}"
    return float("inf"), "never_below_threshold"

# ---- Power-law fit -------------------------------------------------------
def fit_pl(Ns, vals):
    mask = np.array(vals) > 1e-8
    if mask.sum() < 2: return 0.0, 0.0, 0.0
    lN  = np.log(np.array(Ns)[mask].astype(float))
    lv  = np.log(np.array(vals)[mask].astype(float))
    c   = np.polyfit(lN, lv, 1)
    pred = np.polyval(c, lN)
    ss_res = np.sum((lv - pred)**2)
    ss_tot = np.sum((lv - lv.mean())**2)
    R2 = float(1 - ss_res/ss_tot) if ss_tot > 1e-10 else 0.0
    return float(np.exp(c[1])), float(-c[0]), R2

# ---- Main ----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N-max",   type=int,   default=256)
    p.add_argument("--n-chi",   type=int,   default=80)
    p.add_argument("--K",       type=int,   default=500)
    p.add_argument("--thresh-C",  type=float, default=0.01)
    p.add_argument("--thresh-dF", type=float, default=0.01)
    p.add_argument("--out",     default="program_h3_results.json")
    args = p.parse_args()

    # Fine N spacing: every 2 up to 32, then powers of 2 to N_max
    N_fine  = list(range(4, min(33, args.N_max+1), 2))
    N_coarse = []
    N = 64
    while N <= args.N_max:
        N_coarse.append(N); N *= 2
    Ns = sorted(set(N_fine + N_coarse))

    print("="*70, flush=True)
    print("Program H3 -- What Survives After Concurrence Dies?")
    print(f"N values: {Ns}")
    print(f"K={args.K}  n_chi={args.n_chi}  thresh_C={args.thresh_C}  thresh_dF={args.thresh_dF}")
    print("="*70, flush=True)

    print(f"\n{'N':>5}  {'chi*':>7}  {'F_std':>7}  {'F_sch':>7}  {'dF':>7}  "
          f"{'C':>7}  {'Neg':>7}  {'H_T':>6}  {'rank':>4}  {'F(pi)':>7}", flush=True)
    print("-"*85, flush=True)

    t0 = time.time()
    all_results = []
    sweep_catalog = {}

    for N in Ns:
        t1 = time.time()
        best, sweep = sweep_oat(N, n_chi=args.n_chi, K=args.K)
        elapsed = time.time() - t1

        print(f"{N:>5}  {best['chi_star']:>7.4f}  {best['F_std']:>7.4f}  "
              f"{best['F_sch']:>7.4f}  {best['dF']:>7.4f}  "
              f"{best['C']:>7.4f}  {best['Neg']:>7.4f}  {best['H_T']:>6.3f}  "
              f"{best['rank_T']:>4}  {best.get('F_null',0):>7.4f}  "
              f"({elapsed:.1f}s)", flush=True)

        best["N"] = N
        all_results.append(best)
        sweep_catalog[str(N)] = sweep

    total = time.time() - t0

    # ---- Crossover analysis ----------------------------------------------
    Ns_arr  = [r["N"] for r in all_results]
    dFs_arr = [r["dF"] for r in all_results]
    Cs_arr  = [r["C"]  for r in all_results]
    Negs_arr= [r["Neg"] for r in all_results]

    Nc_C,   msg_C   = find_crossover(Ns_arr, Cs_arr,   args.thresh_C,  "C")
    Nc_Neg, msg_Neg = find_crossover(Ns_arr, Negs_arr, args.thresh_C/2, "Neg")
    Nc_dF,  msg_dF  = find_crossover(Ns_arr, dFs_arr,  args.thresh_dF, "dF")

    ratio = (Nc_dF / Nc_C) if Nc_C > 0 else float("inf")

    print("\n" + "="*70, flush=True)
    print("CROSSOVER ANALYSIS", flush=True)
    print(f"  N_c(C)    = {Nc_C:.1f}  [{msg_C}]")
    print(f"  N_c(Neg)  = {Nc_Neg:.1f}  [{msg_Neg}]")
    print(f"  N_c(dF)   = {Nc_dF}  [{msg_dF}]")
    print(f"  N_c(dF) / N_c(C) = {ratio:.1f}x  <<< KEY RATIO", flush=True)

    # ---- Power-law fits --------------------------------------------------
    A_dF, alpha_dF, R2_dF = fit_pl(Ns_arr, dFs_arr)
    A_C,  alpha_C,  R2_C  = fit_pl(Ns_arr, Cs_arr)
    A_Neg,alpha_Neg,R2_Neg= fit_pl(Ns_arr, Negs_arr)

    print("\nPOWER LAWS:", flush=True)
    print(f"  dF      ~ {A_dF:.4f} * N^(-{alpha_dF:.4f})  R2={R2_dF:.4f}")
    print(f"  C       ~ {A_C:.4f}  * N^(-{alpha_C:.4f})   R2={R2_C:.4f}")
    print(f"  Neg     ~ {A_Neg:.4f} * N^(-{alpha_Neg:.4f})  R2={R2_Neg:.4f}")
    print(f"  alpha_dF / alpha_C = {alpha_dF/alpha_C:.3f}  (>1 means C decays faster)", flush=True)

    # PTM rank persistence
    rank_all = [r["rank_T"] for r in all_results]
    rank_3_count = sum(1 for r in rank_all if r == 3)
    print(f"\nPTM rank=3: {rank_3_count}/{len(Ns_arr)} N values  "
          f"({100*rank_3_count//len(Ns_arr)}%)", flush=True)

    # H_T spectral entropy trend
    HTs = [r["H_T"] for r in all_results]
    print(f"H_T range: {min(HTs):.3f} -- {max(HTs):.3f}  "
          f"(log3={math.log(3):.3f} = fully isotropic)", flush=True)

    print(f"\nTotal runtime: {total:.1f}s", flush=True)

    # ---- Save results ----------------------------------------------------
    output = {
        "Ns": Ns_arr,
        "results": all_results,
        "crossover": {
            "Nc_C": Nc_C, "msg_C": msg_C,
            "Nc_Neg": Nc_Neg, "msg_Neg": msg_Neg,
            "Nc_dF": str(Nc_dF), "msg_dF": msg_dF,
            "ratio_dF_over_C": ratio,
            "thresh_C": args.thresh_C,
            "thresh_dF": args.thresh_dF,
        },
        "power_laws": {
            "dF": {"A": A_dF, "alpha": alpha_dF, "R2": R2_dF},
            "C":  {"A": A_C,  "alpha": alpha_C,  "R2": R2_C},
            "Neg":{"A": A_Neg,"alpha": alpha_Neg,"R2": R2_Neg},
            "alpha_ratio_dF_over_C": alpha_dF / alpha_C if alpha_C > 0 else 0.0,
        },
        "PTM_rank_3_fraction": rank_3_count / len(Ns_arr),
        "K": args.K, "n_chi": args.n_chi, "N_max": args.N_max,
    }
    # Sweep catalog saved separately (large)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {args.out}", flush=True)

    sweep_file = args.out.replace(".json", "_sweeps.json")
    with open(sweep_file, "w") as f:
        json.dump(sweep_catalog, f)
    print(f"Saved sweeps: {sweep_file}", flush=True)

    out_zip = args.out.replace(".json", ".zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out)
        zf.write(sweep_file)
    print(f"Zipped: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
