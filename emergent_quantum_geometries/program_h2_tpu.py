"""
program_h2_tpu.py -- Program H2: Universality of Recoverable Transport Scaling
================================================================================
Cross-Hamiltonian test: do exponents alpha_OAT, alpha_XXZ, alpha_Ising cluster?
Measures ΔF/C^β collapse to find universal scaling form.

Models:
  OAT:   H = chi*J_z^A*J_z^B    closed-form rho2, N=4..256
  XXZ:   H = J*(XX+YY) + Dz*ZZ  ground state ED, N=4..12
  Ising: H = -J*ZZ - h*X         ground state ED, N=4..12

Protocol: Schmidt rotation (U_A=U†, U_B=Vh.conj()) -- oat_teleport_v2 convention.
Observables: F_std, F_sch, F_opt, ΔF=F_sch-F_std, C, rank_T, ΔF/C^β.

Upload and run:
  python3 program_h2_tpu.py

Saves: program_h2_results.json + program_h2_results.zip
"""

import json, time, os, zipfile, math, cmath, itertools
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    USE_JAX = True
    print(f"[DEVICE] {jax.devices()}  backend={jax.default_backend()}", flush=True)
except ImportError:
    USE_JAX = False
    print("[DEVICE] JAX not available -- numpy fallback", flush=True)

# ---- Pauli matrices -------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
P0 = np.array([[1,0],[0,0]], dtype=complex)
P1 = np.array([[0,0],[0,1]], dtype=complex)
H2g = np.array([[1,1],[1,-1]], dtype=complex)/math.sqrt(2)

def kron_list(ops):
    result = ops[0]
    for op in ops[1:]: result = np.kron(result, op)
    return result

def site_op(op, site, N):
    """Single-site operator at site index in N-qubit system."""
    ops = [I2]*N; ops[site] = op
    return kron_list(ops)

def two_site_op(op1, op2, i, j, N):
    ops = [I2]*N; ops[i] = op1; ops[j] = op2
    return kron_list(ops)

# ---- Teleportation circuit -----------------------------------------------
CNOT_01 = kron_list([P0,I2,I2]) + kron_list([P1,X,I2])
CIRC    = kron_list([H2g,I2,I2]) @ CNOT_01
CIRC_D  = CIRC.conj().T
PROJ    = {(m1,m2): kron_list([P0 if m1==0 else P1, P0 if m2==0 else P1, I2])
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
    th = math.acos(1-2*u1); ph = 2*math.pi*u2
    return np.array([math.cos(th/2), math.sin(th/2)*cmath.exp(1j*ph)], dtype=complex)

def avg_fid_haar(rho2, K=200, seed=42):
    rng = np.random.default_rng(seed)
    return float(np.mean([teleport_fid(haar_qubit(rng), rho2) for _ in range(K)]))

# ---- Diagnostics ---------------------------------------------------------
def concurrence(rho):
    sy2 = np.kron(Y, Y)
    R   = rho @ (sy2 @ rho.conj() @ sy2)
    eigs = np.sqrt(np.abs(np.linalg.eigvals(R).real).clip(0))
    eigs = np.sort(eigs)[::-1]
    return float(max(0, eigs[0]-eigs[1]-eigs[2]-eigs[3]))

def compute_T(rho):
    T = np.zeros((3,3))
    for i,si in enumerate([X,Y,Z]):
        for j,sj in enumerate([X,Y,Z]):
            T[i,j] = float(np.real(np.trace(np.kron(si,sj)@rho)))
    return T

def f_opt_svd(rho):
    svs = np.linalg.svd(compute_T(rho), compute_uv=False)
    return float((1+svs.sum())/4), svs

def schmidt_rotate(rho2):
    """U_A=U†, U_B=Vh.conj() from dominant eigvec SVD. oat_teleport_v2 convention."""
    _, vecs = np.linalg.eigh(rho2)
    M = vecs[:,-1].reshape(2,2)
    U, sigma, Vh = np.linalg.svd(M)
    U_A = U.conj().T; U_B = Vh.conj()
    Ul  = np.kron(U_A, U_B)
    return Ul@rho2@Ul.conj().T, float(2*abs(sigma[0])*abs(sigma[1])), U_A, U_B

def analyze(rho2, K=200):
    C           = concurrence(rho2)
    fm, svs     = f_opt_svd(rho2)
    F_opt       = (2*fm+1)/3
    T           = compute_T(rho2)
    F_std       = float((1 + T[0,0]/3)/2)   # analytic: (1+T_xx/3)/2
    rho_rot, Cs, _, _ = schmidt_rotate(rho2)
    F_sch       = avg_fid_haar(rho_rot, K=K)
    return {"C":C, "C_sch":Cs, "F_std":F_std, "F_sch":F_sch, "F_opt":F_opt,
            "dF":F_sch-F_std, "f_max":fm, "rank_T":int((svs>1e-3).sum()),
            "T_xx":float(T[0,0]), "svs":svs.tolist()}

# ---- OAT (closed-form) ---------------------------------------------------
IDX = [(0,0),(0,1),(1,0),(1,1)]

def rho2_oat(chi_t, N):
    rho = np.zeros((4,4), dtype=complex)
    ep  = max(0, N//2-1)
    for a,(iL,iR) in enumerate(IDX):
        for b,(jL,jR) in enumerate(IDX):
            ph   = -1j*chi_t*((iL-.5)*(iR-.5)-(jL-.5)*(jR-.5))
            cR   = math.cos((iR-jR)*chi_t/2)**ep if ep>0 else 1.0
            cL   = math.cos((iL-jL)*chi_t/2)**ep if ep>0 else 1.0
            rho[a,b] = 0.25*cmath.exp(ph)*cR*cL
    return rho

def oat_F_null(N):
    """Theorem 4 check: F_std at chi_t=pi should be 0.5."""
    T = compute_T(rho2_oat(math.pi-1e-6, N))
    return float((1+T[0,0]/3)/2)

def sweep_oat(N, n_chi=100, K=200):
    """Adaptive chi_t grid: min = max(0.001, 0.5/N), max = min(pi-0.01, 8/N)."""
    chi_lo = max(0.001, 0.3/N)
    chi_hi = min(math.pi-0.01, max(6/N, 0.5))
    chi_vals = np.concatenate([
        np.linspace(chi_lo, chi_hi, n_chi),
        np.linspace(chi_hi, math.pi-0.01, 10)
    ])
    best = {"dF": -1}; peak_C = 0
    for chi_t in chi_vals:
        r = analyze(rho2_oat(chi_t, N), K=K)
        peak_C = max(peak_C, r["C"])
        if r["dF"] > best["dF"]:
            best = r; best["chi_star"] = float(chi_t)
    above = [c for c in chi_vals
             if (2*float((1+np.linalg.svd(compute_T(rho2_oat(c,N)),compute_uv=False)[0].sum())/4)+1)/3 > 2/3]
    dchi = (max(above)-min(above)) if len(above)>1 else 0.0
    return {**best, "N":N, "model":"OAT", "peak_C":peak_C,
            "delta_chi":float(dchi), "F_null":oat_F_null(N)}

# ---- Ground-state ED models ----------------------------------------------
def build_xxz(N, J_perp=1.0, Dz=1.0):
    """XXZ: H = J_perp*(XX+YY)/2 + Dz*ZZ summed over bonds."""
    dim = 2**N; H = np.zeros((dim, dim), dtype=complex)
    for i in range(N-1):
        H += J_perp/2*(two_site_op(X,X,i,i+1,N)+two_site_op(Y,Y,i,i+1,N))
        H += Dz*two_site_op(Z,Z,i,i+1,N)
    return H

def build_ising(N, J=1.0, h=1.0):
    """TFIM: H = -J*ZZ - h*X."""
    dim = 2**N; H = np.zeros((dim, dim), dtype=complex)
    for i in range(N-1):
        H -= J*two_site_op(Z,Z,i,i+1,N)
    for i in range(N):
        H -= h*site_op(X,i,N)
    return H

def pair_rdm(gs, N, site_i, site_j):
    """
    Compute 4x4 two-site RDM for sites (site_i, site_j) by partial trace.

    Method: permute so desired sites are axes 0,1; reshape to (4, 2^(N-2));
    rho2 = psi @ psi† already traces out the bulk.
    Mirrors OAT cross-half convention: use midchain bond (N//2-1, N//2).
    """
    perm = [site_i, site_j] + [k for k in range(N) if k not in (site_i, site_j)]
    psi  = gs.reshape([2]*N).transpose(perm).reshape(4, 2**(N-2))
    rho2 = psi @ psi.conj().T
    return rho2 / float(np.trace(rho2).real)

def ground_pair_rdm(H, N):
    """Ground state midchain bond RDM (sites N//2-1, N//2) -- cross-half analog."""
    _, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]
    return pair_rdm(gs, N, N//2 - 1, N//2)

def sweep_ed(N, model, param_vals, K=200):
    """Sweep coupling parameter for ED model using midchain bond pair."""
    best = None; peak_C = 0.0
    for p in param_vals:
        try:
            H   = build_xxz(N, J_perp=p) if model == "XXZ" else build_ising(N, h=p)
            rho = ground_pair_rdm(H, N)
            r   = analyze(rho, K=K)
            peak_C = max(peak_C, r["C"])
            if best is None or r["dF"] > best["dF"]:
                best = dict(r); best["param_star"] = float(p)
        except Exception as e:
            print(f"    [WARN] {model} N={N} p={p:.3f}: {type(e).__name__}: {e}", flush=True)
            continue
    if best is None:
        best = {"dF":0.0,"F_std":2/3,"F_sch":2/3,"F_opt":2/3,
                "C":0.0,"C_sch":0.0,"rank_T":0,"T_xx":0.0,
                "svs":[0.0,0.0,0.0],"f_max":0.5,"param_star":0.0}
    return {**best, "N":N, "model":model, "peak_C":peak_C}

# ---- Power-law fit -------------------------------------------------------
def fit_pl(Ns, vals):
    lN = np.log(np.array(Ns,float)); lv = np.log(np.array(vals,float)+1e-12)
    c  = np.polyfit(lN,lv,1); pred=np.polyval(c,lN)
    ss_res=np.sum((lv-pred)**2); ss_tot=np.sum((lv-lv.mean())**2)
    return float(np.exp(c[1])), float(-c[0]), float(1-ss_res/ss_tot) if ss_tot>1e-10 else 0.0

# ---- ΔF/C^β collapse -----------------------------------------------------
def ratio_collapse(data_list, beta_vals=None):
    """Find β minimizing variance of ΔF/C^β across N. Returns best β."""
    if beta_vals is None: beta_vals = np.linspace(0.3, 2.0, 35)
    dFs = np.array([d["dF"] for d in data_list], float)
    Cs  = np.array([d["C"]  for d in data_list], float) + 1e-10
    best_beta=1.0; best_cv=1e9
    for beta in beta_vals:
        ratio = dFs / Cs**beta
        cv = float(np.std(ratio)/np.mean(ratio)) if np.mean(ratio)>1e-10 else 1e9
        if cv < best_cv: best_cv=cv; best_beta=float(beta)
    return best_beta, best_cv

# ---- Main ----------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N-max-oat", type=int, default=128)
    p.add_argument("--N-max-ed",  type=int, default=10)
    p.add_argument("--n-chi",     type=int, default=80)
    p.add_argument("--K",         type=int, default=200)
    p.add_argument("--out",       default="program_h2_results.json")
    args = p.parse_args()

    N_oat = []; N=4
    while N <= args.N_max_oat: N_oat.append(N); N*=2
    N_ed  = list(range(4, args.N_max_ed+1, 2))   # even sizes only for ED

    print("="*70, flush=True)
    print("Program H2 -- Universality of Recoverable Transport Scaling")
    print(f"OAT N={N_oat}  ED N={N_ed}  chi_t/param grid={args.n_chi}  K={args.K}")
    print("="*70, flush=True)

    t0=time.time(); results={}

    # --- OAT sweep --------------------------------------------------------
    print("\n[OAT] Closed-form rho2, adaptive chi_t grid", flush=True)
    oat_data = []
    for N in N_oat:
        r = sweep_oat(N, n_chi=args.n_chi, K=args.K)
        oat_data.append(r)
        print(f"  N={N:>4}  chi*={r.get('chi_star',0):.4f}  "
              f"F_sch={r['F_sch']:.4f}  dF={r['dF']:.4f}  "
              f"C={r['C']:.4f}  rank={r['rank_T']}  F(pi)={r['F_null']:.4f}", flush=True)
    results["OAT"] = oat_data

    oat_ok = [d for d in oat_data if d["C"]>1e-4]
    if len(oat_ok)>=3:
        A,alpha,R2 = fit_pl([d["N"] for d in oat_ok],[d["dF"] for d in oat_ok])
        _,aC,R2C   = fit_pl([d["N"] for d in oat_ok],[d["C"]  for d in oat_ok])
        beta,cv    = ratio_collapse(oat_ok)
        print(f"  OAT:  dF~N^(-{alpha:.3f}) R2={R2:.3f}  "
              f"C~N^(-{aC:.3f}) R2={R2C:.3f}  best_beta={beta:.2f} cv={cv:.3f}")
        results["OAT_fit"] = {"alpha_dF":alpha,"R2_dF":R2,"alpha_C":aC,"R2_C":R2C,
                               "best_beta":beta,"collapse_cv":cv}

    # --- XXZ sweep --------------------------------------------------------
    if N_ed:
        print(f"\n[XXZ] Ground state ED, J_perp sweep 0.05..8.0", flush=True)
        # Dense near J_perp=1..5 where entanglement peaks; extended to 8
        param_xxz = np.concatenate([
            np.linspace(0.05, 1.0, args.n_chi//4),
            np.linspace(1.0,  4.0, args.n_chi//2),
            np.linspace(4.0,  8.0, args.n_chi//4),
        ])
        xxz_data  = []
        for N in N_ed:
            if 2**N > 4096: break   # memory limit
            r = sweep_ed(N, "XXZ", param_xxz, K=args.K)
            xxz_data.append(r)
            print(f"  N={N:>3}  J_perp*={r.get('param_star',0):.3f}  "
                  f"F_sch={r['F_sch']:.4f}  dF={r['dF']:.4f}  C={r['C']:.4f}", flush=True)
        results["XXZ"] = xxz_data
        xxz_ok = [d for d in xxz_data if d["C"] > 0.01]  # entangled only
        if len(xxz_ok)>=2:
            A2,a2,R2_2 = fit_pl([d["N"] for d in xxz_ok],[d["dF"] for d in xxz_ok])
            beta2,cv2  = ratio_collapse(xxz_ok)
            print(f"  XXZ:  dF~N^(-{a2:.3f}) R2={R2_2:.3f}  best_beta={beta2:.2f}")
            results["XXZ_fit"]={
                "alpha_dF":a2,"R2_dF":R2_2,"best_beta":beta2,"cv":cv2,
                "N_entangled":[d["N"] for d in xxz_ok]}

    # --- Ising sweep ------------------------------------------------------
    if N_ed:
        print(f"\n[Ising] Ground state ED, h/J sweep near critical point", flush=True)
        # Dense near h_c=1 (critical), extended to 3.0
        param_ising = np.concatenate([
            np.linspace(0.05, 0.5,  args.n_chi//4),
            np.linspace(0.5,  2.0,  args.n_chi//2),   # critical region
            np.linspace(2.0,  3.0,  args.n_chi//4),
        ])
        ising_data  = []
        for N in N_ed:
            if 2**N > 4096: break
            r = sweep_ed(N, "Ising", param_ising, K=args.K)
            ising_data.append(r)
            print(f"  N={N:>3}  h/J*={r.get('param_star',0):.3f}  "
                  f"F_sch={r['F_sch']:.4f}  dF={r['dF']:.4f}  C={r['C']:.4f}", flush=True)
        results["Ising"] = ising_data
        ising_ok = [d for d in ising_data if d["C"] > 0.01]  # entangled only
        if len(ising_ok)>=2:
            A3,a3,R2_3 = fit_pl([d["N"] for d in ising_ok],[d["dF"] for d in ising_ok])
            beta3,cv3  = ratio_collapse(ising_ok)
            print(f"  Ising: dF~N^(-{a3:.3f}) R2={R2_3:.3f}  best_beta={beta3:.2f}")
            results["Ising_fit"]={
                "alpha_dF":a3,"R2_dF":R2_3,"best_beta":beta3,"cv":cv3,
                "N_entangled":[d["N"] for d in ising_ok]}

    # --- Cross-model summary ----------------------------------------------
    print("\n" + "="*70)
    print("UNIVERSALITY SUMMARY")
    print(f"{'Model':>8}  {'alpha_dF':>9}  {'alpha_C':>9}  {'best_beta':>10}  {'R2_dF':>7}")
    for key in ["OAT_fit","XXZ_fit","Ising_fit"]:
        if key in results:
            r = results[key]
            aC = r.get("alpha_C","--")
            print(f"  {key.split('_')[0]:>8}  {r['alpha_dF']:>9.3f}  "
                  f"{str(aC)[:7]:>9}  {r['best_beta']:>10.3f}  {r['R2_dF']:>7.3f}")

    total = time.time()-t0
    print(f"\nTotal runtime: {total:.1f}s")

    # Save
    def default(o):
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(args.out,"w") as f:
        json.dump(results, f, indent=2, default=default)
    print(f"Saved: {args.out}")

    out_zip = args.out.replace(".json",".zip")
    with zipfile.ZipFile(out_zip,"w",zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out)
    print(f"Zipped: {out_zip}")


if __name__=="__main__":
    main()
