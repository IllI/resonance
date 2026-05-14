"""
program_i0_tpu.py -- Program I0: Dynamic Operator Transport (Pilot)
====================================================================
Moves from static boundary pairs to SPATIOTEMPORAL TRANSPORT.

Core question: Does recoverable operator transport (ΔF>0) persist
at distant sites even after pairwise concurrence vanishes?
Does it travel at the Lieb-Robinson velocity?

Setup:
  - N-qubit chain (N=4..16 exact diag, Hilbert space 2^N)
  - Initial state: all-|↑⟩ product state (no prior entanglement)
  - Evolve under H for times t=0..T_max
  - At each (j, t): extract pair PTM for sites (0, j)
  - Compute full observable suite: ΔF, C, Neg, rank_T, H_T

Models:
  - XXZ critical:  J_perp=1, Dz=0   (XY model, ballistic)
  - XXZ Ising:     J_perp=0.5, Dz=2  (localized-like)
  - TFIM critical: J=1, h=1           (critical, ballistic)

Key outputs:
  F_rec(j,t)   -- recoverable transport fidelity field
  C(j,t)       -- pairwise concurrence field
  dF(j,t)      -- transport enhancement field
  rank_T(j,t)  -- PTM rank field
  H_T(j,t)     -- spectral entropy field
  LR_front     -- Lieb-Robinson velocity estimate from ΔF>thresh wavefront

Feed F_rec(j,t) and rank_T(j,t) into D-LinOSS post-hoc for stratum detection.

nohup python3 program_i0_tpu.py > i0.log 2>&1 &
"""

import json, time, math, cmath, zipfile, argparse
import numpy as np
try:
    from scipy.linalg import expm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import jax
    print(f"[DEVICE] {jax.devices()}  backend={jax.default_backend()}", flush=True)
except ImportError:
    print("[DEVICE] JAX not found -- numpy fallback", flush=True)

# ---- Pauli matrices -------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
P0 = np.array([[1,0],[0,0]], dtype=complex)
P1 = np.array([[0,0],[0,1]], dtype=complex)
H2g = np.array([[1,1],[1,-1]], dtype=complex)/math.sqrt(2)

def kron_list(ops):
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def site_op(op, i, N):
    ops = [I2]*N; ops[i] = op; return kron_list(ops)

def bond_op(o1, o2, i, N):
    ops = [I2]*N; ops[i]=o1; ops[i+1]=o2; return kron_list(ops)

# ---- Hamiltonians --------------------------------------------------------
def build_xxz(N, J_perp=1.0, Dz=0.0):
    dim = 2**N; H = np.zeros((dim,dim), dtype=complex)
    for i in range(N-1):
        H += J_perp/2*(bond_op(X,X,i,N) + bond_op(Y,Y,i,N))
        H += Dz*bond_op(Z,Z,i,N)
    return H

def build_ising(N, J=1.0, h=1.0):
    dim = 2**N; H = np.zeros((dim,dim), dtype=complex)
    for i in range(N-1): H -= J*bond_op(Z,Z,i,N)
    for i in range(N):   H -= h*site_op(X,i,N)
    return H

# ---- Exact time evolution -----------------------------------------------
def time_evolve_precompute(H):
    """Diagonalize H once; return (evals, evecs) for fast U(t) application."""
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs

def apply_U(psi, evals, evecs, t):
    """U(t)|psi> = V exp(-i*evals*t) V^† |psi>."""
    coeffs = evecs.conj().T @ psi
    return evecs @ (np.exp(-1j * evals * t) * coeffs)

# ---- Two-site RDM -------------------------------------------------------
def pair_rdm(psi, N, i, j):
    """4x4 RDM for sites (i,j) by permute-reshape partial trace."""
    perm = [i, j] + [k for k in range(N) if k not in (i,j)]
    bulk = 2**(N-2) if N > 2 else 1
    mat  = psi.reshape([2]*N).transpose(perm).reshape(4, bulk)
    rho2 = mat @ mat.conj().T
    return rho2 / float(rho2.trace().real)

# ---- Observables --------------------------------------------------------
def concurrence(rho):
    sy2 = np.kron(Y,Y)
    R   = rho @ (sy2 @ rho.conj() @ sy2)
    ev  = np.sqrt(np.clip(np.linalg.eigvals(R).real, 0, None))
    ev  = np.sort(ev)[::-1]
    return float(max(0.0, ev[0]-ev[1]-ev[2]-ev[3]))

def negativity(rho):
    r4  = rho.reshape(2,2,2,2)
    rpt = r4.transpose(0,3,2,1).reshape(4,4)
    ev  = np.linalg.eigvalsh(rpt)
    return float(np.sum(np.abs(ev[ev<0])))

def compute_T(rho):
    T = np.zeros((3,3))
    for a,sa in enumerate([X,Y,Z]):
        for b,sb in enumerate([X,Y,Z]):
            T[a,b] = float(np.real(np.trace(np.kron(sa,sb)@rho)))
    return T

def spectral_entropy(svs):
    s2 = np.array(svs)**2; s2=s2[s2>1e-10]
    if not len(s2): return 0.0
    p=s2/s2.sum(); return float(-np.sum(p*np.log(p)))

def von_neumann_entropy(rho):
    """S(rho) = -Tr(rho log rho). Representation-agnostic."""
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log(ev)))

def mutual_information(rho2):
    """I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB).
    Representation-agnostic relational observable."""
    rho_A = np.trace(rho2.reshape(2,2,2,2), axis1=1, axis2=3)  # trace out B
    rho_B = np.trace(rho2.reshape(2,2,2,2), axis1=0, axis2=2)  # trace out A
    S_AB  = von_neumann_entropy(rho2)
    S_A   = von_neumann_entropy(rho_A)
    S_B   = von_neumann_entropy(rho_B)
    return float(max(0.0, S_A + S_B - S_AB))

def schmidt_rotate(rho2):
    _, vecs = np.linalg.eigh(rho2)
    M = vecs[:,-1].reshape(2,2)
    U,sig,Vh = np.linalg.svd(M)
    Ul = np.kron(U.conj().T, Vh.conj())
    return Ul@rho2@Ul.conj().T

# ---- Teleportation circuit -----------------------------------------------
CNOT_01 = np.kron(P0,np.kron(I2,I2)) + np.kron(P1,np.kron(X,I2))
CIRC    = np.kron(H2g,np.kron(I2,I2)) @ CNOT_01
CIRC_D  = CIRC.conj().T
PROJ    = {(m1,m2): np.kron(P0 if m1==0 else P1, np.kron(P0 if m2==0 else P1, I2))
           for m1 in range(2) for m2 in range(2)}
UCORR   = {(0,0):I2, (0,1):X, (1,0):Z, (1,1):X@Z}

def avg_fid_haar(rho2, K=200, seed=42):
    rng = np.random.default_rng(seed)
    def rand_qubit():
        u,v = rng.uniform(0,1,2)
        th=math.acos(1-2*u); ph=2*math.pi*v
        return np.array([math.cos(th/2),math.sin(th/2)*cmath.exp(1j*ph)],dtype=complex)
    def fid(psi_A):
        rt = np.kron(np.outer(psi_A,psi_A.conj()), rho2)
        rb = CIRC@rt@CIRC_D; out=np.zeros((2,2),dtype=complex)
        for (m1,m2),Pi in PROJ.items():
            rp=Pi@rb@Pi; prob=float(rp.trace().real)
            if prob<1e-12: continue
            r=(rp/prob).reshape(2,2,2,2,2,2)
            rq=np.einsum('ijkijl->kl',r)
            out+=prob*(UCORR[(m1,m2)]@rq@UCORR[(m1,m2)].conj().T)
        return float(np.clip((psi_A.conj()@out@psi_A).real,0,1))
    return float(np.mean([fid(rand_qubit()) for _ in range(K)]))

def analyze_pair(rho2, K=200):
    C     = concurrence(rho2)
    Neg   = negativity(rho2)
    T     = compute_T(rho2)
    svs   = np.linalg.svd(T, compute_uv=False)
    fm    = float((1+svs.sum())/4)
    F_std = float((1+T[0,0]/3)/2)
    F_opt = float((2*fm+1)/3)
    H_T   = spectral_entropy(svs)
    rank_T= int((svs>1e-3).sum())
    rho_r = schmidt_rotate(rho2)
    F_sch = avg_fid_haar(rho_r, K=K)
    dF    = F_sch - F_std
    # Representation-agnostic observables
    MI    = mutual_information(rho2)  # I(0:j) -- no PTM geometry assumption
    S_AB  = von_neumann_entropy(rho2) # joint entropy
    return dict(C=C, Neg=Neg, F_std=F_std, F_sch=F_sch, F_opt=F_opt,
                dF=dF, rank_T=rank_T, H_T=H_T,
                sv1=float(svs[0]), sv2=float(svs[1]), sv3=float(svs[2]),
                MI=MI, S_AB=S_AB)

# ---- LR front detection -------------------------------------------------
def estimate_LR_velocity(js, ts, field, thresh=0.005):
    """Find wavefront: largest j where field[j,t] > thresh, per time step.
       Fit slope: j_front ~ v_LR * t."""
    fronts = []
    js_arr = list(js)
    for ti, t in enumerate(ts):
        row = np.array([field[j][ti] for j in js_arr])
        above = [j for j,v in zip(js_arr, row) if v > thresh]
        if above: fronts.append((t, max(above)))
    if len(fronts) < 3: return None, fronts
    ts_f = np.array([f[0] for f in fronts], float)
    js_f = np.array([f[1] for f in fronts], float)
    v = float(np.polyfit(ts_f, js_f, 1)[0])
    return v, fronts

# ---- Main ----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N",       type=int,   default=14)
    p.add_argument("--T-max",   type=float, default=4.0)
    p.add_argument("--n-t",     type=int,   default=40)
    p.add_argument("--K",       type=int,   default=200)
    p.add_argument("--out",     default="program_i0_results.json")
    args = p.parse_args()
    N = args.N

    # Time grid -- linear
    ts = np.linspace(0.0, args.T_max, args.n_t)
    js = list(range(1, N))   # output sites

    # Initial state: |+>_0 x |up>^{N-1}  (single-magnon superposition)
    # |up...up> (all-up) is eigenstate of XY/XXZ -- use superposition
    # psi0 = (|0 00...0> + |1 00...0>) / sqrt(2)
    # The |100...0> magnon component propagates ballistically under XY/XXZ.
    psi0 = np.zeros(2**N, dtype=complex)
    psi0[0]          = 1.0/math.sqrt(2)   # |00...0> = all up
    psi0[2**(N-1)]   = 1.0/math.sqrt(2)   # |10...0> = magnon on site 0

    MODELS = [
        ("XY",    build_xxz(N, J_perp=1.0, Dz=0.0)),
        ("XXZ",   build_xxz(N, J_perp=1.0, Dz=1.0)),
        ("Ising", build_ising(N, J=1.0, h=1.0)),
    ]

    print("="*70, flush=True)
    print("Program I0 -- Dynamic Operator Transport (Pilot)")
    print(f"N={N}  T_max={args.T_max}  n_t={args.n_t}  K={args.K}")
    print(f"Initial state: |up>^N  Models: {[m for m,_ in MODELS]}")
    print("="*70, flush=True)

    all_results = {}
    t0_global = time.time()

    for model_name, H in MODELS:
        print(f"\n[{model_name}] Diagonalizing... ", end="", flush=True)
        evals, evecs = time_evolve_precompute(H)
        print(f"done.", flush=True)

        # Store 2D fields: field[j][t_idx]
        fields = {obs: {j: [] for j in js}
                  for obs in ["dF","C","Neg","rank_T","H_T","F_sch","F_std","MI","S_AB"]}

        print(f"  {'t':>5}  " +
              "  ".join(f"j={j:>2}:dF" for j in js[:min(4,len(js))]) + " ...",
              flush=True)

        for ti, t in enumerate(ts):
            psi_t = apply_U(psi0, evals, evecs, t)
            row_dF = []
            for j in js:
                rho2  = pair_rdm(psi_t, N, 0, j)
                obs   = analyze_pair(rho2, K=args.K)
                for key in fields: fields[key][j].append(obs[key])
                row_dF.append(obs["dF"])

            # Print progress every 5 steps
            if ti % 5 == 0:
                vals = "  ".join(f"{v:>6.3f}" for v in row_dF[:4])
                print(f"  t={t:>5.2f}  {vals} ...", flush=True)

        # LR velocity estimate from excess dF wavefront (above baseline)
        baseline_dF = {j: fields["dF"][j][0] for j in js}  # dF at t=0
        excess_dF   = {j: [fields["dF"][j][ti] - baseline_dF[j]
                           for ti in range(len(ts))] for j in js}
        v_LR_dF, fronts_dF = estimate_LR_velocity(
            js, ts, excess_dF, thresh=0.015)
        v_LR_C, fronts_C = estimate_LR_velocity(
            js, ts, fields["C"], thresh=0.005)
        # MI LR velocity -- representation-agnostic cross-check
        baseline_MI = {j: fields["MI"][j][0] for j in js}
        excess_MI   = {j: [fields["MI"][j][ti] - baseline_MI[j]
                           for ti in range(len(ts))] for j in js}
        v_LR_MI, fronts_MI = estimate_LR_velocity(
            js, ts, excess_MI, thresh=0.01)

        # Crossover: at each site j, find t_c(C) and t_c(dF)
        crossover = {}
        for j in js:
            c_arr  = np.array(fields["C"][j])
            dF_arr = np.array(fields["dF"][j])
            # t when C first rises above 0.005
            c_on   = ts[c_arr  > 0.005][0] if (c_arr  > 0.005).any() else None
            dF_on  = ts[dF_arr > 0.005][0] if (dF_arr > 0.005).any() else None
            # t when C falls back below 0.005 (decoherence front)
            cross_c_off = None
            if c_on is not None:
                after = ts[ts > c_on]
                c_after = c_arr[ts > c_on]
                below = after[c_after < 0.005]
                if len(below): cross_c_off = float(below[0])
            crossover[j] = {"t_c_on": float(c_on) if c_on is not None else None,
                            "t_dF_on": float(dF_on) if dF_on is not None else None,
                            "t_c_off": cross_c_off}

        print(f"\n  [{model_name}] LR velocity from dF front: {v_LR_dF}", flush=True)
        print(f"  [{model_name}] LR velocity from C front:  {v_LR_C}", flush=True)
        print(f"  [{model_name}] LR velocity from MI front: {v_LR_MI}  <-- rep-agnostic", flush=True)

        all_results[model_name] = {
            "ts": ts.tolist(),
            "js": js,
            "fields": {k: {str(j): v for j,v in f.items()}
                       for k,f in fields.items()},
            "LR_velocity_dF": v_LR_dF,
            "LR_velocity_C":  v_LR_C,
            "LR_velocity_MI": v_LR_MI,
            "crossover": {str(j): v for j,v in crossover.items()},
        }

    total = time.time() - t0_global
    print(f"\n\nTotal runtime: {total:.1f}s", flush=True)

    # ---- Summary ----------------------------------------------------------
    print("\n" + "="*70, flush=True)
    print("TRANSPORT SUMMARY -- Representation Cross-Check", flush=True)
    print(f"  v_LR(dF) = PTM geometry front", flush=True)
    print(f"  v_LR(C)  = pairwise entanglement front", flush=True)
    print(f"  v_LR(MI) = mutual information front (representation-agnostic)", flush=True)
    print(f"{'Model':>8}  {'v_LR(dF)':>10}  {'v_LR(C)':>10}  {'v_LR(MI)':>10}", flush=True)
    for mn in all_results:
        r = all_results[mn]
        v1 = f"{r['LR_velocity_dF']:.2f}" if r['LR_velocity_dF'] else "N/A"
        v2 = f"{r['LR_velocity_C']:.2f}"  if r['LR_velocity_C']  else "N/A"
        v3 = f"{r['LR_velocity_MI']:.2f}" if r['LR_velocity_MI'] else "N/A"
        print(f"  {mn:>8}  {v1:>10}  {v2:>10}  {v3:>10}", flush=True)

    output = {"N": N, "T_max": args.T_max, "n_t": args.n_t,
              "K": args.K, "total_runtime": total, "models": all_results}

    with open(args.out,"w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}", flush=True)

    out_zip = args.out.replace(".json",".zip")
    with zipfile.ZipFile(out_zip,"w",zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out)
    print(f"Zipped: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
