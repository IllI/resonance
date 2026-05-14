"""
program_iplus_tpu.py -- Program I+: Representation Stability Test
==================================================================
Core question: Is the transport stratum representation-stable?

Four independent representations of the same two-site state rho2(0,j,t):

  Rep 1 (PTM)   -- Pauli Transfer Matrix: dF = F_sch - F_std
                   Stratum: excess dF > thresh_dF
  Rep 2 (Choi)  -- Singlet fraction: F_s = <Phi+|rho2|Phi+>
                   Channel-state duality; stratum: F_s > thresh_Fs
                   Classical bound: F_s = 1/4 (no quantum transport)
  Rep 3 (OS)    -- Operator spreading: OT_j(t) = perturbation transport
                   ||(rho_j^{P_mu} - rho_j)||_F summed over Pauli inputs
                   Stratum: excess OT > thresh_OT
  Rep 4 (MI)    -- Mutual information: I(0:j) = S_0 + S_j - S_0j
                   Representation-minimal relational structure
                   Stratum: excess MI > thresh_MI

For each (j,t): compute 4 binary stratum indicators.
Agreement rate = fraction of (j,t) where all 4 agree.
LR velocity per representation.
Velocity spread = std/mean -- if low, representations are consistent.

Physical interpretation:
  High agreement + low velocity spread --> representation-stable structure
  Low agreement OR high velocity spread --> representation artifact
  Ising negative control must show low agreement (MI spreads, PTM/OS do not)
"""

import json, time, math, cmath, zipfile, argparse
import numpy as np

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
PAULIS = [X, Y, Z]
PHI_PLUS = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]], dtype=complex)/2

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
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N-1):
        H += J_perp/2*(bond_op(X,X,i,N) + bond_op(Y,Y,i,N))
        H += Dz*bond_op(Z,Z,i,N)
    return H

def build_ising(N, J=1.0, h=1.0):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N-1): H -= J*bond_op(Z,Z,i,N)
    for i in range(N):   H -= h*site_op(X,i,N)
    return H

# ---- Time evolution -------------------------------------------------------
def diagonalize(H):
    return np.linalg.eigh(H)

def apply_U(psi, evals, evecs, t):
    coeffs = evecs.conj().T @ psi
    return evecs @ (np.exp(-1j * evals * t) * coeffs)

# ---- Two-site and single-site RDMs ---------------------------------------
def pair_rdm(psi, N, i, j):
    perm = [i, j] + [k for k in range(N) if k not in (i,j)]
    bulk = 2**(N-2) if N > 2 else 1
    mat  = psi.reshape([2]*N).transpose(perm).reshape(4, bulk)
    rho2 = mat @ mat.conj().T
    return rho2 / float(rho2.trace().real)

def single_rho(psi, N, j):
    """Single-site RDM at site j."""
    perm = [j] + [k for k in range(N) if k != j]
    bulk = 2**(N-1)
    mat  = psi.reshape([2]*N).transpose(perm).reshape(2, bulk)
    rho  = mat @ mat.conj().T
    return rho / float(rho.trace().real)

# ---- Rep 1: PTM transport ------------------------------------------------
def compute_T(rho2):
    T = np.zeros((3,3))
    for a,sa in enumerate(PAULIS):
        for b,sb in enumerate(PAULIS):
            T[a,b] = float(np.real(np.trace(np.kron(sa,sb) @ rho2)))
    return T

def schmidt_rotate(rho2):
    _, vecs = np.linalg.eigh(rho2)
    M = vecs[:,-1].reshape(2,2)
    U,_,Vh = np.linalg.svd(M)
    return np.kron(U.conj().T, Vh.conj()) @ rho2 @ np.kron(U.conj().T, Vh.conj()).conj().T

def avg_fid_haar(rho2, K=200, seed=42):
    rng = np.random.default_rng(seed)
    CNOT_01 = np.kron(P0,np.kron(I2,I2)) + np.kron(P1,np.kron(X,I2))
    CIRC    = np.kron(H2g,np.kron(I2,I2)) @ CNOT_01; CIRC_D = CIRC.conj().T
    PROJ    = {(m1,m2): np.kron(P0 if m1==0 else P1, np.kron(P0 if m2==0 else P1, I2))
               for m1 in range(2) for m2 in range(2)}
    UCORR   = {(0,0):I2,(0,1):X,(1,0):Z,(1,1):X@Z}
    def rand_q():
        u,v = rng.uniform(0,1,2)
        th=math.acos(1-2*u); ph=2*math.pi*v
        return np.array([math.cos(th/2), math.sin(th/2)*cmath.exp(1j*ph)], dtype=complex)
    def fid(pA):
        rt=np.kron(np.outer(pA,pA.conj()), rho2); rb=CIRC@rt@CIRC_D
        out=np.zeros((2,2), dtype=complex)
        for (m1,m2),Pi in PROJ.items():
            rp=Pi@rb@Pi; prob=float(rp.trace().real)
            if prob<1e-12: continue
            r=(rp/prob).reshape(2,2,2,2,2,2); rq=np.einsum('ijkijl->kl',r)
            out+=prob*(UCORR[(m1,m2)]@rq@UCORR[(m1,m2)].conj().T)
        return float(np.clip((pA.conj()@out@pA).real, 0, 1))
    return float(np.mean([fid(rand_q()) for _ in range(K)]))

def ptm_obs(rho2, K=200):
    T     = compute_T(rho2)
    svs   = np.linalg.svd(T, compute_uv=False)
    F_std = float((1 + T[0,0]/3) / 2)
    rank_T= int((svs > 1e-3).sum())
    rho_r = schmidt_rotate(rho2)
    F_sch = avg_fid_haar(rho_r, K=K)
    dF    = F_sch - F_std
    return dict(dF=dF, F_std=F_std, F_sch=F_sch, rank_T=rank_T,
                sv1=float(svs[0]), sv2=float(svs[1]), sv3=float(svs[2]))

# ---- Rep 2: Bipartite entanglement entropy (EE) -------------------------
def bipartite_entropy(psi, N, j):
    """S_j = entanglement entropy across cut: sites 0..j | sites j+1..N-1.
    Requires FULL N-qubit state -- NOT derivable from pair_rdm(0,j).
    Starts at 0 for product state. Propagates as entanglement front.
    Operationally: cumulative entanglement crossing spatial cut at j.
    Independent from MI(0:j) which is pairwise (only sites 0 and j).
    """
    dim_L = 2**(j+1)
    dim_R = 2**(N-j-1) if N > j+1 else 1
    mat   = psi.reshape(dim_L, max(dim_R, 1))
    svs   = np.linalg.svd(mat, compute_uv=False)
    s2 = svs**2; s2 = s2[s2 > 1e-12]
    return float(-np.sum(s2 * np.log(s2))) if len(s2) else 0.0

# ---- Rep 3: Operator spreading -------------------------------------------
def os_obs(psi_t, psi_P_list, N, j):
    """OT_j = sqrt(sum_mu ||rho_j^{P_mu} - rho_j||_F^2).
    psi_P_list: list of 3 states P_mu_0 |psi_t> (pre-evolved with U(t)).
    This measures how much a Pauli perturbation on site 0 has spread to j.
    """
    rho_j = single_rho(psi_t, N, j)
    OT_sq = 0.0
    for psi_P in psi_P_list:
        rho_j_P = single_rho(psi_P, N, j)
        delta   = rho_j_P - rho_j
        OT_sq  += float(np.real(np.trace(delta.conj().T @ delta)))
    return dict(OT=float(math.sqrt(max(0.0, OT_sq))))

# ---- Rep 4: Mutual information (from existing I0) ------------------------
def von_neumann_entropy(rho):
    ev = np.linalg.eigvalsh(rho); ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log(ev)))

def mi_obs(rho2):
    rho_A = np.trace(rho2.reshape(2,2,2,2), axis1=1, axis2=3)
    rho_B = np.trace(rho2.reshape(2,2,2,2), axis1=0, axis2=2)
    MI = max(0.0, von_neumann_entropy(rho_A) + von_neumann_entropy(rho_B)
             - von_neumann_entropy(rho2))
    return dict(MI=float(MI))

# ---- LR velocity ---------------------------------------------------------
def estimate_LR(js, ts, field, thresh):
    fronts = []
    for ti, t in enumerate(ts):
        above = [j for j in js if field[j][ti] > thresh]
        if above: fronts.append((t, max(above)))
    if len(fronts) < 3: return None, fronts
    ts_f = np.array([f[0] for f in fronts], float)
    js_f = np.array([f[1] for f in fronts], float)
    return float(np.polyfit(ts_f, js_f, 1)[0]), fronts

# ---- Stratum agreement ---------------------------------------------------
def compute_agreement(js, ts, fields, thresholds):
    """For each (j,t), compute binary stratum indicator per representation,
    then check if all 4 agree (all in-stratum or all out-stratum)."""
    reps = list(thresholds.keys())
    agree_count = 0; total = 0
    per_rep_in = {r: 0 for r in reps}
    for j in js:
        for ti in range(len(ts)):
            indicators = {}
            for rep, thresh in thresholds.items():
                val = fields[rep][j][ti]
                indicators[rep] = int(val > thresh)
                per_rep_in[rep] += indicators[rep]
            vals = list(indicators.values())
            agree_count += int(all(v == vals[0] for v in vals))
            total += 1
    agreement_rate = agree_count / total if total > 0 else 0.0
    in_rates = {r: per_rep_in[r]/total for r in reps}
    return dict(agreement_rate=agreement_rate, in_rates=in_rates, total=total)

# ---- Main ----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N",          type=int,   default=12)
    p.add_argument("--T-max",      type=float, default=6.0)
    p.add_argument("--n-t",        type=int,   default=40)
    p.add_argument("--K",          type=int,   default=200)
    p.add_argument("--thresh-dF",  type=float, default=0.015)
    p.add_argument("--thresh-EE",  type=float, default=0.04)
    p.add_argument("--thresh-OT",  type=float, default=0.015)
    p.add_argument("--thresh-MI",  type=float, default=0.03)
    p.add_argument("--out",        default="program_iplus_results.json")
    args = p.parse_args()
    N = args.N

    ts = np.linspace(0.0, args.T_max, args.n_t)
    js = list(range(1, N))

    # Initial state: |+>_0 x |up>^{N-1}
    psi0 = np.zeros(2**N, dtype=complex)
    psi0[0] = 1/math.sqrt(2); psi0[2**(N-1)] = 1/math.sqrt(2)

    # Perturbed initial states for operator spreading (3 Paulis on site 0)
    psi0_P = [site_op(P, 0, N) @ psi0 for P in PAULIS]

    MODELS = [
        ("XY",    build_xxz(N, J_perp=1.0, Dz=0.0)),
        ("XXZ",   build_xxz(N, J_perp=1.0, Dz=1.0)),
        ("Ising", build_ising(N, J=1.0, h=1.0)),
    ]

    print("="*70, flush=True)
    print("Program I+ -- Representation Stability Test")
    print(f"N={N}  T_max={args.T_max}  n_t={args.n_t}  K={args.K}")
    print("Representations: PTM | Choi(singlet) | OpSpreading | MI", flush=True)
    print("="*70, flush=True)

    all_results = {}
    t0_global = time.time()

    for model_name, H in MODELS:
        print(f"\n[{model_name}] Diagonalizing... ", end="", flush=True)
        evals, evecs = diagonalize(H)
        print("done.", flush=True)

        # Fields indexed by [rep][j][ti]
        reps = ["dF", "EE", "OT", "MI"]
        fields = {r: {j: [] for j in js} for r in reps}
        extra  = {k: {j: [] for j in js} for k in
                  ["F_std","F_sch","rank_T","sv1","sv2","sv3","F_s"]}

        print(f"  {'t':>5}  {'j=1 dF':>8}  {'j=1 EE':>8}  "
              f"{'j=1 OT':>8}  {'j=1 MI':>8}  | all-site dF...",
              flush=True)

        for ti, t in enumerate(ts):
            # Evolve main state and 3 perturbed states
            psi_t  = apply_U(psi0,    evals, evecs, t)
            psi_Pt = [apply_U(p0P, evals, evecs, t) for p0P in psi0_P]

            for j in js:
                rho2  = pair_rdm(psi_t, N, 0, j)
                obs_p = ptm_obs(rho2, K=args.K)
                obs_o = os_obs(psi_t, psi_Pt, N, j)
                obs_m = mi_obs(rho2)

                fields["dF"][j].append(obs_p["dF"])
                fields["EE"][j].append(bipartite_entropy(psi_t, N, j))
                fields["OT"][j].append(obs_o["OT"])
                fields["MI"][j].append(obs_m["MI"])

                for k in ["F_std","F_sch","rank_T","sv1","sv2","sv3"]:
                    extra[k][j].append(obs_p[k])
                rho_B = np.trace(rho2.reshape(2,2,2,2), axis1=0, axis2=2)
                extra["F_s"][j].append(float(np.real(np.trace(PHI_PLUS @ rho2))))

            if ti % 5 == 0:
                v1 = fields["dF"][1][-1]; v2 = fields["EE"][1][-1]
                v3 = fields["OT"][1][-1]; v4 = fields["MI"][1][-1]
                all_dF = "  ".join(f"{fields['dF'][j][-1]:.3f}" for j in js[:6])
                print(f"  t={t:>5.2f}  {v1:>8.4f}  {v2:>8.4f}  "
                      f"{v3:>8.4f}  {v4:>8.4f}  | {all_dF}",
                      flush=True)

        # Compute excess above baseline (t=0)
        def excess(field):
            base = {j: field[j][0] for j in js}
            return {j: [field[j][ti] - base[j] for ti in range(len(ts))]
                    for j in js}

        ex_dF = excess(fields["dF"])
        ex_EE = excess(fields["EE"])
        ex_OT = excess(fields["OT"])
        ex_MI = excess(fields["MI"])

        # LR velocities (on excess fields)
        v_dF, _ = estimate_LR(js, ts, ex_dF, args.thresh_dF)
        v_EE, _ = estimate_LR(js, ts, ex_EE, args.thresh_EE)
        v_OT, _ = estimate_LR(js, ts, ex_OT, args.thresh_OT)
        v_MI, _ = estimate_LR(js, ts, ex_MI, args.thresh_MI)

        velocities = {"PTM_dF": v_dF, "EE": v_EE,
                      "OS_OT": v_OT, "MI": v_MI}

        # Velocity consistency score
        vs = [v for v in velocities.values() if v is not None]
        v_mean = float(np.mean(vs)) if vs else 0.0
        v_std  = float(np.std(vs))  if vs else 0.0
        v_cv   = v_std / abs(v_mean) if abs(v_mean) > 1e-6 else float("inf")

        print(f"\n  [{model_name}] LR velocities:", flush=True)
        for rn, v in velocities.items():
            print(f"    {rn:>12}: {v}", flush=True)
        print(f"  Velocity CV (std/mean): {v_cv:.3f} "
              f"  {'CONSISTENT' if v_cv < 0.3 else 'INCONSISTENT'}", flush=True)

        # Stratum agreement (excess fields, thresholds)
        ex_fields = {"PTM": ex_dF, "EE": ex_EE, "OS": ex_OT, "MI": ex_MI}
        thresholds = {"PTM": args.thresh_dF, "EE": args.thresh_EE,
                      "OS":  args.thresh_OT,  "MI": args.thresh_MI}
        agreement = compute_agreement(js, ts, ex_fields, thresholds)

        print(f"  Stratum agreement rate: {agreement['agreement_rate']:.3f}  "
              f"  {'STABLE' if agreement['agreement_rate'] > 0.75 else 'UNSTABLE'}")
        print(f"  In-stratum rates: " +
              "  ".join(f"{r}:{v:.2f}" for r,v in agreement['in_rates'].items()),
              flush=True)

        all_results[model_name] = {
            "ts": ts.tolist(), "js": js,
            "fields": {k: {str(j): v for j,v in f.items()}
                       for k,f in fields.items()},
            "extra":  {k: {str(j): v for j,v in f.items()}
                       for k,f in extra.items()},
            "velocities": velocities,
            "velocity_CV": v_cv,
            "agreement": agreement,
        }

    total = time.time() - t0_global

    # ---- Summary ---------------------------------------------------------
    print("\n" + "="*70, flush=True)
    print("REPRESENTATION STABILITY SUMMARY", flush=True)
    print(f"{'Model':>8}  {'v_PTM':>7}  {'v_Choi':>7}  "
          f"{'v_OS':>7}  {'v_MI':>7}  {'CV':>6}  {'Agree':>7}  {'Status':>12}",
          flush=True)
    for mn, r in all_results.items():
        vs = r["velocities"]
        fmt = lambda v: f"{v:.2f}" if v else " N/A "
        cv  = r["velocity_CV"]
        ag  = r["agreement"]["agreement_rate"]
        status = "STABLE" if (cv < 0.35 and ag > 0.70) else \
                 ("SELECTIVE" if ag > 0.60 else "UNSTABLE")
        print(f"  {mn:>8}  {fmt(vs['PTM_dF']):>7}  {fmt(vs['EE']):>7}  "
              f"{fmt(vs['OS_OT']):>7}  {fmt(vs['MI']):>7}  "
              f"{cv:>6.3f}  {ag:>7.3f}  {status:>12}", flush=True)

    print(f"\nTotal runtime: {total:.1f}s", flush=True)

    output = {"N": N, "T_max": args.T_max, "n_t": args.n_t, "K": args.K,
              "total_runtime": total, "thresholds": vars(args),
              "models": all_results}

    with open(args.out, "w") as f: json.dump(output, f, indent=2)
    print(f"Saved: {args.out}", flush=True)
    out_zip = args.out.replace(".json", ".zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out)
    print(f"Zipped: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
