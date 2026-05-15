"""
program_j_tpu.py  --  Program J: Causal Ordering of Quantum Transport
======================================================================
Central question:
  "Does recoverable quantum transport emerge LATER than raw information
   propagation, and can it be predicted without fidelity or Hamiltonian labels?"

Six observables measured per (j, t):
  MI(j,t)      -- mutual information I(0:j)                [info arrival]
  C(j,t)       -- pairwise concurrence                     [entanglement]
  EE(j,t)      -- bipartite entropy S(0..j | rest)         [full-state]
  OS(j,t)      -- operator spreading OT from site 0        [Heisenberg]
  dF(j,t)      -- PTM transport enhancement F_sch-F_std    [channel]
  F_rec(j,t)   -- F_sch = operational recovery fidelity   [verified]

CAUSAL ORDERING TEST:
  For each j: find onset time t_x = first t where observable x > threshold.
  Report: t_rec(j) - t_MI(j), t_rec(j) - t_EE(j), etc.
  Prediction (XXZ):  t_MI <= t_EE <= t_C < t_dF ~ t_rec
  Prediction (Ising): t_MI < t_C < t_EE  but t_dF = inf, t_rec = inf

BLIND STRATUM PREDICTION:
  D-LinOSS proxy -- uses ONLY {EE, OS, MI}, no dF, no F_rec, no Hamiltonian.
  Stratum_blind(j,t) = (excess_EE > th_EE) AND (excess_MI > th_MI)
  Then verify: P(F_rec > 2/3 | Stratum_blind=1) vs P(F_rec > 2/3 | Stratum_blind=0)
  A large difference = blind prediction is informative.
  D-LinOSS never saw fidelity during inference.

FALSIFIABLE PREDICTIONS:
  1. t_rec(j) > t_MI(j) for all j in XXZ  [recovery lags information]
  2. t_dF(j) ~ t_rec(j) for all j in XXZ  [dF predicts recovery onset]
  3. P(F_rec>2/3 | blind=1) >> P(F_rec>2/3 | blind=0) in XXZ
  4. None of the above hold for Ising (negative control)
"""

import json, time, math, cmath, zipfile, argparse
import numpy as np

try:
    import jax
    print(f"[DEVICE] {jax.devices()}  backend={jax.default_backend()}", flush=True)
except ImportError:
    print("[DEVICE] JAX not found -- numpy fallback", flush=True)

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

def build_xxz(N, J_perp=1.0, Dz=0.0):
    H = np.zeros((2**N,2**N), dtype=complex)
    for i in range(N-1):
        H += J_perp/2*(bond_op(X,X,i,N)+bond_op(Y,Y,i,N))
        H += Dz*bond_op(Z,Z,i,N)
    return H

def build_ising(N, J=1.0, h=1.0):
    H = np.zeros((2**N,2**N), dtype=complex)
    for i in range(N-1): H -= J*bond_op(Z,Z,i,N)
    for i in range(N):   H -= h*site_op(X,i,N)
    return H

def apply_U(psi, evals, evecs, t):
    return evecs @ (np.exp(-1j*evals*t) * (evecs.conj().T @ psi))

def pair_rdm(psi, N, i, j):
    perm = [i,j]+[k for k in range(N) if k not in (i,j)]
    mat  = psi.reshape([2]*N).transpose(perm).reshape(4, 2**(N-2) if N>2 else 1)
    rho2 = mat @ mat.conj().T
    return rho2 / float(rho2.trace().real)

def single_rho(psi, N, j):
    perm = [j]+[k for k in range(N) if k!=j]
    mat  = psi.reshape([2]*N).transpose(perm).reshape(2, 2**(N-1))
    rho  = mat @ mat.conj().T
    return rho / float(rho.trace().real)

def von_neumann_entropy(rho):
    ev = np.linalg.eigvalsh(rho); ev = ev[ev>1e-12]
    return float(-np.sum(ev*np.log(ev))) if len(ev) else 0.0

# --- Six observables -------------------------------------------------------

def obs_MI(rho2):
    rA = np.trace(rho2.reshape(2,2,2,2),axis1=1,axis2=3)
    rB = np.trace(rho2.reshape(2,2,2,2),axis1=0,axis2=2)
    return max(0.0, von_neumann_entropy(rA)+von_neumann_entropy(rB)
               -von_neumann_entropy(rho2))

def obs_C(rho2):
    """Concurrence via Wootters formula."""
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    rho_tilde = np.kron(sy,sy) @ rho2.conj() @ np.kron(sy,sy)
    M  = rho2 @ rho_tilde
    ev = np.sqrt(np.clip(np.linalg.eigvals(M).real, 0, None))
    ev = np.sort(ev)[::-1]
    return float(max(0.0, ev[0]-ev[1]-ev[2]-ev[3]))

def obs_EE(psi, N, j):
    dim_L = 2**(j+1); dim_R = 2**(N-j-1) if N>j+1 else 1
    mat = psi.reshape(dim_L, max(dim_R,1))
    svs = np.linalg.svd(mat, compute_uv=False)
    s2  = svs**2; s2 = s2[s2>1e-12]
    return float(-np.sum(s2*np.log(s2))) if len(s2) else 0.0

def obs_OS(psi_t, psi_Pt, N, j):
    rho_j = single_rho(psi_t, N, j)
    OT_sq = 0.0
    for pP in psi_Pt:
        d = single_rho(pP, N, j) - rho_j
        OT_sq += float(np.real(np.trace(d.conj().T @ d)))
    return float(math.sqrt(max(0.0, OT_sq)))

def schmidt_rotate(rho2):
    _, vecs = np.linalg.eigh(rho2)
    M = vecs[:,-1].reshape(2,2)
    U,_,Vh = np.linalg.svd(M)
    R = np.kron(U.conj().T, Vh.conj())
    return R @ rho2 @ R.conj().T

def avg_fid_haar(rho2, K, rng):
    CNOT = np.kron(P0,np.kron(I2,I2)) + np.kron(P1,np.kron(X,I2))
    CIRC = np.kron(H2g,np.kron(I2,I2)) @ CNOT; CD = CIRC.conj().T
    PROJ = {(m1,m2): np.kron(P0 if m1==0 else P1,
                              np.kron(P0 if m2==0 else P1, I2))
            for m1 in range(2) for m2 in range(2)}
    UCORR = {(0,0):I2,(0,1):X,(1,0):Z,(1,1):X@Z}
    def rand_q():
        u,v = rng.uniform(0,1,2)
        th=math.acos(1-2*u); ph=2*math.pi*v
        return np.array([math.cos(th/2),math.sin(th/2)*cmath.exp(1j*ph)],dtype=complex)
    def fid(pA):
        rt=np.kron(np.outer(pA,pA.conj()),rho2); rb=CIRC@rt@CD
        out=np.zeros((2,2),dtype=complex)
        for (m1,m2),Pi in PROJ.items():
            rp=Pi@rb@Pi; prob=float(rp.trace().real)
            if prob<1e-12: continue
            r=(rp/prob).reshape(2,2,2,2,2,2); rq=np.einsum('ijkijl->kl',r)
            out+=prob*(UCORR[(m1,m2)]@rq@UCORR[(m1,m2)].conj().T)
        return float(np.clip((pA.conj()@out@pA).real,0,1))
    return float(np.mean([fid(rand_q()) for _ in range(K)]))

def obs_dF_Frec(rho2, K, rng):
    T = np.zeros((3,3))
    for a,sa in enumerate(PAULIS):
        for b,sb in enumerate(PAULIS):
            T[a,b] = float(np.real(np.trace(np.kron(sa,sb)@rho2)))
    F_std = float((1+T[0,0]/3)/2)
    rho_r = schmidt_rotate(rho2)
    F_sch = avg_fid_haar(rho_r, K, rng)
    return float(F_sch-F_std), float(F_sch)

# --- Onset time detector ---------------------------------------------------

def onset_time(ts, series, thresh):
    """First time series[ti] > thresh. Returns None if never."""
    for ti,t in enumerate(ts):
        if series[ti] > thresh:
            return float(t)
    return None

# --- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N",         type=int,   default=10)
    p.add_argument("--T-max",     type=float, default=5.0)
    p.add_argument("--n-t",       type=int,   default=40)
    p.add_argument("--K",         type=int,   default=200)
    p.add_argument("--th-MI",     type=float, default=0.03)
    p.add_argument("--th-C",      type=float, default=0.005)
    p.add_argument("--th-EE",     type=float, default=0.04)
    p.add_argument("--th-OS",     type=float, default=0.015)
    p.add_argument("--th-dF",     type=float, default=0.015)
    p.add_argument("--th-rec",    type=float, default=2/3)
    p.add_argument("--out",       default="program_j_results.json")
    args = p.parse_args()
    N = args.N

    ts   = np.linspace(0.0, args.T_max, args.n_t)
    js   = list(range(1, N))
    rng  = np.random.default_rng(42)

    psi0   = np.zeros(2**N, dtype=complex)
    psi0[0]          = 1/math.sqrt(2)
    psi0[2**(N-1)]   = 1/math.sqrt(2)
    psi0_P = [site_op(P,0,N)@psi0 for P in PAULIS]

    MODELS = [
        ("XY",    build_xxz(N, J_perp=1.0, Dz=0.0)),
        ("XXZ",   build_xxz(N, J_perp=1.0, Dz=1.0)),
        ("Ising", build_ising(N, J=1.0, h=1.0)),
    ]

    print("="*72, flush=True)
    print("Program J -- Causal Ordering of Quantum Transport")
    print(f"N={N}  T_max={args.T_max}  n_t={args.n_t}  K={args.K}")
    print("Obs: MI | C | EE | OS | dF | F_rec", flush=True)
    print("="*72, flush=True)

    all_results = {}
    t0 = time.time()

    for model_name, H in MODELS:
        print(f"\n[{model_name}] Diagonalizing... ", end="", flush=True)
        evals, evecs = np.linalg.eigh(H)
        print("done.", flush=True)

        obs_names = ["MI","C","EE","OS","dF","F_rec"]
        series = {o: {j: [] for j in js} for o in obs_names}

        for ti, t in enumerate(ts):
            psi_t  = apply_U(psi0, evals, evecs, t)
            psi_Pt = [apply_U(p, evals, evecs, t) for p in psi0_P]

            for j in js:
                rho2 = pair_rdm(psi_t, N, 0, j)
                dF, Frec = obs_dF_Frec(rho2, args.K, rng)
                series["MI"][j].append(obs_MI(rho2))
                series["C"][j].append(obs_C(rho2))
                series["EE"][j].append(obs_EE(psi_t, N, j))
                series["OS"][j].append(obs_OS(psi_t, psi_Pt, N, j))
                series["dF"][j].append(dF)
                series["F_rec"][j].append(Frec)

            if ti % 5 == 0:
                j1 = js[0]
                print(f"  t={t:>5.2f}  "
                      f"MI={series['MI'][j1][-1]:.3f}  "
                      f"EE={series['EE'][j1][-1]:.3f}  "
                      f"OS={series['OS'][j1][-1]:.3f}  "
                      f"dF={series['dF'][j1][-1]:.3f}  "
                      f"F_rec={series['F_rec'][j1][-1]:.3f}",
                      flush=True)

        # --- Excess above t=0 baseline for threshold detection
        def excess(s):
            return {j: [s[j][ti]-s[j][0] for ti in range(len(ts))] for j in js}

        ex = {o: excess(series[o]) for o in ["MI","C","EE","OS","dF"]}
        thresholds = {"MI":args.th_MI,"C":args.th_C,"EE":args.th_EE,
                      "OS":args.th_OS,"dF":args.th_dF}

        # --- Causal ordering: onset times per site
        onset = {o: {} for o in obs_names}
        for j in js:
            for o in ["MI","C","EE","OS","dF"]:
                onset[o][j] = onset_time(ts, ex[o][j], thresholds[o])
            # F_rec: absolute threshold (not excess)
            onset["F_rec"][j] = onset_time(ts, series["F_rec"][j], args.th_rec)

        print(f"\n  [{model_name}] Onset times (None = never reached):", flush=True)
        print(f"  {'j':>3}  {'t_MI':>6}  {'t_C':>6}  {'t_EE':>6}  "
              f"{'t_OS':>6}  {'t_dF':>6}  {'t_rec':>6}  "
              f"{'rec-MI':>7}  {'rec-dF':>7}", flush=True)
        causal_delays = []  # t_rec - t_MI per site
        dF_rec_diffs  = []  # t_rec - t_dF per site
        for j in js[:8]:    # print up to 8 sites
            tMI  = onset["MI"][j];  tC   = onset["C"][j]
            tEE  = onset["EE"][j];  tOS  = onset["OS"][j]
            tdF  = onset["dF"][j];  trec = onset["F_rec"][j]
            fmt  = lambda v: f"{v:.2f}" if v is not None else "  inf"
            lag  = (trec - tMI) if (trec and tMI) else None
            ddF  = (trec - tdF) if (trec and tdF) else None
            if lag is not None:  causal_delays.append(lag)
            if ddF is not None:  dF_rec_diffs.append(ddF)
            print(f"  {j:>3}  {fmt(tMI):>6}  {fmt(tC):>6}  {fmt(tEE):>6}  "
                  f"{fmt(tOS):>6}  {fmt(tdF):>6}  {fmt(trec):>6}  "
                  f"{fmt(lag):>7}  {fmt(ddF):>7}", flush=True)

        mean_lag = float(np.mean(causal_delays)) if causal_delays else None
        mean_ddF = float(np.mean(dF_rec_diffs))  if dF_rec_diffs  else None
        print(f"\n  Mean(t_rec - t_MI) = {mean_lag}", flush=True)
        print(f"  Mean(t_rec - t_dF) = {mean_ddF}", flush=True)

        # --- Blind stratum prediction (no dF, no F_rec, no Hamiltonian)
        # Blind features: EE and MI only
        blind_in  = {"F_rec_above": 0, "F_rec_below": 0,
                     "F_rec_above_count": 0, "total": 0}
        tp=tn=fp=fn=0
        for j in js:
            for ti in range(len(ts)):
                blind_flag = (ex["EE"][j][ti] > args.th_EE and
                              ex["MI"][j][ti] > args.th_MI)
                rec_flag   = series["F_rec"][j][ti] > args.th_rec
                if blind_flag and rec_flag:     tp += 1
                elif blind_flag and not rec_flag: fp += 1
                elif not blind_flag and rec_flag: fn += 1
                else:                             tn += 1
        total = tp+tn+fp+fn
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0  # P(rec>2/3 | blind=1)
        base = (tp+fn)/total if total>0 else 0.0  # baseline P(rec>2/3)
        print(f"\n  [{model_name}] Blind stratum prediction "
              f"(EE+MI only, no dF/F_rec/Hamiltonian):", flush=True)
        print(f"    Precision P(F_rec>2/3 | blind=1): {prec:.3f}", flush=True)
        print(f"    Baseline  P(F_rec>2/3):            {base:.3f}", flush=True)
        lift = prec/base if base > 0 else float("inf")
        print(f"    Lift (precision/baseline):          {lift:.2f}x", flush=True)

        all_results[model_name] = {
            "ts": ts.tolist(), "js": js,
            "series":  {o: {str(j): series[o][j] for j in js}
                        for o in obs_names},
            "onset":   {o: {str(j): onset[o][j]  for j in js}
                        for o in obs_names},
            "causal": {"mean_t_rec_minus_MI":  mean_lag,
                       "mean_t_rec_minus_dF":  mean_ddF,
                       "causal_delays": causal_delays,
                       "dF_rec_diffs":  dF_rec_diffs},
            "blind_prediction": {"tp":tp,"tn":tn,"fp":fp,"fn":fn,
                                 "precision":prec,"baseline":base,"lift":lift},
        }

    total_t = time.time() - t0

    # --- Summary -------------------------------------------------------------
    print("\n" + "="*72, flush=True)
    print("PROGRAM J SUMMARY -- Causal Ordering & Blind Prediction", flush=True)
    print(f"{'Model':>8}  {'t_rec>t_MI':>12}  {'t_rec~t_dF':>12}  "
          f"{'BlindPrec':>10}  {'Base':>6}  {'Lift':>6}", flush=True)
    for mn, r in all_results.items():
        lag  = r["causal"]["mean_t_rec_minus_MI"]
        ddF  = r["causal"]["mean_t_rec_minus_dF"]
        prec = r["blind_prediction"]["precision"]
        base = r["blind_prediction"]["baseline"]
        lift = r["blind_prediction"]["lift"]
        lag_s  = f"{lag:+.2f}" if lag is not None else "N/A"
        ddF_s  = f"{ddF:+.2f}" if ddF is not None else "N/A"
        print(f"  {mn:>8}  {lag_s:>12}  {ddF_s:>12}  "
              f"{prec:>10.3f}  {base:>6.3f}  {lift:>6.2f}x", flush=True)

    print(f"\nTotal runtime: {total_t:.1f}s", flush=True)
    print("\nFalsification checklist:", flush=True)
    for mn, r in all_results.items():
        lag = r["causal"]["mean_t_rec_minus_MI"]
        ddF = r["causal"]["mean_t_rec_minus_dF"]
        lift = r["blind_prediction"]["lift"]
        pred1 = "PASS" if (lag and lag > 0) else "FAIL"
        pred2 = "PASS" if (ddF and abs(ddF) < 1.0) else ("N/A" if ddF is None else "FAIL")
        pred3 = "PASS" if lift > 1.5 else "FAIL"
        print(f"  {mn}: t_rec>t_MI={pred1}  t_rec~t_dF={pred2}  blind_lift={pred3}({lift:.2f}x)")

    output = {"N": N, "T_max": args.T_max, "n_t": args.n_t, "K": args.K,
              "total_runtime": total_t, "models": all_results}
    with open(args.out,"w") as f: json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}", flush=True)
    out_zip = args.out.replace(".json",".zip")
    with zipfile.ZipFile(out_zip,"w",zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out)
    print(f"Zipped: {out_zip}", flush=True)

if __name__ == "__main__":
    main()
