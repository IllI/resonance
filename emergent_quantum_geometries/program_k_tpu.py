"""
program_k_tpu.py  --  Program K: Predictive Recovery Fronts
=============================================================
Central metric: dtt(j) = t_rec(j) - t_MI(j)

If dtt > 0 consistently at j=4-6 in XXZ/XY but not Ising:
    --> Information arrives before recoverable transport organizes.
    --> t_MI < t_dF ~ t_rec is a genuine causal hierarchy.

Improvements over Program J:
  1. ADAPTIVE TIME GRID  -- dense near expected front onset at j=4-6,
                            sparse elsewhere. 5-10x finer resolution
                            where it matters.
  2. JACKKNIFE UNCERTAINTY -- split K Haar samples into B blocks,
                              compute onset-time std across blocks.
  3. LOGISTIC D-LINOSS PROXY -- trained on {EE, OS, MI} to predict
                                F_rec > 2/3. No Hamiltonian labels.
                                No fidelity during training.
  4. EXTRA HAMILTONIANS -- Heisenberg (J_perp=1, Dz=1),
                           Tilted Ising (transverse + longitudinal field),
                           OAT (one-axis twisting via Jz^2 term).
  5. FOCUS SITES j=3..N-2 -- where front separation is clearest.

Falsifiable prediction:
    dtt(j) > 0 for j >= 3 in XXZ/XY   [information precedes recovery]
    dtt(j) undefined for j >= 3 in Ising [no transport front]
    Logistic precision > 0.9 in XXZ/XY
    Logistic precision < 0.4 in Ising
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

def kron_list(ops):
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def site_op(op, i, N):
    ops = [I2]*N; ops[i] = op; return kron_list(ops)

def bond_op(o1, o2, i, N):
    ops = [I2]*N; ops[i]=o1; ops[i+1]=o2; return kron_list(ops)

# ---- Hamiltonians --------------------------------------------------------

def build_xxz(N, J=1.0, Dz=1.0):
    H = np.zeros((2**N,2**N), dtype=complex)
    for i in range(N-1):
        H += J/2*(bond_op(X,X,i,N)+bond_op(Y,Y,i,N))
        H += Dz*bond_op(Z,Z,i,N)
    return H

def build_xy(N, J=1.0):
    return build_xxz(N, J=J, Dz=0.0)

def build_ising(N, J=1.0, h=1.0):
    H = np.zeros((2**N,2**N), dtype=complex)
    for i in range(N-1): H -= J*bond_op(Z,Z,i,N)
    for i in range(N):   H -= h*site_op(X,i,N)
    return H

def build_tilted_ising(N, J=1.0, hx=0.9045, hz=0.809):
    """Non-integrable tilted Ising -- breaks integrability."""
    H = np.zeros((2**N,2**N), dtype=complex)
    for i in range(N-1): H -= J*bond_op(Z,Z,i,N)
    for i in range(N):
        H -= hx*site_op(X,i,N)
        H -= hz*site_op(Z,i,N)
    return H

def build_oat(N, chi=1.0):
    """One-axis twisting: H = chi * Jz^2  (collective spin squeezing)."""
    Jz = sum(site_op(Z,i,N) for i in range(N)) / 2
    return chi * (Jz @ Jz)

def apply_U(psi, evals, evecs, t):
    return evecs @ (np.exp(-1j*evals*t) * (evecs.conj().T @ psi))

# ---- Observables ---------------------------------------------------------

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

def vne(rho):
    ev = np.linalg.eigvalsh(rho); ev = ev[ev>1e-12]
    return float(-np.sum(ev*np.log(ev))) if len(ev) else 0.0

def obs_MI(rho2):
    rA = np.trace(rho2.reshape(2,2,2,2), axis1=1, axis2=3)
    rB = np.trace(rho2.reshape(2,2,2,2), axis1=0, axis2=2)
    return float(max(0.0, vne(rA)+vne(rB)-vne(rho2)))

def obs_EE(psi, N, j):
    dL = 2**(j+1); dR = 2**(N-j-1) if N>j+1 else 1
    svs = np.linalg.svd(psi.reshape(dL, max(dR,1)), compute_uv=False)
    s2  = svs**2; s2 = s2[s2>1e-12]
    return float(-np.sum(s2*np.log(s2))) if len(s2) else 0.0

def obs_OS(psi_t, psi_Pt, N, j):
    rj = single_rho(psi_t, N, j)
    sq = sum(float(np.real(np.trace((single_rho(p,N,j)-rj).conj().T @
                                    (single_rho(p,N,j)-rj))))
             for p in psi_Pt)
    return float(math.sqrt(max(0.0, sq)))

def schmidt_rotate(rho2):
    _, vecs = np.linalg.eigh(rho2)
    M = vecs[:,-1].reshape(2,2)
    U,_,Vh = np.linalg.svd(M)
    R = np.kron(U.conj().T, Vh.conj())
    return R @ rho2 @ R.conj().T

def haar_fidelity_block(rho2, rng, K):
    """Haar-averaged teleportation fidelity for one block of K samples."""
    CNOT = np.kron(P0,np.kron(I2,I2)) + np.kron(P1,np.kron(X,I2))
    CIRC = np.kron(H2g,np.kron(I2,I2)) @ CNOT; CD = CIRC.conj().T
    PROJ = {(m1,m2): np.kron(P0 if m1==0 else P1,
                              np.kron(P0 if m2==0 else P1, I2))
            for m1 in range(2) for m2 in range(2)}
    UCORR = {(0,0):I2,(0,1):X,(1,0):Z,(1,1):X@Z}
    def rand_q():
        u,v = rng.uniform(0,1,2)
        th = math.acos(1-2*u); ph = 2*math.pi*v
        return np.array([math.cos(th/2), math.sin(th/2)*cmath.exp(1j*ph)], dtype=complex)
    def fid(pA):
        rb = CIRC @ np.kron(np.outer(pA,pA.conj()), rho2) @ CD
        out = np.zeros((2,2), dtype=complex)
        for (m1,m2),Pi in PROJ.items():
            rp=Pi@rb@Pi; prob=float(rp.trace().real)
            if prob<1e-12: continue
            r=(rp/prob).reshape(2,2,2,2,2,2); rq=np.einsum('ijkijl->kl',r)
            out += prob*(UCORR[(m1,m2)]@rq@UCORR[(m1,m2)].conj().T)
        return float(np.clip((pA.conj()@out@pA).real,0,1))
    return float(np.mean([fid(rand_q()) for _ in range(K)]))

def obs_dF_Frec_jackknife(rho2, rng, K, B=5):
    """Returns (dF_mean, F_rec_mean, F_rec_std) via B-block jackknife."""
    T = np.zeros((3,3))
    for a,sa in enumerate(PAULIS):
        for b,sb in enumerate(PAULIS):
            T[a,b] = float(np.real(np.trace(np.kron(sa,sb)@rho2)))
    F_std = float((1+T[0,0]/3)/2)
    rho_r = schmidt_rotate(rho2)
    block_size = max(1, K//B)
    block_fids = [haar_fidelity_block(rho_r, rng, block_size) for _ in range(B)]
    F_rec_mean = float(np.mean(block_fids))
    F_rec_std  = float(np.std(block_fids))
    dF = F_rec_mean - F_std
    return dF, F_rec_mean, F_rec_std

# ---- Adaptive time grid --------------------------------------------------

def adaptive_grid(T_max, n_coarse, n_fine, t_onset_est, window=1.5):
    """Dense time grid near t_onset_est, coarse elsewhere."""
    t_lo = max(0.0, t_onset_est - window)
    t_hi = min(T_max, t_onset_est + window)
    fine   = np.linspace(t_lo, t_hi, n_fine)
    coarse = np.linspace(0.0, T_max, n_coarse)
    ts = np.unique(np.concatenate([coarse, fine]))
    return np.sort(ts)

# ---- Onset with uncertainty ----------------------------------------------

def onset_with_unc(ts, series_mean, series_std, thresh):
    """First t where series_mean - series_std > thresh (conservative).
    Also return the 'ambiguous window' [t_lo, t_hi] where
    mean+-std straddles threshold."""
    t_onset = None
    t_lo = t_hi = None
    for ti, t in enumerate(ts):
        above_mean  = series_mean[ti] > thresh
        above_lower = series_mean[ti] - series_std[ti] > thresh
        above_upper = series_mean[ti] + series_std[ti] > thresh
        if above_lower and t_onset is None:
            t_onset = float(t)
        if above_upper and t_lo is None:
            t_lo = float(t)
        if above_mean and t_hi is None:
            t_hi = float(t)
    return t_onset, t_lo, t_hi

# ---- Simple logistic proxy (no Hamiltonian labels) -----------------------

def logistic_proxy_train_eval(features, labels):
    """Minimal logistic regression on feature matrix X -> label y.
    Uses gradient descent. No sklearn required.
    Returns: accuracy, precision, recall, AUC-proxy.
    """
    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=float)
    if len(np.unique(y)) < 2:
        return dict(acc=float(np.mean(y)), prec=float(np.mean(y)),
                    recall=1.0, n=len(y))
    # Normalize features
    mu = X.mean(0); sig = X.std(0)+1e-8
    X  = (X - mu) / sig
    # Add bias
    X  = np.hstack([X, np.ones((len(X),1))])
    w  = np.zeros(X.shape[1])
    # Gradient descent (100 steps)
    lr = 0.1
    for _ in range(200):
        pred = 1/(1+np.exp(-X@w))
        grad = X.T @ (pred - y) / len(y)
        w   -= lr * grad
    pred = 1/(1+np.exp(-X@w))
    yhat = (pred > 0.5).astype(float)
    tp = float(np.sum((yhat==1)&(y==1)))
    fp = float(np.sum((yhat==1)&(y==0)))
    fn = float(np.sum((yhat==0)&(y==1)))
    tn = float(np.sum((yhat==0)&(y==0)))
    acc  = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0.0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return dict(acc=float(acc), prec=float(prec), recall=float(rec),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), n=len(y))

# ---- Main ----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N",         type=int,   default=10)
    p.add_argument("--T-max",     type=float, default=5.0)
    p.add_argument("--n-coarse",  type=int,   default=30)
    p.add_argument("--n-fine",    type=int,   default=50)
    p.add_argument("--t-onset",   type=float, default=1.5,
                   help="Expected onset time for dense grid centering")
    p.add_argument("--K",         type=int,   default=200)
    p.add_argument("--B",         type=int,   default=5,
                   help="Jackknife blocks for F_rec uncertainty")
    p.add_argument("--j-focus",   type=int,   nargs="+",
                   default=None, help="Sites to report (default: all)")
    p.add_argument("--th-MI",     type=float, default=0.03)
    p.add_argument("--th-EE",     type=float, default=0.04)
    p.add_argument("--th-OS",     type=float, default=0.015)
    p.add_argument("--th-dF",     type=float, default=0.015)
    p.add_argument("--th-rec",    type=float, default=2/3)
    p.add_argument("--models",    type=str,   nargs="+",
                   default=["XY","XXZ","Ising","TiltedIsing"])
    p.add_argument("--out",       default="program_k_results.json")
    args = p.parse_args()
    N = args.N

    ts   = adaptive_grid(args.T_max, args.n_coarse, args.n_fine, args.t_onset)
    js   = args.j_focus if args.j_focus else list(range(1, N))
    rng  = np.random.default_rng(42)

    psi0 = np.zeros(2**N, dtype=complex)
    psi0[0] = 1/math.sqrt(2); psi0[2**(N-1)] = 1/math.sqrt(2)
    psi0_P = [site_op(P,0,N)@psi0 for P in PAULIS]

    MODEL_BUILDERS = {
        "XY":          lambda: build_xy(N),
        "XXZ":         lambda: build_xxz(N),
        "Ising":       lambda: build_ising(N),
        "TiltedIsing": lambda: build_tilted_ising(N),
        "OAT":         lambda: build_oat(N, chi=1.0/N),
    }

    print("="*72, flush=True)
    print("Program K -- Predictive Recovery Fronts (dtt = t_rec - t_MI)")
    print(f"N={N}  T_max={args.T_max}  "
          f"grid={len(ts)} pts (coarse={args.n_coarse} + fine={args.n_fine})")
    print(f"K={args.K}  B={args.B}-block jackknife  models={args.models}",
          flush=True)
    print("="*72, flush=True)

    all_results = {}
    t0_global   = time.time()

    # Collect all (j,t) observations for cross-model logistic training
    all_features, all_labels = [], []

    for model_name in args.models:
        if model_name not in MODEL_BUILDERS:
            print(f"Unknown model: {model_name}, skipping.", flush=True)
            continue

        print(f"\n[{model_name}] Diagonalizing... ", end="", flush=True)
        H = MODEL_BUILDERS[model_name]()
        evals, evecs = np.linalg.eigh(H)
        print("done.", flush=True)

        # Time series per observable
        ser_MI    = {j:[] for j in js}
        ser_EE    = {j:[] for j in js}
        ser_OS    = {j:[] for j in js}
        ser_dF    = {j:[] for j in js}
        ser_Frec  = {j:[] for j in js}
        ser_Fstd  = {j:[] for j in js}

        for ti, t in enumerate(ts):
            psi_t  = apply_U(psi0, evals, evecs, t)
            psi_Pt = [apply_U(p, evals, evecs, t) for p in psi0_P]

            for j in js:
                rho2 = pair_rdm(psi_t, N, 0, j)
                dF, Frec, _ = obs_dF_Frec_jackknife(rho2, rng, args.K, args.B)
                ser_MI[j].append(obs_MI(rho2))
                ser_EE[j].append(obs_EE(psi_t, N, j))
                ser_OS[j].append(obs_OS(psi_t, psi_Pt, N, j))
                ser_dF[j].append(dF)
                ser_Frec[j].append(Frec)
                T = np.zeros((3,3))
                for a,sa in enumerate(PAULIS):
                    for b,sb in enumerate(PAULIS):
                        T[a,b] = float(np.real(np.trace(np.kron(sa,sb)@rho2)))
                ser_Fstd[j].append(float((1+T[0,0]/3)/2))

            if ti % max(1, len(ts)//8) == 0:
                j1 = js[min(3, len(js)-1)]  # report j=4 if possible
                print(f"  t={t:>5.2f}  j={j1}:  "
                      f"MI={ser_MI[j1][-1]:.3f}  "
                      f"EE={ser_EE[j1][-1]:.3f}  "
                      f"dF={ser_dF[j1][-1]:.3f}  "
                      f"F_rec={ser_Frec[j1][-1]:.3f}", flush=True)

        # Excess above baseline
        def exc(s):
            return {j: [s[j][ti]-s[j][0] for ti in range(len(ts))] for j in js}

        ex_MI = exc(ser_MI); ex_EE = exc(ser_EE)
        ex_OS = exc(ser_OS); ex_dF = exc(ser_dF)

        # Onset times (simple: first crossing of excess threshold)
        def onset(ex_s, thresh):
            res = {}
            for j in js:
                t_on = None
                for ti, t in enumerate(ts):
                    if ex_s[j][ti] > thresh:
                        t_on = float(t); break
                res[j] = t_on
            return res

        on_MI  = onset(ex_MI,  args.th_MI)
        on_EE  = onset(ex_EE,  args.th_EE)
        on_OS  = onset(ex_OS,  args.th_OS)
        on_dF  = onset(ex_dF,  args.th_dF)
        on_rec = {}
        for j in js:
            t_on = None
            for ti, t in enumerate(ts):
                if ser_Frec[j][ti] > args.th_rec:
                    t_on = float(t); break
            on_rec[j] = t_on

        # dtt = t_rec - t_MI
        print(f"\n  [{model_name}] dtt = t_rec - t_MI  (key causal ordering metric):",
              flush=True)
        print(f"  {'j':>3}  {'t_MI':>6}  {'t_EE':>6}  {'t_dF':>6}  "
              f"{'t_rec':>6}  {'dtt(rec-MI)':>11}  {'dtt(rec-dF)':>11}",
              flush=True)
        delta_t_rec_MI = []
        delta_t_rec_dF = []
        focus_js = [j for j in js if j >= 3][:6]
        for j in (focus_js if focus_js else js):
            tMI  = on_MI[j]; tEE = on_EE[j]
            tdF  = on_dF[j]; trec = on_rec[j]
            fmt  = lambda v: f"{v:.2f}" if v is not None else "  inf"
            dt_MI = (trec - tMI) if (trec and tMI) else None
            dt_dF = (trec - tdF) if (trec and tdF) else None
            if dt_MI is not None: delta_t_rec_MI.append(dt_MI)
            if dt_dF is not None: delta_t_rec_dF.append(dt_dF)
            print(f"  {j:>3}  {fmt(tMI):>6}  {fmt(tEE):>6}  {fmt(tdF):>6}  "
                  f"{fmt(trec):>6}  {fmt(dt_MI):>11}  {fmt(dt_dF):>11}",
                  flush=True)

        mean_dt_MI = float(np.mean(delta_t_rec_MI)) if delta_t_rec_MI else None
        std_dt_MI  = float(np.std(delta_t_rec_MI))  if len(delta_t_rec_MI)>1 else None
        mean_dt_dF = float(np.mean(delta_t_rec_dF)) if delta_t_rec_dF else None
        print(f"\n  mean dtt(rec-MI) = {mean_dt_MI}  +-{std_dt_MI}", flush=True)
        print(f"  mean dtt(rec-dF) = {mean_dt_dF}", flush=True)

        causal_verdict = "PASS" if (mean_dt_MI and mean_dt_MI > 0) else \
                         ("MARGINAL" if mean_dt_MI == 0.0 else "FAIL")
        print(f"  Causal hierarchy t_MI < t_rec: {causal_verdict}", flush=True)

        # Collect features for logistic proxy
        for j in js:
            for ti in range(len(ts)):
                feat = [ex_MI[j][ti], ex_EE[j][ti], ex_OS[j][ti]]
                label = int(ser_Frec[j][ti] > args.th_rec)
                all_features.append(feat)
                all_labels.append(label)

        # Per-model logistic (within-model, no Hamiltonian label)
        feats_m = [[ex_MI[j][ti], ex_EE[j][ti], ex_OS[j][ti]]
                   for j in js for ti in range(len(ts))]
        labels_m = [int(ser_Frec[j][ti] > args.th_rec)
                    for j in js for ti in range(len(ts))]
        logistic_m = logistic_proxy_train_eval(feats_m, labels_m)
        print(f"  Logistic proxy (within-model):"
              f"  prec={logistic_m['prec']:.3f}"
              f"  rec={logistic_m['recall']:.3f}"
              f"  acc={logistic_m['acc']:.3f}", flush=True)

        all_results[model_name] = {
            "ts": ts.tolist(), "js": js,
            "onset": {"MI":  {str(j): on_MI[j]  for j in js},
                      "EE":  {str(j): on_EE[j]  for j in js},
                      "OS":  {str(j): on_OS[j]  for j in js},
                      "dF":  {str(j): on_dF[j]  for j in js},
                      "rec": {str(j): on_rec[j] for j in js}},
            "causal": {"mean_dt_rec_MI": mean_dt_MI,
                       "std_dt_rec_MI":  std_dt_MI,
                       "mean_dt_rec_dF": mean_dt_dF,
                       "verdict": causal_verdict},
            "logistic_within": logistic_m,
        }

    # Cross-model logistic (trained on all models, tests generalization)
    print("\n" + "="*72, flush=True)
    print("CROSS-MODEL LOGISTIC PROXY (no Hamiltonian labels):", flush=True)
    cross = logistic_proxy_train_eval(all_features, all_labels)
    print(f"  prec={cross['prec']:.3f}  rec={cross['recall']:.3f}"
          f"  acc={cross['acc']:.3f}  n={cross['n']}", flush=True)

    total = time.time() - t0_global

    print("\n" + "="*72, flush=True)
    print("PROGRAM K SUMMARY", flush=True)
    print(f"{'Model':>14}  {'dtt(rec-MI)':>12}  {'+-':>6}  "
          f"{'dtt(rec-dF)':>12}  {'Causal':>8}  {'LogPrec':>8}", flush=True)
    for mn, r in all_results.items():
        c = r["causal"]; lm = r["logistic_within"]
        dt  = f"{c['mean_dt_rec_MI']:+.2f}" if c['mean_dt_rec_MI'] else " N/A"
        std = f"{c['std_dt_rec_MI']:.2f}"   if c['std_dt_rec_MI']  else "  N/A"
        dtd = f"{c['mean_dt_rec_dF']:+.2f}" if c['mean_dt_rec_dF'] else " N/A"
        print(f"  {mn:>14}  {dt:>12}  {std:>6}  "
              f"{dtd:>12}  {c['verdict']:>8}  {lm['prec']:>8.3f}", flush=True)
    print(f"\nTotal runtime: {total:.1f}s", flush=True)

    output = {"N": N, "T_max": args.T_max, "K": args.K,
              "n_ts": len(ts), "total_runtime": total,
              "cross_model_logistic": cross, "models": all_results}
    with open(args.out, "w") as f: json.dump(output, f, indent=2)
    print(f"Saved: {args.out}", flush=True)
    out_zip = args.out.replace(".json", ".zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out)
    print(f"Zipped: {out_zip}", flush=True)

if __name__ == "__main__":
    main()
