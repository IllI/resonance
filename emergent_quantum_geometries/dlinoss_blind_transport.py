"""
dlinoss_blind_transport.py — Blind Operator-Geometry Discovery (TPU Program E)
================================================================================
Tests whether D-LinOSS recovers teleportation-capable strata WITHOUT labels.

D-LinOSS sees ONLY: T_ij, singular_values, PTM_eigenvalues, rank,
                    spectral_entropy, dT_dchi, singularity_proximity
D-LinOSS NEVER sees: F_avg, V_Q, kappa_Q, teleportation labels

Success criteria:
  1. A distinct stratum emerges from blind embedding
  2. P(F_avg > 2/3 | TRANSPORT stratum) >> P(F_avg > 2/3)   [post-hoc]
  3. Null models (separable states, scrambled) fail to produce same structure

Usage:
  python dlinoss_blind_transport.py --scale small   # ~600 pts, ~1 min CPU
  python dlinoss_blind_transport.py --scale medium  # ~3k pts
  python dlinoss_blind_transport.py --scale full    # ~15k pts, TPU
"""

import argparse, json, os, time
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from ptm_features import (PTMFeatures, PTMFeatureVector, rho2_exact,
                           compute_T_matrix, f_max_analytic)
from dlinoss_geometry import spectral_entropy

# ── Blind feature set: no F_avg, no V_Q, no labels ───────────────────────────
ALL_NAMES = PTMFeatureVector.feature_names()
BLIND_FEATURES = set([
    "T_xx","T_xy","T_xz","T_yx","T_yy","T_yz","T_zx","T_zy","T_zz",
    "sv_1","sv_2","sv_3",
    "lam_1","lam_2","lam_3","lam_4",
    "T_rank",
    "dT_xx","dT_yy","dT_zz","dT_yz","dT_zy",
    "singularity_proximity",
])

# Column index in X_blind (order = ALL_NAMES filtered to BLIND_FEATURES)
_MASK = np.array([i for i, n in enumerate(ALL_NAMES) if n in BLIND_FEATURES])
_BCOL = {ALL_NAMES[orig]: blind for blind, orig in enumerate(_MASK)}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rand_so3(rng):
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _t_to_rho(T):
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    rho = np.eye(4, dtype=complex) / 4
    for i, si in enumerate([sx, sy, sz]):
        for j, sj in enumerate([sx, sy, sz]):
            rho += T[i, j] * np.kron(si, sj) / 4
    return rho


def _build_row(T, svs, eigs, chi_t, N, Gamma, dT, fm):
    """Build 34-dim row matching PTMFeatureVector layout."""
    T_frob = float(np.linalg.norm(T, "fro"))
    T0_frob = 2.0  # reference at chi_t≈0
    sing_prox = float(max(0.0, min(1.0, 1.0 - T_frob / T0_frob)))
    rank = int((svs > 1e-3).sum())
    F_avg = (2 * fm + 1) / 3
    return np.array([
        chi_t, N, Gamma,
        T[0,0], T[0,1], T[0,2], T[1,0], T[1,1], T[1,2],
        T[2,0], T[2,1], T[2,2],
        svs[0], svs[1], svs[2],
        dT[0,0], dT[0,1], dT[0,2], dT[1,0], dT[1,1], dT[1,2],
        dT[2,0], dT[2,1], dT[2,2],
        eigs[0], eigs[1], eigs[2], eigs[3],
        0.0, 0.0, float(rank),
        fm, F_avg, sing_prox,
    ])


def _ising_boundary(L, J, h):
    dim = 2 ** L
    H = np.zeros((dim, dim))
    for i in range(L - 1):
        for s in range(dim):
            zi  = 1 if (s >> (L-1-i))   & 1 == 0 else -1
            zi1 = 1 if (s >> (L-1-i-1)) & 1 == 0 else -1
            H[s, s] -= J * zi * zi1
    for i in range(L):
        for s in range(dim):
            H[s, s ^ (1 << (L-1-i))] -= h
    vals, vecs = eigh(H, subset_by_index=[0, 0])
    psi = vecs[:, 0]
    psi_mat = psi.reshape(4, dim // 4)
    rho = psi_mat @ psi_mat.conj().T
    return rho / np.trace(rho)


# ── Data generators ───────────────────────────────────────────────────────────
def gen_oat(n_chi, n_gam, N_vals, seed=0):
    rows, meta = [], []
    chi_grid = np.linspace(0.02, np.pi, n_chi)
    gam_grid = np.linspace(0.0, 0.35, n_gam)
    for N in N_vals:
        feat = PTMFeatures(N=N, fast=True)
        for Gamma in gam_grid:
            for chi_t in chi_grid:
                fv = feat.compute(chi_t, Gamma)
                rows.append(fv.to_array())
                meta.append({"source": "oat", "F_avg": fv.F_avg})
    return np.array(rows), meta


def gen_null_spectrum_preserving(n, seed=10):
    """Null A: keep T singular values, randomize orientation."""
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    feat = PTMFeatures(N=4, fast=True)
    for chi_t in np.linspace(0.02, np.pi, n):
        fv = feat.compute(chi_t, 0.0)
        T = np.array([[fv.T_xx, fv.T_xy, fv.T_xz],
                      [fv.T_yx, fv.T_yy, fv.T_yz],
                      [fv.T_zx, fv.T_zy, fv.T_zz]])
        _, s, _ = np.linalg.svd(T)
        T_null = _rand_so3(rng) @ np.diag(s) @ _rand_so3(rng).T
        svs = np.sort(np.linalg.svd(T_null, compute_uv=False))[::-1]
        eigs = np.sort(np.linalg.eigvalsh(_t_to_rho(T_null)))[::-1]
        fm = f_max_analytic(T_null)
        rows.append(_build_row(T_null, svs, eigs, chi_t, 4, 0.0,
                               np.zeros((3,3)), fm))
        meta.append({"source": "null_spec", "F_avg": (2*fm+1)/3})
    return np.array(rows), meta


def gen_null_separable(n, seed=20):
    """Null B: random SEPARABLE states — guaranteed rank(T)=1, F_avg≤2/3."""
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for _ in range(n):
        psiA = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        psiA /= np.linalg.norm(psiA)
        psiB = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        psiB /= np.linalg.norm(psiB)
        rho = np.outer(np.kron(psiA, psiB), np.kron(psiA, psiB).conj())
        T = compute_T_matrix(rho)
        svs = np.sort(np.linalg.svd(T, compute_uv=False))[::-1]
        eigs = np.sort(np.linalg.eigvalsh(rho))[::-1]
        fm = f_max_analytic(T)
        rows.append(_build_row(T, svs, eigs, 0.0, 2, 0.0,
                               np.zeros((3,3)), fm))
        meta.append({"source": "null_sep", "F_avg": (2*fm+1)/3})
    return np.array(rows), meta


def gen_null_scrambled(X_oat, m_oat, seed=30):
    """Null C: shuffle row order — breaks smooth parameter trajectory."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_oat))
    meta = [{"source": "null_scram", "F_avg": m_oat[i]["F_avg"]} for i in idx]
    return X_oat[idx].copy(), meta


def gen_ising(n_g, L=6, seed=40):
    """Transverse Ising boundary states (universality test)."""
    rows, meta = [], []
    for h in np.linspace(0.2, 3.0, n_g):
        try:
            rho = _ising_boundary(L, 1.0, h)
            T = compute_T_matrix(rho)
            svs = np.sort(np.linalg.svd(T, compute_uv=False))[::-1]
            eigs = np.sort(np.linalg.eigvalsh(rho))[::-1]
            fm = f_max_analytic(T)
            rows.append(_build_row(T, svs, eigs, h, L, 0.0,
                                   np.zeros((3,3)), fm))
            meta.append({"source": "ising", "F_avg": (2*fm+1)/3})
        except Exception:
            pass
    return (np.array(rows) if rows else np.zeros((0, 34))), meta


# ── Diffusion map ─────────────────────────────────────────────────────────────
def diffusion_map(X, n_components=3, alpha=0.5):
    D2 = cdist(X, X) ** 2
    eps = float(np.median(D2[D2 > 0]))
    K = np.exp(-D2 / eps)
    d = K.sum(axis=1)
    K = K / np.outer(d**alpha, d**alpha)
    P = K / K.sum(axis=1, keepdims=True)
    vals, vecs = eigh(P)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    emb = vecs[:, 1:n_components+1] * (vals[1:n_components+1] ** 2)
    return emb, vals[1:n_components+1]


# ── DBSCAN on embedding ───────────────────────────────────────────────────────
def dbscan(emb, min_samples=5):
    D = cdist(emb, emb)
    eps = float(np.percentile(D[D > 0], 10))
    n = len(emb)
    visited = np.zeros(n, dtype=bool)
    labels = -np.ones(n, dtype=int)
    cid = 0
    for i in range(n):
        if visited[i]: continue
        visited[i] = True
        nb = np.where(D[i] <= eps)[0]
        if len(nb) < min_samples: continue
        labels[i] = cid
        q = list(nb)
        while q:
            j = q.pop()
            if not visited[j]:
                visited[j] = True
                nb2 = np.where(D[j] <= eps)[0]
                if len(nb2) >= min_samples:
                    q.extend(nb2)
            if labels[j] == -1:
                labels[j] = cid
        cid += 1
    return labels


# ── Strata inference (geometric rules, NO labels) ─────────────────────────────
def infer_strata(X_blind):
    sv1  = X_blind[:, _BCOL["sv_1"]]
    sv2  = X_blind[:, _BCOL["sv_2"]]
    sv3  = X_blind[:, _BCOL["sv_3"]]
    sp   = X_blind[:, _BCOL["singularity_proximity"]]
    rank = X_blind[:, _BCOL["T_rank"]]
    H_T  = np.array([spectral_entropy(X_blind[i, :9]) for i in range(len(X_blind))])

    strata = np.empty(len(X_blind), dtype=object)
    for i in range(len(X_blind)):
        sv_tot = sv1[i] + sv2[i] + sv3[i] + 1e-10
        aniso = sv1[i] / sv_tot > 0.38   # sv_1 dominant (OAT signature)
        if sp[i] > 0.90 and rank[i] < 1:
            strata[i] = "SINGULAR"
        elif H_T[i] > 0.5 and rank[i] >= 3 and aniso and sp[i] < 0.65:
            strata[i] = "TRANSPORT"
        elif sp[i] > 0.70 and rank[i] > 0:
            strata[i] = "BOUNDARY"
        elif H_T[i] < 0.15 or rank[i] < 1:
            strata[i] = "PRODUCT"
        elif not aniso:
            strata[i] = "ISOTROPIC"
        else:
            strata[i] = "CLASSICAL"
    return strata, H_T


# ── Main experiment ───────────────────────────────────────────────────────────
def run_experiment(scale="small", out_dir="blind_experiment"):
    print("=" * 65)
    print("D-LinOSS BLIND TRANSPORT EXPERIMENT (Program E)")
    print("=" * 65)

    S = {
        "small":  dict(n_chi=25, n_gam=5,  N_vals=(4,),    null_n=100, ising_n=12),
        "medium": dict(n_chi=40, n_gam=8,  N_vals=(4,6),   null_n=300, ising_n=20),
        "full":   dict(n_chi=60, n_gam=12, N_vals=(4,6,8), null_n=600, ising_n=25),
    }
    cfg = S[scale]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"\nScale={scale}  blind_dim={len(_MASK)}")

    # Generate
    print("\n[1] OAT channels...")
    X_oat, m_oat = gen_oat(cfg["n_chi"], cfg["n_gam"], cfg["N_vals"])
    print(f"    {len(X_oat)} pts")

    print("[2] Null A: spectrum-preserving...")
    X_nA, m_nA = gen_null_spectrum_preserving(cfg["null_n"])

    print("[3] Null B: separable product states (F<=2/3 guaranteed)...")
    X_nB, m_nB = gen_null_separable(cfg["null_n"])

    print("[4] Null C: scrambled trajectory...")
    X_nC, m_nC = gen_null_scrambled(X_oat, m_oat)

    print("[5] Transverse Ising (universality)...")
    X_ising, m_ising = gen_ising(cfg["ising_n"])
    print(f"    {len(X_ising)} Ising pts")

    # Stack
    parts = [X_oat, X_nA, X_nB, X_nC]
    metas = m_oat + m_nA + m_nB + m_nC
    if len(X_ising) > 0:
        parts.append(X_ising)
        metas += m_ising

    X_all = np.vstack(parts)
    sources = np.array([m["source"] for m in metas])
    F_all   = np.array([m["F_avg"]  for m in metas])
    print(f"\nTotal: {len(X_all)} channels, {len(set(sources))} sources")

    # Blind slice
    X_blind = X_all[:, _MASK]
    print(f"Blind features: {X_blind.shape[1]} (F_avg hidden)")

    # Strata inference (no labels)
    print("\n[6] Blind strata inference...")
    strata, H_T = infer_strata(X_blind)
    unique_s, cnts = np.unique(strata, return_counts=True)
    for s, c in zip(unique_s, cnts):
        print(f"    {s:<15} {c:5d} ({100*c/len(strata):.1f}%)")

    # Diffusion map (blind)
    print("\n[7] Diffusion map (blind, 3 components)...")
    emb, eig_vals = diffusion_map(X_blind)
    print(f"    Eigenvalues: {eig_vals.round(4)}")

    # DBSCAN on embedding
    print("\n[8] DBSCAN on diffusion map...")
    db = dbscan(emb)
    n_cl = len(set(db)) - (1 if -1 in db else 0)
    print(f"    {n_cl} clusters found  ({(db==-1).sum()} outliers)")
    if n_cl > 0:
        base = (F_all > 2/3).mean()
        for cl in sorted(set(db)):
            m = db == cl
            f = (F_all[m] > 2/3).mean()
            dom = max(set(sources[m]), key=lambda s: (sources[m]==s).sum())
            tag = "outlier" if cl == -1 else f"cluster {cl}"
            print(f"    {tag:<12} n={m.sum():4d}  P(F>2/3)={f:.3f}  "
                  f"lift={f/(base+1e-10):.2f}x  dominant={dom}")

    # KEY STATISTIC
    print("\n" + "="*65)
    print("KEY RESULT: P(F_avg > 2/3 | inferred stratum)  [post-hoc]")
    print("="*65)
    base = (F_all > 2/3).mean()
    print(f"Base rate = {base:.3f}")
    stats = {}
    for s in unique_s:
        m = strata == s
        frac = (F_all[m] > 2/3).mean()
        lift = frac / (base + 1e-10)
        print(f"  {s:<15} n={m.sum():5d}  P(F>2/3)={frac:.3f}  lift={lift:.2f}x")
        stats[s] = {"n": int(m.sum()), "frac": float(frac), "lift": float(lift)}

    # Null control
    print("\n" + "="*65)
    print("NULL CONTROL: source mix in TRANSPORT stratum")
    print("="*65)
    t_mask = strata == "TRANSPORT"
    if t_mask.sum() > 0:
        for src in np.unique(sources):
            n_in = (sources[t_mask] == src).sum()
            n_tot = (sources == src).sum()
            print(f"  {src:<15} {n_in:4d}/{n_tot:4d} ({100*n_in/max(n_tot,1):.1f}%)")
    else:
        print("  (no TRANSPORT stratum found)")

    # Counterfactual
    print("\n" + "="*65)
    print("COUNTERFACTUAL: perturb transport PTM")
    print("="*65)
    t_idx = np.where(t_mask)[0]
    if len(t_idx) > 0:
        rng = np.random.default_rng(99)
        idx = t_idx[len(t_idx)//2]
        T_orig = X_all[idx, 3:12].reshape(3, 3)
        print(f"  Original: F={F_all[idx]:.4f}  stratum={strata[idx]}")
        for eps, lab in [(0.01,"tiny"), (0.05,"small"), (0.15,"moderate"), (0.30,"large")]:
            dT = rng.standard_normal((3,3)); dT /= np.linalg.norm(dT)
            T_p = T_orig + eps * dT
            svs_p = np.sort(np.linalg.svd(T_p, compute_uv=False))[::-1]
            fm_p = f_max_analytic(T_p)
            rk_p = int((svs_p > 1e-3).sum())
            sp_p = float(max(0, 1 - np.linalg.norm(T_p,"fro") / 2.0))
            x_p = X_blind[idx].copy()
            for ti, tn in enumerate(["T_xx","T_xy","T_xz","T_yx","T_yy",
                                     "T_yz","T_zx","T_zy","T_zz"]):
                if tn in _BCOL: x_p[_BCOL[tn]] = T_p.flatten()[ti]
            for k, v in [("sv_1",svs_p[0]),("sv_2",svs_p[1]),("sv_3",svs_p[2]),
                         ("T_rank",rk_p),("singularity_proximity",sp_p)]:
                if k in _BCOL: x_p[_BCOL[k]] = v
            st_p, _ = infer_strata(x_p.reshape(1,-1))
            same = st_p[0] == strata[idx]
            print(f"  eps={eps:.2f} ({lab:<9}): F={(2*fm_p+1)/3:.4f}  "
                  f"stratum={st_p[0]}  same={'YES' if same else 'NO '}")

    # Save
    np.savez(os.path.join(out_dir, "blind_experiment.npz"),
             X_all=X_all, X_blind=X_blind, emb=emb,
             strata=strata, sources=sources, F_all=F_all, H_T=H_T)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"scale": scale, "n": len(X_all),
                   "base_rate": float(base), "strata": stats,
                   "diffusion_eigs": eig_vals.tolist(),
                   "runtime_s": round(time.time()-t0,1)}, f, indent=2)
    print(f"\nSaved → {out_dir}/  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="small",
                   choices=["small","medium","full"])
    p.add_argument("--out", default="blind_experiment")
    args = p.parse_args()
    run_experiment(args.scale, args.out)
