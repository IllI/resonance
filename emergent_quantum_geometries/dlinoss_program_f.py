"""
dlinoss_program_f.py — Blind Stratification Under Adversarial Controls
=======================================================================
Attacks the vulnerability: "detector rediscovers smooth parameter continuity."

Four adversarial upgrades over Program E:
  1. Full parameter scrambling — D-LinOSS sees isolated tensor points only
  2. Adversarial nulls — preserve spectrum+rank, randomize tensor orientation
  3. Held-out Hamiltonians — XXZ/Ising tested without retraining
  4. Bootstrap enrichment ratio E = P(F>2/3|TRANSPORT)/P(F>2/3) with 95% CI

Central statistic: Transport Enrichment Ratio E (not silhouette, not purity)
Language: "transport-enriched operator stratum" (not "teleportation phase")

Usage:
  python dlinoss_program_f.py --scale small
  python dlinoss_program_f.py --scale full --out results/program_f/
"""
import argparse, json, os, time
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from ptm_features import (PTMFeatures, PTMFeatureVector, compute_T_matrix,
                           f_max_analytic)
from dlinoss_geometry import spectral_entropy

# ── Blind feature mapping ─────────────────────────────────────────────────────
ALL = PTMFeatureVector.feature_names()
BLIND = {"T_xx","T_xy","T_xz","T_yx","T_yy","T_yz","T_zx","T_zy","T_zz",
         "sv_1","sv_2","sv_3","lam_1","lam_2","lam_3","lam_4",
         "T_rank","dT_xx","dT_yy","dT_zz","dT_yz","dT_zy",
         "singularity_proximity"}
_MASK = np.array([i for i,n in enumerate(ALL) if n in BLIND])
_C = {ALL[o]: b for b, o in enumerate(_MASK)}   # name -> blind col index


def _build(T, svs, eigs, chi_t, N, Gamma, dT, fm):
    sp = float(max(0, min(1, 1 - np.linalg.norm(T,"fro")/2.0)))
    rk = int((svs > 1e-3).sum())
    Fa = (2*fm+1)/3
    return np.array([chi_t,N,Gamma,
                     T[0,0],T[0,1],T[0,2],T[1,0],T[1,1],T[1,2],
                     T[2,0],T[2,1],T[2,2],
                     svs[0],svs[1],svs[2],
                     dT[0,0],dT[0,1],dT[0,2],dT[1,0],dT[1,1],dT[1,2],
                     dT[2,0],dT[2,1],dT[2,2],
                     eigs[0],eigs[1],eigs[2],eigs[3],
                     0.0,0.0,float(rk),fm,Fa,sp])

def _so3(rng):
    Q,R = np.linalg.qr(rng.standard_normal((3,3)))
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q)<0: Q[:,0]*=-1
    return Q


# ── Data generators ───────────────────────────────────────────────────────────
def gen_oat(n_chi, n_gam, N_vals):
    rows, meta = [], []
    for N in N_vals:
        feat = PTMFeatures(N=N, fast=True)
        for G in np.linspace(0,0.35,n_gam):
            for c in np.linspace(0.02,np.pi,n_chi):
                fv = feat.compute(c, G)
                rows.append(fv.to_array())
                meta.append({"src":"oat","F":fv.F_avg})
    return np.array(rows), meta


def gen_xxz(n, seed=5):
    """XXZ spin chain boundary pair (ED, L=6)."""
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for delta in np.linspace(-1.5, 1.5, n):
        try:
            L = 6; dim = 2**L
            H = np.zeros((dim,dim))
            for i in range(L-1):
                for s in range(dim):
                    zi = 1 if (s>>(L-1-i))&1==0 else -1
                    zi1= 1 if (s>>(L-1-i-1))&1==0 else -1
                    H[s,s] += delta*zi*zi1
                    s2 = s^(1<<(L-1-i))^(1<<(L-1-i-1))
                    H[s,s2] += 0.5
            vals,vecs = eigh(H, subset_by_index=[0,0])
            psi = vecs[:,0]; pm = psi.reshape(4, dim//4)
            rho = pm @ pm.conj().T; rho /= np.trace(rho)
            T = compute_T_matrix(rho)
            svs = np.sort(np.linalg.svd(T,compute_uv=False))[::-1]
            eigs= np.sort(np.linalg.eigvalsh(rho))[::-1]
            fm = f_max_analytic(T)
            rows.append(_build(T,svs,eigs,delta,L,0,np.zeros((3,3)),fm))
            meta.append({"src":"xxz","F":(2*fm+1)/3})
        except Exception: pass
    return (np.array(rows) if rows else np.zeros((0,34))), meta


def gen_ising(n, L=6):
    rows, meta = [], []
    for h in np.linspace(0.2,3.0,n):
        try:
            dim=2**L; H=np.zeros((dim,dim))
            for i in range(L-1):
                for s in range(dim):
                    zi=1 if (s>>(L-1-i))&1==0 else -1
                    zi1=1 if (s>>(L-1-i-1))&1==0 else -1
                    H[s,s]-=zi*zi1
            for i in range(L):
                for s in range(dim): H[s,s^(1<<(L-1-i))]-=h
            vals,vecs=eigh(H,subset_by_index=[0,0])
            psi=vecs[:,0]; pm=psi.reshape(4,dim//4)
            rho=pm@pm.conj().T; rho/=np.trace(rho)
            T=compute_T_matrix(rho)
            svs=np.sort(np.linalg.svd(T,compute_uv=False))[::-1]
            eigs=np.sort(np.linalg.eigvalsh(rho))[::-1]
            fm=f_max_analytic(T)
            rows.append(_build(T,svs,eigs,h,L,0,np.zeros((3,3)),fm))
            meta.append({"src":"ising","F":(2*fm+1)/3})
        except Exception: pass
    return (np.array(rows) if rows else np.zeros((0,34))), meta


def gen_null_adversarial_spectrum(X_oat, seed=10):
    """Null A: preserve SVs+rank+Frobenius, randomize eigenvector orientation."""
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for row in X_oat:
        T = row[3:12].reshape(3,3)
        _,s,_ = np.linalg.svd(T)
        T2 = _so3(rng)@np.diag(s)@_so3(rng).T
        svs2 = np.sort(np.linalg.svd(T2,compute_uv=False))[::-1]
        fm2 = f_max_analytic(T2)
        eigs2 = np.sort(np.linalg.eigvalsh(
            np.eye(4,dtype=complex)/4 + sum(
                T2[i,j]*np.kron(P[i],P[j])
                for i in range(3) for j in range(3))/4
        ))[::-1].real
        rows.append(_build(T2,svs2,eigs2,0,4,0,np.zeros((3,3)),fm2))
        meta.append({"src":"null_adv_spec","F":(2*fm2+1)/3})
    return np.array(rows), meta


def gen_null_hif(n, seed=20):
    """Null B: high-F_avg channels but random PTM orientation (adversarial)."""
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    feat = PTMFeatures(N=4, fast=True)
    # Find a high-F_avg OAT point, then randomize its PTM orientation
    fv_star = feat.compute(1.189, 0.0)
    T_star = np.array([[fv_star.T_xx,fv_star.T_xy,fv_star.T_xz],
                       [fv_star.T_yx,fv_star.T_yy,fv_star.T_yz],
                       [fv_star.T_zx,fv_star.T_zy,fv_star.T_zz]])
    _,s,_ = np.linalg.svd(T_star)
    for _ in range(n):
        T2 = _so3(rng)@np.diag(s)@_so3(rng).T
        svs2 = np.sort(np.linalg.svd(T2,compute_uv=False))[::-1]
        fm2 = f_max_analytic(T2)
        rows.append(_build(T2,svs2,np.ones(4)*0.25,0,4,0,np.zeros((3,3)),fm2))
        meta.append({"src":"null_hif","F":(2*fm2+1)/3})
    return np.array(rows), meta


def gen_null_separable(n, seed=30):
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for _ in range(n):
        pA = rng.standard_normal(2)+1j*rng.standard_normal(2); pA/=np.linalg.norm(pA)
        pB = rng.standard_normal(2)+1j*rng.standard_normal(2); pB/=np.linalg.norm(pB)
        rho = np.outer(np.kron(pA,pB),np.kron(pA,pB).conj())
        T = compute_T_matrix(rho)
        svs = np.sort(np.linalg.svd(T,compute_uv=False))[::-1]
        eigs= np.sort(np.linalg.eigvalsh(rho))[::-1]
        fm = f_max_analytic(T)
        rows.append(_build(T,svs,eigs,0,2,0,np.zeros((3,3)),fm))
        meta.append({"src":"null_sep","F":(2*fm+1)/3})
    return np.array(rows), meta


# Pauli matrices for gen_null_adversarial_spectrum
_sx=np.array([[0,1],[1,0]],dtype=complex)
_sy=np.array([[0,-1j],[1j,0]],dtype=complex)
_sz=np.array([[1,0],[0,-1]],dtype=complex)
P=[_sx,_sy,_sz]


# ── Diffusion map ─────────────────────────────────────────────────────────────
def diffusion_map(X, nc=3, alpha=0.5):
    D2 = cdist(X,X)**2
    eps = float(np.median(D2[D2>0]))
    K = np.exp(-D2/eps)
    d = K.sum(1)
    K = K/np.outer(d**alpha,d**alpha)
    P_ = K/K.sum(1,keepdims=True)
    vals,vecs = eigh(P_)
    idx = np.argsort(vals)[::-1]
    vals,vecs = vals[idx],vecs[:,idx]
    return vecs[:,1:nc+1]*(vals[1:nc+1]**2), vals[1:nc+1]


# ── Strata (geometric, no labels) ────────────────────────────────────────────
def infer_strata(Xb):
    sv1=Xb[:,_C["sv_1"]]; sv2=Xb[:,_C["sv_2"]]; sv3=Xb[:,_C["sv_3"]]
    sp=Xb[:,_C["singularity_proximity"]]; rk=Xb[:,_C["T_rank"]]
    HT=np.array([spectral_entropy(Xb[i,:9]) for i in range(len(Xb))])
    out=np.empty(len(Xb),dtype=object)
    for i in range(len(Xb)):
        tot=sv1[i]+sv2[i]+sv3[i]+1e-10
        aniso=sv1[i]/tot>0.38
        if sp[i]>0.90 and rk[i]<1:         out[i]="SINGULAR"
        elif HT[i]>0.5 and rk[i]>=3 and aniso and sp[i]<0.65: out[i]="TRANSPORT"
        elif sp[i]>0.70 and rk[i]>0:        out[i]="BOUNDARY"
        elif HT[i]<0.15 or rk[i]<1:         out[i]="PRODUCT"
        elif not aniso:                       out[i]="ISOTROPIC"
        else:                                 out[i]="CLASSICAL"
    return out, HT


# ── Bootstrap enrichment ratio ────────────────────────────────────────────────
def bootstrap_enrichment(strata, F_all, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(strata)
    E_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0,n,n)
        st_b = strata[idx]; F_b = F_all[idx]
        t_mask = st_b=="TRANSPORT"
        base = (F_b>2/3).mean()
        if t_mask.sum()>0 and base>0:
            E_boot.append((F_b[t_mask]>2/3).mean()/base)
    E_boot = np.array(E_boot)
    return float(E_boot.mean()), float(np.percentile(E_boot,2.5)), float(np.percentile(E_boot,97.5))


# ── Local dimension vs teleportation ─────────────────────────────────────────
def dim_vs_teleportation(Xb, F_all, n_bins=8):
    from dlinoss_geometry import twonn_intrinsic_dim
    D_local = twonn_intrinsic_dim(Xb[:,_C["sv_1"]:_C["sv_1"]+3])
    valid = D_local[D_local>0.1]
    if len(valid) < n_bins*2: return []
    bins = np.percentile(valid, np.linspace(0,100,n_bins+1))
    results = []
    for lo,hi in zip(bins[:-1],bins[1:]):
        m = (D_local>=lo)&(D_local<hi)
        if m.sum()>5:
            results.append({"d_lo":float(lo),"d_hi":float(hi),
                            "n":int(m.sum()),
                            "P_F_above":float((F_all[m]>2/3).mean())})
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def run(scale="small", out="program_f"):
    SCALES={
        "small": dict(n_chi=20,n_gam=4,N_vals=(4,),null_n=80,xxz_n=15,ising_n=12),
        "medium":dict(n_chi=35,n_gam=7,N_vals=(4,6),null_n=250,xxz_n=25,ising_n=20),
        "full":  dict(n_chi=55,n_gam=11,N_vals=(4,6,8),null_n=500,xxz_n=35,ising_n=30),
    }
    cfg=SCALES[scale]; os.makedirs(out,exist_ok=True); t0=time.time()
    print(f"=== Program F  scale={scale} ===")

    # Generate — ALL sources
    print("[1] OAT..."); X_oat,m_oat=gen_oat(cfg["n_chi"],cfg["n_gam"],cfg["N_vals"])
    print(f"    {len(X_oat)} pts")
    print("[2] Null adversarial-spectrum (preserve SVs, randomize orientation)...")
    X_nA,m_nA=gen_null_adversarial_spectrum(X_oat)
    print("[3] Null high-F/random-orientation (adversarial B)...")
    X_nB,m_nB=gen_null_hif(cfg["null_n"])
    print("[4] Null separable (guaranteed F<=2/3)...")
    X_nC,m_nC=gen_null_separable(cfg["null_n"])
    print("[5] XXZ boundary (held-out Hamiltonian)...")
    X_xxz,m_xxz=gen_xxz(cfg["xxz_n"])
    print(f"    {len(X_xxz)} pts")
    print("[6] Ising boundary (held-out Hamiltonian)...")
    X_ising,m_ising=gen_ising(cfg["ising_n"])
    print(f"    {len(X_ising)} pts")

    # Stack and FULLY SCRAMBLE (kill trajectory continuity)
    parts=[X_oat,X_nA,X_nB,X_nC]
    metas=m_oat+m_nA+m_nB+m_nC
    heldout_parts=[]; heldout_meta=[]
    for Xh,mh in [(X_xxz,m_xxz),(X_ising,m_ising)]:
        if len(Xh)>0:
            heldout_parts.append(Xh); heldout_meta+=mh

    X_train=np.vstack(parts); meta_train=metas
    rng=np.random.default_rng(42)
    shuf=rng.permutation(len(X_train))
    X_train=X_train[shuf]; meta_train=[meta_train[i] for i in shuf]
    src_train=np.array([m["src"] for m in meta_train])
    F_train=np.array([m["F"] for m in meta_train])
    print(f"\n[SCRAMBLED] {len(X_train)} training pts (trajectory continuity destroyed)")

    # Blind slice
    Xb=X_train[:,_MASK]

    # Strata inference (no labels)
    print("\n[7] Strata inference (blind)...")
    strata,HT=infer_strata(Xb)
    for s,c in zip(*np.unique(strata,return_counts=True)):
        print(f"    {s:<15} {c:5d} ({100*c/len(strata):.1f}%)")

    # Diffusion map
    print("\n[8] Diffusion map (blind)...")
    emb,eigs=diffusion_map(Xb)
    print(f"    Eigenvalues: {eigs.round(4)}")

    # KEY STATISTIC: Transport Enrichment Ratio E with 95% CI
    print("\n"+"="*60)
    print("TRANSPORT ENRICHMENT RATIO E = P(F>2/3|TRANSPORT)/P(F>2/3)")
    print("="*60)
    base=(F_train>2/3).mean()
    print(f"Base rate P(F>2/3) = {base:.3f}")
    for s in np.unique(strata):
        m=strata==s
        frac=(F_train[m]>2/3).mean()
        E=frac/(base+1e-10)
        print(f"  {s:<15} n={m.sum():5d}  P(F>2/3)={frac:.3f}  E={E:.2f}x")

    E_mean,E_lo,E_hi=bootstrap_enrichment(strata,F_train)
    print(f"\n  TRANSPORT E = {E_mean:.2f}  95%CI [{E_lo:.2f}, {E_hi:.2f}]  (n_boot=200)")

    # Null control: source mix in TRANSPORT
    print("\n"+"="*60)
    print("NULL CONTROL: source mix in TRANSPORT stratum")
    print("="*60)
    t_mask=strata=="TRANSPORT"
    for src in np.unique(src_train):
        n_in=(src_train[t_mask]==src).sum()
        n_tot=(src_train==src).sum()
        print(f"  {src:<20} {n_in:4d}/{n_tot:4d} ({100*n_in/max(n_tot,1):.1f}%)")

    # Adversarial null check: do null_adv_spec or null_hif enter TRANSPORT?
    print("\n  [ADVERSARIAL CHECK]")
    for null_src in ["null_adv_spec","null_hif","null_sep"]:
        m_null=(src_train==null_src)
        if m_null.sum()==0: continue
        frac_in=(strata[m_null]=="TRANSPORT").mean()
        print(f"  {null_src:<20} in TRANSPORT: {100*frac_in:.1f}%  "
              f"(want: low for sep, variable for others)")

    # Held-out Hamiltonian test
    if heldout_parts:
        print("\n"+"="*60)
        print("HELD-OUT HAMILTONIAN GENERALIZATION")
        print("="*60)
        X_hout=np.vstack(heldout_parts); m_hout=heldout_meta
        F_hout=np.array([m["F"] for m in m_hout])
        src_hout=np.array([m["src"] for m in m_hout])
        Xbh=X_hout[:,_MASK]
        st_h,_=infer_strata(Xbh)
        base_h=(F_hout>2/3).mean()
        print(f"Held-out base rate = {base_h:.3f}")
        for src in np.unique(src_hout):
            ms=src_hout==src
            mt=st_h[ms]=="TRANSPORT"
            Fh=F_hout[ms]
            frac_t=mt.mean()
            F_in_t=(Fh[mt]>2/3).mean() if mt.sum()>0 else float("nan")
            E_h=F_in_t/(base_h+1e-10) if not np.isnan(F_in_t) else float("nan")
            print(f"  {src:<8} {(ms).sum():3d} pts  "
                  f"in TRANSPORT={100*frac_t:.0f}%  "
                  f"P(F>2/3|TRANSPORT)={F_in_t:.3f}  E={E_h:.2f}x")

    # Local dimension vs teleportation
    print("\n"+"="*60)
    print("LOCAL DIMENSION vs TELEPORTATION P(F>2/3|D_eff)")
    print("="*60)
    dim_results=dim_vs_teleportation(Xb,F_train)
    for r in dim_results:
        print(f"  D_eff [{r['d_lo']:.2f},{r['d_hi']:.2f}]  "
              f"n={r['n']:4d}  P(F>2/3)={r['P_F_above']:.3f}")

    # Save
    np.savez(os.path.join(out,"program_f.npz"),
             X_train=X_train,Xb=Xb,emb=emb,
             strata=strata,src_train=src_train,F_train=F_train,HT=HT)
    with open(os.path.join(out,"results_f.json"),"w") as f:
        json.dump({"scale":scale,"n_train":len(X_train),
                   "E_mean":E_mean,"E_ci":[E_lo,E_hi],
                   "base_rate":float(base),
                   "dim_vs_F":dim_results,
                   "runtime_s":round(time.time()-t0,1)},f,indent=2)
    print(f"\nDone -> {out}/  ({time.time()-t0:.1f}s)")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--scale",default="small",choices=["small","medium","full"])
    p.add_argument("--out",default="program_f")
    args=p.parse_args()
    run(args.scale,args.out)
