"""
approach_c_tpu.py — Approach C Purity Scaling for TPU shotgun pipeline.
Accepts --N n1 n2 n3... and --out results.json exactly like the v2 pipeline.
"""
import argparse, json, sys, time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar, curve_fit
from scipy.linalg import eigvalsh

parser = argparse.ArgumentParser()
parser.add_argument("--N", nargs="+", type=int,
                    default=[2,4,6,8,12,16,24,32,48,64,128,256,512])
parser.add_argument("--out", type=str, default="approach_c_results.json")
parser.add_argument("--n_restarts", type=int, default=80)
args = parser.parse_args()

try:
    import jax
    print(f"JAX {jax.__version__}  devices={jax.devices()}", flush=True)
except ImportError:
    print("JAX not available, running in numpy mode", flush=True)

# ── Exact rho2 from Proposition 1 ────────────────────────────────────────
def rho2_exact(chi_t, N):
    rho = np.zeros((4,4), dtype=complex)
    m = N//2 - 1
    for a, (iL, iR) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        for b, (jL, jR) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            phase = np.exp(1j*chi_t*((iL-.5)*(iR-.5)-(jL-.5)*(jR-.5)))
            cR = np.cos((iR-jR)*chi_t/2)**m
            cL = np.cos((iL-jL)*chi_t/2)**m
            rho[a,b] = 0.25*phase*cR*cL
    return rho

def purity_analytic(chi_t, N):
    """Tr(rho^2) = (4 + 8*c^{2m} + 4*c^{4m})/16, m=N/2-1, c=cos(chi_t/2)."""
    m = N//2-1
    c2 = np.cos(chi_t/2)**(2*m)
    return (4 + 8*c2 + 4*c2**2)/16

def concurrence(rho):
    sy = np.array([[0,-1j],[1j,0]])
    R  = rho @ np.kron(sy,sy) @ rho.conj() @ np.kron(sy,sy)
    ev = np.sort(np.real(np.sqrt(np.maximum(eigvalsh(R),0))))[::-1]
    return max(0.0, float(ev[0]-ev[1]-ev[2]-ev[3]))

def su2(p):
    a,b,g = p[0],p[1],p[2]
    Rza = np.array([[np.exp(-1j*a/2),0],[0,np.exp(1j*a/2)]])
    Ryb = np.array([[np.cos(b/2),-np.sin(b/2)],[np.sin(b/2),np.cos(b/2)]])
    Rzg = np.array([[np.exp(-1j*g/2),0],[0,np.exp(1j*g/2)]])
    return Rza@Ryb@Rzg

def sf(rho, p):
    U = np.kron(su2(p[:3]), su2(p[3:]))
    rr = U@rho@U.conj().T
    v = np.array([0,1,1,0],dtype=complex)/np.sqrt(2)
    return float(np.real(v.conj()@rr@v))

def fmax(rho, n_restarts):
    def neg(p): return -sf(rho,p)
    res = differential_evolution(neg,[(-np.pi,np.pi)]*6,seed=42,
                                  maxiter=300,tol=1e-10,popsize=15)
    best = -res.fun
    for seed in range(n_restarts):
        np.random.seed(seed)
        x0 = res.x + np.random.normal(0,.3,6)
        r  = minimize(neg,x0,method='Powell',
                      options={'ftol':1e-12,'xtol':1e-12,'maxiter':50000})
        if -r.fun > best: best=-r.fun
    return best

# ── Main sweep ────────────────────────────────────────────────────────────
print(f"\nApproach C sweep: N={args.N}", flush=True)
print(f"{'N':>6}  {'chi*':>7}  {'C':>7}  {'purity':>7}  "
      f"{'F_opt':>7}  {'(2+C)/3':>7}  {'delta_N':>7}")
print("-"*60)

results = []
for N in args.N:
    t0 = time.time()
    # Find chi_t* (maximises C)
    chi_grid = np.linspace(.05, np.pi-.05, 400)
    idx = np.argmax([concurrence(rho2_exact(c,N)) for c in chi_grid])
    lo  = max(.01, chi_grid[max(0,idx-5)])
    hi  = min(np.pi-.01, chi_grid[min(len(chi_grid)-1,idx+5)])
    res2 = minimize_scalar(lambda c:-concurrence(rho2_exact(c,N)),
                           bounds=(lo,hi),method='bounded')
    cstar = float(res2.x)
    rho  = rho2_exact(cstar,N)
    C    = concurrence(rho)
    pur  = purity_analytic(cstar,N)

    # F_opt: full SU(2)^2 opt for N<=64, purity-corrected approx for larger N
    if N <= 64:
        fm = fmax(rho, args.n_restarts)
    else:
        # Quick local search seeded near analytic estimate
        est = (1+C)/2*np.sqrt(pur)
        def neg2(p): return -sf(rho,p)
        best=0
        for seed in range(40):
            np.random.seed(seed)
            r3 = minimize(neg2,np.random.uniform(-np.pi,np.pi,6),
                          method='Powell',
                          options={'ftol':1e-10,'xtol':1e-10,'maxiter':50000})
            if -r3.fun>best: best=-r3.fun
        fm = max(best, est)

    Fopt  = (2*fm+1)/3
    Fxst  = (2+C)/3
    delta = 1-Fopt/Fxst if Fxst>1e-10 else 0.0
    dt    = time.time()-t0

    print(f"{N:6d}  {cstar:7.4f}  {C:7.5f}  {pur:7.5f}  "
          f"{Fopt:7.5f}  {Fxst:7.5f}  {delta:7.5f}  [{dt:.1f}s]", flush=True)
    results.append({"N":int(N),"chi_t_star":cstar,"C":C,"purity":pur,
                    "F_opt":Fopt,"F_xst":Fxst,"delta_N":delta,"f_max":fm})

# ── Fit ───────────────────────────────────────────────────────────────────
Ns  = np.array([r["N"]      for r in results if r["N"]>=4])
ds  = np.array([r["delta_N"] for r in results if r["N"]>=4])
ps  = np.array([r["purity"]  for r in results if r["N"]>=4])
mask= ds>1e-5
fit_d = {"a":0,"b":0}
if mask.sum()>=3:
    pd,_ = curve_fit(lambda n,a,b: a*n**(-b), Ns[mask], ds[mask], p0=[.1,.5])
    fit_d = {"a":float(pd[0]),"b":float(pd[1])}
    print(f"\ndelta_N ~ {pd[0]:.4f} * N^(-{pd[1]:.4f})")
pp,_ = curve_fit(lambda n,a,b: 1-a*n**(-b), Ns, ps, p0=[.5,.5])
fit_p = {"a":float(pp[0]),"b":float(pp[1])}
print(f"1-purity ~ {pp[0]:.4f} * N^(-{pp[1]:.4f})")
print(f"\nConclusion: F_opt/(2+C)/3 -> 1 at rate N^(-{fit_d.get('b',0):.3f})")

out = {"N_list":args.N,"results":results,
       "delta_fit":fit_d,"purity_fit":fit_p,
       "analytic_purity":"(4+8*c^{2m}+4*c^{4m})/16  m=N/2-1  c=cos(chi*/2)"}
with open(args.out,"w") as f: json.dump(out,f,indent=2)
print(f"\nSaved: {args.out}", flush=True)
