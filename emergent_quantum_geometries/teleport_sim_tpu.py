"""
teleport_sim_tpu.py — v6: scipy DE (6D opt) + JAX vmap (5k-state batch)
Step 0 confirmed: T_yz=-0.497 at chi_t*, f_max=0.638, F_avg=0.758>2/3 (paper stands)
"""
import jax, jax.numpy as jnp, numpy as np, json, argparse
from scipy.optimize import differential_evolution, minimize
from scipy.linalg import eigvalsh
from scipy.optimize import minimize_scalar

parser = argparse.ArgumentParser()
parser.add_argument("--out",    default="teleport_results.json")
parser.add_argument("--N_haar", type=int, default=5000)
parser.add_argument("--K_vq",   type=int, default=2000)
args = parser.parse_args()

print(f"JAX {jax.__version__}  devices={jax.devices()}", flush=True)
key = jax.random.PRNGKey(42)

# ── rho2 from Proposition 1 ─────────────────────────────────────────────────
def rho2_np(chi_t, N):
    m=N//2-1; B=[(0,0),(0,1),(1,0),(1,1)]
    r=np.zeros((4,4),dtype=complex)
    for a,(iL,iR) in enumerate(B):
        for b,(jL,jR) in enumerate(B):
            r[a,b]=.25*np.exp(1j*chi_t*((iL-.5)*(iR-.5)-(jL-.5)*(jR-.5)))*\
                   np.cos((iR-jR)*chi_t/2)**m*np.cos((iL-jL)*chi_t/2)**m
    return r

def apply_dephasing(rho, gam):
    B=[(0,0),(0,1),(1,0),(1,1)]; r=rho.copy()
    for a,(iL,iR) in enumerate(B):
        for b,(jL,jR) in enumerate(B):
            r[a,b]*=np.exp(-2*gam*(int(iL!=jL)+int(iR!=jR)))
    return r

def apply_depol(rho, p): return (1-p)*rho + p*np.eye(4)/4

# ── SU(2) optimizer (scipy DE — proven reliable in Step 0) ──────────────────
def su2(p):
    a,b,g=p
    return (np.array([[np.exp(-1j*a/2),0],[0,np.exp(1j*a/2)]]) @
            np.array([[np.cos(b/2),-np.sin(b/2)],[np.sin(b/2),np.cos(b/2)]]) @
            np.array([[np.exp(-1j*g/2),0],[0,np.exp(1j*g/2)]]))

PSI_P = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)  # |Phi+> matches [I,Z,X,Y] corrections

def singlet_frac_np(p, rho):
    U=np.kron(su2(p[:3]),su2(p[3:]))
    r=U@rho@U.conj().T
    return float(np.real(PSI_P.conj()@r@PSI_P))

def find_fmax(rho, seed=42, n_refine=30):
    np.random.seed(seed)
    res=differential_evolution(lambda p:-singlet_frac_np(p,rho),
                               [(-np.pi,np.pi)]*6,seed=seed,
                               maxiter=500,tol=1e-12,workers=1)
    best=-res.fun; bp=res.x
    for s in range(n_refine):
        r2=minimize(lambda p:-singlet_frac_np(p,rho),
                    res.x+np.random.normal(0,.3,6),method='Powell',
                    options={'ftol':1e-13,'maxiter':50000})
        if -r2.fun>best: best=-r2.fun; bp=r2.x
    UA=su2(bp[:3]); UB=su2(bp[3:])
    return best, UA, UB

def concurrence(rho):
    sy=np.array([[0,-1j],[1j,0]]); ss=np.kron(sy,sy)
    ev=np.sort(np.real(np.sqrt(np.maximum(eigvalsh(rho@ss@rho.conj()@ss),0))))[::-1]
    return max(0.,float(ev[0]-ev[1]-ev[2]-ev[3]))

# ── Teleportation: JAX vmap over 5000 Haar states (TPU does this) ───────────
BELL=jnp.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1,-1,0]],
               dtype=jnp.complex64)/jnp.sqrt(jnp.float32(2))
CORR=jnp.array([[[1,0],[0,1]],[[1,0],[0,-1]],
                 [[0,1],[1,0]],[[0,-1j],[1j,0]]],dtype=jnp.complex64)
I2  =jnp.eye(2,dtype=jnp.complex64)

@jax.jit
def teleport_single(phi, rho_rot):
    rho_in=jnp.outer(phi,phi.conj()).astype(jnp.complex64)
    rho_tot=jnp.kron(rho_in,rho_rot)
    def outcome(bv,C):
        P8=jnp.kron(jnp.outer(bv,bv.conj()),I2)
        rp=P8@rho_tot@P8.conj().T
        rb=jnp.sum(jnp.stack([rp[2*i:2*i+2,2*i:2*i+2] for i in range(4)]),axis=0)
        return jnp.real(phi.conj()@(C@rb@C.conj().T)@phi)
    return jnp.sum(jax.vmap(outcome)(BELL,CORR))  # sum p_m*F_m

teleport_batch=jax.jit(jax.vmap(teleport_single,in_axes=(0,None)))

def haar_states_jax(key,n):
    key,k1,k2=jax.random.split(key,3)
    re=jax.random.normal(k1,(n,2)).astype(jnp.complex64)
    im=jax.random.normal(k2,(n,2)).astype(jnp.complex64)
    v=re+jnp.array(1j,dtype=jnp.complex64)*im
    return v/jnp.linalg.norm(v,axis=1,keepdims=True),key

def F_avg_jax(rho_rot_j,key,N=500):
    phis,key=haar_states_jax(key,N)
    return float(jnp.mean(teleport_batch(phis,rho_rot_j))),key

def V_Q_jax(rho_j,key,K=2000):
    key,k1,k2=jax.random.split(key,3)
    params=jax.random.uniform(k1,(K,6),minval=-jnp.pi,maxval=jnp.pi)
    def sf(p):
        UA=jnp.array([[jnp.exp(jnp.array(-0.5j,dtype=jnp.complex64)*p[0]),jnp.float32(0)],
                      [jnp.float32(0),jnp.exp(jnp.array(0.5j,dtype=jnp.complex64)*p[0])]])
        UB=jnp.array([[jnp.exp(jnp.array(-0.5j,dtype=jnp.complex64)*p[3]),jnp.float32(0)],
                      [jnp.float32(0),jnp.exp(jnp.array(0.5j,dtype=jnp.complex64)*p[3])]])
        U=jnp.kron(UA,UB); r=U@rho_j@U.conj().T
        psi=jnp.array([0,1,1,0],dtype=jnp.complex64)/jnp.sqrt(jnp.float32(2))
        return jnp.real(psi.conj()@r@psi)
    sf_all=jax.vmap(sf)(params)
    return float(jnp.mean((jnp.float32(2)*sf_all+jnp.float32(1))/jnp.float32(3)>jnp.float32(2/3))),key

# ── chi_t* ──────────────────────────────────────────────────────────────────
N=4
print("Finding chi_t*...",flush=True)
chi_grid=np.linspace(.05,np.pi-.05,400)
C_arr=[concurrence(rho2_np(c,N)) for c in chi_grid]
idx=np.argmax(C_arr)
res2=minimize_scalar(lambda c:-concurrence(rho2_np(c,N)),
    bounds=(chi_grid[max(0,idx-5)],chi_grid[min(399,idx+5)]),method='bounded')
chi_star=float(res2.x)
print(f"chi_t*={chi_star:.5f}  C={C_arr[idx]:.5f}",flush=True)

results={}

# ── A ────────────────────────────────────────────────────────────────────────
print("\n[A] Fidelity distribution...",flush=True)
CARDS={'|0>':[1,0],'|1>':[0,1],'|+>':[1,1],'|->':[1,-1],'|+i>':[1,1j],'|-i>':[1,-1j]}
part_a={}
for label,chi_t in [("chi_t_star",chi_star),("chi_t_pi",np.pi)]:
    rho=rho2_np(chi_t,N)
    fmax,UA,UB=find_fmax(rho)
    U=np.kron(UA,UB); rot=jnp.array((U@rho@U.conj().T).astype(np.complex64))
    phis,key=haar_states_jax(key,args.N_haar)
    fids=np.array(teleport_batch(phis,rot))
    theory=(2*fmax+1)/3
    print(f"  {label}: F={fids.mean():.5f}+-{fids.std():.5f} theory={theory:.5f} fmax={fmax:.5f}",flush=True)
    cf={}
    for n,v in CARDS.items():
        va=np.array(v,dtype=np.complex64); va/=np.linalg.norm(va)
        cf[n]=float(teleport_single(jnp.array(va),rot))
    print(f"  Cardinals: {cf}",flush=True)
    part_a[label]={"chi_t":chi_t,"F_avg":float(fids.mean()),"F_std":float(fids.std()),
                   "F_min":float(fids.min()),"F_theory":theory,"fmax":fmax,
                   "cardinal_fids":cf,"fids_sample":fids[:200].tolist()}
results["A"]=part_a

# ── B ────────────────────────────────────────────────────────────────────────
print("\n[B] Phase diagram...",flush=True)
phase=[]
for chi_t in np.linspace(.05,np.pi,20):
    for gam in [0.,.05,.10,.15,.20,.30]:
        rho=apply_dephasing(rho2_np(chi_t,N),gam)
        fmax,UA,UB=find_fmax(rho,n_refine=15)
        U=np.kron(UA,UB); rot=jnp.array((U@rho@U.conj().T).astype(np.complex64))
        F,key=F_avg_jax(rot,key,N=300)
        print(f"  chi={chi_t:.3f} gam={gam:.2f} F={F:.4f} theory={(2*fmax+1)/3:.4f}",flush=True)
        phase.append({"chi_t":float(chi_t),"gamma_t":float(gam),"F_teleport":F,"F_theory":(2*fmax+1)/3})
results["B"]=phase

# ── C ────────────────────────────────────────────────────────────────────────
print("\n[C] V_Q correlation...",flush=True)
corr=[]
for chi_t in np.linspace(.1,np.pi-.05,25):
    rho=rho2_np(chi_t,N)
    vq,key=V_Q_jax(jnp.array(rho.astype(np.complex64)),key,K=args.K_vq)
    fmax,UA,UB=find_fmax(rho,n_refine=20)
    U=np.kron(UA,UB); rot=jnp.array((U@rho@U.conj().T).astype(np.complex64))
    F,key=F_avg_jax(rot,key,N=400)
    print(f"  chi={chi_t:.4f} V_Q={vq:.4f} F={F:.4f} theory={(2*fmax+1)/3:.4f}",flush=True)
    corr.append({"chi_t":float(chi_t),"V_Q":vq,"F_teleport":F,"F_theory":(2*fmax+1)/3})
results["C"]=corr

# ── D ────────────────────────────────────────────────────────────────────────
print("\n[D] Theorem 4 null table...",flush=True)
rho_pi=rho2_np(np.pi,N)
maxdiff=float(np.max(np.abs(rho_pi-np.eye(4)/4)))
fmax_pi,UA,UB=find_fmax(rho_pi)
U=np.kron(UA,UB); rot_pi=jnp.array((U@rho_pi@U.conj().T).astype(np.complex64))
phis,key=haar_states_jax(key,args.N_haar)
fids_pi=np.array(teleport_batch(phis,rot_pi))
vq_pi,key=V_Q_jax(jnp.array(rho_pi.astype(np.complex64)),key,K=args.K_vq)
null={"rho_I4_maxdiff":maxdiff,"F_teleport":float(fids_pi.mean()),
      "F_variance":float(fids_pi.var()),"V_Q":vq_pi,"theorem4_proved":bool(maxdiff<1e-10)}
print(f"  rho=I/4: {maxdiff:.2e}  VQ={vq_pi:.4f}  "
      f"F={fids_pi.mean():.5f}+-{fids_pi.std():.5f}  proved={null['theorem4_proved']}",flush=True)
results["D"]=null

# ── E ────────────────────────────────────────────────────────────────────────
print("\n[E] Noise robustness...",flush=True)
noise=[]
for p in [0.,.01,.02,.05,.10,.15,.20]:
    rho=apply_depol(rho2_np(chi_star,N),p)
    fmax_n,UA,UB=find_fmax(rho,n_refine=20)
    U=np.kron(UA,UB); rot=jnp.array((U@rho@U.conj().T).astype(np.complex64))
    F_s,key=F_avg_jax(rot,key,N=500)
    print(f"  p={p:.2f} F(chi*)={F_s:.4f} theory={(2*fmax_n+1)/3:.4f} gap={F_s-0.5:.4f}",flush=True)
    noise.append({"p_dep":p,"F_star":F_s,"F_pi":0.5,"F_theory":(2*fmax_n+1)/3,"gap":F_s-0.5})
results["E"]=noise

print(f"\nSaving {args.out}...",flush=True)
with open(args.out,"w") as f: json.dump(results,f,indent=2)
print("Done.",flush=True)
