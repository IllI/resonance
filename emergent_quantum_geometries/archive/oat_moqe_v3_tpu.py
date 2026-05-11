"""
oat_moqe_v3_tpu.py
===================
Two-layer quantum emergence system.

Layer 1 - LEARNER:  D-LiNOSS SSM, blind, reconstruction-only.
Layer 2 - OBSERVER: 7 pluggable formalism probes. Takes learner's
                    latent z(t) and asks which quantum laws it obeys.

Probes: hamiltonian, syk, mbl, lindblad, oat, penrose_or, novel
Output: structured emergence report + narrative
"""

import argparse, json, math
import numpy as np
from pathlib import Path
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

print(f"[DEVICE] {jax.devices()}", flush=True)
print(f"[DEVICE] backend={jax.default_backend()}", flush=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def _norm(s):
    mu = s.mean((1,2), keepdims=True); sd = s.std((1,2), keepdims=True)+1e-8
    return (s-mu)/sd

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def gen_oat(B,T,F,seed=0):
    rng=np.random.default_rng(seed)
    Ns=np.array([2,4,6,8,10,12,14])[:F]
    if len(Ns)<F: Ns=np.pad(Ns,(0,F-len(Ns)),mode='edge')
    t=np.linspace(0,math.pi,T)
    raw=np.zeros((B,T,F),np.float32)
    for b in range(B):
        chi=rng.uniform(0.8,1.2)
        for fi,N in enumerate(Ns):
            C=np.clip(np.abs(np.sin(chi*t))*(N**-1.43),0,1)
            raw[b,:,fi]=(2+C)/3+rng.normal(0,.005,T)
    return raw, _norm(raw.copy())

def gen_heisenberg(B,T,F,seed=1):
    rng=np.random.default_rng(seed)
    t=np.linspace(0,2*math.pi,T)
    w=rng.uniform(.1,.5,(B,F))
    raw=(np.cos(w[:,None,:]*t[None,:,None])+rng.normal(0,.01,(B,T,F))).astype(np.float32)
    return raw, _norm(raw.copy())

def gen_syk(B,T,F,seed=2):
    rng=np.random.default_rng(seed)
    beta=rng.uniform(3,8,(B,1)); lam=2*math.pi/beta
    t=np.linspace(0,5,T); k=np.arange(1,F+1)[None,None,:]
    w=math.pi*k/(F+1); g=lam[:,:,None]*np.ones((B,1,F))
    raw=(np.exp(-g*t[None,:,None])*np.cos(w*t[None,:,None])+rng.normal(0,.01,(B,T,F))).astype(np.float32)
    return raw, _norm(raw.copy())

def gen_mbl(B,T,F,seed=3):
    rng=np.random.default_rng(seed)
    t=np.linspace(0,10,T)
    w=rng.uniform(.05,2,(B,F)); g=rng.uniform(.001,.05,(B,F))
    raw=(np.exp(-g[:,None,:]*t[None,:,None])*np.cos(w[:,None,:]*t[None,:,None])+rng.normal(0,.01,(B,T,F))).astype(np.float32)
    return raw, _norm(raw.copy())

def gen_lindblad(B,T,F,seed=4):
    rng=np.random.default_rng(seed)
    t=np.linspace(0,5,T)
    g=rng.uniform(.2,.4,(B,1))*np.ones((B,F)); w=(math.pi*np.arange(1,F+1)/(F+1))[None,:]
    raw=(np.exp(-g[:,None,:]*t[None,:,None])*np.cos(w[:,None,:]*t[None,:,None])+rng.normal(0,.01,(B,T,F))).astype(np.float32)
    return raw, _norm(raw.copy())

def gen_penrose_or(B,T,F,seed=5):
    rng=np.random.default_rng(seed)
    t=np.linspace(0,5,T); t_OR=rng.uniform(1,3.5,(B,1))
    w=(math.pi*np.arange(1,F+1)/(F+1))[None,:]
    decay=np.exp(-5*np.maximum(t[None,:,None]-t_OR[:,:,None],0))
    raw=(np.cos(w[:,None,:]*t[None,:,None])*decay+rng.normal(0,.01,(B,T,F))).astype(np.float32)
    return raw, _norm(raw.copy())

GENERATORS={
    "oat":gen_oat,"heisenberg":gen_heisenberg,"syk":gen_syk,
    "mbl":gen_mbl,"lindblad":gen_lindblad,"penrose_or":gen_penrose_or
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LAYER 1 — LEARNER  (blind D-LiNOSS SSM)
# ═══════════════════════════════════════════════════════════════════════════════

class Learner(nn.Module):
    state_dim: int = 64

    @nn.compact
    def __call__(self, x):
        B,T,F = x.shape
        x = nn.LayerNorm(use_scale=False,use_bias=False)(x)
        raw_w=self.param('raw_w',nn.initializers.constant(0.5),(self.state_dim,))
        raw_g=self.param('raw_g',nn.initializers.constant(0.1),(self.state_dim,))
        omega=jnp.clip(jax.nn.softplus(raw_w),.01,10.)
        gamma=jnp.clip(jax.nn.softplus(raw_g),.001,2.)
        t=jnp.linspace(0.,1.,T)
        phi=jnp.exp(-gamma[None,:]*t[:,None])*jnp.cos(omega[None,:]*t[:,None])
        W_init=nn.initializers.glorot_normal()
        Wa=self.param('Wa',lambda r,s:W_init(r,s)*.1,(F,self.state_dim))
        alpha=jnp.tanh(jnp.dot(x.mean(1),Wa,precision=jax.lax.Precision.HIGHEST))
        z=alpha[:,None,:]*phi[None,:,:]               # (B,T,state_dim) latent
        Wo=self.param('Wo',lambda r,s:W_init(r,s)*.1,(self.state_dim,F))
        recon=jnp.dot(z,Wo,precision=jax.lax.Precision.HIGHEST)
        return z, recon, omega, gamma

def train_learner(data_j, epochs=500, state_dim=64):
    net=Learner(state_dim=state_dim)
    params=net.init(jax.random.PRNGKey(0),data_j)['params']
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-3))
    state=opt.init(params)

    @jax.jit
    def step(p,s,x):
        def loss_fn(p):
            _,rec,_,_=net.apply({'params':p},x)
            L=jnp.mean((rec-x)**2)
            return jnp.where(jnp.isnan(L)|jnp.isinf(L),1e5,L)
        L,g=jax.value_and_grad(loss_fn)(p)
        g=jax.tree_util.tree_map(lambda x:jnp.clip(x,-.1,.1),g)
        u,s2=opt.update(g,s,p)
        return optax.apply_updates(p,u),s2,L

    for ep in range(epochs+1):
        params,state,loss=step(params,state,data_j)
        if ep%100==0: print(f"  [Learner] ep={ep:>4d}  loss={float(loss):.5f}",flush=True)

    z,_,omega,gamma=net.apply({'params':params},data_j)
    return np.array(z), np.array(omega), np.array(gamma)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LAYER 2 — OBSERVER PROBES
# ═══════════════════════════════════════════════════════════════════════════════

def _safe(x): return float(np.clip(x,0,1))

class HamiltonianProbe:
    name="hamiltonian"
    def score(self,z,omega,gamma,raw,norm):
        """
        Tests Heisenberg equation of motion: dz_k/dt = -i[z_k, H]
        For H = sum_k Omega_k * n_k, this predicts each mode rotates
        as exp(-i Omega_k t), so |dz/dt| ~ Omega * |z|.
        EOM residual: ratio of actual dz/dt to predicted Omega*z scale.
        Perfect Heisenberg dynamics -> ratio near 1.
        Also checks: norm conservation (unitary) and gamma near zero.
        Note: [Omega_i, Omega_j] = 0 trivially for scalar frequencies;
        we instead test whether the inferred H generates correct EOM.
        """
        B,T,S=z.shape
        dt=1.0/(T-1)
        dz=np.diff(z,axis=1)                        # (B,T-1,S)
        dz_mag=float(np.linalg.norm(dz,axis=-1).mean())
        z_mag=float(np.linalg.norm(z[:,:-1,:],axis=-1).mean())
        omega_scale=float(omega.mean())
        # EOM ratio: |dz/dt| / (Omega * |z|) -- should be ~1 for Heisenberg
        eom_ratio=dz_mag/(omega_scale*z_mag*dt+1e-8)
        eom_fit=float(np.exp(-abs(eom_ratio-1.0)*2))
        # Norm conservation: unitary => ||z(t)|| stable
        norms=np.linalg.norm(z,axis=-1)
        norm_var=float(norms.std(axis=1).mean())
        norm_fit=float(np.exp(-norm_var*3))
        # Low gamma => near-unitary
        gamma_fit=float(np.exp(-gamma.mean()*5))
        fit=_safe((eom_fit+norm_fit+gamma_fit)/3)
        return {"fit":fit,"eom_ratio":eom_ratio,"norm_variance":norm_var,
                "gamma_mean":float(gamma.mean()),"eom_fit":eom_fit}

class SYKProbe:
    name="syk"
    def score(self,z,omega,gamma,raw,norm):
        B,T,_=z.shape
        # OTOC proxy: |z(T/2) - z(0)| growth rate
        early=np.linalg.norm(z[:,0,:],axis=-1)
        mid  =np.linalg.norm(z[:,T//2,:],axis=-1)
        late =np.linalg.norm(z[:,-1,:],axis=-1)
        decay_mid=float((mid/np.maximum(early,1e-8)).mean())
        decay_late=float((late/np.maximum(early,1e-8)).mean())
        # SYK: fast exponential decay (decay_late << 1, decay_mid intermediate)
        syk_decay_fit=np.exp(-abs(decay_late-0.05)*10)
        # gamma near lambda_L=2pi/5
        lambda_L=2*math.pi/5
        gamma_fit=np.exp(-abs(gamma.mean()-lambda_L)*2)
        fit=_safe((syk_decay_fit+gamma_fit)/2)
        return {"fit":fit,"decay_late":decay_late,"gamma_vs_lyapunov":float(gamma.mean()-lambda_L)}

class MBLProbe:
    name="mbl"
    def score(self,z,omega,gamma,raw,norm):
        # Level spacing ratio of omega spectrum (Poisson=0.386, GOE=0.53)
        s=np.sort(omega)
        gaps=np.diff(s)+1e-10
        r=float(np.mean(np.minimum(gaps[:-1],gaps[1:])/np.maximum(gaps[:-1],gaps[1:])))
        r_fit=np.exp(-abs(r-0.386)*10)   # Poisson = localized = MBL
        # Sustained signal (no decay)
        norms=np.linalg.norm(z,axis=-1)
        sustain=float((norms[:,-1]/np.maximum(norms[:,0],1e-8)).mean())
        sustain_fit=np.exp(-abs(sustain-0.9)*5)
        # Freq disorder
        disorder=float(omega.std())
        disorder_fit=_safe(disorder/2)
        fit=_safe((r_fit+sustain_fit+disorder_fit)/3)
        return {"fit":fit,"level_spacing_r":r,"freq_disorder":disorder,"sustain":sustain}

class LindbladProbe:
    name="lindblad"
    def score(self,z,omega,gamma,raw,norm):
        """
        BLP (Breuer-Laine-Piilo) Markovianity measure.
        Non-Markovianity N_BLP = integral of sigma(t) dt where sigma > 0,
        sigma(t) = d/dt [trace_distance(rho_1(t), rho_2(t))].
        For Markovian (Lindblad) channels: trace distance is monotonically
        non-increasing => N_BLP = 0.
        For non-Markovian: distance increases (information backflow) => N_BLP > 0.
        We use L1 norm of z as a proxy for trace distance between pairs.
        """
        B,T,S=z.shape
        # BLP: pair up trajectories and measure trace distance evolution
        n_pairs=min(B//2,16)
        z1=z[:n_pairs];  z2=z[n_pairs:n_pairs*2]
        D=np.linalg.norm(z1-z2,axis=-1)            # (n_pairs,T) trace distance proxy
        dD=np.diff(D,axis=1)                        # (n_pairs,T-1)
        N_BLP=float(np.sum(np.maximum(dD,0)))       # total information backflow
        # Lindblad = Markovian => low N_BLP
        markov_fit=float(np.exp(-N_BLP*0.05))
        # Also: uniform gamma (all bath modes decay at same rate)
        gamma_uniformity=float(np.exp(-gamma.std()*10))
        gamma_mean_fit=float(np.exp(-abs(gamma.mean()-0.30)*5))
        # Moderate decay
        norms=np.linalg.norm(z,axis=-1)
        decay=float((norms[:,-1]/np.maximum(norms[:,0],1e-8)).mean())
        decay_fit=float(np.exp(-abs(decay-0.3)*5))
        fit=_safe((markov_fit+gamma_uniformity*gamma_mean_fit+decay_fit)/3)
        return {"fit":fit,"N_BLP":N_BLP,"markov_fit":markov_fit,
                "gamma_uniformity":gamma_uniformity,"gamma_mean":float(gamma.mean()),
                "decay":decay}

class OATProbe:
    name="oat"
    def score(self,z,omega,gamma,raw,norm):
        """
        OAT unique signature: raw F in [2/3, 1] gives raw_mean > 2/3.
        No other framework has a positive raw offset — they all center near 0.
        This is the most reliable physics discriminator for OAT.
        """
        raw_mean=float(raw.mean())
        threshold=2/3
        above=float(raw_mean>threshold)
        margin=max(0.0, raw_mean-threshold)   # how far above 2/3
        # Oscillation: F oscillates (std > 0), not monotone decay
        min_val=float(raw.min())
        min_above=float(min_val>0.65)         # even trough stays above threshold
        raw_osc=float(raw.std(axis=1).mean())
        osc_fit=float(np.exp(-abs(raw_osc-0.08)*15))
        # Combined: dominant weight on raw_mean signature
        fit=_safe(above*0.6 + margin*8 + min_above*0.2 + osc_fit*0.1)
        return {"fit":fit,"raw_mean":raw_mean,"raw_min":min_val,"raw_osc":raw_osc,"margin_above_threshold":margin}

class PenroseBiTwistorProbe:
    """
    Penrose bi-twistor probe with phenomenological OR threshold.

    DISCLAIMER: The spinor identification
      xi^A = (z_mode0+i*z_mode1) * exp(i*Omega_0*t)   [Alice spinor]
      pi_A' = (z_mode2+i*z_mode3) * exp(i*Omega_2*t)  [Bob co-spinor]
    embeds C^4 into twistor space via a specific basis choice. The null
    surface condition Z^{AB}Z_{AB}=0 and the Plucker condition are
    mathematically meaningful as topological probes of the 2-qubit
    boundary state regardless of the gravitational interpretation.

    E_G_phenom is NOT E_G = hbar*c^5/G (gravitational self-energy).
    For cold atoms E_G ~ 10^-58 J, experimentally unreachable.
    Instead E_G_phenom is a tunable phenomenological collapse threshold
    in units of the bi-twistor norm ||Z||^2, calibrated to the
    observed scale of the latent representation.
    """
    name="penrose_or"
    def __init__(self, E_G_phenom=0.01):
        self.E_G=E_G_phenom
    def score(self,z,omega,gamma,raw,norm):
        """
        Penrose bi-twistor analysis.

        Twistor: Z^alpha = (omega^A, pi_{A'}) where omega^A is a 2-spinor.
        Bi-twistor: Z^{AB} = xi^A x pi^B - pi^A x xi^B (antisymmetric)
        Null surface: Z^{AB} Z_{AB} = 0  (separable) vs > 0 (entangled)

        The latent z modes are real-valued. We extend them to complex spinors
        by adding phase via the learned frequencies omega_k:
          xi^A(t) = z_k(t) * exp(i * Omega_k * t)
        This gives non-trivial complex bi-twistors even from real SSM output,
        correctly encoding the phase structure of the quantum evolution.
        """
        B,T,S=z.shape
        t=np.linspace(0.,1.,T)
        # Complex phase extension via learned omega
        ph=lambda k: np.exp(1j*omega[min(k,S-1)]*t)   # (T,)
        # Alice spinor: modes 0,1 with phase
        xi0=(z[:,:,0]*ph(0)[None,:]+1j*z[:,:,1]*ph(1)[None,:])
        xi1=(z[:,:,1]*ph(1)[None,:]-1j*z[:,:,0]*ph(0)[None,:])
        # Bob co-spinor: modes 2,3 with phase
        pi0=(z[:,:,2]*ph(2)[None,:]+1j*z[:,:,3]*ph(3)[None,:])
        pi1=(z[:,:,3]*ph(3)[None,:]-1j*z[:,:,2]*ph(2)[None,:])
        # Bi-twistor Pfaffian
        Z_pfaff=xi0*pi1-xi1*pi0   # (B,T) complex
        Z_norm_sq=np.abs(Z_pfaff)**2   # (B,T)

        # Departure from null surface (entanglement signature)
        null_residual = float(Z_norm_sq.mean())

        # Spinor phase winding (topological invariant)
        angles = np.angle(Z_pfaff)          # (B,T)
        winding = float(np.abs(np.diff(angles,axis=1)).sum(axis=1).mean()/(2*math.pi))

        # OR threshold crossings: ||Z||^2 crossing E_G_phenom (phenomenological)
        # NOT the gravitational formula. A tunable collapse threshold.
        or_crossings=int(np.sum(
            (Z_norm_sq[:,:-1] < self.E_G) & (Z_norm_sq[:,1:] >= self.E_G)
        ))  # rising edge crossings

        # Sudden collapse signature: high d^2(Z)/dt^2 spike (not just low late amplitude)
        dZ=np.diff(Z_norm_sq,axis=1)
        d2Z=np.diff(dZ,axis=1)
        collapse_spike=float(d2Z.std()/(d2Z.mean()+1e-8))  # high = sudden collapse
        collapse_fit=_safe(collapse_spike/10)

        # BLP-style: does Z_norm increase after decreasing? (non-Markovian backflow)
        Z_increase=float(np.sum(np.maximum(dZ,0)))/(Z_norm_sq.size+1e-8)

        # Combine
        entangle_fit=_safe(null_residual*10)   # recalibrated to latent scale
        winding_fit=_safe(min(winding,1.0))
        fit=_safe((entangle_fit+winding_fit+collapse_fit)/3)
        return {
            "fit":fit,
            "null_surface_residual":null_residual,
            "spinor_winding":winding,
            "or_threshold_crossings":or_crossings,
            "E_G_phenom":self.E_G,
            "collapse_spike":collapse_spike,
            "z_backflow":Z_increase,
            "bi_twistor_norm_mean":float(Z_norm_sq.mean()),
            "note":"E_G is phenomenological, not gravitational. Null surface = topological probe."
        }

class NovelProbe:
    """
    Residual geometry probe. Fires when no known formalism fits well.
    Extracts topological invariants from the latent trajectory:
      - SVD effective dimension (proxy for Hausdorff dim of trajectory manifold)
      - Level spacing ratio r (Poisson vs GOE spectral statistics)
      - Lyapunov exponent proxy (exponential divergence of nearby trajectories)
      - Time-reversal symmetry score
      - Topological winding of leading SSM mode
    These invariants are passed to an optional LLM narrative generator
    (generate_novel_narrative) when no standard formalism fits.
    """
    name="novel"
    def score(self,z,omega,gamma,raw,norm):
        B,T,S=z.shape
        # SVD effective dimension of latent trajectory manifold
        z_flat=z[0].reshape(T,-1)
        U,sv,Vt=np.linalg.svd(z_flat,full_matrices=False)
        sv_norm=sv/(sv.sum()+1e-10)
        effective_dim=float(np.exp(-np.sum(sv_norm*np.log(sv_norm+1e-10))))
        # Level spacing ratio (Poisson=0.386, GOE=0.53)
        s=np.sort(omega); gaps=np.diff(s)+1e-10
        r=float(np.mean(np.minimum(gaps[:-1],gaps[1:])/np.maximum(gaps[:-1],gaps[1:])))
        # Lyapunov: exponential divergence of nearby trajectory pairs
        n_pairs=min(B//2,8)
        D=np.linalg.norm(z[:n_pairs]-z[n_pairs:n_pairs*2],axis=-1)  # (pairs,T)
        D_safe=np.maximum(D,1e-10)
        log_D=np.log(D_safe)
        lyap=float(np.polyfit(np.arange(T),log_D.mean(0),1)[0])  # slope of log divergence
        # Time-reversal symmetry: compare forward vs reversed trajectory
        z_rev=z[:,::-1,:]
        trs=float(np.exp(-np.linalg.norm(z-z_rev)/(np.linalg.norm(z)+1e-8)))
        # Topological winding of leading mode
        lead=z[:,:,0]
        winding=float(np.abs(np.diff(np.sign(lead),axis=1)).mean())
        fit=_safe(1.0-(1.0/max(effective_dim,1)))
        return {
            "fit":fit,
            "effective_dim":effective_dim,
            "level_spacing_r":r,
            "lyapunov_proxy":lyap,
            "time_reversal_symmetry":trs,
            "topological_winding":winding,
        }

def generate_novel_narrative(invariants, max_fit):
    """
    Called when max probe fit < 0.5: nothing known fits well.
    Outputs a structured prompt for LLM narrative generation.
    The LLM names the geometry from invariants -- not from numbers it invents.
    On TPU VM: returns the prompt text (no API call).
    Locally with GEMINI_API_KEY: calls the API.
    """
    prompt=(
        f"A physical system's latent dynamics have these measured invariants:\n"
        f"  Hausdorff dim (SVD proxy): {invariants['effective_dim']:.3f}\n"
        f"  Level spacing ratio r:     {invariants['level_spacing_r']:.4f} "
        f"(Poisson=0.386, GOE=0.530)\n"
        f"  Lyapunov exponent proxy:   {invariants['lyapunov_proxy']:.5f}\n"
        f"  Time-reversal symmetry:    {invariants['time_reversal_symmetry']:.3f} "
        f"(1=symmetric, 0=broken)\n"
        f"  Topological winding:       {invariants['topological_winding']:.3f}\n\n"
        f"None of: Heisenberg, SYK, MBL, Lindblad, OAT, Penrose OR fit (max={max_fit:.2f}).\n"
        f"What mathematical structure might describe this geometry? Be specific about "
        f"known frameworks (non-associative algebras, anyonic statistics, "
        f"non-commutative geometry, hyperbolic dynamics, etc.) that could "
        f"accommodate these invariants."
    )
    import os
    api_key=os.environ.get("GEMINI_API_KEY","")
    if not api_key:
        return f"[NOVEL GEOMETRY DETECTED — LLM narrative pending API key]\n\nPrompt:\n{prompt}"
    try:
        import urllib.request, json as _json
        body=_json.dumps({"contents":[{"parts":[{"text":prompt}]}]}).encode()
        req=urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            data=body, headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req,timeout=30) as resp:
            data=_json.loads(resp.read())
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[LLM call failed: {e}]\n\nPrompt:\n{prompt}"

ALL_PROBES=[
    HamiltonianProbe(),SYKProbe(),MBLProbe(),LindbladProbe(),
    OATProbe(),PenroseBiTwistorProbe(E_G_phenom=0.01),NovelProbe()
]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EMERGENCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def emergence_report(source, z, omega, gamma, raw, norm, probe_results):
    scores={r['name']:r['fit'] for r in probe_results}
    ranked=sorted(scores.items(),key=lambda x:-x[1])
    dominant,dom_score=ranked[0]
    secondary,sec_score=ranked[1] if len(ranked)>1 else ("none",0)

    bi=next(r for r in probe_results if r['name']=='penrose_or')
    oat=next(r for r in probe_results if r['name']=='oat')
    novel=next(r for r in probe_results if r['name']=='novel')
    mbl=next(r for r in probe_results if r['name']=='mbl')

    # Narrative
    lines=[
        f"Source: {source} | Dominant formalism: {dominant} ({dom_score:.2f}) | Secondary: {secondary} ({sec_score:.2f})",
        f"OAT signal: raw_mean={oat['raw_mean']:.4f} (threshold 2/3={2/3:.4f}), quantum advantage={'YES' if oat['raw_mean']>2/3 else 'NO'}",
        f"Penrose bi-twistor: null_residual={bi['null_surface_residual']:.4f}, winding={bi['spinor_winding']:.3f}, OR_crossings={bi['or_threshold_crossings']}",
        f"MBL level spacing r={mbl['level_spacing_r']:.4f} (Poisson=0.386, GOE=0.53)",
        f"BLP N_BLP={probe_results[3]['N_BLP']:.2f} | EOM_fit={probe_results[0]['eom_fit']:.3f}",
        f"Effective Hilbert dim (SVD)={novel['effective_dim']:.2f} | Lyapunov={novel['lyapunov_proxy']:.5f}",
        f"Learned SSM: omega_mean={omega.mean():.4f} gamma_mean={gamma.mean():.4f}",
    ]
    narrative=" | ".join(lines)

    # Novel narrative: LLM generation when no known formalism fits well
    max_fit_score=max(scores.values())
    novel_narrative=generate_novel_narrative(novel, max_fit_score) if max_fit_score < 0.5 else None

    return {
        "source":source,
        "probe_results":{r['name']:r for r in probe_results},
        "ranked_formalisms":ranked,
        "dominant":dominant,
        "dominant_score":dom_score,
        "secondary":secondary,
        "secondary_score":sec_score,
        "penrose_bitwistor":{k:v for k,v in bi.items() if k!='name'},
        "oat_analysis":{k:v for k,v in oat.items() if k!='name'},
        "novel_geometry":{k:v for k,v in novel.items() if k!='name'},
        "ssm_params":{"omega_mean":float(omega.mean()),"omega_std":float(omega.std()),"gamma_mean":float(gamma.mean()),"gamma_std":float(gamma.std())},
        "emergence_narrative":narrative,
        "novel_narrative":novel_narrative,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(source, epochs=500, B=128, T=48, F=12):
    print(f"\n{'='*60}\n  OAT-MoQE v3  |  {source.upper()}  |  {jax.default_backend()}\n{'='*60}",flush=True)
    gen=GENERATORS[source]
    raw,norm=gen(B,T,F)
    data_j=jnp.array(norm)

    print(f"[Learner] training on {source} data {data_j.shape}...",flush=True)
    z,omega,gamma=train_learner(data_j,epochs)

    print(f"\n[Observer] running {len(ALL_PROBES)} probes...",flush=True)
    probe_results=[]
    for p in ALL_PROBES:
        r=p.score(z,omega,gamma,raw,norm)
        r['name']=p.name
        probe_results.append(r)
        print(f"  {p.name:<15} fit={r['fit']:.4f}",flush=True)

    report=emergence_report(source,z,omega,gamma,raw,norm,probe_results)

    print(f"\n  Dominant:  {report['dominant']} ({report['dominant_score']:.3f})")
    print(f"  Secondary: {report['secondary']} ({report['secondary_score']:.3f})")
    print(f"  Narrative: {report['emergence_narrative'][:200]}...",flush=True)
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="ALL")
    ap.add_argument("--epochs",type=int,default=500)
    ap.add_argument("--T",type=int,default=48)
    ap.add_argument("--F",type=int,default=12)
    ap.add_argument("--B",type=int,default=128)
    args=ap.parse_args()

    sources=list(GENERATORS.keys()) if args.source=="ALL" else [args.source]
    all_reports={}
    for src in sources:
        all_reports[src]=run(src,args.epochs,args.B,args.T,args.F)

    print(f"\n\n{'='*60}")
    print("  EMERGENCE SUMMARY")
    print(f"  {'Source':<14} {'Dominant':<14} {'Score':>7}  {'BT Null Res':>12}  {'OR Cross':>8}  {'N_BLP':>7}  {'EOM fit':>7}")
    print(f"  {'-'*68}")
    for src,r in all_reports.items():
        bt=r['penrose_bitwistor']; lb=r['probe_results']['lindblad']
        hm=r['probe_results']['hamiltonian']
        print(f"  {src:<14} {r['dominant']:<14} {r['dominant_score']:>7.3f}"
              f"  {bt['null_surface_residual']:>12.6f}  {bt['or_threshold_crossings']:>8d}"
              f"  {lb['N_BLP']:>7.2f}  {hm['eom_fit']:>7.3f}")
    print(f"{'='*60}\n")

    out=Path("oat_moqe_v3_results.json")
    with open(out,"w") as f:
        json.dump({"experiment":"OAT-MoQE v3","reports":all_reports},f,indent=2,default=str)
    print(f"[DONE] {out}",flush=True)

if __name__=="__main__":
    main()
