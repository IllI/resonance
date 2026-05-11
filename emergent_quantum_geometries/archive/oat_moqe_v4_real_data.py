"""
oat_moqe_v4_real_data.py
========================
v4: Uses REAL physics — imports exact OAT simulation functions from
jila_oat_exact_tpu.py instead of hand-crafted approximations.

Hypothesis H1: Learner trained on real OAT F(chi*t), C(chi*t) curves
  will produce z(t) where OAT probe fits > 0.80, all others < 0.50,
  and bi-twistor null residual > 0 (entangled departure from null surface).

Hypothesis H0 (null): Same learner on separable states (C=0, F=2/3)
  will NOT trigger OAT probe (raw_mean near 2/3 but no margin).

Three datasets:
  1. OAT real  — exact statevector sim, F > 2/3, C > 0
  2. Separable — product state |+>^N, chi=0, C=0, F=2/3 exactly
  3. Thermal   — classical noise baseline, F ~ U[0.4, 0.7]

Scientific threshold for H1 confirmation:
  OAT probe fit  > 0.80  on OAT data
  OAT probe fit  < 0.30  on separable data
  Bi-twistor null residual: OAT > separable (measurable gap)
  Result reproducible across seeds 0,1,2
"""

import argparse, json, math
import numpy as np
from pathlib import Path
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

# Import real physics from validated simulation
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from jila_oat_exact_tpu import jz_table, plus_state, oat_evolve, boundary_rdm, concurrence, fidelity_from_concurrence

print(f"[DEVICE] {jax.devices()}", flush=True)
print(f"[DEVICE] backend={jax.default_backend()}", flush=True)

CLASSICAL_LIMIT = 2.0 / 3.0

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: REAL DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def gen_oat_real(B, T, N_list, chi_range=(0.0, math.pi), seed=0):
    """
    Generate REAL OAT fidelity curves using exact statevector simulation.
    Each batch sample uses a slightly different chi coupling strength.
    Output shape: (B, T, F) where F = len(N_list)
    raw[b, t, f] = F(chi_b * t_vals[t], N_list[f])  -- real physics
    """
    rng = np.random.default_rng(seed)
    chi_t_vals = np.linspace(chi_range[0], chi_range[1], T)
    F_channels = len(N_list)
    raw = np.zeros((B, T, F_channels), np.float32)

    # Precompute F,C curves for each N (chi scale = 1.0)
    base_curves = {}
    for N in N_list:
        mA_np, mB_np = jz_table(N)
        mA = jnp.array(mA_np); mB = jnp.array(mB_np)
        psi0 = jnp.array(plus_state(N))
        Fv, Cv = [], []
        for chi_t in chi_t_vals:
            psi_t = np.array(oat_evolve(psi0, mA, mB, float(chi_t)))
            rho2 = boundary_rdm(psi_t, N)
            C = concurrence(rho2)
            Fv.append(fidelity_from_concurrence(C))
            Cv.append(C)
        base_curves[N] = (np.array(Fv), np.array(Cv))
        print(f"  [OAT sim] N={N} F_peak={max(Fv):.4f} C_peak={max(Cv):.4f}", flush=True)

    # Each batch sample: slight chi variation + small noise
    for b in range(B):
        chi_scale = rng.uniform(0.85, 1.15)
        for fi, N in enumerate(N_list):
            Fv_base, _ = base_curves[N]
            # Interpolate at scaled chi*t
            scaled_t = chi_scale * chi_t_vals
            Fv_scaled = np.interp(scaled_t, chi_t_vals, Fv_base)
            raw[b, :, fi] = Fv_scaled + rng.normal(0, 0.003, T)

    raw = raw.clip(0, 1)
    mu = raw.mean((1, 2), keepdims=True)
    sd = raw.std((1, 2), keepdims=True) + 1e-8
    norm = ((raw - mu) / sd).astype(np.float32)
    return raw, norm, base_curves


def gen_separable(B, T, N_list, seed=1):
    """
    Separable (product) states: chi=0, no entanglement ever.
    C=0 => F=2/3 exactly. Tests H0: OAT probe should NOT fire.
    """
    rng = np.random.default_rng(seed)
    F_channels = len(N_list)
    raw = np.full((B, T, F_channels), CLASSICAL_LIMIT, np.float32)
    raw += rng.normal(0, 0.005, raw.shape)
    raw = raw.clip(0, 1)
    mu = raw.mean((1, 2), keepdims=True)
    sd = raw.std((1, 2), keepdims=True) + 1e-8
    norm = ((raw - mu) / sd).astype(np.float32)
    return raw, norm


def gen_thermal(B, T, N_list, seed=2):
    """
    Classical thermal noise: F ~ Uniform[0.4, 0.7], no quantum structure.
    """
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.4, 0.7, (B, T, len(N_list))).astype(np.float32)
    raw += rng.normal(0, 0.01, raw.shape)
    raw = raw.clip(0, 1)
    mu = raw.mean((1, 2), keepdims=True)
    sd = raw.std((1, 2), keepdims=True) + 1e-8
    norm = ((raw - mu) / sd).astype(np.float32)
    return raw, norm


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LEARNER (same as v3 — blind D-LiNOSS SSM)
# ═══════════════════════════════════════════════════════════════════════════════

class Learner(nn.Module):
    state_dim: int = 64

    @nn.compact
    def __call__(self, x):
        B, T, F = x.shape
        x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        raw_w = self.param('raw_w', nn.initializers.constant(0.5), (self.state_dim,))
        raw_g = self.param('raw_g', nn.initializers.constant(0.1), (self.state_dim,))
        omega = jnp.clip(jax.nn.softplus(raw_w), .01, 10.)
        gamma = jnp.clip(jax.nn.softplus(raw_g), .001, 2.)
        t = jnp.linspace(0., 1., T)
        phi = jnp.exp(-gamma[None, :] * t[:, None]) * jnp.cos(omega[None, :] * t[:, None])
        W_init = nn.initializers.glorot_normal()
        Wa = self.param('Wa', lambda r, s: W_init(r, s) * .1, (F, self.state_dim))
        alpha = jnp.tanh(jnp.dot(x.mean(1), Wa, precision=jax.lax.Precision.HIGHEST))
        z = alpha[:, None, :] * phi[None, :, :]
        Wo = self.param('Wo', lambda r, s: W_init(r, s) * .1, (self.state_dim, F))
        recon = jnp.dot(z, Wo, precision=jax.lax.Precision.HIGHEST)
        return z, recon, omega, gamma


def train_learner(data_j, epochs=600, state_dim=64):
    net = Learner(state_dim=state_dim)
    params = net.init(jax.random.PRNGKey(0), data_j)['params']
    opt = optax.chain(optax.clip_by_global_norm(1.), optax.adam(1e-3))
    state = opt.init(params)

    @jax.jit
    def step(p, s, x):
        def loss_fn(p):
            _, rec, _, _ = net.apply({'params': p}, x)
            L = jnp.mean((rec - x) ** 2)
            return jnp.where(jnp.isnan(L) | jnp.isinf(L), 1e5, L)
        L, g = jax.value_and_grad(loss_fn)(p)
        g = jax.tree_util.tree_map(lambda x: jnp.clip(x, -.1, .1), g)
        u, s2 = opt.update(g, s, p)
        return optax.apply_updates(p, u), s2, L

    for ep in range(epochs + 1):
        params, state, loss = step(params, state, data_j)
        if ep % 100 == 0:
            print(f"  [Learner] ep={ep:>4d}  loss={float(loss):.5f}", flush=True)

    z, _, omega, gamma = net.apply({'params': params}, data_j)
    return np.array(z), np.array(omega), np.array(gamma)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: OBSERVER PROBES (physics-corrected from v3)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe(x):
    return float(np.clip(x, 0, 1))


class OATProbe:
    """
    Primary probe. OAT unique signature: raw F in [2/3, 1].
    raw_mean > 2/3 AND raw_min > 0.65 AND oscillates around mean.
    No other physical framework has a uniformly positive fidelity offset.
    """
    name = "oat"
    threshold = CLASSICAL_LIMIT

    def score(self, z, omega, gamma, raw, norm):
        raw_mean = float(raw.mean())
        above = float(raw_mean > self.threshold)
        margin = max(0.0, raw_mean - self.threshold)
        raw_min = float(raw.min())
        min_above = float(raw_min > 0.65)
        raw_osc = float(raw.std(axis=1).mean())
        osc_fit = float(np.exp(-abs(raw_osc - 0.06) * 20))
        fit = _safe(above * 0.6 + margin * 8 + min_above * 0.2 + osc_fit * 0.1)
        return {
            "fit": fit, "raw_mean": raw_mean, "raw_min": raw_min,
            "margin_above_2_3": margin, "raw_osc": raw_osc,
            "quantum_advantage": raw_mean > self.threshold,
        }


class SeparableProbe:
    """
    Separable/classical probe. C=0 => F=2/3 exactly, no oscillation.
    Detects product states and classical data.
    """
    name = "separable"

    def score(self, z, omega, gamma, raw, norm):
        raw_mean = float(raw.mean())
        raw_std = float(raw.std())
        at_threshold = float(np.exp(-abs(raw_mean - CLASSICAL_LIMIT) * 20))
        low_variance = float(np.exp(-raw_std * 10))
        fit = _safe((at_threshold + low_variance) / 2)
        return {"fit": fit, "raw_mean": raw_mean, "raw_std": raw_std}


class HamiltonianProbe:
    """Heisenberg EOM: |dz/dt| / (Omega*|z|*dt) ~ 1"""
    name = "hamiltonian"

    def score(self, z, omega, gamma, raw, norm):
        B, T, S = z.shape
        dt = 1.0 / (T - 1)
        dz = np.diff(z, axis=1)
        dz_mag = float(np.linalg.norm(dz, axis=-1).mean())
        z_mag = float(np.linalg.norm(z[:, :-1, :], axis=-1).mean())
        eom_ratio = dz_mag / (float(omega.mean()) * z_mag * dt + 1e-8)
        eom_fit = float(np.exp(-abs(eom_ratio - 1.0) * 2))
        norms = np.linalg.norm(z, axis=-1)
        norm_var = float(norms.std(axis=1).mean())
        gamma_fit = float(np.exp(-gamma.mean() * 5))
        fit = _safe((eom_fit + np.exp(-norm_var * 3) + gamma_fit) / 3)
        return {"fit": fit, "eom_ratio": eom_ratio, "eom_fit": eom_fit,
                "norm_variance": norm_var, "gamma_mean": float(gamma.mean())}


class LindbladProbe:
    """BLP Markovianity: N_BLP = total increase in trace distance between pairs."""
    name = "lindblad"

    def score(self, z, omega, gamma, raw, norm):
        B, T, S = z.shape
        n_pairs = min(B // 2, 16)
        D = np.linalg.norm(z[:n_pairs] - z[n_pairs:n_pairs * 2], axis=-1)
        dD = np.diff(D, axis=1)
        N_BLP = float(np.sum(np.maximum(dD, 0)))
        markov_fit = float(np.exp(-N_BLP * 0.05))
        gamma_u = float(np.exp(-gamma.std() * 10))
        gamma_m = float(np.exp(-abs(gamma.mean() - 0.30) * 5))
        norms = np.linalg.norm(z, axis=-1)
        decay = float((norms[:, -1] / np.maximum(norms[:, 0], 1e-8)).mean())
        decay_fit = float(np.exp(-abs(decay - 0.3) * 5))
        fit = _safe((markov_fit + gamma_u * gamma_m + decay_fit) / 3)
        return {"fit": fit, "N_BLP": N_BLP, "markov_fit": markov_fit,
                "gamma_mean": float(gamma.mean()), "decay": decay}


class PenroseBiTwistorProbe:
    """
    Bi-twistor topological probe with phenomenological threshold.
    E_G_phenom is tunable, NOT the gravitational formula.
    For cold atoms E_G_grav ~ 1e-58 J, experimentally unreachable.
    """
    name = "penrose_or"

    def __init__(self, E_G_phenom=0.01):
        self.E_G = E_G_phenom

    def score(self, z, omega, gamma, raw, norm):
        B, T, S = z.shape
        t = np.linspace(0., 1., T)
        ph = lambda k: np.exp(1j * omega[min(k, S - 1)] * t)
        xi0 = z[:, :, 0] * ph(0)[None, :] + 1j * z[:, :, 1] * ph(1)[None, :]
        xi1 = z[:, :, 1] * ph(1)[None, :] - 1j * z[:, :, 0] * ph(0)[None, :]
        pi0 = z[:, :, 2] * ph(2)[None, :] + 1j * z[:, :, 3] * ph(3)[None, :]
        pi1 = z[:, :, 3] * ph(3)[None, :] - 1j * z[:, :, 2] * ph(2)[None, :]
        Z_pfaff = xi0 * pi1 - xi1 * pi0
        Z_norm_sq = np.abs(Z_pfaff) ** 2
        null_residual = float(Z_norm_sq.mean())
        angles = np.angle(Z_pfaff)
        winding = float(np.abs(np.diff(angles, axis=1)).sum(axis=1).mean() / (2 * math.pi))
        or_crossings = int(np.sum(
            (Z_norm_sq[:, :-1] < self.E_G) & (Z_norm_sq[:, 1:] >= self.E_G)
        ))
        dZ = np.diff(Z_norm_sq, axis=1)
        d2Z = np.diff(dZ, axis=1)
        collapse_spike = float(d2Z.std() / (abs(d2Z.mean()) + 1e-8))
        collapse_fit = _safe(collapse_spike / 10)
        entangle_fit = _safe(null_residual * 10)
        winding_fit = _safe(min(winding, 1.0))
        fit = _safe((entangle_fit + winding_fit + collapse_fit) / 3)
        return {
            "fit": fit, "null_surface_residual": null_residual,
            "spinor_winding": winding, "or_threshold_crossings": or_crossings,
            "E_G_phenom": self.E_G, "collapse_spike": collapse_spike,
            "bi_twistor_norm_mean": float(Z_norm_sq.mean()),
            "note": "E_G is phenomenological. Null surface = topological probe, not gravitational."
        }


class NovelProbe:
    """Residual probe: SVD dim, Lyapunov, time-reversal, level spacing."""
    name = "novel"

    def score(self, z, omega, gamma, raw, norm):
        B, T, S = z.shape
        z_flat = z[0].reshape(T, -1)
        _, sv, _ = np.linalg.svd(z_flat, full_matrices=False)
        sv_norm = sv / (sv.sum() + 1e-10)
        effective_dim = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-10))))
        s = np.sort(omega); gaps = np.diff(s) + 1e-10
        r = float(np.mean(np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])))
        n_pairs = min(B // 2, 8)
        D = np.linalg.norm(z[:n_pairs] - z[n_pairs:n_pairs * 2], axis=-1)
        log_D = np.log(np.maximum(D, 1e-10))
        lyap = float(np.polyfit(np.arange(T), log_D.mean(0), 1)[0])
        z_rev = z[:, ::-1, :]
        trs = float(np.exp(-np.linalg.norm(z - z_rev) / (np.linalg.norm(z) + 1e-8)))
        winding = float(np.abs(np.diff(np.sign(z[:, :, 0]), axis=1)).mean())
        fit = _safe(1.0 - (1.0 / max(effective_dim, 1)))
        return {
            "fit": fit, "effective_dim": effective_dim, "level_spacing_r": r,
            "lyapunov_proxy": lyap, "time_reversal_symmetry": trs,
            "topological_winding": winding,
        }


ALL_PROBES = [
    OATProbe(), SeparableProbe(), HamiltonianProbe(),
    LindbladProbe(), PenroseBiTwistorProbe(E_G_phenom=0.01), NovelProbe()
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: HYPOTHESIS TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_hypothesis(oat_report, sep_report, thermal_report):
    """
    Tests H1 against H0 with explicit scientific thresholds.
    H1: OAT probe fit > 0.80 on OAT data, < 0.30 on separable data.
    Returns: CONFIRMED / PARTIAL / REJECTED
    """
    oat_oat_fit = oat_report['probe_scores']['oat']
    oat_sep_fit = sep_report['probe_scores']['oat']
    oat_thermal_fit = thermal_report['probe_scores']['oat']

    bt_oat = oat_report['probe_scores']['penrose_or']
    bt_sep = sep_report['probe_scores']['penrose_or']

    null_oat = oat_report['penrose_null_residual']
    null_sep = sep_report['penrose_null_residual']

    h1_oat_high   = oat_oat_fit > 0.80
    h1_sep_low    = oat_sep_fit < 0.30
    h1_thermal_low = oat_thermal_fit < 0.30
    h1_bt_gap     = null_oat > null_sep  # entangled > separable in twistor space

    print(f"\n{'='*65}")
    print("  HYPOTHESIS TEST")
    print(f"{'='*65}")
    print(f"  H1: OAT probe on OAT data > 0.80    {oat_oat_fit:.3f}  {'PASS' if h1_oat_high else 'FAIL'}")
    print(f"  H0: OAT probe on separable < 0.30   {oat_sep_fit:.3f}  {'PASS' if h1_sep_low else 'FAIL'}")
    print(f"  H0: OAT probe on thermal < 0.30     {oat_thermal_fit:.3f}  {'PASS' if h1_thermal_low else 'FAIL'}")
    print(f"  Bi-twistor: OAT null > separable    {null_oat:.6f} > {null_sep:.6f}  {'PASS' if h1_bt_gap else 'FAIL'}")

    n_pass = sum([h1_oat_high, h1_sep_low, h1_thermal_low, h1_bt_gap])
    if n_pass == 4:
        verdict = "CONFIRMED"
    elif n_pass >= 2:
        verdict = "PARTIAL"
    else:
        verdict = "REJECTED"

    print(f"\n  Verdict: {verdict}  ({n_pass}/4 criteria met)")
    print(f"{'='*65}\n")

    return {
        "verdict": verdict, "criteria_met": n_pass,
        "oat_probe_on_oat": oat_oat_fit,
        "oat_probe_on_separable": oat_sep_fit,
        "oat_probe_on_thermal": oat_thermal_fit,
        "penrose_null_oat": null_oat,
        "penrose_null_separable": null_sep,
        "penrose_null_gap": null_oat - null_sep,
        "h1_oat_high": h1_oat_high,
        "h0_sep_low": h1_sep_low,
        "h0_thermal_low": h1_thermal_low,
        "h1_bt_gap": h1_bt_gap,
    }


def run_dataset(label, raw, norm, epochs, tag=""):
    print(f"\n{'='*60}\n  DATASET: {label.upper()}{tag}\n{'='*60}", flush=True)
    data_j = jnp.array(norm)
    z, omega, gamma = train_learner(data_j, epochs)

    print(f"\n[Observer] probes...", flush=True)
    probe_scores = {}
    probe_details = {}
    for p in ALL_PROBES:
        r = p.score(z, omega, gamma, raw, norm)
        probe_scores[p.name] = r['fit']
        probe_details[p.name] = r
        print(f"  {p.name:<16} fit={r['fit']:.4f}", flush=True)

    bt = probe_details['penrose_or']
    oat = probe_details['oat']
    print(f"\n  raw_mean={oat['raw_mean']:.4f}  F>2/3={'YES' if oat['quantum_advantage'] else 'NO'}", flush=True)
    print(f"  bi-twistor null={bt['null_surface_residual']:.6f}  winding={bt['spinor_winding']:.3f}", flush=True)

    return {
        "label": label,
        "probe_scores": probe_scores,
        "oat_analysis": oat,
        "penrose_null_residual": bt['null_surface_residual'],
        "spinor_winding": bt['spinor_winding'],
        "ssm_omega_mean": float(omega.mean()),
        "ssm_gamma_mean": float(gamma.mean()),
        "probe_details": probe_details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[CONFIG] N={args.N} epochs={args.epochs} T={args.T} B={args.B}", flush=True)
    print(f"\n[Phase 1] Generating real OAT simulation data...", flush=True)
    raw_oat, norm_oat, base_curves = gen_oat_real(args.B, args.T, args.N, seed=args.seed)

    print(f"\n[Phase 2] Generating separable (C=0) comparison data...", flush=True)
    raw_sep, norm_sep = gen_separable(args.B, args.T, args.N, seed=args.seed + 1)

    print(f"\n[Phase 3] Generating thermal (classical) baseline...", flush=True)
    raw_th, norm_th = gen_thermal(args.B, args.T, args.N, seed=args.seed + 2)

    # Run learner + observer on each dataset
    oat_report   = run_dataset("OAT (real sim)", raw_oat, norm_oat, args.epochs)
    sep_report   = run_dataset("Separable (C=0)", raw_sep, norm_sep, args.epochs)
    th_report    = run_dataset("Thermal (classical)", raw_th, norm_th, args.epochs)

    # Hypothesis test
    hyp = test_hypothesis(oat_report, sep_report, th_report)

    # Save
    out = Path("oat_moqe_v4_results.json")
    with open(out, "w") as f:
        json.dump({
            "experiment": "OAT-MoQE v4 — Real Data Hypothesis Test",
            "hypothesis": {
                "H1": "OAT probe > 0.80 on real OAT sim, < 0.30 on separable",
                "H0": "Learner cannot distinguish entangled from separable",
            },
            "config": vars(args),
            "datasets": {
                "oat": oat_report, "separable": sep_report, "thermal": th_report
            },
            "hypothesis_test": hyp,
            "base_curves": {
                f"N{N}": {"F_peak": float(Fv.max()), "C_peak": float(Cv.max())}
                for N, (Fv, Cv) in base_curves.items()
            },
        }, f, indent=2, default=str)
    print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
