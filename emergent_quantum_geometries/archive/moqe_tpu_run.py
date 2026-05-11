"""
moqe_tpu_run.py
================
Self-contained Mixture of Quantum Experts (MoQE) runner for TRC TPUs.
Bundles all ingestion, D-LiNOSS SSM, and MoQE training logic.

Usage on TPU VM:
    python3 moqe_tpu_run.py --source JILA_MBL
    python3 moqe_tpu_run.py --source JILA_TELEPORTATION
    python3 moqe_tpu_run.py --source SYK_ED
    python3 moqe_tpu_run.py --source ALL   # runs all sources sequentially
"""

import argparse
import json
import math
import sys
import urllib.request
import zipfile
import io
import glob
import os
import numpy as np
from pathlib import Path

# ─── JAX / Flax / Optax ───────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

DATA_DIR = Path("./data")

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_jila_mbl_data():
    """Download Moire Lattice MBL data from Zenodo if not already present."""
    real_dir = DATA_DIR / "jila_mbl" / "real" / "fig2_data"
    if real_dir.exists() and list(real_dir.glob("*.txt")):
        print("[DATA] JILA_MBL already downloaded.")
        return
    real_dir.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/api/records/15337680/files/data.zip/content"
    print("[DATA] Downloading JILA_MBL from Zenodo...")
    try:
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req, timeout=60)
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            z.extractall(real_dir)
        print("[DATA] JILA_MBL extracted successfully.")
    except Exception as e:
        print(f"[DATA] Download failed: {e} — will use synthetic fallback.")


def _synthetic_jila_teleportation(B, T, F, dt=0.1, chi=0.05):
    """Simulate JILA Ana Maria Rey 2025 EPR teleportation protocol."""
    t = np.arange(T) * dt
    rng = np.random.default_rng(7)
    N_ions_set = np.logspace(1, 2.5, F).astype(int)
    kappa = 0.005
    omega_N = chi * N_ions_set[None, :]
    fidelity = np.cos(omega_N * t[:, None])**2
    fidelity *= np.exp(-kappa * t[:, None])
    squeezing_boost = 0.1 * np.exp(-((t - T * dt * 0.5)**2) / (2 * (T * dt * 0.15)**2))
    fidelity += squeezing_boost[:, None]
    fidelity = np.clip(fidelity, 0.0, 1.0)
    fidelity -= fidelity.mean(axis=0, keepdims=True)
    fidelity /= (fidelity.std(axis=0, keepdims=True) + 1e-8)
    noise = rng.normal(0, 0.02, (B, T, F))
    return fidelity[None] + noise


def _synthetic_syk(B, T, F, dt=0.1, beta=5.0):
    t = np.arange(T) * dt
    lam_L = 2 * np.pi / beta
    k = np.arange(1, F + 1)
    omegas = np.pi * k / (F + 1)
    gammas = lam_L * np.ones(F)
    sig = np.sum(np.exp(-gammas[None, :] * t[:, None]) * np.cos(omegas[None, :] * t[:, None]), axis=1)
    sig /= (np.std(sig) + 1e-8)
    rng = np.random.default_rng(42)
    sig_2d = np.tile(sig[:T, None], (1, F))
    noise = rng.normal(0, 0.02, (B, T, F))
    return sig_2d[None] + noise


def _load_jila_mbl(B, T, F, dt=0.1):
    """Load real Zenodo MBL data with physics-informed exponential interpolation."""
    real_dir = DATA_DIR / "jila_mbl" / "real" / "fig2_data"
    files = sorted(glob.glob(str(real_dir / "imbalanceTimeDecay*.txt")))
    if not files:
        print("[DATA] No JILA_MBL files — using synthetic MBL fallback.")
        return _synthetic_mbl(B, T, F, dt)
    print(f"[DATA] Loaded {len(files)} JILA_MBL voltage sweep(s), interpolating to T={T}...")
    interp_signals = []
    for f in files:
        try:
            data = np.loadtxt(f, skiprows=1)
            if data.ndim < 2 or data.shape[0] < 3:
                continue
            t_raw, I_raw = data[:, 0], data[:, 1]
            C_floor = max(I_raw[-1] - 0.02, 0.0)
            I_shifted = np.clip(I_raw - C_floor, 1e-6, None)
            coeffs = np.polyfit(t_raw, np.log(I_shifted), 1)
            Gamma, A = -coeffs[0], np.exp(coeffs[1])
            t_interp = np.linspace(t_raw[0], t_raw[-1], T)
            I_interp = np.clip(A * np.exp(-Gamma * t_interp) + C_floor, 0.0, 1.0)
            interp_signals.append(I_interp)
        except Exception:
            continue
    spectral_raw = np.stack(interp_signals, axis=1)  # (T, n_sweeps)
    # Tile or truncate to F features
    if spectral_raw.shape[1] < F:
        pad = np.tile(spectral_raw, (1, F // spectral_raw.shape[1] + 1))[:, :F]
        spectral = pad
    else:
        spectral = spectral_raw[:, :F]
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, (B, T, F))
    return spectral[None] + noise


def _synthetic_mbl(B, T, F, dt=0.1):
    t = np.arange(T) * dt
    rng = np.random.default_rng(1)
    omegas = rng.uniform(0.5, 2.5, F)
    gammas = rng.exponential(0.01, F)
    sig = np.sum(np.exp(-gammas[None, :] * t[:, None]) * np.cos(omegas[None, :] * t[:, None]), axis=1)
    sig /= (np.std(sig) + 1e-8)
    sig_2d = np.tile(sig[:T, None], (1, F))
    rng2 = np.random.default_rng(99)
    noise = rng2.normal(0, 0.03, (B, T, F))
    return sig_2d[None] + noise


def load_data(source: str, batch_size=128, T=32, F=32, dt=0.1):
    if source == "JILA_MBL":
        fetch_jila_mbl_data()
        return _load_jila_mbl(batch_size, T, F, dt)
    elif source == "JILA_TELEPORTATION":
        return _synthetic_jila_teleportation(batch_size, T, F, dt)
    elif source == "SYK_ED":
        return _synthetic_syk(batch_size, T, F, dt)
    else:
        raise ValueError(f"Unknown source: {source}")


# ═══════════════════════════════════════════════════════════════════════════════
#  D-LiNOSS QUANTUM DAMPER (MoQE)
# ═══════════════════════════════════════════════════════════════════════════════

class DLinOSSQuantumDamper(nn.Module):
    """
    D-LiNOSS Quantum Damper with learnable (omega_k, gamma_k) per mode.

    Physics:
      phi_k(t) = exp(-gamma_k * t) * cos(omega_k * t)   [damped oscillator basis]

    gamma_k is the quantum damping rate — compared against:
      - Heisenberg: gamma_k ≈ 0     (unitary, no dissipation)
      - Lindblad:   gamma_k = const  (uniform thermal bath)
      - Penrose OR: gamma_k = E_G/ħ  (gravity-driven collapse)
      - SYK:        gamma_k = λ_L   (Lyapunov / max-chaos rate)
      - MBL:        gamma_k ≈ 0, omega_k disordered (localization)

    The NaN-stable implementation:
      - Normalises t to [0, 1] so exp(-gamma * t) never underflows for gamma < 4
      - Initialises log_gamma = -2 → gamma ≈ 0.13 (small physical damping)
      - Projects temporal-mean of input to mode coefficients via a single W
        (avoids the large (B,T,state_dim) × (T,state_dim) einsum that caused
        gradient explosion in the original sequential formulation)
    """
    state_dim:    int = 32
    n_frameworks: int = 5

    @nn.compact
    def __call__(self, x):
        B, T, F = x.shape

        # ── Signal Normalization ─────────────────────────────────────────────
        # Crucial for TPU stability to keep activations in a reasonable range
        x = nn.LayerNorm(use_scale=False, use_bias=False)(x)

        # ── Learned quantum SSM parameters ───────────────────────────────────
        # Using softplus + clip is more stable than exp() on TPU
        raw_omega = self.param('raw_omega', nn.initializers.constant(0.5), (self.state_dim,))
        raw_gamma = self.param('raw_gamma', nn.initializers.constant(0.1), (self.state_dim,))

        # Physical ranges: omega in [0.01, 10.0], gamma in [0.001, 2.0]
        omega_k = jnp.clip(jax.nn.softplus(raw_omega), 0.01, 10.0)
        gamma_k = jnp.clip(jax.nn.softplus(raw_gamma), 0.001, 2.0)

        # Normalised time axis [0, 1]
        t = jnp.linspace(0.0, 1.0, T)              # (T,)

        # ── Damped oscillator basis (the SSM impulse response) ───────────────
        # phi_k(t) = exp(-gamma_k * t) * cos(omega_k * t)
        phi = (jnp.exp(-gamma_k[None, :] * t[:, None])   # (T, state_dim)
               * jnp.cos(omega_k[None, :] * t[:, None]))

        # ── Input → mode coefficients ─────────────────────────────────────────
        # Project temporal mean of input; tanh bounds coefficients → no blow-up
        x_mean = jnp.mean(x, axis=1)               # (B, F)
        # Scale initial weights down for stability
        W_init = nn.initializers.glorot_normal()
        W_alpha = self.param('W_alpha', lambda rng, shape: W_init(rng, shape) * 0.1, (F, self.state_dim))
        alpha = jnp.tanh(jnp.dot(x_mean, W_alpha, precision=jax.lax.Precision.HIGHEST))

        # ── Reconstruct signal via SSM basis ──────────────────────────────────
        # y_latent(b, t, k) = alpha(b, k) * phi(t, k)
        y_latent = alpha[:, None, :] * phi[None, :, :]  # (B, T, state_dim)

        W_out = self.param('W_out', lambda rng, shape: W_init(rng, shape) * 0.1, (self.state_dim, F))
        y_recon = jnp.dot(y_latent, W_out, precision=jax.lax.Precision.HIGHEST)

        # ── MoQE logits: derived from SSM latent state (quantum-grounded) ────
        latent_mean = jnp.mean(y_latent, axis=(0, 1))   # (state_dim,)
        W_moqe = self.param('W_moqe', lambda rng, shape: W_init(rng, shape) * 0.1, (self.state_dim, self.n_frameworks))
        moqe_logits = jnp.dot(latent_mean, W_moqe, precision=jax.lax.Precision.HIGHEST)

        return y_latent, y_recon, omega_k, gamma_k, moqe_logits


# ═══════════════════════════════════════════════════════════════════════════════
#  MoQE LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_qpc_losses(omega_k, gamma_k, beta=5.0):
    lambda_L = 2 * jnp.pi / beta
    eg_threshold = 0.8
    qpc_coherence = jnp.mean(jnp.exp(-jnp.clip(gamma_k, 0, 5.0)))

    losses = jnp.array([
        jnp.mean(jnp.clip(gamma_k, 0, 5)**2),                              # Heisenberg: γ→0
        jnp.mean((jnp.clip(gamma_k, 0, 5) - 0.30)**2),                     # Lindblad:   γ=uniform
        jnp.abs(qpc_coherence - eg_threshold),                              # Penrose OR: QPC→E_G
        jnp.mean((jnp.clip(gamma_k, 0, 5) - lambda_L)**2),                 # SYK:        γ=λ_L
        jnp.mean(jnp.clip(gamma_k, 0, 5)**2) + 0.1 * jnp.std(omega_k),    # MBL:        γ≈0, ω disordered
    ])
    return losses


def train_step(variables, opt_state, batch, optimizer, damper_net):
    def loss_fn(params):
        _, y_recon, omega_k, gamma_k, moqe_logits = damper_net.apply(
            {'params': params}, batch
        )
        loss_recon = jnp.mean((y_recon - batch)**2)
        # Use log_softmax for numerical stability
        log_probs = jax.nn.log_softmax(moqe_logits)
        probs = jnp.exp(log_probs)
        qpc_losses = compute_qpc_losses(omega_k, gamma_k)
        loss_moqe = jnp.dot(probs, qpc_losses)
        total = loss_recon + 0.1 * loss_moqe
        # Final NaN safety guard
        return jnp.where(jnp.logical_or(jnp.isnan(total), jnp.isinf(total)), 1e5, total)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(variables['params'])
    # Strict gradient clipping for TPU
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -0.1, 0.1), grads)
    updates, new_opt_state = optimizer.update(grads, opt_state, variables['params'])
    new_params = optax.apply_updates(variables['params'], updates)
    return {'params': new_params}, new_opt_state, loss


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

FRAMEWORK_NAMES = ["heisenberg", "lindblad", "penrose_or", "syk", "mbl"]
FRAMEWORK_DESCS = [
    "Unitary evolution, no decoherence.",
    "Thermal Lindblad bath.",
    "Penrose OR. Gravitational collapse.",
    "SYK holographic chaos.",
    "Many-body localized.",
]

def run_experiment(source: str, epochs=200, batch_size=128, T=32, F=32, temperature=0.05):
    print("\n" + "="*60)
    print(f"  MoQE Experiment — Source: {source}")
    print(f"  JAX devices: {jax.devices()}")
    print("="*60)

    # Phase 1: Load data
    print("\n[Phase 1] Ingesting data...")
    data_np = load_data(source, batch_size, T, F)
    drift_data = jnp.array(data_np, dtype=jnp.float32)
    print(f"  Tensor shape: {drift_data.shape}")
    print(f"  Signal range: [{float(jnp.min(drift_data)):.4f}, {float(jnp.max(drift_data)):.4f}]")

    # Phase 2: Init model
    print("\n[Phase 2] Initializing D-LiNOSS MoQE...")
    damper_net = DLinOSSQuantumDamper()
    rng = jax.random.PRNGKey(0)
    variables = damper_net.init(rng, drift_data)

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.1),  # Stricter norm clipping
        optax.adam(learning_rate=1e-4)   # Lower LR for stability
    )
    opt_state = optimizer.init(variables['params'])

    # Phase 3: Training loop
    print(f"\n[Phase 3] Training for {epochs} epochs...")
    train_step_jit = jax.jit(lambda v, o, b: train_step(v, o, b, optimizer, damper_net))

    for epoch in range(epochs):
        variables, opt_state, loss = train_step_jit(variables, opt_state, drift_data)
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {float(loss):.4f}")

    # Phase 4: Extract MoQE probabilities
    print("\n[Phase 4] Extracting emergent framework probabilities...")
    _, _, omega_k, gamma_k, moqe_logits = damper_net.apply(variables, drift_data)
    # Use standard softmax without temperature to avoid underflow
    emergent_probs = jax.nn.softmax(moqe_logits)
    # Apply temperature post-hoc for display sharpening
    sharpened = jax.nn.softmax(moqe_logits / temperature)
    probs_np = np.where(np.isnan(np.array(sharpened)), np.array(emergent_probs), np.array(sharpened))
    # Final NaN guard — fall back to uniform if still NaN
    if np.any(np.isnan(probs_np)):
        probs_np = np.array(emergent_probs)
    if np.any(np.isnan(probs_np)):
        probs_np = np.ones(len(FRAMEWORK_NAMES)) / len(FRAMEWORK_NAMES)

    # Phase 5: Compute entropy & confidence
    entropy = float(-jnp.sum(emergent_probs * jnp.log(emergent_probs + 1e-9)))
    max_entropy = math.log(len(FRAMEWORK_NAMES))
    confidence = 1.0 - entropy / max_entropy

    winner_idx = int(np.argmax(probs_np))

    print("\n" + "="*60)
    print("  MIXTURE OF QUANTUM EXPERTS  —  Emergent Probabilities")
    print("="*60)
    print(f"  {'Framework':<18} {'P(match)':>10}  Description")
    print("-"*60)
    sorted_idx = np.argsort(probs_np)[::-1]
    for i in sorted_idx:
        winner_marker = "  ◄ WINNER" if i == winner_idx else ""
        print(f"  {FRAMEWORK_NAMES[i]:<18} {probs_np[i]*100:>8.2f}%  {FRAMEWORK_DESCS[i]}{winner_marker}")
    print("-"*60)
    print(f"  Decision entropy: {entropy:.4f}  (max = {max_entropy:.4f} for {len(FRAMEWORK_NAMES)} frameworks)")
    print(f"  Confidence:       {confidence*100:.1f}%")
    print("="*60)

    # Physical observables
    omega_np = np.array(omega_k)
    gamma_np = np.array(gamma_k)
    print(f"\n  Learned ω_k: mean={omega_np.mean():.4f}  std={omega_np.std():.4f}")
    print(f"  Learned γ_k: mean={gamma_np.mean():.4f}  std={gamma_np.std():.4f}")
    lambda_L = 2 * math.pi / 5.0
    print(f"  SYK Lyapunov λ_L = {lambda_L:.4f}  |  γ_mean - λ_L = {abs(gamma_np.mean() - lambda_L):.4f}")

    result = {
        "source": source,
        "devices": str(jax.devices()),
        "probabilities": {FRAMEWORK_NAMES[i]: float(probs_np[i]) for i in range(len(FRAMEWORK_NAMES))},
        "winner": FRAMEWORK_NAMES[winner_idx],
        "confidence": confidence,
        "entropy": entropy,
        "omega_mean": float(omega_np.mean()),
        "gamma_mean": float(gamma_np.mean()),
        "lambda_L": lambda_L,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="ALL",
                        choices=["JILA_MBL", "JILA_TELEPORTATION", "SYK_ED", "ALL"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch",  type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()

    sources = ["JILA_MBL", "JILA_TELEPORTATION", "SYK_ED"] if args.source == "ALL" else [args.source]

    all_results = {}
    for src in sources:
        result = run_experiment(src, epochs=args.epochs, batch_size=args.batch,
                                temperature=args.temperature)
        all_results[src] = result

    # Save results to JSON
    out_path = Path("moqe_tpu_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Results saved to {out_path}")
    print(json.dumps(all_results, indent=2))
