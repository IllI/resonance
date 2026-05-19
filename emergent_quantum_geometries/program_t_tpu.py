"""
Program T -- Spatially Structured Recoverability (TPU-native)
=============================================================
3x4 grid Hamiltonian with heterogeneous couplings.
Alice (top-left corner) injects a hidden state.
Bob (configurable distance) attempts reconstruction.
Probe tomography conditions controller priors.
CRF (Conditional Recovery Fidelity) is measured post-hoc only.
Controller sees ONLY local Alice-region observables (causal locality).
"""
import argparse, json, os, sys, time
import numpy as np
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_l_tpu import (
    _get_eigh, CLIFFORDS, N_CLIFF,
    select_backend, _jax_device_summary,
    kron_site, von_neumann,
)
from program_s1_tpu import (
    _apply_U_psi_jax, _apply_site_gate_jax,
    _apply_site_gate_static_site, _apply_noise_jax, _get_coherence,
)

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Grid Hamiltonian
# ---------------------------------------------------------------------------

def build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=0.0,
                            W_disorder=0.0, disorder_sites=None,
                            protected_bonds=None, seed=42):
    N = Lx * Ly
    dim = 2 ** N
    rng = np.random.default_rng(seed)
    H = np.zeros((dim, dim), dtype=np.complex64)

    def site(x, y): return y * Lx + x

    _sx = np.array([[0,1],[1,0]], dtype=np.complex64)
    _sy = np.array([[0,-1j],[1j,0]], dtype=np.complex64)
    _sz = np.array([[1,0],[0,-1]], dtype=np.complex64)

    bonds = []
    for y in range(Ly):
        for x in range(Lx):
            s = site(x, y)
            if x + 1 < Lx: bonds.append((s, site(x+1, y)))
            if y + 1 < Ly: bonds.append((s, site(x, y+1)))

    for (i, j) in bonds:
        if protected_bonds and (i, j) in protected_bonds:
            J = J_mean
        else:
            J = float(rng.normal(J_mean, J_std)) if J_std > 0 else J_mean
        J = max(0.1, J)
        XX = kron_site(_sx, i, N) @ kron_site(_sx, j, N)
        YY = kron_site(_sy, i, N) @ kron_site(_sy, j, N)
        ZZ = kron_site(_sz, i, N) @ kron_site(_sz, j, N)
        H += J * (XX + YY + ZZ)

    d_sites = disorder_sites if disorder_sites else []
    for s in d_sites:
        h = float(rng.uniform(-W_disorder, W_disorder))
        H += h * kron_site(_sz, s, N)

    return H.astype(np.complex64)


GRID_MODELS = {
    "GridXXZ_uniform": lambda N, Lx, Ly, seed:
        build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=0.0, seed=seed),
    "GridXXZ_hetero": lambda N, Lx, Ly, seed:
        build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=0.5, seed=seed),
    "GridXXZ_pockets": lambda N, Lx, Ly, seed:
        build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=0.3, W_disorder=2.0,
                               disorder_sites=[0,1,Lx,Lx+1], seed=seed),
    "GridXXZ_protected": lambda N, Lx, Ly, seed:
        build_grid_hamiltonian(Lx, Ly, J_mean=1.0, J_std=0.5,
                               protected_bonds=[(0,1),(0,Lx),(1,Lx+1)], seed=seed),
}

# ---------------------------------------------------------------------------
# Alice / Bob regions and distance classes
# ---------------------------------------------------------------------------

def get_alice_sites(Lx): return [0, 1]

def get_bob_candidates(Lx, Ly):
    def site(x, y): return y * Lx + x
    def mdist(x, y): return x + y
    short, medium, long_ = [], [], []
    for y in range(Ly):
        for x in range(Lx):
            d = mdist(x, y)
            entry = (site(x, y), d)
            if d <= 2 and d > 0: short.append(entry)
            elif 3 <= d <= 5: medium.append(entry)
            elif d >= 6: long_.append(entry)
    return {"short": short, "medium": medium, "long": long_}

def pick_bob_anchor(candidates, rng):
    result = {}
    for cls, lst in candidates.items():
        if lst:
            chosen = lst[len(lst)//2]
            result[cls] = chosen[0], chosen[1]
    return result

# ---------------------------------------------------------------------------
# Probe Tomography (Phase T0)
# ---------------------------------------------------------------------------

def run_probe_stage(evals_np, evecs_np, N, dt, alice_sites):
    dim = 2 ** N
    eps_angles = [np.pi/8, np.pi/4, np.pi/2]
    n_obs = 2 * N  # local X and Z for all qubits

    def evolve_psi(psi, t):
        p_e = evecs_np.conj().T @ psi
        p_e = p_e * np.exp(-1j * evals_np * t)
        return evecs_np @ p_e

    def local_coherence(psi):
        obs = []
        for i in range(N):
            stride = 2 ** (N - 1 - i)
            pr = psi.reshape(-1, 2 * stride)
            lo, hi = pr[:, :stride], pr[:, stride:]
            obs.append(float(2.0 * np.real(np.sum(lo * np.conj(hi)))))
        idx = np.arange(dim)
        for i in range(N):
            bm = ((idx >> (N - 1 - i)) & 1).astype(float)
            obs.append(float(np.real(np.sum(psi * np.conj(psi) * (1 - 2*bm)))))
        return np.array(obs)

    psi_ref = np.zeros(dim, dtype=np.complex64)
    psi_ref[0] = 1.0
    O_ref = local_coherence(evolve_psi(psi_ref, dt))

    susceptibility = np.zeros((len(alice_sites), n_obs), dtype=np.float32)
    damping_rates = np.zeros(len(alice_sites), dtype=np.float32)

    for ai, site in enumerate(alice_sites):
        resp_sum = np.zeros(n_obs)
        for eps in eps_angles:
            rz = np.array([[np.exp(-1j*eps/2), 0],
                           [0, np.exp(1j*eps/2)]], dtype=np.complex64)
            stride = 2 ** (N - 1 - site)
            psi_p = psi_ref.reshape(-1, 2*stride).copy()
            lo, hi = psi_p[:, :stride].copy(), psi_p[:, stride:].copy()
            psi_p[:, :stride] = rz[0,0]*lo + rz[0,1]*hi
            psi_p[:, stride:] = rz[1,0]*lo + rz[1,1]*hi
            psi_p = psi_p.reshape(dim)
            O_p = local_coherence(evolve_psi(psi_p, dt))
            resp_sum += np.abs(O_p - O_ref) / (eps + 1e-9)
        susceptibility[ai] = resp_sum / len(eps_angles)
        O_late = local_coherence(evolve_psi(psi_ref, 2*dt))
        damping_rates[ai] = float(np.mean(np.abs(O_late - O_ref)))

    J_frob = float(np.linalg.norm(susceptibility, "fro"))
    return {
        "susceptibility": susceptibility.tolist(),
        "damping_rates": damping_rates.tolist(),
        "J_frob": J_frob,
    }

# ---------------------------------------------------------------------------
# CRF: Conditional Recovery Fidelity (post-hoc only)
# ---------------------------------------------------------------------------

def compute_crf(psi_echoed, psi_alice_hidden, N, bob_site):
    dim = 2 ** N
    psi = np.array(psi_echoed)
    stride = 2 ** (N - 1 - bob_site)
    rho_loc = np.zeros((2, 2), dtype=np.complex64)
    for s0 in range(dim):
        amp_s = psi[s0]
        b0 = (s0 >> (N - 1 - bob_site)) & 1
        s1_base = s0 ^ (b0 << (N - 1 - bob_site))
        for b1 in range(2):
            s1 = s1_base | (b1 << (N - 1 - bob_site))
            rho_loc[b0, b1] += amp_s * np.conj(psi[s1])
    phi = np.array(psi_alice_hidden, dtype=np.complex64)
    crf = float(np.real(phi.conj() @ rho_loc @ phi))
    return max(0.0, min(1.0, crf))

# ---------------------------------------------------------------------------
# Controllers (causal: Alice-local observables only)
# ---------------------------------------------------------------------------

def ctrl_static_t(step, **_):
    if step == 0: return (1, 0)
    if step % 8 == 0: return (3, 0)
    return (0, 0)

def ctrl_dd_t(step, **_):
    seq = [1, 1, 2, 2]
    if step % 4 == 0: return (seq[(step // 4) % 4], 0)
    return (0, 0)

def ctrl_random_t(step, rng, **_):
    if rng.random() < 0.15:
        ci = int(rng.integers(1, N_CLIFF))
        si = int(rng.integers(0, 2))
        return (ci, si)
    return (0, 0)

def ctrl_sparse_predictive_t(step, obs_hist, cooldown_ref, tau, **_):
    if len(obs_hist) < 3 or cooldown_ref[0] > 0 or step < 3:
        if cooldown_ref[0] > 0: cooldown_ref[0] -= 1
        return (0, 0)
    O_t = np.array(obs_hist[-1])
    O_t1 = np.array(obs_hist[-2])
    O_t2 = np.array(obs_hist[-3])
    pred = O_t1 + (O_t1 - O_t2)
    err = float(np.max(np.abs(O_t - pred)))
    if err > tau:
        cooldown_ref[0] = 4
        return (3, 0)
    return (0, 0)

def ctrl_probe_sparse_t(step, obs_hist, cooldown_ref, tau, probe_info, **_):
    J_frob = probe_info.get("J_frob", 0.0)
    tau_adapted = tau / (1.0 + J_frob)
    return ctrl_sparse_predictive_t(step, obs_hist, cooldown_ref, tau_adapted)

CONTROLLERS_T = {
    "static":            ctrl_static_t,
    "dd":               ctrl_dd_t,
    "random":           ctrl_random_t,
    "sparse_predictive": ctrl_sparse_predictive_t,
    "probe_sparse":     ctrl_probe_sparse_t,
}

# ---------------------------------------------------------------------------
# Single trajectory runner (numpy, vmapped externally via batch loop)
# ---------------------------------------------------------------------------

def run_one_trajectory_t(psi0, psi_alice_hidden, evals_np, evecs_np,
                          N, dt, n_steps, noise_seq, ctrl_name,
                          tau, probe_info, bob_site, rng):
    cliffords_np = [np.eye(2, dtype=np.complex64)] + \
                   [np.array(g, dtype=np.complex64) for g in CLIFFORDS[1:]]

    def apply_cliff(psi, ci, site):
        g = cliffords_np[ci]
        return _apply_site_gate_static_site(psi, g, N, site)

    def evolve(psi, dt_):
        p_e = evecs_np.conj().T @ psi
        p_e = p_e * np.exp(-1j * evals_np * dt_)
        return evecs_np @ p_e

    def get_obs(psi):
        obs = []
        for site in [0, 1]:
            stride = 2 ** (N - 1 - site)
            pr = psi.reshape(-1, 2*stride)
            lo, hi = pr[:, :stride], pr[:, stride:]
            obs.append(float(2.0 * np.real(np.sum(lo * np.conj(hi)))))
        idx = np.arange(2**N)
        for site in [0, 1]:
            bm = ((idx >> (N - 1 - site)) & 1).astype(float)
            obs.append(float(np.real(np.sum(np.abs(psi)**2 * (1 - 2*bm)))))
        return obs

    def apply_noise_simple(psi, noise_row):
        gamma = np.clip(noise_row[0] * dt, 0.0, 0.5)
        dim = 2**N
        idx = np.arange(dim)
        n_ones = np.zeros(dim, dtype=np.float32)
        for q in range(N):
            n_ones += ((idx >> (N-1-q)) & 1).astype(np.float32)
        damp = (1.0 - gamma) ** n_ones
        psi = psi * damp.astype(np.complex64)
        nrm = np.linalg.norm(psi)
        return psi / nrm if nrm > 1e-12 else psi

    psi = psi0.copy()
    obs_hist = [get_obs(psi)]
    cooldown_ref = [0]
    n_interv = 0
    ctrl_fn = CONTROLLERS_T[ctrl_name]

    for step in range(n_steps):
        kwargs = dict(step=step, obs_hist=obs_hist, cooldown_ref=cooldown_ref,
                      tau=tau, probe_info=probe_info, rng=rng)
        ci, si = ctrl_fn(**kwargs)
        if ci > 0:
            psi = apply_cliff(psi, ci, si)
            n_interv += 1
        psi = evolve(psi, dt)
        psi = apply_noise_simple(psi, noise_seq[step])
        obs_hist.append(get_obs(psi))

    psi_final = psi.copy()
    psi_echoed = evolve(psi_final, -dt * n_steps)
    psi_ideal = evolve(psi0, dt * n_steps)

    fidelity = float(np.abs(np.vdot(psi_ideal, psi_final))**2)
    F_recover = float(np.abs(np.vdot(psi0, psi_echoed))**2)
    crf = compute_crf(psi_echoed, psi_alice_hidden, N, bob_site)

    return fidelity, F_recover, crf, n_interv, obs_hist

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment_t(args):
    t0 = time.time()
    backend, backend_info = select_backend("jax", require_tpu=getattr(args, "require_tpu", False))

    print("=" * 70)
    print("Program T -- Spatially Structured Recoverability")
    print("=" * 70)
    Lx, Ly = args.Lx, args.Ly
    N = Lx * Ly
    print(f"Grid: {Lx}x{Ly} = {N} qubits  T_max={args.T_max}  n_steps={args.n_steps}")

    dt = args.T_max / args.n_steps
    rng = np.random.default_rng(args.seed)

    alice_sites = get_alice_sites(Lx)
    bob_cands = get_bob_candidates(Lx, Ly)

    all_records = []
    summary = {}

    for model_name in args.models:
        print(f"\n== Model: {model_name} ==")
        H = GRID_MODELS[model_name](N, Lx, Ly, args.seed)
        evals_np, evecs_np = _get_eigh(H, (model_name, N, args.seed))

        print(f"  [PROBE] Running probe tomography...")
        probe_info = run_probe_stage(evals_np, evecs_np, N, dt, alice_sites)
        tau_adapted = args.tau_thresh / (1.0 + probe_info["J_frob"])
        print(f"  [PROBE] J_frob={probe_info['J_frob']:.3f}  tau_adapted={tau_adapted:.4f}")

        for dist_class, bob_list in bob_cands.items():
            if not bob_list: continue
            bob_site, bob_dist = bob_list[len(bob_list)//2]
            print(f"\n  -- distance={dist_class} (Bob=site {bob_site}, dist={bob_dist}) --")

            ctrl_records = {c: [] for c in args.controllers}

            for b in range(args.B):
                seed_b = args.seed + b * 1000
                pair_rng = np.random.default_rng(seed_b + 111)
                noise_rng = np.random.default_rng(seed_b + 222)

                theta = float(pair_rng.uniform(0, np.pi))
                phi_angle = float(pair_rng.uniform(0, 2*np.pi))
                psi_alice_1q = np.array([np.cos(theta/2),
                                          np.exp(1j*phi_angle)*np.sin(theta/2)],
                                         dtype=np.complex64)

                dim = 2 ** N
                psi0 = np.zeros(dim, dtype=np.complex64)
                psi0[0] = psi_alice_1q[0]
                psi0[dim//2] = psi_alice_1q[1]
                psi0 /= np.linalg.norm(psi0)

                gamma = float(rng.uniform(0.005, 0.02))
                noise_seq = np.zeros((args.n_steps, 8), dtype=np.float32)
                noise_seq[:, 0] = noise_rng.normal(gamma, 0.003, args.n_steps).clip(0)

                for ctrl_name in args.controllers:
                    ctrl_rng = np.random.default_rng(seed_b + hash(ctrl_name) % 10000)
                    fid, F_rec, crf, n_int, obs_hist = run_one_trajectory_t(
                        psi0, psi_alice_1q, evals_np, evecs_np,
                        N, dt, args.n_steps, noise_seq,
                        ctrl_name, args.tau_thresh if ctrl_name != "probe_sparse" else tau_adapted,
                        probe_info, bob_site, ctrl_rng
                    )
                    rho_path = float(bob_dist) / (n_int + 1e-9)
                    rec = {
                        "model": model_name, "controller": ctrl_name,
                        "dist_class": dist_class, "bob_site": bob_site,
                        "bob_dist": bob_dist, "B": b,
                        "fidelity": fid, "F_recover": F_rec,
                        "CRF": crf, "n_interventions": n_int,
                        "rho_path": rho_path,
                        "J_frob": probe_info["J_frob"],
                        "tau_adapted": tau_adapted,
                    }
                    ctrl_records[ctrl_name].append(rec)
                    all_records.append(rec)
                    print(f"    [{ctrl_name:20s}] b={b} fid={fid:.3f} F_rec={F_rec:.3f} CRF={crf:.3f} n_int={n_int} rho={rho_path:.1f}")

            key = f"{model_name}/{dist_class}"
            for ctrl_name in args.controllers:
                recs = ctrl_records[ctrl_name]
                summary[f"{key}/{ctrl_name}"] = {
                    "fidelity": float(np.mean([r["fidelity"] for r in recs])),
                    "F_recover": float(np.mean([r["F_recover"] for r in recs])),
                    "CRF": float(np.mean([r["CRF"] for r in recs])),
                    "n_interventions": float(np.mean([r["n_interventions"] for r in recs])),
                    "rho_path": float(np.mean([r["rho_path"] for r in recs])),
                    "n": len(recs),
                }

            if "sparse_predictive" in args.controllers and "probe_sparse" in args.controllers:
                sp_crf = summary.get(f"{key}/sparse_predictive", {}).get("CRF", 0.0)
                ps_crf = summary.get(f"{key}/probe_sparse", {}).get("CRF", 0.0)
                G_probe = ps_crf - sp_crf
                st_crf = summary.get(f"{key}/static", {}).get("CRF", 0.0)
                print(f"  [SUMMARY] {key}  G_probe={G_probe:+.3f}  CRF_static={st_crf:.3f}  CRF_probe={ps_crf:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_t_N{N}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args), "summary": summary,
            "records": all_records,
            "runtime_s": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=4)
    p.add_argument("--T-max", type=float, default=8.0, dest="T_max")
    p.add_argument("--n-steps", type=int, default=40, dest="n_steps")
    p.add_argument("--tau-thresh", type=float, default=0.08, dest="tau_thresh")
    p.add_argument("--B", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+",
                   default=["GridXXZ_uniform", "GridXXZ_hetero",
                            "GridXXZ_pockets", "GridXXZ_protected"])
    p.add_argument("--controllers", nargs="+",
                   default=["static", "dd", "random",
                            "sparse_predictive", "probe_sparse"])
    p.add_argument("--out-dir", default="program_t_results", dest="out_dir")
    p.add_argument("--require-tpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.Lx, args.Ly = 2, 4
        args.n_steps = 15
        args.B = 1
        args.models = ["GridXXZ_hetero"]
        args.controllers = ["static", "sparse_predictive", "probe_sparse"]

    run_experiment_t(args)
