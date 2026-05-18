"""
analyze_program_n10a.py
=======================
N10a: W3 confidence characterization on B=50 seeds.

Computes:
  1. Bootstrap CI for delta_tau (W3 vs static)
  2. Intervention efficiency: eta_ctrl = delta_tau / N_nonidentity per seed
  3. Intervention timing distribution (which steps fire)
  4. Gate-conditioned observable drift vs no-gate drift

Usage:
    python analyze_program_n10a.py <results.json>
"""

import json, sys
import numpy as np

OBS_NAMES = [
    "ptm_sv1","ptm_sv2","ptm_sv3","ptm_H","EE","MI",
    "OS","H_T","D_eff","front_v","noise_slope","basis_invar"
]
HORIZON = 3

def load(path):
    with open(path) as f:
        return json.load(f)

def get_runs(data, model, controller):
    return [r for r in data["results"]
            if r["model"] == model and r["controller"] == controller]

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_program_n10a.py <results.json>")
        sys.exit(1)

    data    = load(sys.argv[1])
    model   = "DisorderedXXZ_W3"
    ag_runs = get_runs(data, model, "agnostic")
    st_runs = get_runs(data, model, "static")

    if not ag_runs:
        print(f"[ERROR] No agnostic runs found for {model}")
        sys.exit(1)

    B = len(ag_runs)
    print(f"\n{'='*70}")
    print(f"  N10a: W3 CONFIDENCE CHARACTERIZATION  (B={B} seeds)")
    print(f"{'='*70}")

    # Pair by seed for per-seed delta_tau
    st_tau  = {r["seed"]: float(r["tau_transport"]) for r in st_runs}
    ag_data = []
    for r in ag_runs:
        seed     = r["seed"]
        tau_a    = float(r["tau_transport"])
        tau_s    = st_tau.get(seed, float("nan"))
        n_gates  = sum(1 for a in r.get("ctrl_actions", []) if a[0] != 0)
        delta    = tau_a - tau_s
        ag_data.append({
            "seed": seed, "tau_a": tau_a, "tau_s": tau_s,
            "delta": delta, "n_gates": n_gates,
            "obs":  r.get("obs_history", []),
            "acts": r.get("ctrl_actions", [])
        })

    deltas  = np.array([d["delta"]   for d in ag_data])
    n_gates = np.array([d["n_gates"] for d in ag_data])

    # ── 1. Bootstrap CI for delta_tau ────────────────────────────────────────
    print(f"\n[1] Bootstrap CI for delta_tau (W3 agnostic - static)")
    mean_d = float(np.mean(deltas))
    std_d  = float(np.std(deltas, ddof=1))
    sem_d  = std_d / np.sqrt(B)
    ci95_lo = mean_d - 1.96 * sem_d
    ci95_hi = mean_d + 1.96 * sem_d
    ci99_lo = mean_d - 2.576 * sem_d
    ci99_hi = mean_d + 2.576 * sem_d
    frac_pos = float(np.mean(deltas > 0))
    print(f"    B={B} seeds")
    print(f"    mean(delta_tau)  = {mean_d:+.4f}")
    print(f"    std              = {std_d:.4f}")
    print(f"    SEM              = {sem_d:.4f}")
    print(f"    95% CI           = [{ci95_lo:+.4f}, {ci95_hi:+.4f}]")
    print(f"    99% CI           = [{ci99_lo:+.4f}, {ci99_hi:+.4f}]")
    print(f"    frac_pos         = {frac_pos:.2f} ({int(frac_pos*B)}/{B} seeds positive)")

    if ci95_hi < 0:
        verdict = "GENUINELY NEGATIVE -- intervention-fragile regime confirmed"
    elif ci95_lo > 0:
        verdict = "GENUINELY POSITIVE -- intervention beneficial"
    else:
        verdict = "EFFECTIVELY NEUTRAL -- CI overlaps zero (intervention boundary)"
    print(f"    --> VERDICT: {verdict}")

    # ── 2. Intervention efficiency ────────────────────────────────────────────
    print(f"\n[2] Intervention efficiency: eta_ctrl = delta_tau / N_nonidentity")
    eta_vals = []
    for d in ag_data:
        if d["n_gates"] > 0:
            eta_vals.append(d["delta"] / d["n_gates"])
    print(f"    Mean gates per run:  {np.mean(n_gates):.2f}")
    print(f"    Gate count quartiles: Q25={np.percentile(n_gates,25):.0f}  "
          f"Q50={np.percentile(n_gates,50):.0f}  Q75={np.percentile(n_gates,75):.0f}")
    if eta_vals:
        print(f"    eta_ctrl (gate-active runs only, n={len(eta_vals)}): "
              f"mean={np.mean(eta_vals):+.4f}  std={np.std(eta_vals):.4f}")
    # Per-gate-count analysis
    print(f"    delta_tau by gate count:")
    for ng in sorted(set(n_gates.tolist())):
        mask = n_gates == ng
        sub  = deltas[mask]
        print(f"      n_gates={int(ng)}: "
              f"n={int(mask.sum())}  mean_delta={np.mean(sub):+.3f}  std={np.std(sub):.3f}")

    # ── 3. Intervention timing ────────────────────────────────────────────────
    print(f"\n[3] Intervention timing distribution")
    fire_steps = []
    for d in ag_data:
        acts = d["acts"]
        for t, a in enumerate(acts):
            if a[0] != 0:
                fire_steps.append(t)
    n_steps = max(len(d["acts"]) for d in ag_data if d["acts"]) if ag_data else 0
    if fire_steps:
        print(f"    Total gate fires: {len(fire_steps)} across {B} runs")
        print(f"    Mean fire step:   {np.mean(fire_steps):.1f} / {n_steps}")
        print(f"    Early (t<10):     {sum(1 for s in fire_steps if s < 10)}")
        print(f"    Mid (10<=t<20):   {sum(1 for s in fire_steps if 10 <= s < 20)}")
        print(f"    Late (t>=20):     {sum(1 for s in fire_steps if s >= 20)}")
        # Step histogram
        hist = np.zeros(n_steps, dtype=int)
        for s in fire_steps:
            if s < n_steps:
                hist[s] += 1
        print(f"    Step histogram (fires/seed):")
        for t in range(0, n_steps, 5):
            bar = "#" * int(hist[t:t+5].sum())
            print(f"      t={t:2d}-{t+4:2d}: {bar} ({hist[t:t+5].sum()})")
    else:
        print("    No gate fires recorded.")

    # ── 4. Gate-conditioned observable drift ─────────────────────────────────
    print(f"\n[4] Gate-conditioned observable drift (k=1,2,3 steps after gate)")
    gate_records  = []   # (obs_t, future_obs_list)
    idle_records  = []
    for d in ag_data:
        obs  = d["obs"]
        acts = d["acts"]
        T    = min(len(obs), len(acts))
        for t in range(T):
            x_t    = np.array(obs[t], dtype=float)
            a_t    = int(acts[t][0])
            future = [np.array(obs[t+k], dtype=float) for k in range(1, HORIZON+1) if t+k < len(obs)]
            if a_t != 0:
                gate_records.append((x_t, future))
            else:
                idle_records.append((x_t, future))

    for k in range(1, HORIZON+1):
        gate_d = [r[1][k-1] - r[0] for r in gate_records if len(r[1]) >= k]
        idle_d = [r[1][k-1] - r[0] for r in idle_records if len(r[1]) >= k]
        if not gate_d or not idle_d:
            continue
        gm = np.stack(gate_d); im = np.stack(idle_d)
        print(f"  k={k}  (n_gate={len(gate_d)}, n_idle={len(idle_d)}):")
        for name in ["D_eff","front_v","noise_slope","basis_invar","OS"]:
            i = OBS_NAMES.index(name)
            ai = float(np.mean(np.abs(gm[:,i])))
            ao = float(np.mean(np.abs(im[:,i])))
            sgn_ai = float(np.mean(gm[:,i]))
            sgn_ao = float(np.mean(im[:,i]))
            ratio = ai/ao if ao > 1e-8 else 999.0
            flag = " <<< DESTABILIZED" if ratio > 1.4 else ""
            print(f"    {name:<12} gate_drift={sgn_ai:+.4f}({ai:.4f})  "
                  f"idle_drift={sgn_ao:+.4f}({ao:.4f})  ratio={ratio:.2f}{flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  N10a SUMMARY")
    print(f"{'='*70}")
    print(f"  W3 delta_tau: {mean_d:+.3f}  95%CI=[{ci95_lo:+.3f},{ci95_hi:+.3f}]")
    print(f"  Mean gates:   {np.mean(n_gates):.2f}")
    if eta_vals:
        print(f"  eta_ctrl:     {np.mean(eta_vals):+.4f} per gate")
    print(f"  Verdict:      {verdict}")

    # Save
    out_path = sys.argv[1].replace("_results.json", "_n10a_w3.json")
    out = {
        "model": model, "B": B,
        "mean_delta": mean_d, "std_delta": std_d, "sem": sem_d,
        "ci95": [ci95_lo, ci95_hi], "ci99": [ci99_lo, ci99_hi],
        "frac_pos": frac_pos,
        "mean_gates": float(np.mean(n_gates)),
        "eta_ctrl": float(np.mean(eta_vals)) if eta_vals else None,
        "verdict": verdict
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()
