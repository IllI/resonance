"""
analyze_w3_fragility.py  (v2 -- correct schema)
================================================
W3 fragility diagnostic on stored obs trajectories from Program L results.

Schema discovered:
  results: list of dicts
    model, controller, tau_transport, obs_history (T x 12), ctrl_actions ([[gate,qubit],...])

Produces:
  1. Intervention conditionality: channel stats at fired vs idle steps
  2. Gate consequence: Dx_{t+1:t+k} after intervention vs idle
  3. Intervention density vs lifetime correlation
  4. Fragility veto threshold calibration (noise_slope theta1, basis_invar theta2)
"""

import json, sys
import numpy as np

OBS_NAMES = [
    "ptm_sv1","ptm_sv2","ptm_sv3","ptm_H","EE","MI",
    "OS","H_T","D_eff","front_v","noise_slope","basis_invar"
]
IDX = {n: i for i, n in enumerate(OBS_NAMES)}
HORIZON = 3

def load(path):
    with open(path) as f:
        return json.load(f)

def get_runs(data, model, controller="agnostic"):
    return [r for r in data["results"]
            if r["model"] == model and r["controller"] == controller]

def analyse(runs, label):
    print(f"\n{'='*70}")
    print(f"  FRAGILITY DIAGNOSTIC: {label}")
    print(f"{'='*70}")

    # Flatten records: (x_t, action_t, future_obs, tau, run_seed)
    records, taus, gate_counts = [], [], []
    for run in runs:
        obs  = run.get("obs_history", [])
        acts = run.get("ctrl_actions", [])
        tau  = float(run.get("tau_transport", float("nan")))
        if not obs or not acts: continue
        taus.append(tau)
        gate_counts.append(sum(1 for a in acts if a[0] != 0))
        T = min(len(obs), len(acts))
        for t in range(T):
            x_t    = np.array(obs[t], dtype=float)
            a_t    = int(acts[t][0])
            future = [np.array(obs[t+k], dtype=float) for k in range(1, HORIZON+1) if t+k < len(obs)]
            records.append((x_t, a_t, future, tau))

    taus        = np.array(taus)
    gate_counts = np.array(gate_counts, dtype=float)
    print(f"  Runs: {len(taus)}  |  tau mean={np.mean(taus):.3f}  gate_count mean={np.mean(gate_counts):.1f}")

    # ── 1. Intervention conditionality ───────────────────────────────────────
    int_obs  = np.stack([r[0] for r in records if r[1] != 0]) if any(r[1]!=0 for r in records) else None
    idle_obs = np.stack([r[0] for r in records if r[1] == 0]) if any(r[1]==0 for r in records) else None

    print(f"\n[1] Intervention conditionality")
    print(f"    Fired: {len(int_obs) if int_obs is not None else 0}   "
          f"Idle: {len(idle_obs) if idle_obs is not None else 0}")
    if int_obs is not None and idle_obs is not None:
        print(f"    {'Channel':<14} {'@fired':>10} {'@idle':>10} {'diff':>10}  flag")
        for name in ["D_eff","noise_slope","basis_invar","front_v","OS"]:
            i = IDX[name]
            mf = float(np.mean(int_obs[:,i]))
            mi = float(np.mean(idle_obs[:,i]))
            diff = mf - mi
            flag = " <-- HIGHER AT FIRE" if abs(diff) > 0.05 and diff > 0 else \
                   " <-- LOWER AT FIRE"  if abs(diff) > 0.05 and diff < 0 else ""
            print(f"    {name:<14} {mf:>10.4f} {mi:>10.4f} {diff:>10.4f}  {flag}")

    # ── 2. Gate consequence ───────────────────────────────────────────────────
    print(f"\n[2] Gate consequence: mean |Dx_{{t+k}}| after intervention vs idle")
    for k in range(1, HORIZON+1):
        int_d  = [r[2][k-1] - r[0] for r in records if r[1]!=0 and len(r[2])>=k]
        idle_d = [r[2][k-1] - r[0] for r in records if r[1]==0 and len(r[2])>=k]
        if not int_d or not idle_d: continue
        im = np.stack(int_d);  ilm = np.stack(idle_d)
        print(f"  k={k}:")
        for name in ["noise_slope","basis_invar","D_eff","front_v"]:
            i = IDX[name]
            ai = float(np.mean(np.abs(im[:,i])))
            ao = float(np.mean(np.abs(ilm[:,i])))
            ratio = ai/ao if ao>1e-8 else 999.0
            flag = " <<< DESTABILIZED" if ratio>1.4 else ""
            print(f"    {name:<14} after_int={ai:.4f}  after_idle={ao:.4f}  ratio={ratio:.2f}{flag}")

    # ── 3. Intervention density vs lifetime ───────────────────────────────────
    print(f"\n[3] Intervention density vs lifetime")
    if len(taus) >= 4:
        corr = float(np.corrcoef(gate_counts, taus)[0,1]) if np.std(gate_counts)>0 else 0.0
        print(f"    corr(n_gates, tau) = {corr:+.3f}")
        q25, q75 = np.percentile(gate_counts, [25,75])
        low  = taus[gate_counts <= q25]
        high = taus[gate_counts >= q75]
        if len(low):  print(f"    Low  gates (<=Q25={q25:.0f}): tau = {np.mean(low):.3f} +/- {np.std(low):.3f}")
        if len(high): print(f"    High gates (>=Q75={q75:.0f}): tau = {np.mean(high):.3f} +/- {np.std(high):.3f}")
        if corr < -0.2:
            print(f"    --> MONOTONIC DEGRADATION with intervention count (near-critical fragility confirmed)")

    # ── 4. Fragility veto calibration ─────────────────────────────────────────
    print(f"\n[4] Fragility veto threshold calibration")
    tau_med = float(np.median(taus))
    print(f"    Median tau: {tau_med:.3f}")

    ns_pcts = np.percentile([r[0][IDX["noise_slope"]] for r in records], [25,40,55,65,75])
    bi_pcts = np.percentile([r[0][IDX["basis_invar"]]  for r in records], [25,35,45,55,65])

    best_score, best = -1, (None,None)
    print(f"\n    {'theta_ns':>10} {'theta_bi':>10} {'veto_frac':>10} {'bad_int_rt':>12} {'score':>8}")
    for tns in ns_pcts:
        for tbi in bi_pcts:
            fragile = [(r[1], r[3]) for r in records
                       if r[0][IDX["noise_slope"]] > tns and r[0][IDX["basis_invar"]] < tbi]
            if len(fragile) < 5: continue
            veto_frac    = len(fragile) / len(records)
            bad_int      = sum(1 for a,t in fragile if a!=0 and t < tau_med)
            all_int      = sum(1 for a,t in fragile if a!=0)
            bad_int_rate = bad_int/all_int if all_int>0 else 0.0
            score        = bad_int_rate * min(1.0, 0.45/max(veto_frac,0.01))
            print(f"    {tns:>10.4f} {tbi:>10.4f} {veto_frac:>10.3f} {bad_int_rate:>12.3f} {score:>8.3f}")
            if score > best_score:
                best_score, best = score, (tns, tbi)

    print(f"\n    RECOMMENDED N8 VETO THRESHOLDS for {label}:")
    print(f"      noise_slope > {best[0]:.4f}  AND  basis_invar < {best[1]:.4f}")
    print(f"      (score={best_score:.3f})")
    return {"noise_slope_thresh": best[0], "basis_invar_thresh": best[1], "median_tau": tau_med}

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_w3_fragility.py <results.json>"); sys.exit(1)

    data = load(sys.argv[1])
    out = {}

    for model in ["DisorderedXXZ_W3","XXZ","DisorderedXXZ_W1","OAT"]:
        runs = get_runs(data, model)
        if runs:
            out[model] = analyse(runs, model)
        else:
            print(f"\n[SKIP] {model}: no agnostic runs found")

    # Print calibration summary
    print(f"\n{'='*70}")
    print("  N8 CALIBRATION SUMMARY")
    print(f"{'='*70}")
    for m, r in out.items():
        if r.get("noise_slope_thresh") is not None:
            print(f"  {m:<22}: noise_slope>{r['noise_slope_thresh']:.4f}  basis_invar<{r['basis_invar_thresh']:.4f}  (med_tau={r['median_tau']:.3f})")

    out_path = sys.argv[1].replace("_results.json","_w3_fragility.json")
    with open(out_path,"w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()
