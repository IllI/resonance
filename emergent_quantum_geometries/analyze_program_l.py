#!/usr/bin/env python3
"""
analyze_program_l.py  --  Program L Results Analyzer (v2)
==========================================================
Primary metric:  Δτ = tau_agnostic - tau_static  (denominator-free, stable)
Secondary metric: norm_gain = Δτ / (tau_haware - tau_static)
                  -- only computed when tau_haware > tau_static + eps
                  -- NaN otherwise (oracle weak or underperforms static)

Scientific framing:
  The key result is whether the agnostic controller extends recoverability in
  transport-capable models while null models remain operationally null.
  Gain > 1 is expected and secondary — the oracle is intentionally myopic.

Metrics:
  A. Δτ per model, per bootstrap (primary)
  B. norm_gain per model (secondary, oracle_valid only)
  C. CV_latent of tau_agnostic across non-null models
  D. Control entropy (when ctrl_actions logged via --store-obs)
  E. Representation mutual predictability (when obs_history logged)
  F. Falsification against pre-registered criteria
"""
import argparse
import json
import os
import sys
import numpy as np

# ── Pre-registered success criteria ──────────────────────────────────────────
MAX_CV_LATENT   = 0.3
MIN_MODELS_PASS = 4        # non-null models that must show delta_tau > 0
NULL_MODELS     = {"Ising", "TiltedIsing"}
NULL_DELTA_TAU_THRESHOLD = 0.5   # null models must have |delta_tau| < this
_EPS            = 1e-3     # oracle-valid threshold and denominator floor

OBS_NAMES = [
    "ptm_sv1", "ptm_sv2", "ptm_sv3", "ptm_H",
    "EE", "MI", "OS", "H_T",
    "D_eff", "front_v", "noise_slope", "basis_invar",
]

# Observable group labels for ablation reporting
OBS_GROUPS = {
    "PTM":  [0, 1, 2, 3],   # ptm_sv1, ptm_sv2, ptm_sv3, ptm_H
    "EE":   [4],
    "MI":   [5],
    "OS":   [6],
    "H_T":  [7],
    "Kinematics": [8, 9, 10, 11],  # D_eff, front_v, noise_slope, basis_invar
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def compute_per_model_stats(data):
    """
    Metric A: delta_tau = tau_agnostic - tau_static  (primary, stable)
    Metric B: norm_gain = delta_tau / (tau_haware - tau_static)
              -- only when oracle_valid (tau_haware > tau_static + _EPS)
    Metric C: AAR = (tau_agnostic - tau_dd) / (|tau_dd - tau_static| + eps)
    Metric D: vs_random = tau_agnostic - tau_random
              -- only when random controller present
              -- >0 means adaptation beats random intervention
    """
    records = data.get("results", [])
    models  = sorted({r["model"] for r in records})
    stats   = {}

    ctls_present = {r["controller"] for r in records}

    # Detect DD controller
    dd_ctrl = None
    for candidate in ("dd_xy8", "dd_cpmg"):
        if candidate in ctls_present:
            dd_ctrl = candidate
            break

    # Detect random controller
    has_random = "random" in ctls_present

    for model in models:
        recs = [r for r in records if r["model"] == model]
        all_ctls = ["static", "haware", "agnostic"]
        if dd_ctrl:
            all_ctls.append(dd_ctrl)
        if has_random:
            all_ctls.append("random")

        taus = {c: [r["tau_transport"] for r in recs if r["controller"] == c]
                for c in all_ctls}

        tau_s = float(np.mean(taus["static"]))   if taus["static"]   else 0.0
        tau_h = float(np.mean(taus["haware"]))   if taus["haware"]   else 0.0
        tau_a = float(np.mean(taus["agnostic"])) if taus["agnostic"] else 0.0
        tau_s_std = float(np.std(taus["static"]))   if len(taus["static"]) > 1   else 0.0
        tau_a_std = float(np.std(taus["agnostic"])) if len(taus["agnostic"]) > 1 else 0.0

        # DD baseline
        tau_d     = float(np.mean(taus[dd_ctrl])) if (dd_ctrl and taus[dd_ctrl]) else float("nan")
        tau_d_std = float(np.std(taus[dd_ctrl]))  if (dd_ctrl and len(taus[dd_ctrl]) > 1) else float("nan")

        # Random baseline (Metric D)
        tau_r     = float(np.mean(taus["random"])) if (has_random and taus["random"]) else float("nan")
        tau_r_std = float(np.std(taus["random"]))  if (has_random and len(taus["random"]) > 1) else float("nan")

        # Metric A — primary
        delta_tau = tau_a - tau_s

        # Metric B — oracle-relative
        oracle_valid = bool(tau_h > tau_s + _EPS)
        norm_gain = float(delta_tau / (tau_h - tau_s)) if oracle_valid else float("nan")

        # Metric C — AAR vs DD
        if not np.isnan(tau_d):
            aar = float((tau_a - tau_d) / (abs(tau_d - tau_s) + _EPS))
        else:
            aar = float("nan")

        # Metric D — adaptive vs random
        delta_tau_vs_random = float(tau_a - tau_r) if not np.isnan(tau_r) else float("nan")
        # random_aar: how much agnostic exceeds random relative to random's own gain over static
        if not np.isnan(tau_r):
            random_gain = tau_r - tau_s
            random_aar  = float((tau_a - tau_r) / (abs(random_gain) + _EPS))
        else:
            random_aar = float("nan")

        beats_oracle = bool(tau_a > tau_h)
        beats_dd     = bool(tau_a > tau_d)     if not np.isnan(tau_d) else None
        beats_random = bool(tau_a > tau_r)     if not np.isnan(tau_r) else None

        stats[model] = {
            "tau_static":            tau_s,
            "tau_haware":            tau_h,
            "tau_agnostic":          tau_a,
            "tau_dd":                tau_d,
            "tau_random":            tau_r,
            "tau_static_std":        tau_s_std,
            "tau_agnostic_std":      tau_a_std,
            "tau_dd_std":            tau_d_std,
            "tau_random_std":        tau_r_std,
            "delta_tau":             float(delta_tau),
            "delta_tau_vs_dd":       float(tau_a - tau_d)   if not np.isnan(tau_d) else float("nan"),
            "delta_tau_vs_random":   delta_tau_vs_random,
            "norm_gain":             norm_gain,
            "aar":                   aar,
            "random_aar":            random_aar,
            "oracle_valid":          oracle_valid,
            "beats_oracle":          int(beats_oracle),
            "beats_dd":              beats_dd,
            "beats_random":          beats_random,
            "n_boots":               len(taus["static"]),
            "dd_ctrl":               dd_ctrl,
            "has_random":            has_random,
        }

    return stats


def compute_alignment_metrics(data):
    """
    Program N1 metrics — response gradient alignment analysis.

    Requires --store-obs (obs_history + ctrl_actions in results JSON).

    cos_theta: alignment accuracy between predicted Clifford effect and
               actual observed response. Measures how well the controller
               models the local response direction.

               cos(θ) = (Δx_pred · Δx_obs) / (||Δx_pred|| ||Δx_obs|| + ε)

               >0: controller predicted constructive direction correctly
               <0: controller anti-aligned (predicted wrong direction)

    eta_align: intervention efficiency = Δτ / |non-identity actions|
               How much transport gain per unit of intervention.
               High η_align = selective constructive intervention.
               Low η_align = wasteful / indiscriminate.

    Also computes response_jacobian: mean Δx per observable channel per model,
    capturing which observables are most responsive to intervention.
    """
    records = data.get("results", [])
    models  = sorted({r["model"] for r in records})
    result  = {}

    for model in models:
        agnostic_recs = [r for r in records
                         if r["model"] == model and r["controller"] == "agnostic"]
        static_recs   = [r for r in records
                         if r["model"] == model and r["controller"] == "static"]

        cos_thetas    = []
        eta_aligns    = []
        jacobian_cols = []   # list of Δx vectors at action steps

        for rec in agnostic_recs:
            obs_h   = rec.get("obs_history")
            actions = rec.get("ctrl_actions")
            tau_a   = rec.get("tau_transport", 0.0)

            if not obs_h or not actions or len(obs_h) < 3:
                continue

            obs_arr = np.array(obs_h)   # (T, N_OBS)
            n_steps = len(obs_arr)
            n_acts  = 0
            thetas  = []

            for t, (gate_idx, _) in enumerate(actions):
                if gate_idx == 0 or t == 0 or t >= n_steps - 1:
                    continue   # skip identity and boundary steps
                n_acts += 1

                # Observed response: Δx_obs = x_{t+1} - x_{t-1}
                dx_obs  = obs_arr[min(t+1, n_steps-1)] - obs_arr[t-1]
                # Predicted response: from _CLIFF_OBS_DELTAS (gate physics model)
                # Reconstruct as signed delta vector in obs space
                dx_pred = np.zeros(len(obs_arr[0]))
                from program_l_tpu import _CLIFF_OBS_DELTAS, OBS_NAMES
                for obs_name, delta in _CLIFF_OBS_DELTAS[gate_idx].items():
                    if obs_name in OBS_NAMES:
                        dx_pred[OBS_NAMES.index(obs_name)] = delta

                obs_norm  = float(np.linalg.norm(dx_obs))
                pred_norm = float(np.linalg.norm(dx_pred))
                if obs_norm > 1e-8 and pred_norm > 1e-8:
                    cos_t = float(np.dot(dx_pred, dx_obs) / (pred_norm * obs_norm))
                    thetas.append(cos_t)
                    jacobian_cols.append(dx_obs)

            if thetas:
                cos_thetas.append(float(np.mean(thetas)))

            # eta_align: Δτ per non-identity action
            static_tau = np.mean([r["tau_transport"] for r in static_recs]) if static_recs else 0.0
            if n_acts > 0:
                eta_aligns.append(float((tau_a - static_tau) / n_acts))

        # Response Jacobian: mean |Δx| per obs channel across all action steps
        jacobian = {}
        if jacobian_cols:
            J = np.array(jacobian_cols)   # (n_actions_total, N_OBS)
            from program_l_tpu import OBS_NAMES
            for i, name in enumerate(OBS_NAMES):
                jacobian[name] = float(np.mean(np.abs(J[:, i])))

        result[model] = {
            "cos_theta_mean": float(np.mean(cos_thetas))  if cos_thetas  else float("nan"),
            "cos_theta_std":  float(np.std(cos_thetas))   if len(cos_thetas) > 1 else float("nan"),
            "eta_align_mean": float(np.mean(eta_aligns))  if eta_aligns  else float("nan"),
            "eta_align_std":  float(np.std(eta_aligns))   if len(eta_aligns) > 1 else float("nan"),
            "n_trajectories": len(agnostic_recs),
            "jacobian":       jacobian,
        }

    return result



def compute_cv_latent(stats):
    """CV of tau_agnostic across non-null models."""
    taus = [v["tau_agnostic"] for k, v in stats.items() if k not in NULL_MODELS]
    if len(taus) < 2:
        return float("nan")
    return float(np.std(taus) / (np.mean(taus) + 1e-12))


def compute_delta_tau_histogram(data):
    """Per-model, per-bootstrap Δτ distribution."""
    records = data.get("results", [])
    models  = sorted({r["model"] for r in records})
    hist    = {}

    for model in models:
        recs      = [r for r in records if r["model"] == model]
        tau_s_list = [r["tau_transport"] for r in recs if r["controller"] == "static"]
        tau_a_list = [r["tau_transport"] for r in recs if r["controller"] == "agnostic"]
        if tau_s_list and tau_a_list and len(tau_s_list) == len(tau_a_list):
            deltas = [a - s for a, s in zip(tau_a_list, tau_s_list)]
            hist[model] = {
                "deltas":       deltas,
                "mean":         float(np.mean(deltas)),
                "std":          float(np.std(deltas)) if len(deltas) > 1 else 0.0,
                "min":          float(np.min(deltas)),
                "max":          float(np.max(deltas)),
                "frac_positive": float(np.mean([d > 0 for d in deltas])),
            }
    return hist


def compute_control_entropy(data):
    """
    Shannon entropy over Clifford action choices (non-identity only).
    Measures whether controller converges to stable pulse families
    or chaotic switching. Only available when ctrl_actions stored.
    """
    records = data.get("results", [])
    if not any("ctrl_actions" in r for r in records):
        return None

    from collections import Counter
    models = sorted({r["model"] for r in records})
    result = {}

    for model in models:
        result[model] = {}
        for ctrl in ("static", "haware", "agnostic"):
            recs    = [r for r in records if r["model"] == model and r["controller"] == ctrl]
            actions = []
            for r in recs:
                acts = r.get("ctrl_actions", [])
                actions.extend([a[0] for a in acts if a[0] != 0])  # skip identity
            if not actions:
                result[model][ctrl] = {"entropy": 0.0, "n_actions": 0}
                continue
            counts  = Counter(actions)
            total   = sum(counts.values())
            probs   = [c / total for c in counts.values()]
            entropy = float(-sum(p * np.log2(p + 1e-15) for p in probs))
            result[model][ctrl] = {
                "entropy":   entropy,
                "n_actions": total,
                "dist":      dict(counts),
            }
    return result


def check_null_models(stats):
    """Null models must have |delta_tau| < threshold (operationally null)."""
    issues = []
    for m in NULL_MODELS:
        if m not in stats:
            continue
        dt = stats[m]["delta_tau"]
        if abs(dt) >= NULL_DELTA_TAU_THRESHOLD:
            issues.append(f"{m}: delta_tau={dt:+.3f} (expected ~0)")
    return issues


def falsification_check(stats, cv, hist=None):
    """
    Falsification criteria — Metric A (delta_tau) is primary.
    Metric B (norm_gain) is informational only.
    """
    lines  = []
    passed = True
    non_null = {k: v for k, v in stats.items() if k not in NULL_MODELS}

    # 1. Primary: delta_tau > 0 in >= MIN_MODELS_PASS non-null models
    n_positive = sum(1 for v in non_null.values() if v["delta_tau"] > 0)
    ok1 = n_positive >= min(MIN_MODELS_PASS, len(non_null))
    lines.append(f"  [{'PASS' if ok1 else 'FAIL'}] Metric A: delta_tau > 0 in "
                 f"{n_positive}/{len(non_null)} non-null models")
    passed = passed and ok1

    # 2. CV_latent
    ok2 = np.isnan(cv) or cv < MAX_CV_LATENT
    lines.append(f"  [{'PASS' if ok2 else 'FAIL'}] CV_latent={cv:.3f} "
                 f"(threshold < {MAX_CV_LATENT})")
    passed = passed and ok2

    # 3. Null models operationally null
    null_issues = check_null_models(stats)
    ok3 = len(null_issues) == 0
    if null_issues:
        for issue in null_issues:
            lines.append(f"  [FAIL] Null model violation: {issue}")
    else:
        lines.append("  [PASS] Null models: |delta_tau| ~ 0 (operationally null)")
    passed = passed and ok3

    # 4. Agnostic > static (raw count)
    n_above = sum(1 for v in non_null.values() if v["tau_agnostic"] > v["tau_static"])
    ok4 = n_above >= min(MIN_MODELS_PASS, len(non_null))
    lines.append(f"  [{'PASS' if ok4 else 'FAIL'}] agnostic > static in "
                 f"{n_above}/{len(non_null)} non-null models")
    passed = passed and ok4

    # [INFO] Metric B — secondary, informational only
    valid_models = [(k, v) for k, v in non_null.items() if v["oracle_valid"]]
    if valid_models:
        n_gain_pass = sum(1 for _, v in valid_models if v.get("norm_gain", 0) >= 0.5)
        lines.append(f"  [INFO] Metric B: norm_gain >= 0.5 in "
                     f"{n_gain_pass}/{len(valid_models)} oracle-valid non-null models")
        weak = [k for k, v in non_null.items() if not v["oracle_valid"]]
        if weak:
            lines.append(f"  [INFO] Oracle weak (tau_h <= tau_s): {', '.join(weak)} "
                         f"— Metric B not computed for these")
    else:
        lines.append("  [INFO] Metric B: oracle underperforms static in all non-null "
                     "models — Metric A is the only valid criterion this run")

    # [INFO] Oracle beating
    n_beats = sum(1 for v in non_null.values() if v["beats_oracle"])
    lines.append(f"  [INFO] agnostic beats myopic oracle in {n_beats}/{len(non_null)} "
                 "non-null models (expected: oracle 1-step greedy, agnostic long-horizon)")

    return passed, lines


def compute_predictability_matrix(data):
    """12×12 pairwise R² matrix. Requires obs_history in results (--store-obs)."""
    records   = data.get("results", [])
    histories = []
    for r in records:
        h = r.get("obs_history")
        if h is not None:
            histories.extend(h)
    if not histories:
        return None

    X = np.array(histories)
    n = X.shape[1]
    R2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                R2[i, j] = 1.0
                continue
            xi = X[:, i].reshape(-1, 1)
            yj = X[:, j]
            A   = np.hstack([xi, np.ones_like(xi)])
            beta, _, _, _ = np.linalg.lstsq(A, yj, rcond=None)
            y_hat  = xi.ravel() * beta[0] + beta[1]
            ss_res = np.sum((yj - y_hat) ** 2)
            ss_tot = np.sum((yj - yj.mean()) ** 2) + 1e-15
            R2[i, j] = max(0.0, 1.0 - ss_res / ss_tot)
    return R2


def print_summary(stats, cv, falsification, out_path,
                  hist=None, ctrl_entropy=None,
                  align_metrics=None, pred_matrix=None):
    print("=" * 72)
    print("Program L  --  Analysis Results  (v2: Metric A primary)")
    print("=" * 72)

    # Main table — extended for DD + random baselines when present
    has_dd     = any(not np.isnan(v.get("tau_dd",     float("nan"))) for v in stats.values())
    has_random = any(not np.isnan(v.get("tau_random", float("nan"))) for v in stats.values())

    if has_dd or has_random:
        hdr = f"\n{'Model':<22} {'τ_s':>6}"
        if has_dd:     hdr += f" {'τ_dd':>6}"
        if has_random: hdr += f" {'τ_rnd':>6}"
        hdr += f" {'τ_h':>6} {'τ_a':>6} {'Δτ':>7}"
        if has_dd:     hdr += f" {'vs_DD':>6}"
        if has_random: hdr += f" {'vs_rnd':>6}"
        hdr += "  n"
        print(hdr)
        print("-" * len(hdr.expandtabs()))
        for model, v in sorted(stats.items()):
            dt     = f"{v['delta_tau']:+.3f}"
            tau_d  = f"{v['tau_dd']:.3f}"     if not np.isnan(v.get("tau_dd",     float("nan"))) else "   n/a"
            tau_r  = f"{v['tau_random']:.3f}" if not np.isnan(v.get("tau_random", float("nan"))) else "   n/a"
            vs_dd  = f"{v['delta_tau_vs_dd']:+.3f}"     if not np.isnan(v.get("delta_tau_vs_dd",     float("nan"))) else "  n/a"
            vs_rnd = f"{v['delta_tau_vs_random']:+.3f}" if not np.isnan(v.get("delta_tau_vs_random", float("nan"))) else "  n/a"
            null_t = " [null]" if model in NULL_MODELS else ""
            row = f"  {model:<20} {v['tau_static']:>6.3f}"
            if has_dd:     row += f" {tau_d:>6}"
            if has_random: row += f" {tau_r:>6}"
            row += f" {v['tau_haware']:>6.3f} {v['tau_agnostic']:>6.3f} {dt:>7}"
            if has_dd:     row += f" {vs_dd:>6}"
            if has_random: row += f" {vs_rnd:>6}"
            row += f"  {v['n_boots']:>2}{null_t}"
            print(row)

    else:
        print(f"\n{'Model':<22} {'τ_s':>6} {'τ_h':>6} {'τ_a':>6} {'Δτ':>7}  "
              f"{'oracle':>6}  {'norm_g':>7}   n")
        print("-" * 72)
        for model, v in sorted(stats.items()):
            dt    = f"{v['delta_tau']:+.3f}"
            ov    = "valid" if v["oracle_valid"] else "WEAK"
            ng    = f"{v['norm_gain']:.3f}" if v["oracle_valid"] else "   n/a"
            null_t = " [null]" if model in NULL_MODELS else ""
            print(f"  {model:<20} {v['tau_static']:>6.3f} {v['tau_haware']:>6.3f} "
                  f"{v['tau_agnostic']:>6.3f} {dt:>7}  {ov:>6}  {ng:>7}  "
                  f"{v['n_boots']:>2}{null_t}")


    print(f"\nCV_latent (non-null) = {cv:.3f}  (threshold < {MAX_CV_LATENT})")

    # Delta_tau histogram
    if hist:
        print("\n-- Δτ per-bootstrap distribution --")
        for model, h in sorted(hist.items()):
            pos = sum(1 for d in h["deltas"] if d > 0)
            neg = len(h["deltas"]) - pos
            bar = "+" * pos + "-" * neg
            null_t = " [null]" if model in NULL_MODELS else ""
            print(f"  {model:<22} mean={h['mean']:+.3f}  std={h['std']:.3f}  "
                  f"frac_pos={h['frac_positive']:.0%}  [{bar}]{null_t}")

    # Control entropy
    if ctrl_entropy:
        print("\n-- Control entropy (non-identity actions) --")
        for model, ctls in sorted(ctrl_entropy.items()):
            ag = ctls.get("agnostic", {})
            st = ctls.get("static",   {})
            print(f"  {model:<22} agnostic H={ag.get('entropy', 0):.3f} "
                  f"({ag.get('n_actions', 0)} acts)  "
                  f"static H={st.get('entropy', 0):.3f}")

    # Program N1 alignment metrics
    if align_metrics:
        print("\n-- Response gradient alignment (Program N1) --")
        print(f"  {'Model':<22} {'cos(θ)':>8} {'±':>5} {'η_align':>8} {'±':>5}  meaning")
        print("  " + "-" * 68)
        for model, v in sorted(align_metrics.items()):
            cos_m = v["cos_theta_mean"]
            cos_s = v["cos_theta_std"]
            eta_m = v["eta_align_mean"]
            eta_s = v["eta_align_std"]
            null_t = " [null]" if model in NULL_MODELS else ""

            if np.isnan(cos_m):
                cos_str = "    n/a"
                cos_e   = "     "
            else:
                cos_str = f"{cos_m:+.3f}"
                cos_e   = f"{cos_s:.3f}" if not np.isnan(cos_s) else "  n/a"

            if np.isnan(eta_m):
                eta_str = "    n/a"
                eta_e   = "     "
            else:
                eta_str = f"{eta_m:+.3f}"
                eta_e   = f"{eta_s:.3f}" if not np.isnan(eta_s) else "  n/a"

            # Alignment quality label
            if not np.isnan(cos_m):
                if cos_m > 0.3:   qual = "well-aligned"
                elif cos_m > 0.0: qual = "weakly aligned"
                elif cos_m > -0.2: qual = "neutral"
                else:              qual = "anti-aligned"
            else:
                qual = "no obs_history"

            print(f"  {model:<22} {cos_str:>8} {cos_e:>5} {eta_str:>8} {eta_e:>5}"
                  f"  {qual}{null_t}")

        # Jacobian summary: which observables are most response-active?
        all_jac = {}
        for v in align_metrics.values():
            for obs_name, mag in v.get("jacobian", {}).items():
                all_jac.setdefault(obs_name, []).append(mag)
        if all_jac:
            print("\n  Response Jacobian (mean |Δx| per channel across all models):")
            for obs_name in sorted(all_jac, key=lambda k: -np.mean(all_jac[k])):
                print(f"    {obs_name:<15} |Δx|={np.mean(all_jac[obs_name]):.4f}")

    # Representation mutual predictability

    if pred_matrix is not None:
        print("\n-- Representation Mutual Predictability (mean R² per channel) --")
        mean_r2 = pred_matrix.mean(axis=1)
        for i, name in enumerate(OBS_NAMES):
            flag = " [REDUNDANT?]" if mean_r2[i] > 0.80 else ""
            print(f"  {name:<15} mean_R2={mean_r2[i]:.3f}{flag}")
        print("\n  Group summary:")
        for grp, idxs in OBS_GROUPS.items():
            grp_r2 = float(np.mean([mean_r2[i] for i in idxs]))
            print(f"    {grp:<12} mean_R2={grp_r2:.3f}")

    # Falsification
    print("\n-- Falsification Check --")
    passed, lines = falsification
    for line in lines:
        print(line)
    verdict = ("EXPERIMENT PASSES pre-registered criteria"
               if passed else "EXPERIMENT FAILS pre-registered criteria")
    print(f"\n>>> {verdict}\n")

    # Interpretation
    print("-- Interpretation --")
    non_null_pos = [(m, v["delta_tau"]) for m, v in stats.items()
                    if m not in NULL_MODELS and v["delta_tau"] > 0]
    non_null_neg = [(m, v["delta_tau"]) for m, v in stats.items()
                    if m not in NULL_MODELS and v["delta_tau"] <= 0]
    null_ok      = [(m, v["delta_tau"]) for m, v in stats.items()
                    if m in NULL_MODELS and abs(v["delta_tau"]) < NULL_DELTA_TAU_THRESHOLD]
    null_fail    = [(m, v["delta_tau"]) for m, v in stats.items()
                    if m in NULL_MODELS and abs(v["delta_tau"]) >= NULL_DELTA_TAU_THRESHOLD]

    if non_null_pos:
        print(f"  Adaptive extension (Δτ>0): "
              + ", ".join(f"{m}({dt:+.3f})" for m, dt in non_null_pos))
    if non_null_neg:
        print(f"  No improvement    (Δτ≤0): "
              + ", ".join(f"{m}({dt:+.3f})" for m, dt in non_null_neg))
    if null_ok:
        print(f"  Operationally null (Δτ≈0): "
              + ", ".join(f"{m}({dt:+.3f})" for m, dt in null_ok))
    if null_fail:
        print(f"  Null model FAILED  (Δτ large): "
              + ", ".join(f"{m}({dt:+.3f})" for m, dt in null_fail))
    print()

    if out_path:
        result = {
            "stats":                stats,
            "cv_latent":            cv,
            "falsification_passed": passed,
            "falsification_lines":  lines,
            "delta_tau_hist":       hist,
            "ctrl_entropy":         ctrl_entropy,
            "pred_matrix":          pred_matrix.tolist() if pred_matrix is not None else None,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Analysis saved -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Program L Analyzer v2")
    p.add_argument("results_json", help="Path to program_l_N*_results.json")
    p.add_argument("--out", default=None, help="Optional output JSON path")
    args = p.parse_args()

    if not os.path.exists(args.results_json):
        print(f"Error: {args.results_json} not found")
        sys.exit(1)

    data        = load_results(args.results_json)
    stats       = compute_per_model_stats(data)
    cv          = compute_cv_latent(stats)
    hist        = compute_delta_tau_histogram(data)
    ctrl_ent    = compute_control_entropy(data)
    align_met   = compute_alignment_metrics(data)
    pred_matrix = compute_predictability_matrix(data)
    falsi       = falsification_check(stats, cv, hist)
    out_p       = args.out or args.results_json.replace(".json", "_analysis.json")

    print_summary(stats, cv, falsi, out_p,
                  hist=hist, ctrl_entropy=ctrl_ent,
                  align_metrics=align_met, pred_matrix=pred_matrix)


if __name__ == "__main__":
    main()
