"""
generate_dlinoss_input.py — Generate C(chi_t) and multi-observable time series
for D-LinOSS analysis.

Outputs:
  dlinoss_input.json: per-N time series of [chi_t, C, S_ent, <Sz_A>, <Sz_B>, F_sch]
  dlinoss_meta.json:  metadata for the observer

D-LinOSS will learn the frequency structure of the boundary entanglement
dynamics. Key questions:
  - How many oscillation modes drive C(chi_t)?
  - Do the dominant frequencies scale with N? (area vs volume law)
  - Are there resonant chi_t values with anomalously high C?
"""
import math, json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho, avg_fidelity
from jila_oat_exact_tpu import concurrence

CHI_T_STEPS = 64
CHI_T_MAX   = math.pi * 1.2   # slightly beyond pi to capture full oscillation
N_VALUES    = [2, 4, 6, 8, 10, 12, 16, 20]


def von_neumann_entropy(rho2):
    """S = -Tr[rho log rho] for the boundary pair."""
    eigs = np.real(np.linalg.eigvalsh(rho2))
    eigs = eigs[eigs > 1e-14]
    return float(-np.sum(eigs * np.log2(eigs)))


def sz_expectation(rho2):
    """<Sz_A>, <Sz_B> from boundary RDM."""
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    rho_A = np.trace(rho2.reshape(2, 2, 2, 2), axis1=1, axis2=3)  # trace B
    rho_B = np.trace(rho2.reshape(2, 2, 2, 2), axis1=0, axis2=2)  # trace A
    sz_A = float(np.real(np.trace(rho_A @ sz)))
    sz_B = float(np.real(np.trace(rho_B @ sz)))
    return sz_A, sz_B


def main():
    print(f"Generating D-LinOSS input: N={N_VALUES}, steps={CHI_T_STEPS}")
    chi_t_grid = np.linspace(0.02, CHI_T_MAX, CHI_T_STEPS)

    all_series = {}

    for N in N_VALUES:
        n = N // 2
        print(f"  N={N:>3d}...", end="", flush=True)
        series = []

        for chi_t in chi_t_grid:
            tensors = oat_mps(N, float(chi_t))
            rho2    = extract_boundary_rho(tensors, n)
            C       = concurrence(rho2)
            S_ent   = von_neumann_entropy(rho2)
            sz_A, sz_B = sz_expectation(rho2)
            F_pred  = (2 + C) / 3

            series.append({
                "chi_t":   float(chi_t),
                "C":       float(C),
                "S_ent":   S_ent,
                "sz_A":    sz_A,
                "sz_B":    sz_B,
                "F_pred":  float(F_pred),
                # CHSH analytic
                "S_chsh":  float(2 * math.sqrt(1 + C**2)),
            })

        # Find optimal chi_t
        C_vals = [s["C"] for s in series]
        opt_idx = int(np.argmax(C_vals))
        print(f" C_max={max(C_vals):.4f} at chi_t={series[opt_idx]['chi_t']:.3f}")

        all_series[str(N)] = {
            "N": N, "n": n, "bond_dim": n + 1,
            "C_max": max(C_vals),
            "chi_t_opt": series[opt_idx]["chi_t"],
            "time_series": series
        }

    output = {
        "description": "OAT boundary entanglement time series for D-LinOSS",
        "chi_t_grid_steps": CHI_T_STEPS,
        "chi_t_max": CHI_T_MAX,
        "N_values": N_VALUES,
        "channels": ["chi_t", "C", "S_ent", "sz_A", "sz_B", "F_pred", "S_chsh"],
        "channel_descriptions": {
            "chi_t":  "OAT interaction time (dimensionless)",
            "C":      "Wootters concurrence of boundary pair",
            "S_ent":  "von Neumann entropy of boundary pair (bits)",
            "sz_A":   "Expectation <Sz> of Alice boundary qubit",
            "sz_B":   "Expectation <Sz> of Bob boundary qubit",
            "F_pred": "Predicted teleportation fidelity (2+C)/3",
            "S_chsh": "Max CHSH value 2*sqrt(1+C^2)"
        },
        "per_N": all_series
    }

    with open("dlinoss_input.json", "w") as f:
        json.dump(output, f, indent=2)

    # Also write a flat CSV for easier plotting
    with open("dlinoss_input.csv", "w") as f:
        f.write("N,chi_t,C,S_ent,sz_A,sz_B,F_pred,S_chsh\n")
        for N_str, nd in all_series.items():
            N = nd["N"]
            for s in nd["time_series"]:
                f.write(f"{N},{s['chi_t']:.6f},{s['C']:.8f},"
                        f"{s['S_ent']:.8f},{s['sz_A']:.8f},"
                        f"{s['sz_B']:.8f},{s['F_pred']:.8f},{s['S_chsh']:.8f}\n")

    print(f"\n[DONE] dlinoss_input.json  ({CHI_T_STEPS*len(N_VALUES)} rows)")
    print(f"[DONE] dlinoss_input.csv")
    print(f"\nD-LinOSS input channels: {output['channels']}")
    print("Feed dlinoss_input.csv to the observer as a multivariate time series.")
    print("Key question: how many oscillation modes does the model need to fit C(chi_t)?")


if __name__ == "__main__":
    main()
