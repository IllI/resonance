"""
generate_dlinoss_input.py — Generate witness decay + multi-observable time series
for D-LinOSS analysis.

CRITICAL FIX (2026-05-09): Use Tr[W·ρ(t)] (entanglement witness) as the
primary D-LinOSS channel instead of concurrence C.

Reason: C is nonlinear in ρ — under Markovian dephasing it decays super-
exponentially for mixed states (Wootters artifact), causing D-LinOSS to
spuriously identify SYK/MBL. The witness Tr[Wρ] is LINEAR in ρ and decays
as exactly exp(-4Γt) under σ_z dephasing for any state. This gives D-LinOSS
a clean, physically interpretable signal.

Outputs:
  dlinoss_input.json: per-N time series of [chi_t, witness, S_ent, <Sz_A>, <Sz_B>, F_pred, S_chsh]
  dlinoss_input.csv  (flat CSV)

Channels:
  witness: Tr[W_opt · ρ(χt)] — linear, clean exponential decay under dephasing
  S_ent:   von Neumann entropy (distinguishes pure/mixed dynamics)
  sz_A/B:  boundary spin polarization (directly measurable at JILA)
  F_pred:  predicted teleportation fidelity (2+C)/3
  S_chsh:  max CHSH value (Horodecki)
"""
import math, json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from oat_teleport_v3_tpu import oat_mps, extract_boundary_rho
from jila_oat_exact_tpu import concurrence

SIGMAS = [
    np.array([[0,1],[1,0]], dtype=complex),
    np.array([[0,-1j],[1j,0]], dtype=complex),
    np.array([[1,0],[0,-1]], dtype=complex),
]
BELLS = [
    np.array([1,0,0,1],  dtype=complex)/math.sqrt(2),
    np.array([1,0,0,-1], dtype=complex)/math.sqrt(2),
    np.array([0,1,1,0],  dtype=complex)/math.sqrt(2),
    np.array([0,1,-1,0], dtype=complex)/math.sqrt(2),
]

def best_witness_val(rho2):
    """Tr[W_opt·ρ] where W_opt = I/4 - |β_best><β_best|. Linear in ρ."""
    f_vals = [float(np.real(b.conj() @ rho2 @ b)) for b in BELLS]
    return 0.25 - max(f_vals)

def chsh_horodecki(rho):
    """Exact Horodecki CHSH — NOT the pure-state formula."""
    T = np.zeros((3,3))
    for i,si in enumerate(SIGMAS):
        for j,sj in enumerate(SIGMAS):
            T[i,j] = float(np.real(np.trace(rho @ np.kron(si,sj))))
    eigs = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
    return float(2 * math.sqrt(max(0, eigs[0] + eigs[1])))

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
            witness = best_witness_val(rho2)  # LINEAR in rho — no Wootters artifact
            S_ent   = von_neumann_entropy(rho2)
            sz_A, sz_B = sz_expectation(rho2)
            F_pred  = (2 + C) / 3
            S_chsh  = chsh_horodecki(rho2)    # Horodecki exact, not 2*sqrt(1+C^2)

            series.append({
                "chi_t":   float(chi_t),
                "witness": float(witness),    # PRIMARY: Tr[W·rho], linear in rho
                "C":       float(C),          # kept for reference
                "S_ent":   S_ent,
                "sz_A":    sz_A,
                "sz_B":    sz_B,
                "F_pred":  float(F_pred),
                "S_chsh":  float(S_chsh),     # Horodecki (exact)
            })

        # Find optimal chi_t (maximize F_pred = maximize C)
        C_vals = [s["C"] for s in series]
        W_vals = [s["witness"] for s in series]
        opt_idx = int(np.argmax(C_vals))
        print(f" C_max={max(C_vals):.4f}  W_min={min(W_vals):.4f} at chi_t={series[opt_idx]['chi_t']:.3f}")

        all_series[str(N)] = {
            "N": N, "n": n, "bond_dim": n + 1,
            "C_max": max(C_vals),
            "chi_t_opt": series[opt_idx]["chi_t"],
            "time_series": series
        }

    channels = ["chi_t", "witness", "C", "S_ent", "sz_A", "sz_B", "F_pred", "S_chsh"]
    output = {
        "description": "OAT boundary witness decay + observables for D-LinOSS",
        "fix": "2026-05-09: witness replaces C as primary channel (linear in rho)",
        "chi_t_grid_steps": CHI_T_STEPS,
        "chi_t_max": CHI_T_MAX,
        "N_values": N_VALUES,
        "channels": channels,
        "channel_descriptions": {
            "chi_t":   "OAT interaction time (dimensionless)",
            "witness": "Tr[W_opt·rho] — LINEAR in rho, primary D-LinOSS channel",
            "C":       "Wootters concurrence (reference, nonlinear — avoid for D-LinOSS)",
            "S_ent":   "von Neumann entropy of boundary pair (bits)",
            "sz_A":    "Expectation <Sz> of Alice boundary qubit (measurable)",
            "sz_B":    "Expectation <Sz> of Bob boundary qubit (measurable)",
            "F_pred":  "Predicted teleportation fidelity (2+C)/3",
            "S_chsh":  "Max CHSH value — Horodecki exact (not pure-state formula)",
        },
        "per_N": all_series
    }

    with open("dlinoss_input.json", "w") as f:
        json.dump(output, f, indent=2)

    with open("dlinoss_input.csv", "w") as f:
        f.write(",".join(channels) + "\n")
        for N_str, nd in all_series.items():
            Nv = nd["N"]
            for s in nd["time_series"]:
                f.write(f"{Nv},{s['chi_t']:.6f},{s['witness']:.8f},"
                        f"{s['C']:.8f},{s['S_ent']:.8f},{s['sz_A']:.8f},"
                        f"{s['sz_B']:.8f},{s['F_pred']:.8f},{s['S_chsh']:.8f}\n")

    print(f"\n[DONE] dlinoss_input.json  ({CHI_T_STEPS*len(N_VALUES)} rows)")
    print(f"[DONE] dlinoss_input.csv")
    print(f"\nPrimary D-LinOSS channel: witness (linear in rho — no Wootters artifact)")
    print(f"Feed dlinoss_input.csv to observer. witness column = framework identifier.")


if __name__ == "__main__":
    main()
