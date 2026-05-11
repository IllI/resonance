"""
generate_figures.py
====================
Generates publication-quality figures from jila_oat_exact_results.json
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ── Load results ──────────────────────────────────────────────────────────────
with open("results/simulation/jila_oat_exact_results.json") as f:
    data = json.load(f)

results = data["results"]
CL = data["classical_limit"]
N_list = [v["N_ions"] for v in results.values()]
colors = cm.plasma(np.linspace(0.15, 0.9, len(N_list)))

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "legend.framealpha": 0.3,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "DejaVu Sans",
}
plt.rcParams.update(STYLE)

# ════════════════════════════════════════════════════════════════
# FIGURE 1: Fidelity F vs chi*t for all N
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(CL, color="#f78166", linewidth=1.5, linestyle="--",
           label=f"Classical limit  F = 2/3 = {CL:.4f}", zorder=2)
ax.axhspan(0.0, CL, alpha=0.08, color="#f78166")

for (k, v), c in zip(results.items(), colors):
    chi_t = np.array(v["chi_t_curve"])
    F     = np.array(v["F_curve"])
    N     = v["N_ions"]
    ax.plot(chi_t, F, color=c, linewidth=2.0, label=f"N = {N} ions")
    # Mark peak
    idx = int(np.argmax(F))
    ax.scatter(chi_t[idx], F[idx], color=c, s=60, zorder=5, marker="*")

ax.set_xlabel(r"Coupling parameter  $\chi t$", fontsize=13)
ax.set_ylabel(r"Teleportation fidelity  $F$", fontsize=13)
ax.set_title("JILA OAT Quantum Teleportation — Fidelity vs Coupling\n"
             r"$H = \chi J_z^A J_z^B$, initial state $|+\rangle^N$  [TPU v6e-8, europe-west4-a]",
             fontsize=12, pad=12)
ax.legend(fontsize=9, ncol=2, loc="upper right")
ax.set_xlim(0, math.pi)
ax.set_ylim(0.60, 1.02)
ax.grid(True)
ax.set_xticks([0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi])
ax.set_xticklabels(["0", "π/4", "π/2", "3π/4", "π"])

# Annotation
ax.text(2.5, 0.94, "QUANTUM\nADVANTAGE\nREGION", color="#7ee787",
        fontsize=9, ha="center", alpha=0.7)

plt.tight_layout()
fig.savefig(out_dir / "fig1_fidelity_vs_chi_t.png", dpi=180, bbox_inches="tight")
plt.close()
print("  [OK] fig1_fidelity_vs_chi_t.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 2: Concurrence C vs chi*t
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(0, color="#f78166", linewidth=1.2, linestyle="--",
           label="Entanglement threshold  C = 0", zorder=2)

for (k, v), c in zip(results.items(), colors):
    chi_t = np.array(v["chi_t_curve"])
    C     = np.array(v["C_curve"])
    N     = v["N_ions"]
    ax.plot(chi_t, C, color=c, linewidth=2.0, label=f"N = {N}")

ax.set_xlabel(r"Coupling parameter  $\chi t$", fontsize=13)
ax.set_ylabel(r"Wootters Concurrence  $C$", fontsize=13)
ax.set_title("OAT-Generated Boundary Qubit Entanglement\n"
             r"Concurrence of qubit pair $(A_{N/2-1},\, B_0)$ after evolution",
             fontsize=12, pad=12)
ax.legend(fontsize=9, ncol=2, loc="upper right")
ax.set_xlim(0, math.pi)
ax.set_ylim(-0.02, 1.05)
ax.grid(True)
ax.set_xticks([0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi])
ax.set_xticklabels(["0", "π/4", "π/2", "3π/4", "π"])

plt.tight_layout()
fig.savefig(out_dir / "fig2_concurrence_vs_chi_t.png", dpi=180, bbox_inches="tight")
plt.close()
print("  [OK] fig2_concurrence_vs_chi_t.png")

# ════════════════════════════════════════════════════════════════
# FIGURE 3: Scaling — F_peak and C_peak vs N
# ════════════════════════════════════════════════════════════════
N_arr   = np.array([v["N_ions"] for v in results.values()])
F_peaks = np.array([v["F_peak"] for v in results.values()])
C_peaks = np.array([v["C_peak"] for v in results.values()])
t_opts  = np.array([v["chi_t_optimal"] for v in results.values()])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: F_peak vs N
axes[0].plot(N_arr, F_peaks, "o-", color="#79c0ff", linewidth=2,
             markersize=8, label="F_peak (TPU)")
axes[0].axhline(CL, color="#f78166", linestyle="--", linewidth=1.5,
                label=f"Classical limit = {CL:.4f}")
axes[0].set_xlabel("Total ion number  N", fontsize=12)
axes[0].set_ylabel("Peak fidelity  F_peak", fontsize=12)
axes[0].set_title("Peak Teleportation Fidelity vs N", fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(True)

# Panel B: C_peak vs N with power-law fit
axes[1].loglog(N_arr, C_peaks, "s-", color="#d2a8ff", linewidth=2,
               markersize=8, label="C_peak (TPU)")
# Fit power law C ~ N^alpha
log_N = np.log(N_arr)
log_C = np.log(C_peaks)
coeffs = np.polyfit(log_N, log_C, 1)
alpha = coeffs[0]
N_fit = np.linspace(N_arr.min(), N_arr.max(), 100)
C_fit = np.exp(coeffs[1]) * N_fit**alpha
axes[1].loglog(N_fit, C_fit, "--", color="#f0883e", linewidth=1.5,
               label=fr"Power law: $C \propto N^{{{alpha:.2f}}}$")
axes[1].set_xlabel("Total ion number  N  (log scale)", fontsize=12)
axes[1].set_ylabel("Peak concurrence  C_peak  (log)", fontsize=12)
axes[1].set_title("Concurrence Scaling with N", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(True, which="both")

# Panel C: Optimal time chi*t* vs N
axes[2].plot(N_arr, t_opts, "^-", color="#7ee787", linewidth=2,
             markersize=8, label=r"$\chi t^*$ (TPU)")
# Fit chi*t* ~ N^beta
log_t = np.log(t_opts)
coeffs2 = np.polyfit(log_N, log_t, 1)
beta = coeffs2[0]
t_fit = np.exp(coeffs2[1]) * N_fit**beta
axes[2].plot(N_fit, t_fit, "--", color="#f0883e", linewidth=1.5,
             label=fr"Power law: $\chi t^* \propto N^{{{beta:.2f}}}$")
axes[2].set_xlabel("Total ion number  N", fontsize=12)
axes[2].set_ylabel(r"Optimal coupling  $\chi t^*$", fontsize=12)
axes[2].set_title("Optimal Coupling Time Scaling", fontsize=11)
axes[2].legend(fontsize=9)
axes[2].grid(True)

for ax in axes:
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")

plt.suptitle("JILA OAT Teleportation — N-Scaling Analysis  [JAX/TPU v6e-8]",
             fontsize=13, y=1.02, color="#e6edf3")
plt.tight_layout()
fig.savefig(out_dir / "fig3_scaling_analysis.png", dpi=180, bbox_inches="tight")
plt.close()
print("  [OK] fig3_scaling_analysis.png")
print(f"\n  Concurrence power law: C ~ N^{alpha:.3f}")
print(f"  Optimal time power law: chi*t* ~ N^{beta:.3f}")
print(f"\n[DONE] All figures saved to {out_dir}/")
