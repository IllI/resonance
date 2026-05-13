"""
Update paper: add §IX.G (topology replication) and reframe title/abstract/conclusion.
"""

path = r"docs\PAPER_DRAFT_v1.md"
with open(path, encoding="utf-8") as f:
    content = f.read()

# ── 1. Update title ───────────────────────────────────────────────────────
OLD_TITLE = "# Geometric Phase Structure of Recoverable Quantum Transport in One-Axis Twisting and Dicke Boundary Channels"
NEW_TITLE = "# Reduced-State Correlation Structure in Cross-Half ZZ OAT Dynamics: Tensor-Network Analysis and Superconducting Hardware Validation"
content = content.replace(OLD_TITLE, NEW_TITLE)

# ── 2. Update version line ────────────────────────────────────────────────
OLD_VER = "**Draft v8 -- IBM hardware validation added**"
NEW_VER = "**Draft v9 -- Topology replication complete; framing corrected**"
content = content.replace(OLD_VER, NEW_VER)
OLD_SUB = "*(§IX: 3-point PTM protocol, pre-registered predictions, results pending)*"
NEW_SUB = "*(§IX: Runs 1–3 complete; R²=0.986/0.976; null <0.5σ; layout-independent)*"
content = content.replace(OLD_SUB, NEW_SUB)

# ── 3. Reframe abstract opening ───────────────────────────────────────────
OLD_ABS = "We identify a geometric phase structure governing operationally recoverable quantum transport in one-axis twisting (OAT) and Dicke boundary channels."
NEW_ABS = "We analyze the reduced-state correlation structure of cross-half ZZ one-axis twisting (OAT) dynamics and validate the predicted observable structure on superconducting quantum hardware."
content = content.replace(OLD_ABS, NEW_ABS)

# ── 4. Add §IX.G before Related Work ─────────────────────────────────────
IX_G = """
### IX.G Run 3 — Topology-Controlled Replication

> [!IMPORTANT]
> **Layout independence confirmed. The transpiler/routing-artifact objection is answered.**

**Design:** Identical 9-point protocol on disjoint physical chain [4,5,6,7] (Layout B), same `optimization_level=0`, same readout-matrix mitigation. Chain [4,5,6,7] is fully disjoint from Layout A [0,1,2,3].

**Jobs:** cal `d81tqfvoha1c73bks33g` · sweep `d81tqhfoha1c73bks370` · null `d81tqhntjchs73bnccvg`

**Layout B results** (readout-matrix mitigated):

| χt/π | T_xx raw | T_xx mitigated | A·cos²(χt/2)/2 |
|---|---|---|---|
| 0.000 | 0.4520 | 0.4551 | 0.4562 |
| 0.125 | 0.4460 | 0.4493 | 0.4374 |
| 0.250 | 0.3920 | 0.3950 | 0.3880 |
| 0.375 | 0.3040 | 0.3058 | 0.3143 |
| 0.500 | 0.1640 | 0.1649 | 0.2273 |
| 0.625 | 0.1620 | 0.1631 | 0.1403 |
| 0.750 | 0.0820 | 0.0825 | 0.0666 |
| 0.875 | 0.0040 | 0.0032 | 0.0173 |
| **1.000** | **−0.0005** | **−0.0009** | **0.0000** ← NULL |

**Layout A vs Layout B comparison:**

| Metric | Layout A [0,1,2,3] | Layout B [4,5,6,7] |
|---|---|---|
| Circuit depth | 76 | **64** |
| Attenuation $A$ | 0.9089 | **0.9033** |
| $R^2$ | 0.9856 | **0.9758** |
| $T_{xx}(\pi)$ mitigated | +0.0034 | −0.0009 |
| Null $\sigma$ from zero | 0.4σ | **−0.12σ** |
| Signed ordering | ✅ | ✅ |
| Rz angles exact (`opt_level=0`) | ✅ | ✅ |

**Key findings:**

1. **Different attenuation, same phase:** $A_A = 0.909$ vs $A_B = 0.903$ (0.6% difference). The attenuation reflects qubit-specific noise ($T_1$/$T_2$), not the circuit structure — Layout B is shallower (depth 64 vs 76) yet has slightly lower $A$, ruling out a simple depth→attenuation relationship.

2. **Both nulls within 0.5σ:** $+0.4\sigma$ (Layout A) and $-0.12\sigma$ (Layout B). The null is not a property of specific qubits — it is a structural feature of the cross-half ZZ Hamiltonian at $\chi t = \pi$.

3. **Functional form layout-independent:** $R^2 > 0.975$ on both disjoint chains. The curve $T_{xx}(\chi t) \propto \cos^2(\chi t/2)$ is not a routing or calibration artifact.

**What this establishes:**

> *The predicted correlation curve $T_{xx} = A\cos^2(\chi t/2)/2$ and internal null $T_{xx}(\pi) \approx 0$ were recovered on two disjoint 4-qubit subgraphs of ibm_marrakesh with consistent curve shape ($R^2 > 0.975$) and null structure ($< 0.5\sigma$) across both layouts. The hardware observation is consistent with the reduced-state PTM structure predicted by Theorem 3 and is layout-independent within shot-noise uncertainty.*

**Summary of all three runs:**

| Run | Layout | Depth | $A$ | $R^2$ | Null $\sigma$ | Shots at $\pi$ |
|---|---|---|---|---|---|---|
| 1 | [0,1,2,3] | 43* | — | — | 2.1σ† | 500 |
| 2 | [0,1,2,3] | 76 | 0.909 | 0.986 | 0.4σ | 4000 |
| **3** | **[4,5,6,7]** | **64** | **0.903** | **0.976** | **0.12σ** | **4000** |

*Run 1 used `opt_level=2`; angles were shifted by compiler. Run 2 corrected to `opt_level=0`.
†Run 1 null was affected by compiler angle shift + readout asymmetry; resolved in Runs 2–3.

"""

related_pos = content.find("\n## Related Work")
content = content[:related_pos] + IX_G + content[related_pos:]

# ── 5. Update §VIII conclusion's IBM bullet ───────────────────────────────
OLD_IBM = "- **IBM quantum (§IX, pre-registered):** Cross-half OAT ($N=4$), 3-point PTM protocol ($\\chi t\\in\\{0,\\chi t^*,\\pi\\}$), measure $T_{xx}=\\langle X_1 X_2\\rangle/2$. Internal null at $\\chi t=\\pi$. Predicted separation: $\\sim14\\sigma$. Results pending."
NEW_IBM = "- **IBM quantum (§IX, complete):** Three runs on ibm_marrakesh. 9-point curve $T_{xx}(\\chi t)=A\\cos^2(\\chi t/2)/2$ confirmed on two disjoint layouts (R²=0.986/0.976). Internal null $T_{xx}(\\pi)$ within 0.5σ of zero on both layouts. Layout-independent: $A_A=0.909$, $A_B=0.903$."
content = content.replace(OLD_IBM, NEW_IBM)

# ── 6. Reframe the conclusion primary claim ───────────────────────────────
OLD_CLAIM = "**Primary claim:** *We identify a geometric phase structure governing operationally recoverable quantum transport. OAT and Dicke boundary channels possess finite-volume super-classical recovery basins ($V_Q>0$) that collapse at singular interaction phases ($\\chi t=\\pi$) and under dephasing — a property absent from all classical anisotropic channels and from product states.*"
NEW_CLAIM = "**Primary claim:** *We analyze the reduced-state correlation structure of cross-half ZZ OAT dynamics and validate key predictions on superconducting hardware. The PTM observable $T_{xx}(\\chi t) = \\cos^{N-2}(\\chi t/2)/2$ — proved analytically, reproduced exactly in tensor-network simulation, and recovered on ibm_marrakesh across two disjoint qubit layouts with $R^2 > 0.975$ — provides a calibration-transparent, layout-independent witness of the predicted cross-half correlation structure. The internal null at $\\chi t = \\pi$ is confirmed within 0.5σ on both layouts, consistent with the rank-collapse structure of Theorem 3. The recovery basin conjecture ($V_Q > 0$) remains a simulation-level observation motivating future operational tests.*"
content = content.replace(OLD_CLAIM, NEW_CLAIM)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

# ── Verify ────────────────────────────────────────────────────────────────
with open(path, encoding="utf-8") as f:
    txt = f.read()

checks = [
    ("New title",              "Reduced-State Correlation" in txt),
    ("Draft v9",               "Draft v9" in txt),
    ("Reframed abstract",      "validate the predicted observable structure" in txt),
    ("§IX.G present",          "IX.G Run 3" in txt),
    ("Comparison table",       "Layout A [0,1,2,3]" in txt),
    ("Run 3 job ID",           "d81tqhntjchs73bnccvg" in txt),
    ("Conclusion reframed",    "calibration-transparent, layout-independent witness" in txt),
    ("IBM bullet updated",     "two disjoint layouts" in txt),
    ("Related Work intact",    "## Related Work" in txt),
]
print("Paper verification:")
for label, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'}  {label}")
print("Done.")
