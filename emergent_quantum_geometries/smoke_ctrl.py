"""Quick smoke test: verify agnostic controller makes non-identity choices."""
import sys
sys.path.insert(0, '.')
import program_l_tpu as p

GATE_NAMES = ["I", "X", "Y", "Z", "H", "S"]

for model in ["XXZ", "Ising"]:
    rec = p.simulate_trajectory(
        model_name=model, N=6, controller_name="agnostic",
        noise_params={"T1_rate": 0.005, "T2_rate": 0.01},
        T_max=6.0, n_steps=30, seed=42,
        use_dropout=True, backend="numpy", store_obs=True
    )
    actions   = rec["ctrl_actions"]
    non_id    = [(i, a) for i, a in enumerate(actions) if a[0] != 0]
    gate_dist = {}
    for _, a in non_id:
        gate_dist[GATE_NAMES[a[0]]] = gate_dist.get(GATE_NAMES[a[0]], 0) + 1

    print(f"\n--- {model} ---")
    print(f"  steps={len(actions)}  non-identity={len(non_id)}")
    print(f"  gate distribution: {gate_dist}")
    print(f"  tau_transport: {rec['tau_transport']:.3f}")
    for step, act in non_id[:5]:
        print(f"    step {step:2d}: {GATE_NAMES[act[0]]} on site {act[1]}")
