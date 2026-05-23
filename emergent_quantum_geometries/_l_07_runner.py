# _l_07_runner.py  —  Trajectory simulator, experiment runner, main


def simulate_trajectory(model_name, N, controller_name, noise_params,
                        T_max, n_steps, seed, use_dropout=True):
    """
    Simulate one noisy trajectory under one controller.
    Returns dict with F_rec time series and tau_transport.
    """
    rng    = np.random.default_rng(seed)
    d_rng  = np.random.default_rng(seed + 10000)   # separate for dropout
    H      = MODEL_BUILDERS[model_name](N, seed)
    dt     = T_max / n_steps
    dim    = 2 ** N

    # Initial state: Neel-like product state
    rho = np.zeros((dim, dim), dtype=complex)
    neel_idx = sum(1 << i for i in range(0, N, 2))
    rho[neel_idx, neel_idx] = 1.0

    obs_history = []
    t_history   = []
    f_rec_series = []
    in_transport = []

    controller = get_controller(controller_name)
    rho_prev   = None

    for step in range(n_steps):
        t_cur = step * dt
        noise_params["_t_cur"] = t_cur

        # Controller decision
        if controller_name == "static":
            ci, site = controller(step, N)
        elif controller_name == "haware":
            ci, site = controller(step, N, rho=rho, H=H, dt=dt,
                                  noise_params=noise_params)
        else:  # agnostic
            ci, site = controller(step, N, rho=rho,
                                  obs_history=obs_history,
                                  t_history=t_history,
                                  dropout_rng=d_rng,
                                  use_dropout=use_dropout)

        # Apply pulse if not identity
        if ci != 0:
            rho = apply_clifford(rho, N, ci, site)

        # Time evolve
        U   = make_U(H, dt)
        rho = evolve_rho(rho, U)
        rho = apply_noise(rho, N, noise_params, dt)

        # Compute observables
        x_obs = compute_observables(rho, N, rho_prev, dt, rng)
        obs_history.append(x_obs.tolist())
        t_history.append(t_cur)

        # F_rec (post-hoc, NOT seen by agnostic controller)
        rho4   = partial_trace_boundary(rho, N)
        T_mat  = compute_T_matrix(rho4)
        f_rec  = f_rec_from_T(T_mat)
        f_rec_series.append(float(f_rec))
        in_transport.append(int(infer_transport_strata(x_obs)))

        rho_prev = rho.copy()

    # Compute tau_transport: last step where F_rec > TH_F_REC
    above = [i for i, f in enumerate(f_rec_series) if f > TH_F_REC]
    tau_transport = float(above[-1] * dt) if above else 0.0

    return {
        "model":       model_name,
        "controller":  controller_name,
        "N":           N,
        "seed":        seed,
        "tau_transport": tau_transport,
        "f_rec_series":  f_rec_series,
        "in_transport":  in_transport,
        "T_max":       T_max,
        "n_steps":     n_steps,
        "noise_params": {k: v for k, v in noise_params.items()
                         if not k.startswith("_")},
    }


def sample_noise_params(rng):
    """Sample noise parameters from pre-registered ranges."""
    return {
        "T1_rate":    float(rng.uniform(0.005, 0.05)),
        "T2_rate":    float(rng.uniform(0.005, 0.10)),
        "drift_amp":  float(rng.uniform(0.0,   0.05)),
        "drift_freq": float(rng.uniform(0.05,  0.5)),
    }


def run_experiment(args):
    models      = args.models
    controllers = ["static", "haware", "agnostic"]
    results     = []
    t0          = time.time()

    print("=" * 60)
    print("Program L — Representation-Agnostic Adaptive Recovery")
    print("=" * 60)
    print(f"N={args.N}  T_max={args.T_max}  n_steps={args.n_steps}")
    print(f"Models: {models}")
    print(f"Bootstrap samples: {args.B}")

    rng_noise = np.random.default_rng(args.seed)

    for model in models:
        print(f"\n── Model: {model} ──")
        for b in range(args.B):
            noise_params = sample_noise_params(rng_noise)
            seed_b = args.seed + b * 1000
            for ctrl in controllers:
                t1 = time.time()
                rec = simulate_trajectory(
                    model_name=model,
                    N=args.N,
                    controller_name=ctrl,
                    noise_params=noise_params.copy(),
                    T_max=args.T_max,
                    n_steps=args.n_steps,
                    seed=seed_b,
                    use_dropout=(ctrl == "agnostic"),
                )
                results.append(rec)
                dt_run = time.time() - t1
                print(f"  [{ctrl:8s}] B={b}  tau={rec['tau_transport']:.3f}  "
                      f"F_final={rec['f_rec_series'][-1]:.3f}  ({dt_run:.1f}s)")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GAIN SUMMARY  (gain = (agnostic-static)/(haware-static))")
    print("=" * 60)
    summary = {}
    for model in models:
        recs = [r for r in results if r["model"] == model]
        taus = {c: [r["tau_transport"] for r in recs if r["controller"] == c]
                for c in controllers}
        tau_s = float(np.mean(taus["static"]))   if taus["static"]   else 0.0
        tau_h = float(np.mean(taus["haware"]))   if taus["haware"]   else 0.0
        tau_a = float(np.mean(taus["agnostic"])) if taus["agnostic"] else 0.0
        denom = tau_h - tau_s
        gain  = float((tau_a - tau_s) / denom) if abs(denom) > 1e-9 else float("nan")
        summary[model] = {
            "tau_static": tau_s, "tau_haware": tau_h,
            "tau_agnostic": tau_a, "gain": gain,
        }
        print(f"  {model:20s}  tau_s={tau_s:.3f}  tau_h={tau_h:.3f}  "
              f"tau_a={tau_a:.3f}  gain={gain:.3f}")

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"program_l_N{args.N}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": summary,
            "results": [{k: v for k, v in r.items() if k != "f_rec_series"}
                        for r in results],
            "runtime_s": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {out_path}  ({time.time()-t0:.1f}s total)")
    return summary


def main():
    p = argparse.ArgumentParser(description="Program L: Agnostic Transport Recovery")
    p.add_argument("--N",       type=int,   default=8)
    p.add_argument("--T-max",   type=float, default=6.0, dest="T_max")
    p.add_argument("--n-steps", type=int,   default=30,  dest="n_steps")
    p.add_argument("--B",       type=int,   default=3,
                   help="Bootstrap samples per model")
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--models",  nargs="+",
                   default=["XXZ", "Ising"],
                   choices=list(MODEL_BUILDERS.keys()))
    p.add_argument("--out-dir", default="program_l_results", dest="out_dir")
    p.add_argument("--smoke-test", action="store_true",
                   help="Quick validation: N=6, 2 models, B=1, 15 steps")
    args = p.parse_args()

    if args.smoke_test:
        args.N       = 6
        args.n_steps = 15
        args.B       = 1
        args.T_max   = 4.0
        args.models  = ["XXZ", "Ising"]
        print("SMOKE TEST MODE")

    run_experiment(args)


if __name__ == "__main__":
    main()
