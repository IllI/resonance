"""
run_v2_experiment_updated.py
Drop-in replacement for run_v2_experiment.py.

Changes from original:
  Phase 1  —  ExperimentalDataIngester replaces IntraSliceJitterStreamer
  Phase 5  —  framework_residuals.run_framework_comparison() replaces
               NQS meta-learner (can run both for comparison)

Quick-start
-----------
# Calibration run (synthetic SYK — known answer):
python run_v2_experiment_updated.py --source SYNTHETIC --framework syk

# Real SYK ED data (download from github.com/BlackHatLKJH/SYK-model):
python run_v2_experiment_updated.py --source SYK_ED --data_dir ./data

# Live Rydberg hardware (requires Amazon Braket credentials):
python run_v2_experiment_updated.py --source RYDBERG --data_dir ./data

# Google Sycamore RCS bitstrings:
python run_v2_experiment_updated.py --source SYCAMORE_RCS --data_dir ./data
"""

import argparse
import jax
import jax.numpy as jnp
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from importlib.util import spec_from_file_location, module_from_spec

def load_module(name, path):
    spec = spec_from_file_location(name, path)
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── existing modules (unchanged) ──────────────────────────────────
mapping    = load_module("mapping",    os.path.join(base_dir, "2_state_mapping/dlinoss_quantum_damper.py"))
heisenberg = load_module("heisenberg", os.path.join(base_dir, "3_projection/heisenberg_space.py"))
bitwistor  = load_module("bitwistor",  os.path.join(base_dir, "3_projection/bitwistor_space.py"))
otoc_mod   = load_module("otoc",       os.path.join(base_dir, "4_observation/otoc_metrics.py"))
kink_mod   = load_module("kink",       os.path.join(base_dir, "4_observation/or_kink_detection.py"))
meta_mod   = load_module("meta",       os.path.join(base_dir, "5_classification/nqs_meta_learner.py"))

# ── NEW modules ────────────────────────────────────────────────────
from experimental_data_ingester import ExperimentalDataIngester
from framework_residuals import run_framework_comparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",    default="SYNTHETIC",
                   choices=["SYNTHETIC", "SYK_ED", "SYCAMORE_RCS", "RYDBERG"])
    p.add_argument("--framework", default="syk",
                   choices=["heisenberg", "lindblad", "penrose_or", "syk", "mbl"],
                   help="Only used when --source SYNTHETIC")
    p.add_argument("--data_dir",  default="./data")
    p.add_argument("--batch",     type=int, default=128)
    p.add_argument("--dt",        type=float, default=0.1,
                   help="Physical timestep in units of J^{-1}")
    p.add_argument("--beta",      type=float, default=5.0,
                   help="Inverse temperature (SYK Lyapunov = 2π/beta)")
    p.add_argument("--both-classifiers", action="store_true",
                   help="Run both NQS meta-learner AND residual comparator")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 56)
    print("  Emergent Quantum Geometries via D-LinOSS  (V2+)")
    print(f"  Data source : {args.source}"
          + (f" [{args.framework}]" if args.source == "SYNTHETIC" else ""))
    print(f"  TPU backend : {jax.default_backend()}")
    print("=" * 56)

    # ── Phase 1: Ingestion ─────────────────────────────────────────
    print("\n[Phase 1] Ingesting experimental data...")
    ingester = ExperimentalDataIngester(
        source     = args.source,
        framework  = args.framework,
        data_dir   = args.data_dir,
        dt         = args.dt,
    )
    drift_data = ingester.stream_phase_data(batch_size=args.batch)
    print(f"  Data tensor shape: {drift_data.shape}")
    print(f"  Signal range:      [{float(drift_data.min()):.4f}, {float(drift_data.max()):.4f}]")

    # ── Phase 2: D-LiNOSS state mapping ───────────────────────────
    print("\n[Phase 2] Mapping to Lindbladian damper via D-LiNOSS...")
    damper_net  = mapping.DLinOSSQuantumDamper()
    rng         = jax.random.PRNGKey(0)
    variables   = damper_net.init(rng, drift_data)
    damping_matrix = damper_net.apply(variables, drift_data)
    print(f"  Damping matrix shape: {damping_matrix.shape}")

    # ── Phase 3: Projection ────────────────────────────────────────
    print("\n[Phase 3] Projecting into competing quantum spaces...")
    initial_state = jnp.ones(32) / jnp.sqrt(32)
    h_space = heisenberg.HeisenbergSimulator()
    b_space = bitwistor.BiTwistorSimulator(threshold=0.8)
    h_state = h_space.apply_damper(initial_state, damping_matrix[0])
    b_state = b_space.apply_damper(initial_state, damping_matrix[0], current_energy=0.9)
    print("  Projection complete.")

    # ── Phase 4: Observation ───────────────────────────────────────
    print("\n[Phase 4] Computing physical observables...")
    otoc_calc = otoc_mod.OTOCCalculator()
    kink_det  = kink_mod.ORKinkDetector()
    h_history = jnp.stack([initial_state, h_state])
    b_history = jnp.stack([initial_state, b_state])
    otoc_curve = otoc_calc.calculate_otoc_decay(h_history)
    has_kink   = kink_det.detect_kink(b_history)
    print(f"  OTOC curve length : {len(otoc_curve)}")
    print(f"  OR kink detected  : {has_kink}")

    # ── Phase 5A: Residual Framework Comparator (new) ──────────────
    print("\n[Phase 5A] Framework Residual Comparison...")
    result = run_framework_comparison(
        damper_net     = damper_net,
        variables      = variables,
        damping_matrix = damping_matrix,
        dt             = args.dt,
        beta           = args.beta,
    )
    probs_residual = result["probabilities"]
    H_residual     = result["entropy"]

    # ── Phase 5B: Original NQS meta-learner (optional) ─────────────
    if args.both_classifiers:
        print("\n[Phase 5B] NQS Meta-Learner (original, for comparison)...")
        meta_net  = meta_mod.NQSMetaLearner()
        meta_vars = meta_net.init(rng, otoc_curve, jnp.array([has_kink]))
        probs_nqs = meta_net.apply(meta_vars, otoc_curve, jnp.array([has_kink]))
        H_nqs     = meta_mod.calculate_entropy(probs_nqs)

        print("\n  NQS meta-learner result:")
        print(f"    P(Heisenberg)    = {float(probs_nqs[0]):.4f}")
        print(f"    P(Bi-Twistor OR) = {float(probs_nqs[1]):.4f}")
        print(f"    Entropy          = {H_nqs:.4f}")
        print()
        print("  Comparison of entropy (lower = more decisive):")
        print(f"    Residual comparator : {H_residual:.4f}")
        print(f"    NQS meta-learner    : {H_nqs:.4f}")

    # ── Final summary ──────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  FINAL SUMMARY")
    print("=" * 56)
    print(f"  Source          : {args.source}")
    print(f"  Winning frame.  : {result['winner'].upper()}")
    print(f"  Entropy         : {H_residual:.4f}  "
          f"(max {__import__('numpy').log(5):.4f})")
    print(f"  Confidence      : {(1 - H_residual / __import__('numpy').log(5)):.1%}")
    print("=" * 56)

    # ── Calibration note ───────────────────────────────────────────
    if args.source == "SYNTHETIC":
        expected = args.framework
        actual   = result["winner"]
        passed   = "✓ PASS" if actual == expected else "✗ FAIL"
        print(f"\n  [CALIBRATION CHECK] Expected={expected}  Got={actual}  {passed}")
        if actual != expected:
            print("  → D-LiNOSS parameter extraction may need key path adjustment.")
            print("    See extract_dlinoss_params() in framework_residuals.py")
    print()


if __name__ == "__main__":
    main()
