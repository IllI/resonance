#!/usr/bin/env python3
"""
Test ResonanceNet on real Haxby fMRI BOLD data.
Generates resonance results and coherence visualization.
"""

import sys
import os
import numpy as np
import nibabel as nib
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from resonance_network import ResonanceNet, ResonanceConfig


def main():
    print("=" * 70)
    print("  RESONANCE on REAL fMRI BOLD DATA (Haxby Visual Recognition)")
    print("=" * 70)

    # ---- Load real data ----
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "haxby2001", "subj1")
    bold_path = os.path.join(data_dir, "bold.nii.gz")
    mask_path = os.path.join(data_dir, "mask4_vt.nii.gz")
    labels_path = os.path.join(data_dir, "labels.txt")

    print("\n[1/6] Loading BOLD fMRI data...")
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()
    print(f"  BOLD shape: {bold_data.shape}  (x, y, z, time)")
    print(f"  TR = {bold_img.header.get_zooms()[3]:.1f}s")

    print("\n[2/6] Applying ventral temporal cortex mask...")
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()
    masked_data = bold_data[mask > 0]  # (n_voxels, n_timepoints)
    print(f"  Masked shape: {masked_data.shape}  ({masked_data.shape[0]} VT voxels)")

    # Load labels
    import pandas as pd
    labels_df = pd.read_csv(labels_path, sep=" ")
    unique_labels = labels_df["labels"].unique()
    print(f"  Stimulus categories: {list(unique_labels)}")

    # ---- Use a meaningful subset ----
    # Take first 300 timepoints (~12.5 minutes of scan)
    n_tp = min(300, masked_data.shape[1])
    data_subset = masked_data[:, :n_tp]
    print(f"\n[3/6] Using first {n_tp} timepoints for resonance")
    print(f"  Subset shape: {data_subset.shape}")
    print(f"  Data range: [{np.min(data_subset):.1f}, {np.max(data_subset):.1f}]")

    # ---- Configure ResonanceNet ----
    config = ResonanceConfig(
        input_dim=n_tp,
        n_complex_neurons=64,
        n_spiking_neurons=128,
        n_harmonics=12,
        n_resonance_epochs=30,
        convergence_threshold=0.90,
        coupling_strength=0.15,
        learning_rate=0.02,
        damping=0.02,
        output_dir=os.path.join(os.path.dirname(__file__), "..", "resonance_results")
    )

    print("\n[4/6] Initializing ResonanceNet...")
    net = ResonanceNet(config)

    # ---- Resonate ----
    print("\n[5/6] Beginning resonance on VT cortex BOLD signal...")
    results = net.resonate_on_data(data_subset)

    # ---- Results ----
    print("\n" + "=" * 70)
    print("  RESONANCE RESULTS")
    print("=" * 70)
    print(f"  Converged:       {results['converged']}")
    print(f"  Final coherence: {results['final_coherence']:.4f}")
    print(f"  Epochs:          {results['epochs_completed']}")
    print(f"  Duration:        {results['duration_seconds']:.2f}s")

    print("\n  Epoch progression:")
    for e in results["epoch_metrics"]:
        bar_len = int(e["mean_coherence"] * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        print(f"    Epoch {e['epoch']+1:3d}: [{bar}] {e['mean_coherence']:.4f}")

    # ---- Network state snapshot ----
    state = net._get_network_state()
    print(f"\n  Network State:")
    print(f"    Mean weight amplitude:  {state['mean_weight_amplitude']:.4f}")
    print(f"    Mean natural frequency: {state['mean_natural_frequency']:.6f} Hz")
    print(f"    Harmonic frequencies:   {[f'{f:.4f}' for f in state['harmonic_frequencies'][:6]]}...")
    print(f"    Astrocyte calcium:      {[f'{c:.4f}' for c in state['astrocyte_calcium']]}")

    # ---- Save ----
    print("\n[6/6] Saving...")
    net.save()

    # Also save a summary
    summary = {
        "data_source": "haxby2001/subj1/bold.nii.gz",
        "mask": "mask4_vt.nii.gz",
        "n_voxels": int(masked_data.shape[0]),
        "n_timepoints_used": n_tp,
        "TR_seconds": float(bold_img.header.get_zooms()[3]),
        "converged": results["converged"],
        "final_coherence": results["final_coherence"],
        "epochs_completed": results["epochs_completed"],
        "duration_seconds": results["duration_seconds"],
        "epoch_coherences": [e["mean_coherence"] for e in results["epoch_metrics"]],
        "stimulus_categories": list(unique_labels)
    }
    summary_path = os.path.join(config.output_dir, "resonance_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    # ---- Generate coherence plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("ResonanceNet — Haxby BOLD fMRI Resonance", fontsize=16, fontweight="bold")

        # 1. Coherence over epochs
        coherences = [e["mean_coherence"] for e in results["epoch_metrics"]]
        max_coh = [e["max_coherence"] for e in results["epoch_metrics"]]
        min_coh = [e["min_coherence"] for e in results["epoch_metrics"]]
        epochs = range(1, len(coherences) + 1)

        axes[0, 0].fill_between(epochs, min_coh, max_coh, alpha=0.3, color="steelblue")
        axes[0, 0].plot(epochs, coherences, "o-", color="steelblue", linewidth=2, label="Mean")
        axes[0, 0].axhline(y=config.convergence_threshold, color="red", linestyle="--", alpha=0.7, label="Target")
        axes[0, 0].set_xlabel("Resonance Epoch")
        axes[0, 0].set_ylabel("Phase Coherence")
        axes[0, 0].set_title("Resonance Convergence")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Harmonic filter frequencies
        freqs = state["harmonic_frequencies"]
        amps = state["harmonic_amplitudes"]
        axes[0, 1].bar(range(len(freqs)), freqs, color="coral", alpha=0.8)
        axes[0, 1].set_xlabel("Harmonic Index")
        axes[0, 1].set_ylabel("Frequency (Hz)")
        axes[0, 1].set_title("Learned Harmonic Frequencies")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Complex neuron resonance scores
        neuron_scores = [n.resonance_score for n in net.complex_neurons]
        axes[1, 0].hist(neuron_scores, bins=20, color="seagreen", alpha=0.8, edgecolor="black")
        axes[1, 0].set_xlabel("Final Resonance Score")
        axes[1, 0].set_ylabel("Neuron Count")
        axes[1, 0].set_title("Complex Neuron Resonance Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Sample BOLD time series with resonance overlay
        sample_idx = np.argmax(np.var(data_subset, axis=1))
        sample_signal = data_subset[sample_idx, :100]
        sample_signal_norm = (sample_signal - np.mean(sample_signal)) / np.std(sample_signal)
        axes[1, 1].plot(sample_signal_norm, color="gray", alpha=0.5, label="BOLD signal")
        # Show harmonic reconstruction
        h_result = net.harmonic_layer.convolve(sample_signal_norm)
        axes[1, 1].plot(h_result["interference_pattern"][:100], color="red", linewidth=1.5, label="Harmonic interference")
        axes[1, 1].set_xlabel("Time (TR)")
        axes[1, 1].set_ylabel("Amplitude")
        axes[1, 1].set_title("BOLD Signal vs Harmonic Interference")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(config.output_dir, "resonance_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved to {plot_path}")

    except ImportError as e:
        print(f"  (matplotlib not available for plotting: {e})")

    print("\n" + "=" * 70)
    print("  Resonance complete.")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
