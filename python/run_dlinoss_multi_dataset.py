"""
Run D-LinOSS fMRI analysis across multiple datasets and subjects.
Compares Haxby (task-based visual) vs OpenNeuro MPI (resting-state).
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dlinoss_fmri import DLinOSSfMRI, DLinOSSConfig


def analyze_volume(name: str, bold_path: str, mask_path: str = None,
                   TR: float = 2.0, max_tp: int = 100, max_voxels: int = 600):
    """Run D-LinOSS on a single volume."""
    import nibabel as nib

    print(f"\n{'#'*70}")
    print(f"  DATASET: {name}")
    print(f"  Path: {bold_path}")
    print(f"{'#'*70}")

    img = nib.load(bold_path)
    bold = img.get_fdata()
    print(f"  Volume shape: {bold.shape}")

    if mask_path and os.path.exists(mask_path):
        mask = nib.load(mask_path).get_fdata()
        data = bold[mask > 0]
        print(f"  Using mask: {mask_path} -> {data.shape[0]} voxels")
    else:
        # No mask — pick highest-variance voxels from whole brain
        flat = bold.reshape(-1, bold.shape[-1])
        # Filter out zero/near-zero variance voxels (background)
        variances = np.var(flat, axis=1)
        threshold = np.percentile(variances[variances > 0], 50)  # top 50% of nonzero
        valid_idx = np.where(variances > threshold)[0]
        if len(valid_idx) > max_voxels:
            top_idx = np.argsort(variances[valid_idx])[-max_voxels:]
            valid_idx = valid_idx[top_idx]
        data = flat[valid_idx]
        print(f"  Auto-selected {data.shape[0]} high-variance voxels (no mask)")

    # Limit timepoints
    n_tp = min(max_tp, data.shape[1])
    data = data[:, :n_tp].astype(np.float64)

    # Detect TR from header if available
    pixdim = img.header.get('pixdim', None)
    if pixdim is not None and len(pixdim) > 4 and pixdim[4] > 0.1:
        TR = float(pixdim[4])
        print(f"  TR from header: {TR}s")
    else:
        print(f"  Using default TR: {TR}s")

    # Configure
    config = DLinOSSConfig(
        state_dim=32,
        hidden_dim=64,
        n_blocks=3,
        model_dim=32,
        n_features=data.shape[0],
        TR=TR,
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'dlinoss_results', name.replace(' ', '_'))
    )

    model = DLinOSSfMRI(config)
    t0 = time.time()
    results = model.process_bold_volume(data)
    elapsed = time.time() - t0

    # Generate plots
    model.generate_plots(data, results)

    results['meta'] = {
        'name': name,
        'bold_path': bold_path,
        'volume_shape': list(bold.shape),
        'n_voxels_used': int(data.shape[0]),
        'n_timepoints_used': n_tp,
        'TR': TR,
        'elapsed_seconds': elapsed
    }

    return results


def main():
    base = os.path.join(os.path.dirname(__file__), '..')
    all_results = {}

    # ============================================================
    # Dataset 1: Haxby 2001 — Task-based visual object recognition
    # ============================================================
    haxby_bold = os.path.join(base, 'data', 'haxby2001', 'subj1', 'bold.nii.gz')
    haxby_mask = os.path.join(base, 'data', 'haxby2001', 'subj1', 'mask4_vt.nii.gz')

    if os.path.exists(haxby_bold):
        r = analyze_volume("Haxby_VT", haxby_bold, haxby_mask, TR=2.5, max_tp=300)
        all_results['haxby_vt'] = r

        # Also try face-specific and house-specific masks
        face_mask = os.path.join(base, 'data', 'haxby2001', 'subj1', 'mask8_face_vt.nii.gz')
        house_mask = os.path.join(base, 'data', 'haxby2001', 'subj1', 'mask8_house_vt.nii.gz')

        if os.path.exists(face_mask):
            r = analyze_volume("Haxby_FaceRegion", haxby_bold, face_mask, TR=2.5, max_tp=300)
            all_results['haxby_face'] = r

        if os.path.exists(house_mask):
            r = analyze_volume("Haxby_HouseRegion", haxby_bold, house_mask, TR=2.5, max_tp=300)
            all_results['haxby_house'] = r

    # ============================================================
    # Dataset 2: OpenNeuro MPI — Resting-state fMRI (5 subjects)
    # ============================================================
    mpi_base = os.path.join(base, 'python', 'brain_datasets', 'openneuro_mpi')
    for subj_num in range(1, 6):
        subj_id = f"sub-{subj_num:04d}"
        bold_path = os.path.join(mpi_base, subj_id, 'func', f'{subj_id}_task-rest_bold.nii.gz')
        if os.path.exists(bold_path):
            r = analyze_volume(
                f"MPI_RestingState_{subj_id}",
                bold_path,
                mask_path=None,
                TR=1.0,  # Resting state often 1s TR; will auto-detect
                max_tp=100,
                max_voxels=500
            )
            all_results[f'mpi_{subj_id}'] = r

    # ============================================================
    # Summary comparison
    # ============================================================
    print("\n" + "=" * 80)
    print("  CROSS-DATASET COMPARISON")
    print("=" * 80)
    print(f"\n{'Dataset':<30} {'GNR':>8} {'Ensheath':>10} {'PreFlux':>10} "
          f"{'Coherence':>10} {'SuperPos':>10} {'Time':>8}")
    print("-" * 92)

    for key, r in all_results.items():
        name = r.get('meta', {}).get('name', key)[:28]
        gnr = r.get('astrocyte', {}).get('mean_gnr', 0)
        ens = r.get('astrocyte', {}).get('mean_ensheathment', 0)
        pf = r.get('astrocyte', {}).get('n_preflux_events', 0)
        coh = r.get('dlinoss', {}).get('output_coherence', 0)
        sp = r.get('superposition', {}).get('n_represented', 0)
        elapsed = r.get('meta', {}).get('elapsed_seconds', 0)
        print(f"  {name:<28} {gnr:>8.3f} {ens:>10.4f} {pf:>10d} "
              f"{coh:>10.4f} {sp:>10d} {elapsed:>7.1f}s")

    # Astrocyte insight: compare task vs resting state
    haxby_gnrs = [r['astrocyte']['mean_gnr'] for k, r in all_results.items() if 'haxby' in k]
    mpi_gnrs = [r['astrocyte']['mean_gnr'] for k, r in all_results.items() if 'mpi' in k]

    if haxby_gnrs and mpi_gnrs:
        print(f"\n  TASK vs REST GNR comparison:")
        print(f"    Task-based (Haxby) mean GNR:     {np.mean(haxby_gnrs):.3f} ± {np.std(haxby_gnrs):.3f}")
        print(f"    Resting-state (MPI) mean GNR:    {np.mean(mpi_gnrs):.3f} ± {np.std(mpi_gnrs):.3f}")
        ratio = np.mean(mpi_gnrs) / np.mean(haxby_gnrs) if np.mean(haxby_gnrs) > 0 else 0
        print(f"    Resting/Task GNR ratio:          {ratio:.2f}x")
        if ratio > 1:
            print(f"    >> Astrocyte activity is {ratio:.1f}x STRONGER in resting state!")
            print(f"       This is expected: at rest, the 'background hum' of the astrocyte")
            print(f"       network dominates without stimulus-locked neuronal spikes.")
        else:
            print(f"    >> Task-based GNR is higher (task engages both neuron + astrocyte)")
            print(f"       With TR=1.0s (MPI), the astrocyte band (0.01-0.1Hz) is narrow")
            print(f"       relative to the neuronal band (0.1-0.5Hz), shifting the ratio.")

    # Save combined results
    out_path = os.path.join(base, 'dlinoss_results', 'multi_dataset_comparison.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Make JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n  Combined results saved to: {out_path}")

    return all_results


if __name__ == "__main__":
    main()
