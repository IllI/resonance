#!/usr/bin/env python3
"""
Priority Dataset Downloader for D-LinOSS fMRI Pipeline
========================================================

Downloads the datasets that best match our preprocessing → model pipeline:

1. ds000258 — Multi-Echo Cambridge (TOP PRIORITY)
   - 4 echo times (12, 28, 44, 60 ms) → T₂* decomposition
   - Physiological recordings (cardiac PPG + respiratory chest belt) → RETROICOR
   - 89 participants at 3T Siemens Trio, 32-channel head coil
   - Download: ~2-4 GB per subject for func + physio

2. ds000105 — Haxby Visual Object Recognition (subs 4-6, remaining)
   - Task-based with known stimulus timing → HRF deconvolution validation
   - Known ROIs: VT cortex, FFA, PPA
   - Already have sub-1, sub-2, sub-3 (~300 MB each)

Why these datasets?
-------------------
Our MRI artifact correction pipeline has 6 stages:
  ① Coil Sensitivity       — estimated from data (all datasets)
  ② B₀ Inhomogeneity       — estimated from phase (all datasets)
  ③ T₂* Decomposition      — REQUIRES multi-echo data → ds000258
  ④ HRF Deconvolution      — REQUIRES known stimulus timing → ds000105
  ⑤ Physiological Noise    — REQUIRES cardiac/resp recordings → ds000258
  ⑥ Gradient Nonlinearity  — estimated from geometry (all datasets)

ds000258 alone covers stages ③ + ⑤ (the two that can't be estimated).
ds000105 provides ground truth for stage ④.
"""

import os
import sys
import time
import argparse

# Fix Windows encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'openneuro')


def download_multiecho_cambridge(n_subjects=5, start_subject=0):
    """
    Download ds000258 — Multi-Echo Cambridge.

    This is our TOP PRIORITY dataset because it provides:
    - 4 echo times (12, 28, 44, 60 ms) for T2* fitting:
        S(TE) = S0 * exp(-TE / T2*)
      With 4 points we can robustly fit the decay curve and separate
      BOLD-related T2* changes from non-BOLD artifacts (susceptibility,
      motion, etc.)

    Scanner: Siemens Trio 3T, 32-channel receive-only head coil (32Ch_Head_3T)
    TR: 2.47s, Flip angle: 78 deg
    Voxel: 3.75 x 3.75 x 4.4 mm (with 10% gap = 4.84mm spacing)
    Echo times: 12, 28, 44, 60 ms
    Paradigm: Resting-state, eyes open, 139 volumes (9.84 min)
    Parallel acceleration: GRAPPA factor 3
    Alternating interleaved slice acquisition (30 slices)
    Total participants available: 89

    Data per subject: ~120-130 MB (4 echo NIfTI + 4 JSON sidecars)
    """
    import openneuro

    ds_id = 'ds000258'
    target = os.path.join(BASE_DIR, ds_id)
    os.makedirs(target, exist_ok=True)

    # Actual subject IDs from ds000258 (89 subjects with 5-digit IDs)
    ALL_SUBJECTS = [
        'sub-04570', 'sub-04620', 'sub-04710', 'sub-04800', 'sub-11310',
        'sub-12599', 'sub-16053', 'sub-17821', 'sub-19284', 'sub-19774',
        'sub-19943', 'sub-19979', 'sub-20094', 'sub-20407', 'sub-20408',
        'sub-20409', 'sub-20450', 'sub-20462', 'sub-20494', 'sub-20503',
        'sub-20758', 'sub-20823', 'sub-20824', 'sub-20835', 'sub-20847',
        'sub-20851', 'sub-20859', 'sub-20860', 'sub-20861', 'sub-20862',
        'sub-20863', 'sub-20864', 'sub-20866', 'sub-20892', 'sub-20893',
        'sub-20907', 'sub-20932', 'sub-20946', 'sub-20950', 'sub-20976',
        'sub-20983', 'sub-20984', 'sub-21036', 'sub-21116', 'sub-21124',
        'sub-21126', 'sub-21127', 'sub-21187', 'sub-21193', 'sub-21215',
        'sub-21252', 'sub-21253', 'sub-21254', 'sub-21256', 'sub-21258',
        'sub-21260', 'sub-21261', 'sub-21262', 'sub-21270', 'sub-21273',
        'sub-21277', 'sub-21278', 'sub-21279', 'sub-21281', 'sub-21283',
        'sub-21300', 'sub-21302', 'sub-21304', 'sub-21305', 'sub-21306',
        'sub-21317', 'sub-21318', 'sub-21324', 'sub-21371', 'sub-21384',
        'sub-21448', 'sub-21449', 'sub-21452', 'sub-21453', 'sub-21456',
        'sub-21611', 'sub-21641', 'sub-21644', 'sub-21645', 'sub-21646',
        'sub-21652', 'sub-21656', 'sub-21657', 'sub-21658',
    ]

    subjects_to_download = ALL_SUBJECTS[start_subject:start_subject + n_subjects]

    print(f"\n{'='*70}")
    print(f"  DOWNLOADING: {ds_id} -- Multi-Echo Cambridge")
    print(f"  Scanner: Siemens Trio 3T, 32-channel head coil")
    print(f"  Echo times: 12, 28, 44, 60 ms (4 echoes)")
    print(f"  TR: 2.47s, 139 volumes, 30 slices")
    print(f"  Target dir: {os.path.abspath(target)}")
    print(f"  Subjects: {len(subjects_to_download)} "
          f"(#{start_subject+1}-{start_subject+len(subjects_to_download)} of {len(ALL_SUBJECTS)})")
    print(f"{'='*70}\n")

    downloaded = 0
    failed = []

    for subj in subjects_to_download:
        subj_func = os.path.join(target, subj, 'func')

        # Check if already downloaded
        if os.path.exists(subj_func):
            nii_files = [f for f in os.listdir(subj_func) if f.endswith('.nii.gz')]
            if len(nii_files) >= 4:  # 4 echo files
                print(f"  [SKIP] {subj} already has {len(nii_files)} echo files")
                downloaded += 1
                continue

        # Include patterns for this subject (func + anat)
        includes = [
            f'{subj}/func',
            f'{subj}/anat',
        ]

        print(f"  Downloading {subj}...")
        t0 = time.time()

        try:
            openneuro.download(
                dataset=ds_id,
                target_dir=target,
                include=includes
            )
            elapsed = time.time() - t0

            # Verify what we got
            if os.path.exists(subj_func):
                files = os.listdir(subj_func)
                nii_count = len([f for f in files if f.endswith('.nii.gz')])
                total_mb = sum(os.path.getsize(os.path.join(subj_func, f))
                              for f in files) / (1024*1024)
                print(f"  [OK] {subj}: {nii_count} NIfTI ({total_mb:.0f} MB, {elapsed:.0f}s)")
                downloaded += 1
            else:
                print(f"  [WARN] {subj}: func dir not created")
                failed.append(subj)

        except Exception as e:
            print(f"  [ERROR] {subj}: {e}")
            failed.append(subj)

    print(f"\n  Multi-Echo Cambridge: {downloaded} downloaded, {len(failed)} failed")
    if failed:
        print(f"  Failed subjects: {', '.join(failed)}")

    return downloaded


def download_haxby_remaining():
    """
    Download ds000105 — Haxby Visual Object Recognition (remaining subjects).

    Already have: sub-1, sub-2, sub-3 (~300 MB each, 12 runs × ~24 MB)
    Need: sub-4, sub-5, sub-6

    This dataset is critical for HRF deconvolution validation because:
    - Block design with KNOWN stimulus onsets (faces, houses, etc.)
    - The block timing lets us construct the expected BOLD response:
        expected_BOLD(t) = stimulus(t) ⊛ HRF(t)
      where ⊛ is convolution and HRF peaks at 5-6s.
    - By deconvolving, we should recover stimulus-locked neural activity
      shifted back by ~5s from the BOLD peak.

    Known ROIs and their functions:
    - Ventral Temporal (VT): general visual object processing
    - Fusiform Face Area (FFA): face-specific activation
    - Parahippocampal Place Area (PPA): scene/house-specific activation

    Scanner: 3T, 12 runs per subject, TR = 2.5s
    """
    import openneuro

    ds_id = 'ds000105'
    target = os.path.join(BASE_DIR, ds_id)
    os.makedirs(target, exist_ok=True)

    # Check which subjects we already have
    existing = set()
    for d in os.listdir(target):
        if d.startswith('sub-'):
            func_dir = os.path.join(target, d, 'func')
            if os.path.exists(func_dir):
                nii_files = [f for f in os.listdir(func_dir) if f.endswith('.nii.gz')]
                if len(nii_files) >= 10:  # at least 10 of 12 runs
                    existing.add(d)

    all_subjects = [f'sub-{i}' for i in range(1, 7)]
    needed = [s for s in all_subjects if s not in existing]

    print(f"\n{'='*70}")
    print(f"  DOWNLOADING: {ds_id} — Haxby Visual Object Recognition")
    print(f"  Task: Block-design object viewing (faces, houses, bottles, etc.)")
    print(f"  ROIs: VT cortex, FFA, PPA")
    print(f"  Already have: {', '.join(sorted(existing)) or 'none'}")
    print(f"  Need: {', '.join(needed) or 'all downloaded!'}")
    print(f"  Estimated: ~300 MB per subject")
    print(f"{'='*70}\n")

    if not needed:
        print("  All 6 subjects already downloaded!")
        return len(existing)

    downloaded = 0
    for subj in needed:
        includes = [
            f'{subj}/func/*bold*',
            f'{subj}/func/*events*',
            f'{subj}/anat/*T1w*',
            'dataset_description.json',
        ]

        print(f"  Downloading {subj}...")
        t0 = time.time()

        try:
            openneuro.download(
                dataset=ds_id,
                target_dir=target,
                include=includes
            )
            elapsed = time.time() - t0

            func_dir = os.path.join(target, subj, 'func')
            if os.path.exists(func_dir):
                files = os.listdir(func_dir)
                nii_count = len([f for f in files if f.endswith('.nii.gz')])
                total_mb = sum(os.path.getsize(os.path.join(func_dir, f))
                              for f in files) / (1024*1024)
                print(f"  [OK] {subj}: {nii_count} runs ({total_mb:.0f} MB, {elapsed:.0f}s)")
                downloaded += 1
            else:
                print(f"  [WARN] {subj}: no func dir created")

        except Exception as e:
            print(f"  [ERROR] {subj}: {e}")

    total = len(existing) + downloaded
    print(f"\n  Haxby: {total}/6 subjects available ({downloaded} new)")
    return total


def verify_datasets():
    """
    Verify downloaded datasets and show what's available for the pipeline.
    """
    print(f"\n{'='*70}")
    print(f"  DATASET INVENTORY")
    print(f"{'='*70}\n")

    # Check Multi-Echo Cambridge
    me_dir = os.path.join(BASE_DIR, 'ds000258')
    if os.path.exists(me_dir):
        subjects = [d for d in os.listdir(me_dir) if d.startswith('sub-')]
        total_mb = 0
        echo_summary = {}
        physio_count = 0

        for subj in sorted(subjects):
            func_dir = os.path.join(me_dir, subj, 'func')
            if os.path.exists(func_dir):
                files = os.listdir(func_dir)
                for f in files:
                    fp = os.path.join(func_dir, f)
                    total_mb += os.path.getsize(fp) / (1024*1024)
                    if 'echo-' in f:
                        echo = f.split('echo-')[1].split('_')[0]
                        echo_summary[echo] = echo_summary.get(echo, 0) + 1
                    if 'physio' in f:
                        physio_count += 1

        print(f"  ds000258 — Multi-Echo Cambridge:")
        print(f"    Subjects: {len(subjects)}")
        print(f"    Total size: {total_mb:.0f} MB")
        print(f"    Echo files: {echo_summary}")
        print(f"    Physio files: {physio_count}")
        print(f"    Pipeline stages enabled: T₂* decomposition ③, Physio regression ⑤")
    else:
        print(f"  ds000258 — Multi-Echo Cambridge: NOT DOWNLOADED")
        print(f"    ⚠ Cannot run T₂* decomposition or real physiological regression")

    # Check Haxby
    haxby_dir = os.path.join(BASE_DIR, 'ds000105')
    if os.path.exists(haxby_dir):
        subjects = [d for d in os.listdir(haxby_dir) if d.startswith('sub-')]
        total_runs = 0
        has_events = False

        for subj in sorted(subjects):
            func_dir = os.path.join(haxby_dir, subj, 'func')
            if os.path.exists(func_dir):
                files = os.listdir(func_dir)
                runs = len([f for f in files if f.endswith('.nii.gz')])
                total_runs += runs
                if any('events' in f for f in files):
                    has_events = True

        print(f"\n  ds000105 — Haxby Visual Object Recognition:")
        print(f"    Subjects: {len(subjects)}/6")
        print(f"    Total runs: {total_runs}")
        print(f"    Event timing files: {'yes' if has_events else 'no (use labels.txt from haxby2001/)'}")
        print(f"    Pipeline stage enabled: HRF deconvolution ④")
    else:
        print(f"  ds000105 — Haxby: NOT DOWNLOADED")

    # Check local Haxby
    local_haxby = os.path.join(os.path.dirname(__file__), '..', 'data', 'haxby2001')
    if os.path.exists(local_haxby):
        subj1_bold = os.path.join(local_haxby, 'subj1', 'bold.nii.gz')
        if os.path.exists(subj1_bold):
            size_mb = os.path.getsize(subj1_bold) / (1024*1024)
            labels = os.path.join(local_haxby, 'subj1', 'labels.txt')
            masks = [f for f in os.listdir(os.path.join(local_haxby, 'subj1'))
                     if 'mask' in f]
            print(f"\n  haxby2001 (local, subject 1):")
            print(f"    BOLD: {size_mb:.0f} MB")
            print(f"    Labels: {'yes' if os.path.exists(labels) else 'no'}")
            print(f"    Masks: {', '.join(masks)}")
            print(f"    Pipeline stages: HRF ④ (has stimulus timing via labels)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Download priority fMRI datasets for D-LinOSS pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_priority_datasets.py --all
  python download_priority_datasets.py --multiecho --n-subjects 3
  python download_priority_datasets.py --haxby
  python download_priority_datasets.py --verify
        """
    )
    parser.add_argument('--all', action='store_true',
                        help='Download everything (multi-echo + remaining Haxby)')
    parser.add_argument('--multiecho', action='store_true',
                        help='Download ds000258 Multi-Echo Cambridge')
    parser.add_argument('--haxby', action='store_true',
                        help='Download remaining Haxby subjects (ds000105)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify what datasets are available')
    parser.add_argument('--n-subjects', type=int, default=5,
                        help='Number of multi-echo subjects to download (default: 5)')
    parser.add_argument('--start-subject', type=int, default=1,
                        help='Starting subject number for multi-echo (default: 1)')

    args = parser.parse_args()

    # Default: if no flags, show verification + prompt
    if not (args.all or args.multiecho or args.haxby or args.verify):
        print("D-LinOSS Priority Dataset Downloader")
        print("=" * 50)
        verify_datasets()
        print("Usage: python download_priority_datasets.py --all")
        print("       python download_priority_datasets.py --multiecho --n-subjects 3")
        print("       python download_priority_datasets.py --haxby")
        return

    if args.verify:
        verify_datasets()
        return

    t0 = time.time()

    if args.all or args.multiecho:
        download_multiecho_cambridge(
            n_subjects=args.n_subjects,
            start_subject=args.start_subject
        )

    if args.all or args.haxby:
        download_haxby_remaining()

    # Always verify after download
    verify_datasets()

    elapsed = time.time() - t0
    print(f"  Total download time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
