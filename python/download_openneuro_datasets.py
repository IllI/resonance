"""
Download additional fMRI datasets from OpenNeuro for D-LinOSS analysis.
Uses the openneuro-py package to fetch BOLD functional data.
"""
import os
import sys
import subprocess

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'openneuro')
os.makedirs(BASE_DIR, exist_ok=True)

# Datasets with functional BOLD data suitable for our analysis
DATASETS = {
    # ds000228: MRI data of 3-12 year old children and adults during movie viewing
    # Good for astrocyte analysis: movie-watching = naturalistic, rich slow-wave content
    'ds000228': {
        'desc': 'Children+Adults movie watching (Richardson 2018)',
        'subjects': ['sub-pixar001', 'sub-pixar002', 'sub-pixar003'],
        'includes': ['func/*bold*', 'func/*events*', 'anat/*T1w*'],
    },
    # ds000171: Sleep deprivation study - resting state
    # Perfect for astrocyte: sleep deprivation changes glial function
    'ds000171': {
        'desc': 'Sleep deprivation resting-state (Zhu 2015)',
        'subjects': ['sub-01', 'sub-02', 'sub-03'],
        'includes': ['func/*bold*', 'func/*events*', 'anat/*T1w*'],
    },
    # ds000030: UCLA Consortium for Neuropsychiatric Phenomics
    # Task-based: Balloon Analog Risk Task, Stop Signal Task
    'ds000030': {
        'desc': 'UCLA CNP - multiple cognitive tasks (Poldrack 2016)',
        'subjects': ['sub-10159', 'sub-10171', 'sub-10189'],
        'includes': ['func/*bold*', 'func/*events*', 'anat/*T1w*'],
    },
    # ds000105: Haxby faces/objects (6 subjects, different from our subj1)
    'ds000105': {
        'desc': 'Haxby visual object recognition (6 subjects)',
        'subjects': ['sub-1', 'sub-2', 'sub-3'],
        'includes': ['func/*bold*', 'anat/*T1w*'],
    },
    # ds000117: Multi-subject, multi-modal neuroimaging
    'ds000117': {
        'desc': 'Multi-modal faces/scrambled (Wakeman & Henson)',
        'subjects': ['sub-01', 'sub-02'],
        'includes': ['func/*bold*', 'anat/*T1w*'],
    },
}


def download_dataset(ds_id, info):
    """Download a dataset using openneuro-py API."""
    target = os.path.join(BASE_DIR, ds_id)

    print(f"\n{'='*60}")
    print(f"  Downloading {ds_id}: {info['desc']}")
    print(f"  Target: {target}")
    print(f"  Subjects: {info['subjects']}")
    print(f"{'='*60}")

    import openneuro

    for subj in info['subjects']:
        subj_dir = os.path.join(target, subj)
        if os.path.exists(subj_dir):
            # Check if BOLD data exists
            func_dir = os.path.join(subj_dir, 'func')
            if os.path.exists(func_dir) and any(f.endswith('.nii.gz') for f in os.listdir(func_dir) if 'bold' in f):
                print(f"  [SKIP] {subj} already downloaded")
                continue

        # Build include patterns for this subject
        includes = []
        for pattern in info['includes']:
            includes.append(f'{subj}/{pattern}')
        
        # Also include top-level dataset description for metadata
        if 'dataset_description.json' not in includes:
             includes.append('dataset_description.json')

        print(f"  Downloading {subj}...")
        try:
            # Using the python API directly
            # Note: openneuro.download returns boolean or None typically
            openneuro.download(dataset=ds_id, target_dir=target, include=includes)
            print(f"  [OK] {subj} downloaded")
        except Exception as e:
            print(f"  [ERROR] {subj}: {e}")


def main():
    # First ensure openneuro-py is installed
    try:
        import openneuro
        print("openneuro-py is available")
    except ImportError:
        print("Installing openneuro-py...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'openneuro-py'],
                      capture_output=True, timeout=120)

    for ds_id, info in DATASETS.items():
        download_dataset(ds_id, info)

    # List what we got
    print(f"\n{'='*60}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*60}")

    total_bold = 0
    for ds_id in DATASETS:
        ds_path = os.path.join(BASE_DIR, ds_id)
        if not os.path.exists(ds_path):
            print(f"  {ds_id}: NOT DOWNLOADED")
            continue

        bold_files = []
        for root, dirs, files in os.walk(ds_path):
            for f in files:
                if 'bold' in f and f.endswith('.nii.gz'):
                    bold_files.append(os.path.join(root, f))

        print(f"  {ds_id}: {len(bold_files)} BOLD files")
        for bf in bold_files[:5]:
            rel = os.path.relpath(bf, ds_path)
            size_mb = os.path.getsize(bf) / (1024*1024)
            print(f"    - {rel} ({size_mb:.1f} MB)")
        total_bold += len(bold_files)

    print(f"\n  Total BOLD files available: {total_bold}")


if __name__ == "__main__":
    main()
