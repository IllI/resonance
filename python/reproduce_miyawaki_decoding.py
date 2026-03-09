import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def stream_ds006661_data(sub_id="sub-001", run_num="01", base_dir=r"c:\Users\cityz\IllI\newer_all\data\openneuro\ds006661"):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    
    local_func_dir = os.path.join(base_dir, sub_id, "func")
    os.makedirs(local_func_dir, exist_ok=True)
    
    run_name = f"task-main_run-{run_num}"
    tsv_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_events.tsv")
    nii_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_bold.nii.gz")
    
    tsv_key = f"ds006661/{sub_id}/func/{sub_id}_{run_name}_events.tsv"
    nii_key = f"ds006661/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz"
    
    if not os.path.exists(tsv_path):
        try: s3.download_file(bucket, tsv_key, tsv_path)
        except: return None, None
            
    if not os.path.exists(nii_path):
        try: s3.download_file(bucket, nii_key, nii_path)
        except: return None, None
            
    return nii_path, tsv_path

def reproduce_miyawaki_decoding():
    tr_sec = 0.125
    print("=====================================================================")
    print(" REPRODUCING MIYAWAKI ET AL. (2025) - RAPID TIME-RESOLVED DECODING")
    print(" Can we decode visual category from ignored flashes within 2 seconds?")
    print("=====================================================================\n")
    
    nii_path, tsv_path = stream_ds006661_data()
    if not nii_path:
        print("Failed to get data.")
        return
        
    print("Loading 7T BOLD and Events...")
    img = nib.load(nii_path)
    data_raw = img.get_fdata()
    events = pd.read_csv(tsv_path, sep='\t')
    
    n_trs = data_raw.shape[-1]
    
    # Fast masking: Get the visual cortex (back of the brain) or top 10k varying voxels
    # Since we don't have a perfect mask, we'll take the top 5000 voxels by variance
    var_map = np.var(data_raw, axis=-1)
    threshold = np.percentile(var_map, 95)
    brain_mask = var_map > threshold
    active_voxels = data_raw[brain_mask]
    
    print(f"Masking to top varying voxels. Voxel count: {active_voxels.shape[0]}")
    data_z = zscore(active_voxels, axis=1)
    data_z = np.nan_to_num(data_z)
    
    # Define mapping to super-categories for robustness
    # Animate: Faces and Animals. Inanimate: Vehicles (Planes/Cars)
    animate_cats = ['male', 'female', 'SiberianHusky', 'toy_poodle']
    inanimate_cats = ['sportscar', 'fighter', 'airliner', 'jeep']
    
    # We will extract a window of -4 TRs (-0.5s) to +24 TRs (+3.0s) around each image flash
    window_start_tr = -4
    window_end_tr = 24
    time_points = np.arange(window_start_tr, window_end_tr) * tr_sec
    
    trials_data = [] # shape: (n_trials, n_timepoints, n_voxels)
    labels = []
    
    image_events = events[events['trial_type'] == 'image'].copy()
    print(f"Found {len(image_events)} image presentation events.")
    
    for i, row in image_events.iterrows():
        onset_sec = row['onset']
        cat = row['image_category']
        
        onset_tr = int(onset_sec / tr_sec)
        
        if onset_tr + window_start_tr < 0 or onset_tr + window_end_tr >= n_trs:
            continue
            
        if cat in animate_cats:
            label = 1 # Animate
        elif cat in inanimate_cats:
            label = 0 # Inanimate
        else:
            continue
            
        # Extract the window
        window = data_z[:, onset_tr + window_start_tr : onset_tr + window_end_tr].T # (n_timepoints, n_voxels)
        trials_data.append(window)
        labels.append(label)
        
    trials_data = np.array(trials_data)
    labels = np.array(labels)
    
    print(f"Extracted {trials_data.shape[0]} valid trials.")
    print(f"Class 0 (Inanimate): {np.sum(labels==0)}, Class 1 (Animate): {np.sum(labels==1)}")
    
    print("\nRunning Time-Resolved SVM Decoding (Animate vs Inanimate)...")
    print(f"Chance accuracy is 50.0%.")
    print("Time(s) | Cross-Val Accuracy (%)")
    print("-" * 35)
    
    # Train a separate SVM for each time point relative to stimulus onset
    accuracies = []
    for t_idx, t_sec in enumerate(time_points):
        # The feature vector for this timepoint across all trials
        X_t = trials_data[:, t_idx, :] # (n_trials, n_voxels)
        
        clf = SVC(kernel='linear', C=1.0)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_t, labels, cv=cv, n_jobs=-1)
        
        acc = np.mean(scores) * 100
        accuracies.append(acc)
        
        marker = "***" if acc > 60.0 else ""
        print(f" {t_sec:>6.3f} | {acc:>18.1f}% {marker}")
        
    peak_acc = np.max(accuracies)
    peak_time = time_points[np.argmax(accuracies)]
    
    print("\n--- CONCLUSION ---")
    print(f"Peak Decoding Accuracy: {peak_acc:.1f}% at {peak_time:.3f} seconds post-stimulus.")
    
    if peak_acc > 60.0 and peak_time < 3.0:
        print("REPRODUCTION SUCCESSFUL: The brain inherently processed the visual category of the flashes, even while behaviorally ignoring them to focus on the crosshair. The non-local 'grokking' occurred structurally, and the ultra-fast TR captured the information representation rapidly.")
    else:
        print("REPRODUCTION FAILED: The SVM could not consistently decode the image category above chance in the expected rapid timeframe. The subject's behavioral ignorance destroyed the signal.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    reproduce_miyawaki_decoding()
