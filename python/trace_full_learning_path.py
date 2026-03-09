import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import zscore
import boto3
from botocore import UNSIGNED
from botocore.config import Config

sys.path.append(r"c:\Users\cityz\IllI\newer_all\python")
from electromagnetic_spectral_map import ElectromagneticSpectralMap

def download_run_data(sub_id, run_idx, out_dir):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    
    nii_key = f"ds002813/{sub_id}/func/{sub_id}_task-train_run-{run_idx}_bold.nii.gz"
    tsv_key = f"ds002813/{sub_id}/func/{sub_id}_task-train_run-{run_idx}_events.tsv"
    
    for key in [nii_key, tsv_key]:
        rel_path = key.replace('ds002813/', '')
        out_path = os.path.join(out_dir, *rel_path.split('/'))
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            print(f"Downloading {key}...")
            try:
                s3.download_file(bucket, key, out_path)
            except Exception as e:
                pass # Event files might be missing for some, but typically training has 8 runs

def extract_mean_integral(nii_path, tsv_path, tr=2.0, window_len=6):
    try:
        data = nib.load(nii_path).get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
    except Exception:
        return np.nan, np.nan
        
    data_z = zscore(data, axis=-1)
    data_z = np.nan_to_num(data_z)
    
    brain_mask = np.var(data, axis=-1) > 10
    var_map = np.var(data_z[brain_mask], axis=-1)
    # Focus on the highly active regions to track the peak footprint
    top_10_thresh = np.percentile(var_map, 90)
    active_mask = (np.var(data_z, axis=-1) > top_10_thresh) & brain_mask
    
    event_ints = []
    
    for _, row in events.iterrows():
        onset_sec = row['onset']
        tr_start = int(onset_sec / tr)
        tr_end = tr_start + window_len
        if tr_end > data_z.shape[-1]:
            break
        event_block = data_z[active_mask, tr_start:tr_end]
        event_ints.append(np.mean(np.abs(event_block)) * 10.0) 
        
    accuracy = np.nan
    if 'correct' in events.columns:
        accuracy = pd.to_numeric(events['correct'], errors='coerce').dropna().mean()
        
    return np.mean(event_ints), accuracy

def trace_learning_path(sub_id, base_dir):
    print(f"\n=======================================================")
    print(f"  TRACING FULL LEARNING PATH: {sub_id}")
    print(f"=======================================================")
    
    out_dir = base_dir
    mapper = ElectromagneticSpectralMap()
    
    results = []
    for run in range(1, 9):
        download_run_data(sub_id, run, out_dir)
        func_dir = os.path.join(base_dir, sub_id, "func")
        nii_path = os.path.join(func_dir, f"{sub_id}_task-train_run-{run}_bold.nii.gz")
        tsv_path = os.path.join(func_dir, f"{sub_id}_task-train_run-{run}_events.tsv")
        
        integral, accuracy = extract_mean_integral(nii_path, tsv_path)
        
        if not np.isnan(integral):
            # To mirror the Little & Thulborn findings: Initial recruitment vs Specialization
            # We track the biological volume (neurons) over the learning curve
            res = mapper.evaluate_resonance_spike("occipital_cortex", "3T", integral, 12)
            results.append({
                "Run": run,
                "Accuracy": accuracy,
                "BOLD_Integral": integral,
                "Active_Neurons": res['estimated_neurons']
            })
            
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    print("Downloading intermediate run NIfTIs to trace the exact learning path...")
    df_master = trace_learning_path("sub-206", base_dir)
    df_struggler = trace_learning_path("sub-233", base_dir)

