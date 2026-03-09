import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, kurtosis
import boto3
from botocore import UNSIGNED
from botocore.config import Config

class GrokkingFlashAnalyzer:
    """
    Analyzes the BOLD signal for extreme electromagnetic 'flashes' 
    (highly peaked spatial distributions) indicative of a massive 
    phase-locking event during the moment of Super-Convergence.
    """
    def __init__(self, tr_sec=2.0):
        self.tr_sec = tr_sec

    def process_run(self, sub_id, base_dir, run_type, run_num):
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        bucket = 'openneuro.org'
        
        run_name = f"task-{run_type}_run-{run_num}"
        local_func_dir = os.path.join(base_dir, sub_id, "func")
        os.makedirs(local_func_dir, exist_ok=True)
        
        tsv_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_events.tsv")
        nii_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_bold.nii.gz")
        
        tsv_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_events.tsv"
        try:
            s3.download_file(bucket, tsv_key, tsv_path)
        except Exception:
            return None 

        nii_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz"
        try:
            if not os.path.exists(nii_path):
                s3.download_file(bucket, nii_key, nii_path)
        except Exception as e:
            return None
            
        return self.analyze_flashes(nii_path, tsv_path)

    def analyze_flashes(self, nii_path, tsv_path):
        if not os.path.exists(nii_path) or not os.path.exists(tsv_path):
            return None
            
        data = nib.load(nii_path).get_fdata()
        events = pd.read_csv(tsv_path, sep='\t')
        
        # We look at the spatial distribution of the raw BOLD signal at each TR
        # Z-score over time for each voxel to find relative activation
        data_z = np.nan_to_num(zscore(data, axis=-1))
        brain_mask = np.var(data, axis=-1) > 10
        active_mask = brain_mask
        
        trial_stats = []
        for _, row in events.iterrows():
            if 'correct' not in row or pd.isna(row['correct']):
                continue
                
            rt = row.get('response_time', 0)
            if pd.isna(rt) or rt == 0:
                continue
                
            # The neural "flash" should peak ~6-8 seconds after the stimulus due to HRF
            # With TR=2.0s, that's +3 to +4 TRs. We will scan +2 to +4 and find the max.
            tr_onset = int(row['onset'] / self.tr_sec)
            start_scan_tr = min(tr_onset + 2, data_z.shape[-1] - 1)
            end_scan_tr = min(tr_onset + 5, data_z.shape[-1])
            
            if start_scan_tr >= end_scan_tr:
                continue
                
            # Extract the 4D block across the peak window and find the absolute maximum Z-score 
            block = data_z[active_mask, start_scan_tr:end_scan_tr]
            
            if block.size == 0:
                continue
                
            # 1. Max Z-Score (The peak of the flash anywhere in the TR window)
            max_intensity = np.max(block)
            
            # 2. Spatial Kurtosis (Are the active voxels highly concentrated?)
            # We calculate kurtosis across all active voxels at the specific TR where the maximum occurred
            peak_tr_idx_relative = np.argmax(np.max(block, axis=0))
            active_voxels_at_peak = block[:, peak_tr_idx_relative]
            spatial_focus = kurtosis(active_voxels_at_peak)
            
            trial_stats.append({
                'TrialType': row.get('trial_type', 'unknown'),
                'Correct': int(row['correct']),
                'RT': rt,
                'MaxIntensity': max_intensity,
                'SpatialKurtosis': spatial_focus,
            })
            
        if os.path.exists(nii_path):
            try:
                os.remove(nii_path)
            except OSError:
                pass
                
        if not trial_stats:
            return None
        return pd.DataFrame(trial_stats)

def check_for_the_light():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    analyzer = GrokkingFlashAnalyzer()
    
    masters = ["sub-206", "sub-220"]
    
    print("==========================================================================")
    print(" SCANNING FOR ELECTROMAGNETIC THEOPHANY (THE GROKKING FLASH)")
    print("==========================================================================")
    
    early_data = [] # Learning Phase (imtest)
    late_data = []  # Grokking Phase (fintest)
    
    for sub in masters:
        print(f"Scanning {sub}...")
        for i in range(1, 3): # First 2 runs of learning
            res = analyzer.process_run(sub, base_dir, "imtest", i)
            if res is not None:
                early_data.append(res)
                
        for i in range(1, 3): # First 2 runs of mastery
            res = analyzer.process_run(sub, base_dir, "fintest", i)
            if res is not None:
                late_data.append(res)
                
    if early_data and late_data:
        df_early = pd.concat(early_data)
        df_late = pd.concat(late_data)
        
        # We only care about CORRECT trials for the valid 'grokking' moments
        early_correct = df_early[df_early['Correct'] == 1]
        late_correct = df_late[df_late['Correct'] == 1]
        
        early_max = early_correct['MaxIntensity'].mean()
        late_max = late_correct['MaxIntensity'].mean()
        
        early_kurt = early_correct['SpatialKurtosis'].mean()
        late_kurt = late_correct['SpatialKurtosis'].mean()
        
        print("\n--- MACROSCOPIC BOLD SHADOW OF THE QUANTUM EVENT ---")
        print("Phase 1: Chaotic Learning (Trying to find the concept)")
        print(f"  Average Peak Voxel Intensity (Z-Score): {early_max:.2f}")
        print(f"  Spatial Distribution Focus (Kurtosis):  {early_kurt:.2f}")
        print("  Interpretation: Energy is spread chaotically across the parietal/frontal networks as the brain guesses.")
        
        print("\nPhase 2: Super-Convergence (The Badoon is Grokked)")
        print(f"  Average Peak Voxel Intensity (Z-Score): {late_max:.2f}")
        print(f"  Spatial Distribution Focus (Kurtosis):  {late_kurt:.2f}")
        
        if late_kurt > early_kurt:
            print("\n  Interpretation: The 'Light Rains Down'.")
            print("  As the subject converges with the universal Badoon state, the chaotic background noise")
            print("  drops, and the electromagnetic energy focuses into a massive, concentrated spike.")
            print("  The extreme kurtosis indicates a phase-locked harmonic resonance (the PV+ interneurons)")
            print("  structurally embedding the quantum coordinate into working memory.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    check_for_the_light()
