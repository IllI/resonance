import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
from scipy.signal import correlate
import torch
import torch.nn as nn
import boto3
from botocore import UNSIGNED
from botocore.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stream_data(sub_id, base_dir, run_type, run_num):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    run_name = f"task-{run_type}_run-{run_num}"
    local_func_dir = os.path.join(base_dir, sub_id, "func")
    os.makedirs(local_func_dir, exist_ok=True)
    tsv_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_events.tsv")
    nii_path = os.path.join(local_func_dir, f"{sub_id}_{run_name}_bold.nii.gz")
    tsv_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_events.tsv"
    try:
        if not os.path.exists(tsv_path):
            s3.download_file(bucket, tsv_key, tsv_path)
    except:
        return None, None
    nii_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_bold.nii.gz"
    try:
        if not os.path.exists(nii_path):
            s3.download_file(bucket, nii_key, nii_path)
    except:
        return None, None
    return nii_path, tsv_path

def define_key_rois(shape):
    nx, ny, nz = shape[:3]
    rois = {}
    rois['PFC'] = np.zeros(shape[:3], dtype=bool)
    rois['PFC'][:, int(ny*0.75):, int(nz*0.6):] = True
    rois['Parahippocampus'] = np.zeros(shape[:3], dtype=bool)
    rois['Parahippocampus'][int(nx*0.25):int(nx*0.75), int(ny*0.35):int(ny*0.65), :int(nz*0.3)] = True
    return rois

class EuclideanAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Linear(input_dim, latent_dim)
        self.dec = nn.Linear(latent_dim, input_dim)
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

class ProjectiveAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Linear(input_dim, latent_dim)
        self.dec = nn.Linear(latent_dim, input_dim)
    def forward(self, x):
        z = self.enc(x)
        # Projective geometry constraint: normalize to unit hypersphere (ray representation in projective space)
        z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-8)
        return self.dec(z)

def train_and_eval(model, train_data, test_data, epochs=100):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    X_train = torch.FloatTensor(train_data).to(device)
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, X_train)
        loss.backward()
        optimizer.step()
        
    # Eval
    model.eval()
    with torch.no_grad():
        X_test = torch.FloatTensor(test_data).to(device)
        out = model(X_test)
        loss = criterion(out, X_test).item()
    return loss

def run_battle():
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    # Testing on 3 clear grokkers for the first round
    subjects = ["sub-201", "sub-220", "sub-226"]
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    tr_sec = 2.0
    
    print("=================================================================", flush=True)
    print(" GROKKING GEOMETRY BATTLE: EUCLIDEAN vs NON-EUCLIDEAN", flush=True)
    print("=================================================================\n", flush=True)
    
    for sub in subjects:
        print(f"Processing {sub}...", flush=True)
        
        explore_data = []
        grok_data = []
        all_trials = []
        
        for run_type, run_num in run_sequence:
            nii_path, tsv_path = stream_data(sub, base_dir, run_type, run_num)
            if not nii_path: continue
            
            try:
                data_raw = nib.load(nii_path).get_fdata()
                events = pd.read_csv(tsv_path, sep='\t')
            except:
                continue
                
            data_z = np.nan_to_num(zscore(data_raw, axis=-1))
            
            brain_mask = np.var(data_raw, axis=-1) > 20
            var_vals = np.var(data_raw, axis=-1)[brain_mask]
            if len(var_vals) == 0: continue
            thresh = np.percentile(var_vals, 95)
            active_mask = brain_mask & (np.var(data_raw, axis=-1) > thresh)
            
            flat_data = data_z[active_mask, :] # (voxels, TRs)
            if flat_data.shape[0] > 1000:
                flat_data = flat_data[:1000, :]
            
            n_trs = flat_data.shape[1]
            rois = define_key_rois(data_raw.shape)
            mask_pfc = rois['PFC'] & brain_mask
            mask_para = rois['Parahippocampus'] & brain_mask
            
            ts_pfc = np.mean(data_z[mask_pfc, :], axis=0) if np.sum(mask_pfc)>0 else np.zeros(n_trs)
            ts_para = np.mean(data_z[mask_para, :], axis=0) if np.sum(mask_para)>0 else np.zeros(n_trs)
            
            for _, row in events.iterrows():
                if 'correct' not in row or pd.isna(row['correct']): continue
                correct = int(row['correct'])
                tr_onset = int(row['onset'] / tr_sec)
                
                # 3-TR window
                start_tr = max(0, tr_onset)
                end_tr = min(n_trs, tr_onset + 3)
                if end_tr - start_tr < 3: continue
                
                snap = flat_data[:, start_tr:end_tr].flatten()
                trial_info = {'Correct': correct, 'snap': snap, 
                              'pfc_window': ts_pfc[start_tr:end_tr+3], # Slightly wider for lag
                              'para_window': ts_para[start_tr:end_tr+3]}
                all_trials.append(trial_info)
                
            # cleanup
            if os.path.exists(nii_path): os.remove(nii_path)
                
        if len(all_trials) < 5:
            print(f"  Not enough trials for {sub}.", flush=True)
            continue
            
        n = len(all_trials)
        correctness = [t['Correct'] for t in all_trials]
        grok_idx = n
        streak_start = n
        for i in range(n - 1, -1, -1):
            if correctness[i] == 1:
                streak_start = i
            else:
                break
        if n - streak_start >= 4:
            grok_idx = streak_start
        
        if grok_idx >= n or grok_idx < 3:
            print(f"  No valid grokking point found for {sub}.", flush=True)
            continue
            
        print(f"  Grokking identified at trial {grok_idx}.", flush=True)
        
        for i, t in enumerate(all_trials):
            if i < grok_idx - 1:
                explore_data.append(t['snap'])
            elif abs(i - grok_idx) <= 1:
                grok_data.append(t['snap'])
                
        if len(explore_data) < 3 or len(grok_data) < 2:
            print("  Not enough data split.", flush=True)
            continue
            
        explore_data = np.stack(explore_data)
        grok_data = np.stack(grok_data)
        
        input_dim = explore_data.shape[1]
        latent_dim = max(2, input_dim // 50)
        
        # --- ROUND 1 ---
        print("\n  [ROUND 1: Representational Efficiency]", flush=True)
        model_euclidean = EuclideanAE(input_dim, latent_dim)
        model_projective = ProjectiveAE(input_dim, latent_dim)
        
        loss_euc = train_and_eval(model_euclidean, explore_data, grok_data)
        loss_proj = train_and_eval(model_projective, explore_data, grok_data)
        
        print(f"    Euclidean Loss (Classical):   {loss_euc:.4f}", flush=True)
        print(f"    Projective Loss (Non-Local):  {loss_proj:.4f}", flush=True)
        
        if loss_proj < loss_euc * 0.9:
            print("    -> WINNER: Projective (Non-Euclidean). Grokking state prefers projective geometry.", flush=True)
        elif loss_euc < loss_proj * 0.9:
            print("    -> WINNER: Euclidean (Classical). Grokking state prefers standard linear geometry.", flush=True)
        else:
            print("    -> TIE: Both geometries compress similarly.", flush=True)

        # --- ROUND 3 ---
        print("\n  [ROUND 3: Propagation vs Simultaneity]", flush=True)
        lags = []
        for i in range(max(0, grok_idx-1), min(len(all_trials), grok_idx+2)):
            pfc = all_trials[i]['pfc_window']
            para = all_trials[i]['para_window']
            if len(pfc) < 4 or np.std(pfc) == 0 or np.std(para) == 0: continue
            
            cross_corr = correlate(zscore(pfc), zscore(para), mode='full')
            lag = np.argmax(cross_corr) - (len(pfc) - 1)
            lags.append(abs(lag) * tr_sec)
            
        if len(lags) > 0:
            mean_lag = np.mean(lags)
            print(f"    Mean absolute lag between PFC and Parahippocampus at grokking: {mean_lag:.2f} seconds", flush=True)
            if mean_lag == 0:
                print("    -> WINNER: Simultaneity (Non-Local). Regions synchronized instantly.", flush=True)
            else:
                print("    -> WINNER: Propagation (Classical). Signal traveled across space.", flush=True)
        else:
            print("    Could not compute lag.", flush=True)
        print("-" * 65, flush=True)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    run_battle()
