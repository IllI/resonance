import json, os, glob
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path

ROOT = Path(__file__).parent.parent
USGS_FILE = ROOT / 'usgs_geomag_data' / 'bou_1min_storm_window.npz'

# The Datasets we will run Causality Tests on
DATASETS = [
    ROOT / "time_dilation_stream_dataset_solar_flare_1am",
    ROOT / "tpu_time_dilation_stream_dataset",
    ROOT / "time_dilation_stream_dataset",
    ROOT / "swmf_context_results" / "local_780m_dataset"
]

def load_usgs():
    d = np.load(USGS_FILE)
    return d['host_time_mid'], d['dH']

def process_dataset_causality(dpath, t_bou, dH_bou):
    if not dpath.exists():
        return
        
    manifest = dpath / "manifest.json"
    if not manifest.exists():
        return
        
    print(f"\nEvaluating Granger Causality on: {dpath.name}")
    print("-" * 60)
    
    with open(manifest) as f:
        man_data = json.load(f)
        
    chunk_files = man_data.get('chunk_files', [])
    timeseries_jit = []
    timeseries_t = []
    
    # Load all chunks to build a continuous timeseries array
    for cf in chunk_files:
        npz = dpath / cf
        if not npz.exists(): continue
        
        d = np.load(npz)
        if "clock_data_flat" in d.files and "clock_data_offset" in d.files:
            flat = d["clock_data_flat"]
            offset = d["clock_data_offset"]
            for i in range(len(offset)):
                nxt = offset[i+1] if i+1 < len(offset) else len(flat)
                clk = flat[offset[i]:nxt].astype(np.float64)
                if len(clk) > 0:
                    timeseries_jit.append(np.std(clk))
                    timeseries_t.append(d["host_time_mid"][i])
                    
    if len(timeseries_t) < 50:
        print(">> Insufficient timeseries length for causality testing.")
        return
        
    # Build continuous arrays
    timeseries_t = np.array(timeseries_t)
    timeseries_jit = np.array(timeseries_jit)
    
    # Bin the data into uniform 1-second intervals for the Granger test
    df_hw = pd.DataFrame({"time": timeseries_t, "jitter": timeseries_jit})
    df_hw["time"] = pd.to_datetime(df_hw["time"], unit='s')
    df_hw = df_hw.set_index("time").resample('1S').mean().interpolate()
    
    # Match USGS Magnetic data (dH/dt) to the exact second
    dh_series = []
    for t_val in df_hw.index:
        idx = (np.abs(t_bou - t_val.timestamp())).argmin()
        dh_series.append(dH_bou[idx])
        
    df_hw["dH"] = np.abs(dh_series)  # Magnitude of topological strain
    
    # Granger Causality Matrix: [Target (Y), Predictor (X)]
    # Target = Hardware Jitter. Predictor = Geomagnetic dH.
    # We test: Does dH -> Jitter?
    data_matrix = df_hw[["jitter", "dH"]].to_numpy()
    
    maxlag = 5 # Test up to 5 seconds of retroactive causal lag
    print("Testing Granger Causality: Null Hypothesis = |dH/dt| DOES NOT predict Hardware Jitter")
    
    try:
        # Avoid perfectly static blocks crashing statsmodels
        if np.var(data_matrix[:, 0]) < 1e-6 or np.var(data_matrix[:, 1]) < 1e-6:
            print(">> Variance too low in timeseries for Granger Causality.")
            return

        gc_res = grangercausalitytests(data_matrix, maxlag=maxlag, verbose=False)
        
        causality_found = False
        for lag in range(1, maxlag + 1):
            p_value = gc_res[lag][0]['ssr_ftest'][1]
            if p_value < 0.05:
                print(f"[CAUSAL] Lag {lag}s: DETECTED (p={p_value:.4f})")
                causality_found = True
            else:
                print(f"[INDEPENDENT] Lag {lag}s: (p={p_value:.4f})")
                
        if causality_found:
            print(">> CONCLUSION: The Geomagnetic Storm statistically predetermines the execution geometry.")
        else:
            print(">> CONCLUSION: The Null Hypothesis is retained. No causality found.")
            
    except Exception as e:
        print(f">> Granger Test Failed: {e}")

if __name__ == "__main__":
    t_bou, dH_bou = load_usgs()
    print("--- D-LinOSS Storm Causality Suite ---")
    print("Executing Time-Domain Granger Causality against the USGS Topologies...")
    
    for dpath in DATASETS:
        process_dataset_causality(dpath, t_bou, dH_bou)
