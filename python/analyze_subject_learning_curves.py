import os
import glob
import pandas as pd
import numpy as np

def analyze_all_test_curves(base_dir):
    print("===========================================================================")
    print(" D-LinOSS: SUBJECT-BY-SUBJECT EMPIRICAL LEARNING CURVE CLASSIFICATION")
    print("===========================================================================")
    
    sub_dirs = glob.glob(os.path.join(base_dir, "sub-*"))
    sub_ids = sorted([os.path.basename(sub) for sub in sub_dirs if os.path.isdir(sub)])
    
    results = []
    
    for sub in sub_ids:
        func_dir = os.path.join(base_dir, sub, "func")
        # Gather any event files that actually record behavioral accuracy
        test_events = sorted(glob.glob(os.path.join(func_dir, "*task-imtest*events.tsv"))) + \
                      sorted(glob.glob(os.path.join(func_dir, "*task-fintest*events.tsv")))
        
        if not test_events:
            continue
            
        run_accuracies = []
        for run_file in test_events:
            try:
                df = pd.read_csv(run_file, sep='\t')
                if 'correct' in df.columns:
                    # 'correct' is typically 1 for correct, 0 for incorrect
                    acc = pd.to_numeric(df['correct'], errors='coerce').dropna().mean()
                    run_accuracies.append(acc)
            except Exception:
                pass
                
        if len(run_accuracies) >= 2:
            initial_acc = run_accuracies[0]
            final_acc = run_accuracies[-1]
            learning_delta = final_acc - initial_acc
            
            # Check for sustained high accuracy (the plateau of Grokking)
            sustained_mastery = all(acc >= 0.80 for acc in run_accuracies[-2:]) if len(run_accuracies) >= 3 else (final_acc >= 0.8)
            
            results.append({
                "Subject": sub,
                "Run1_Acc": initial_acc,
                "Final_Acc": final_acc,
                "Delta": learning_delta,
                "Sustained_Mastery": sustained_mastery,
                "Trajectory": np.round(run_accuracies, 2).tolist()
            })
            
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_sorted = df_res.sort_values(by="Final_Acc", ascending=False)
        
        masters = df_sorted[df_sorted['Sustained_Mastery'] == True]
        strugglers = df_sorted[df_sorted['Sustained_Mastery'] == False].sort_values(by="Final_Acc", ascending=True)
        
        print(f"\n[+] TRUE MASTERS (Consolidation into Category Superposition Achieved): {len(masters)} subjects")
        print("These subjects successfully moved past Working Memory and maintained >80% accuracy.")
        print(masters[['Subject', 'Run1_Acc', 'Final_Acc', 'Trajectory']].head(10).to_string(index=False))
        
        print(f"\n[-] STRUGGLING LEARNERS (Trapped in Working Memory / Noise): {len(strugglers)} subjects")
        print("These subjects failed to construct the latent rule, falling back on high-ATP sensory guessing.")
        print(strugglers[['Subject', 'Run1_Acc', 'Final_Acc', 'Trajectory']].head(10).to_string(index=False))

if __name__ == "__main__":
    base_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"
    analyze_all_test_curves(base_dir)
