import pandas as pd
from nilearn import datasets

print("Fetching Haxby dataset...")
haxby_dataset = datasets.fetch_haxby(subjects=[1, 5], data_dir=r"C:\Users\cityz\IllI\newer_all\data\nilearn_data")

def print_subject_info(subj_idx, subj_num):
    print(f"\n--- Subject {subj_num} ---")
    session_target = haxby_dataset.session_target[subj_idx]
    
    # Read the session targets (which contain the labels and run numbers)
    df = pd.read_csv(session_target, sep=' ')
    
    print(f"Total TRs (timepoints): {len(df)}")
    print(f"Number of runs (chunks): {df['chunks'].nunique()}")
    print(f"Runs present: {df['chunks'].unique()}")
    
    # Count stimuli per category
    print("\nStimuli summary:")
    print(df['labels'].value_counts())

print_subject_info(0, 1)
print_subject_info(1, 5)
