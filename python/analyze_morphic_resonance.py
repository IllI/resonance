"""
analyze_morphic_resonance.py
==================================================
Tests the hypothesis that the "Badoon" concept became easier to grok
for later subjects as the QPC (Quantum Phantom Cauldron) was reinforced
by earlier subjects (a phenomenon similar to Morphic Resonance).

1. Fetches the list of all subjects in ds002813.
2. For each subject, analyzes their behavioral events.tsv to find the Trial of Grokking (Grok_Idx).
3. Analyzes the trend: do subjects later in the experiment grok the concept faster?
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt

def get_all_subjects():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    prefix = 'ds002813/'
    
    # We can list directories to find subjects
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
    
    subjects = []
    for page in pages:
        if 'CommonPrefixes' in page:
            for prefix_info in page['CommonPrefixes']:
                folder = prefix_info['Prefix'].split('/')[-2]
                if folder.startswith('sub-'):
                    subjects.append(folder)
    return sorted(subjects)

def get_subject_grok_idx(sub_id, run_sequence):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    
    all_trials = []
    
    for run_type, run_num in run_sequence:
        run_name = f"task-{run_type}_run-{run_num}"
        tsv_key = f"ds002813/{sub_id}/func/{sub_id}_{run_name}_events.tsv"
        
        try:
            response = s3.get_object(Bucket=bucket, Key=tsv_key)
            events = pd.read_csv(response['Body'], sep='\t')
            for _, row in events.iterrows():
                if 'correct' in row and not pd.isna(row['correct']):
                    all_trials.append(int(row['correct']))
        except Exception as e:
            pass
            
    if not all_trials:
        return None, 0, 0, 0
        
    n = len(all_trials)
    total_correct = sum(all_trials)
    
    # Calculate longest streak
    longest_streak = 0
    current_streak = 0
    for val in all_trials:
        if val == 1:
            current_streak += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
        else:
            current_streak = 0

    return sum(all_trials), n, sum(all_trials)/n, longest_streak

def run_analysis():
    print("Fetching subjects...")
    subjects = get_all_subjects()
    print(f"Found {len(subjects)} subjects.")
    
    run_sequence = [("imtest", i) for i in range(1, 5)] + [("fintest", i) for i in range(1, 5)]
    
    results = []
    
    print("\nAnalyzing behavioral grokking timeline across cohort...")
    print(f"{'Subject':<10} | {'Total Correct':<15} | {'Accuracy':<10} | {'Longest Streak'}")
    print("-" * 60)
    
    for sub in subjects:
        total_correct, total_trials, accuracy, longest_streak = get_subject_grok_idx(sub, run_sequence)
        
        if total_correct is not None and total_trials > 0:
            sub_num = int(sub.split('-')[1])
            results.append({
                'Subject': sub,
                'Sub_Num': sub_num,
                'Total_Correct': total_correct,
                'Accuracy': accuracy,
                'Longest_Streak': longest_streak
            })
            print(f"{sub:<10} | {total_correct:<15} | {accuracy:.1%}    | {longest_streak}")
            
    df = pd.DataFrame(results)
    
    # Sort chronologically by Subject Number
    df = df.sort_values('Sub_Num')
    
    # Ensure there's variance
    if len(df) > 1:
        corr, p_val = spearmanr(df['Sub_Num'], df['Accuracy'])
        print("\n" + "=" * 60)
        print(" MORPHIC RESONANCE HYPOTHESIS TEST")
        print("=" * 60)
        print(f"Correlation between Chronological Subject ID and Overall Accuracy:")
        print(f"Spearman r: {corr:.4f}")
        print(f"p-value:    {p_val:.4f}")
        
        if corr > 0 and p_val < 0.05:
            print("\n>>> SIGNIFICANT EVIDENCE OF LEARNING ACCELERATION")
            print("Later subjects were significantly more accurate than earlier subjects.")
        elif corr > 0 and p_val < 0.1:
            print("\n>>> MARGINAL EVIDENCE OF LEARNING ACCELERATION")
            print("There is a trend suggesting later subjects were more accurate.")
        else:
            print("\n>>> NO SIGNIFICANT EVIDENCE OF UNIFORM ACCELERATION")
            print("The accuracy did not linearly increase across the experiment.")
            
        # Group into quintiles or halves to see macroscopic shift
        df['Cohort_Half'] = pd.qcut(df['Sub_Num'], 2, labels=['First Half', 'Second Half'])
        means = df.groupby('Cohort_Half')['Accuracy'].mean()
        print(f"\nMean Accuracy by Cohort Half:")
        print(f"  First Half Subjects:  {means['First Half']:.1%}")
        print(f"  Second Half Subjects: {means['Second Half']:.1%}")

if __name__ == "__main__":
    run_analysis()
