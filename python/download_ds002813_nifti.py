import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

bucket_name = 'openneuro.org'
out_dir = r'C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813'
os.makedirs(out_dir, exist_ok=True)

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Download sub-201 func/anat NIfTI files for train run 1 (Early) and run 8 (Late)
keys_to_download = [
    # Anatomical
    'ds002813/sub-201/anat/sub-201_T1w.nii.gz',
    'ds002813/sub-201/anat/sub-201_T1w.json',
    
    # Early Learning (Run 1)
    'ds002813/sub-201/func/sub-201_task-train_run-1_bold.nii.gz',
    'ds002813/sub-201/func/sub-201_task-train_run-1_events.tsv',
    
    # Late Learning (Run 4 or 8)
    'ds002813/sub-201/func/sub-201_task-train_run-8_bold.nii.gz',
    'ds002813/sub-201/func/sub-201_task-train_run-8_events.tsv'
]

print("Starting heavy data download for ds002813 (Early vs Late learning)...")
for key in keys_to_download:
    filename = os.path.basename(key)
    rel_path = key.replace('ds002813/', '')
    out_path = os.path.join(out_dir, *rel_path.split('/'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    if os.path.exists(out_path):
        print(f"Already downloaded: {filename}")
        continue
        
    print(f"Downloading {filename}...")
    try:
        s3.download_file(bucket_name, key, out_path)
    except Exception as e:
        print(f"Error downloading {key}: {e}")

print("Download complete.")
