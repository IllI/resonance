import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = 'openneuro.org'
out_dir = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813"

sub = "sub-206"
key_bold = f"ds002813/{sub}/func/{sub}_task-fintest_run-1_bold.nii.gz"
key_events = f"ds002813/{sub}/func/{sub}_task-fintest_run-1_events.tsv"

out_bold = os.path.join(out_dir, sub, "func", os.path.basename(key_bold))
out_events = os.path.join(out_dir, sub, "func", os.path.basename(key_events))

os.makedirs(os.path.dirname(out_bold), exist_ok=True)

if not os.path.exists(out_bold):
    print(f"Downloading {key_bold}...")
    s3.download_file(bucket, key_bold, out_bold)

if not os.path.exists(out_events):
    print(f"Downloading {key_events}...")
    s3.download_file(bucket, key_events, out_events)

sub2 = "sub-233"
key_bold2 = f"ds002813/{sub2}/func/{sub2}_task-fintest_run-1_bold.nii.gz"
key_events2 = f"ds002813/{sub2}/func/{sub2}_task-fintest_run-1_events.tsv"

out_bold2 = os.path.join(out_dir, sub2, "func", os.path.basename(key_bold2))
out_events2 = os.path.join(out_dir, sub2, "func", os.path.basename(key_events2))

os.makedirs(os.path.dirname(out_bold2), exist_ok=True)

if not os.path.exists(out_bold2):
    print(f"Downloading {key_bold2}...")
    s3.download_file(bucket, key_bold2, out_bold2)

if not os.path.exists(out_events2):
    print(f"Downloading {key_events2}...")
    s3.download_file(bucket, key_events2, out_events2)

print("Setup Complete.")
