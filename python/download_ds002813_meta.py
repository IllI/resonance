import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

bucket_name = 'openneuro.org'
prefix = 'ds002813/'
out_dir = r'C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813'
os.makedirs(out_dir, exist_ok=True)

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Let's get root level files first
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
for obj in response.get('Contents', []):
    key = obj['Key']
    if key.endswith('/'): continue
    filename = os.path.basename(key)
    out_path = os.path.join(out_dir, filename)
    print(f"Downloading {filename}...")
    s3.download_file(bucket_name, key, out_path)

# Download sub-01 func events to see the paradigm
sub_prefix = f"ds002813/sub-01/func/"
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=sub_prefix)
for obj in response.get('Contents', []):
    key = obj['Key']
    if 'events.tsv' in key:
        filename = os.path.basename(key)
        out_func_dir = os.path.join(out_dir, 'sub-01', 'func')
        os.makedirs(out_func_dir, exist_ok=True)
        out_path = os.path.join(out_func_dir, filename)
        print(f"Downloading {filename}...")
        s3.download_file(bucket_name, key, out_path)

print("Metadata download complete.")
