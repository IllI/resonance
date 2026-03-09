import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

bucket_name = 'openneuro.org'
out_dir = r'C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813'
os.makedirs(out_dir, exist_ok=True)

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Download sub-201 events
sub_prefix = "ds002813/sub-201/"
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=sub_prefix)
for obj in response.get('Contents', []):
    key = obj['Key']
    if key.endswith('.tsv') or key.endswith('.json'):
        filename = os.path.basename(key)
        
        # recreate subdirs
        rel_path = key.replace('ds002813/', '')
        out_path = os.path.join(out_dir, *rel_path.split('/'))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        print(f"Downloading {rel_path}...")
        s3.download_file(bucket_name, key, out_path)

print("sub-201 metadata download complete.")
