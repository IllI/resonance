import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

bucket_name = 'openneuro.org'
prefix = 'ds002813/'
out_dir = r'C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813'

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
print("Finding all .tsv behavioral event files for all subjects...")

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

download_count = 0
for page in pages:
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.endswith('events.tsv'):
            # Recreate local structure
            rel_path = key.replace('ds002813/', '')
            out_path = os.path.join(out_dir, *rel_path.split('/'))
            
            # Skip if already downloaded
            if os.path.exists(out_path):
                continue
                
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            s3.download_file(bucket_name, key, out_path)
            download_count += 1
            if download_count % 100 == 0:
                print(f"Downloaded {download_count} behavior files...")

print(f"Finished downloading all {download_count} new behavioral event files.")
