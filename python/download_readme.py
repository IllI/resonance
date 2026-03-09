import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = 'openneuro.org'
out_path = r'C:\Users\cityz\IllI\newer_all\data\openneuro\ds002813\README'

try:
    s3.download_file(bucket, 'ds002813/README', out_path)
    print("README downloaded successfully!")
    with open(out_path, 'r') as f:
        print(f.read()[:2000])
except Exception as e:
    print(f"Error: {e}")
