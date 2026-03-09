import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

bucket_name = 'openneuro.org'
prefix = 'ds002813/sub-01/func/'
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
print("Files in sub-01/func/")
for obj in response.get('Contents', []):
    print(obj['Key'])
