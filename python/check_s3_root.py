import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

response = s3.list_objects_v2(Bucket='openneuro.org', Prefix='ds002813/', Delimiter='/')
print("Common prefixes in ds002813/:")
for p in response.get('CommonPrefixes', []):
    print(p['Prefix'])

print("\nFiles in root:")
for c in response.get('Contents', []):
    print(c['Key'])
