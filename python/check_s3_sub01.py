import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

response = s3.list_objects_v2(Bucket='openneuro.org', Prefix='ds002813/sub-01/', Delimiter='/')
# Also list first 10 files
res2 = s3.list_objects_v2(Bucket='openneuro.org', Prefix='ds002813/sub-01/')
for o in res2.get('Contents', [])[:20]:
    print(o['Key'])
