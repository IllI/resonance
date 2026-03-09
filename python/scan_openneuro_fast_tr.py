import boto3
from botocore import UNSIGNED
from botocore.config import Config
import json
import os
import concurrent.futures

def check_dataset(ds_prefix):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    try:
        # Get the first bold.json file we can find
        res = s3.list_objects_v2(Bucket=bucket, Prefix=ds_prefix, MaxKeys=100)
        for obj in res.get('Contents', []):
            if obj['Key'].endswith('bold.json'):
                res_obj = s3.get_object(Bucket=bucket, Key=obj['Key'])
                data = json.loads(res_obj['Body'].read().decode('utf-8'))
                
                tr = data.get('RepetitionTime')
                tesla = data.get('MagneticFieldStrength')
                
                if tr and tesla:
                    tr = float(tr)
                    tesla = float(tesla)
                    # Look for high field (>=7T) AND fast TR (<=0.8s) 
                    # or ultra-fast TR (<= 0.5s) at any field
                    if (tesla >= 7.0 and tr <= 0.8) or (tr <= 0.5):
                        return f"MATCH: {ds_prefix} | TR: {tr}s | Tesla: {tesla}T"
        return None
    except Exception as e:
        return None

def run_openneuro_scan():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    
    print("Initiating deep scan of OpenNeuro.org for High-Temporal/High-Field datasets...", flush=True)
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix='ds', Delimiter='/')
    
    candidates = []
    for page in pages:
        if 'CommonPrefixes' in page:
            for prefix in page['CommonPrefixes']:
                candidates.append(prefix['Prefix'])
                
    print(f"Found {len(candidates)} total datasets. Scanning metadata via ThreadPool...", flush=True)
    
    matches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_dataset, ds): ds for ds in candidates}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                print(f"[FOUND] {result}", flush=True)
                matches.append(result)
            
            if (i+1) % 200 == 0:
                print(f"  Scanned {i+1}/{len(candidates)} datasets...", flush=True)
                
    print("\n--- SCAN COMPLETE ---", flush=True)
    if not matches:
        print("No datasets driving a 7T scanner at TR < 500ms found on OpenNeuro. We may need to specifically request the MGH-USC HCP 7T dataset or use a 3T TR=400ms scan.")
    else:
        for m in matches:
            print(m)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_openneuro_scan()
