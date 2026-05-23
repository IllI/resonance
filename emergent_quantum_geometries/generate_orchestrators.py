import os

with open('run_program_aj_shotgun.ps1', 'r') as f:
    base = f.read()

configs = [
    ('run_us_v5.ps1', 'us-central1-a', 'v5litepod-8', 'v2-alpha-tpuv5-lite', 'program-aj-qr-v5-us', 'program-aj-node-v5-us', 'prog_aj_us_v5.log'),
    ('run_eu_v5.ps1', 'europe-west4-b', 'v5litepod-8', 'v2-alpha-tpuv5-lite', 'program-aj-qr-v5-eu', 'program-aj-node-v5-eu', 'prog_aj_eu_v5.log')
]

for filename, zone, ttype, runtime, qr, node, log in configs:
    content = base.replace('$Zone    = "europe-west4-a"', f'$Zone    = "{zone}"')
    content = content.replace('$Type    = "v6e-8"', f'$Type    = "{ttype}"')
    content = content.replace('$Runtime = "v2-alpha-tpuv6e"', f'$Runtime = "{runtime}"')
    content = content.replace('$QRName  = "program-aj-qr-v6"', f'$QRName  = "{qr}"')
    content = content.replace('$NodeId  = "program-aj-node-v6"', f'$NodeId  = "{node}"')
    content = content.replace('$LogFile   = "prog_aj_run.log"', f'$LogFile   = "{log}"')
    
    with open(filename, 'w') as f:
        f.write(content)
