"""validate_scramble_chunks.py — checks alignment and schema of v2 chunk data."""
import argparse, json, pathlib, sys
import numpy as np

def load_manifest(d):
    return json.loads((pathlib.Path(d) / 'manifest.json').read_text())

def check_dir(label, d):
    d = pathlib.Path(d)
    m = load_manifest(d)
    chunks = m['chunk_files']
    print(f'\n[{label}]  role={m["role"]}  chunks={len(chunks)}  t0={m["t0_utc"]:.1f}')
    print(f'  zone={m["infrastructure"].get("gcp_zone","?")}  '
          f'ip={m["infrastructure"].get("ip","?")}  '
          f'tz={m["infrastructure"].get("timezone","?")}')

    lam_vals, missing, bad_cov = [], [], []
    for cf in chunks:
        p = d / cf
        if not p.exists():
            missing.append(cf); continue
        npz = np.load(str(p))
        if 'covariance_matrix' not in npz.files:
            bad_cov.append(cf); continue
        lam_vals.append(float(npz['lambda_val']))

    lam_vals = np.array(lam_vals)
    print(f'  λ range: [{lam_vals.min():.3f}, {lam_vals.max():.3f}]  '
          f'ON fraction (λ>0.5): {(lam_vals>0.5).mean()*100:.0f}%')
    print(f'  Missing chunks: {len(missing)}  Bad cov: {len(bad_cov)}')
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--alice', required=True)
    p.add_argument('--bob',   required=True)
    args = p.parse_args()

    ma = check_dir('ALICE', args.alice)
    mb = check_dir('BOB',   args.bob)

    # Alignment check
    dt_t0 = abs(ma['t0_utc'] - mb['t0_utc'])
    print(f'\nAlignment:  |t0_alice - t0_bob| = {dt_t0:.2f}s')
    if dt_t0 > 30:
        print('  WARNING: sessions started >30s apart. λ schedule will be misaligned.')
    else:
        print('  OK: starts within 30s.')

    # Zone check
    za = ma['infrastructure'].get('gcp_zone')
    zb = mb['infrastructure'].get('gcp_zone')
    if za and zb and za == zb:
        print(f'\n  !! SAME ZONE DETECTED: {za} — shared-resource confound possible.')
    else:
        print(f'\n  Zones: Alice={za}  Bob={zb}  → different ✓')

    print('\nValidation complete.')

if __name__ == '__main__':
    main()
