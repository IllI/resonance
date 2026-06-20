# AQ-DLINOSS-MERA-SPECTRAL-CAPACITY-TEST-0

Synthetic architecture gate only. No JWST or astrophysical claim is authorized.

The test compares four equal-supervision arms on the locked 96-bin masked synthetic shard:

1. MPS spectral compression + physics-locked D-LinOSS
2. MPS spectral compression + linear SSM
3. physics-locked D-LinOSS without compression
4. linear SSM without compression

The spectral compressor is an open-boundary matrix-product contraction over 12 contiguous eight-bin sites. Each site receives masked local moments and a spectral slope. Both compressed arms use the same encoder contract.

Promotion requires MPS+D-LinOSS leave-window-out correlation at least 0.75, residual correlation at least 0.88, at least 5% improvement over the uncompressed linear SSM, mask and spectral-window shuffle degradation at least 20%, and false-positive correlation at most 0.05.

## Result

The 48-configuration sweep completed on eight TPU devices with verdict `DO_NOT_PROMOTE`.

| Arm | Residual corr | Latent corr | Leave-window-out | Mask shuffle | Spectral shuffle |
|---|---:|---:|---:|---:|---:|
| MPS + D-LinOSS | 0.9887 | 0.9873 | 0.7462 | 0.0768 | 0.9228 |
| MPS + linear SSM | 0.9872 | 0.9866 | 0.7054 | 0.0764 | 0.8051 |
| D-LinOSS | 0.9934 | 0.9924 | 0.4359 | 0.1097 | 1.2058 |
| linear SSM | 0.9132 | 0.8747 | 0.6472 | 0.1225 | 1.0141 |

MPS compression materially improved D-LinOSS leave-window-out recovery from 0.4359 to 0.7462, but remained below the predeclared 0.75 threshold. More importantly, mask-shuffle degradation was only 7.68%, far below the required 20%. The model therefore learned a useful compressed spectral representation without demonstrating robust mask-conditioned assimilation.

The independent GLIMPSE-17775 audit returned `PASS_LRD_DATA`, but the real radiative-history run remains blocked because the synthetic architecture gate did not pass. The relevant public spectrum is from DDT Program 9223 rather than imaging Program 3293.
