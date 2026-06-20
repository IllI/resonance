# AQ-DLINOSS-MERA-SPECTRAL-CAPACITY-TEST-0

Synthetic architecture gate only. No JWST or astrophysical claim is authorized.

The test compares four equal-supervision arms on the locked 96-bin masked synthetic shard:

1. MPS spectral compression + physics-locked D-LinOSS
2. MPS spectral compression + linear SSM
3. physics-locked D-LinOSS without compression
4. linear SSM without compression

The spectral compressor is an open-boundary matrix-product contraction over 12 contiguous eight-bin sites. Each site receives masked local moments and a spectral slope. Both compressed arms use the same encoder contract.

Promotion requires MPS+D-LinOSS leave-window-out correlation at least 0.75, residual correlation at least 0.88, at least 5% improvement over the uncompressed linear SSM, mask and spectral-window shuffle degradation at least 20%, and false-positive correlation at most 0.05.
