# Program AQ-FFT-7 Remote Analysis - 2026-06-02

## Run Summary

AQ-FFT-7 completed successfully on the TRC TPU VM in `europe-west4-a` on `program-ap-eu-node-v1`.

Launch mode:

```text
--aq-fft-7
```

Runtime envelope:

- noise: `0.00`
- seed: `11`
- depth: `3`
- fixed core: `K=6`
- residual: `residual_K=4`
- controller: `complex_pair_norm`

TRC-safe handling:

- result was inspected from TPU stdout only
- no result artifact, checkpoint, array, directory, or summary payload was downloaded

## Observed Result

| controller | total K | core K | residual K | motif | ifft PSNR | oracle PSNR | coeff mag cos | coeff real cos | coeff imag cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex_pair_norm` | `10` | `6` | `4` | `1.0000` | `26.3560 dB` | `31.1144 dB` | `0.8746` | `0.8747` | `0.5113` |

Additional readout:

| phase cos | freq overlap | top1 freq | topK recall | energy-weighted phase | energy-weighted MSE | pair norm error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.4639` | `0.7875` | `1.0000` | `0.8750` | `0.9992` | `0.0777` | `0.0741` |

Hermitian diagnostics:

| Hermitian before | Hermitian after | self-conjugate imaginary energy | pair consistency |
| ---: | ---: | ---: | ---: |
| `0.0047` | `0.0000` | `0.0000` | `0.9953` |

Layer/header readout stayed stable:

| sanity | L1 | L2 | L3 | L4 | L5 | leak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.9990` | `0.8750` | `0.8749` | `0.9975` | `0.8747` | `0.9943` | `0.0087` |

## Capacity Scaling

The important comparison is against the repeated residual_K=2 reference:

| codec | residual K | realized PSNR | oracle PSNR | motif/header |
| --- | ---: | ---: | ---: | ---: |
| `complex_pair_norm` | `2` | `23.1818 dB` | `25.0480 dB` | `1.0000` |
| `complex_pair_norm` | `4` | `26.3560 dB` | `31.1144 dB` | `1.0000` |

The oracle rose by `6.0664 dB` and the realized reconstruction rose by `3.1742 dB`.

That is the clearest positive result in the AQ-FFT sequence so far: increasing sparse residual coefficient capacity increased both the theoretical sparse-code ceiling and the realized D-LinOSS/unfold reconstruction.

## Experimental Journey

The path to AQ-FFT-7 matters scientifically because several plausible failure explanations were ruled out in sequence.

### Operator grammar failure

AQ-1c showed that semantic operator transport was not yet identifiable. The system could preserve some folded semantic structure, but the operator was not a reusable compositional rule. That forced a reset: before claiming transported semantic operators, the experiment had to prove that source content and operator action were both recoverable.

The key bug was that marker payloads had displaced the actual source motif sequence. Once endpoint leakage was removed, the legitimate semantic content path was also removed. This established an important methodological rule:

> D-LinOSS must be given the actual semantic state or a physically meaningful continuous representation of it; marker-only payloads can create classifier-like correlations but not compositional recovery.

### Sequential semantic state construction

AQ-SEQ-0 then showed that a recovered state at time `t` could derive the subsequent semantic state at `t+1` at the clean ceiling. That did not solve arbitrary operator transport, but it did show that sequential state assembly can be coherent in the recovered semantic space.

This became the conceptual bridge to image/video representations: rather than asking one symbolic operator to generate everything, represent a field as a sequence of recoverable local states.

### Image patch vocabulary was insufficient

The first image patch runs showed that motif/header recovery was robust, but pixel reconstruction was weak. That was not a failure of source recovery; it was a representation mismatch. Abstract sequence motifs were too coarse for image-domain reconstruction.

The experiment therefore moved from symbolic motifs to k-space coefficients.

### K-space codec became the right representation

AQ-FFT-1 established that fixed low-frequency plus sparse residual k-space was the correct direction, but the original magnitude/phase split created a reconstruction bottleneck:

```math
\operatorname{Im}(\hat{F}_k)=|\hat{F}_k|\sin(\hat{\phi}_k)
```

Correlated magnitude and phase errors can degrade the imaginary component even if each channel is only moderately wrong.

AQ-FFT-3 fixed this by injecting complex coefficients directly into D-LinOSS oscillator slots:

```math
x_0[k] = F_k
```

That raised realized PSNR from roughly `17 dB` to `20.6478 dB` while keeping motif recovery perfect.

### Complex-pair normalization became the reference codec

AQ-FFT-4 tested whether complex orientation was the issue. `complex_pair_norm` treated each coefficient as a coupled real/imaginary pair and carried magnitude as a side signal. This lifted realized PSNR to `23.1818 dB`.

That run made `complex_pair_norm` the best reference codec.

### Gauge and Hermitian hypotheses were ruled out

AQ-FFT-5 tested global complex gauge mismatch. It found:

- `gauge_err = 0.0000`
- PSNR before and after alignment unchanged
- aligned phase/imaginary metrics unchanged

So weak raw phase/imaginary cosine was not a global phase rotation problem.

AQ-FFT-6 tested Hermitian mismatch for real-image FFTs:

```math
F[-k] = \overline{F[k]}
```

It found:

- Hermitian error already tiny: `0.0070`
- projection reduced it to `0.0000`
- PSNR did not move
- energy-weighted phase was excellent: `0.9992`

So the weak raw imaginary/phase metrics were not the first-order reconstruction blocker. They mostly reflected low-energy quadrature components that barely affect the inverse FFT.

### AQ-FFT-7 proved capacity scaling

Once the misleading phase bottleneck was deprioritized, AQ-FFT-7 increased residual coefficient capacity. The result was decisive:

```text
residual_K=2 -> 23.18 dB realized, 25.05 dB oracle
residual_K=4 -> 26.36 dB realized, 31.11 dB oracle
```

That means the folded codec did not merely memorize a fixed small reconstruction. It scaled when the sparse k-space body was expanded.

## Scientifically Defensible Claim

The current empirical claim should be narrow and reproducible:

> In this simulator, D-LinOSS complex oscillator transport can preserve a folded semantic/header state while carrying enough continuous k-space coefficient structure to reconstruct an unfolded `4 x 4` image patch field. The reconstruction improves when sparse residual k-space capacity is increased, while semantic/header recovery remains perfect.

This is stronger than label recovery and stronger than motif classification. The recovered object is not merely a class identity; it contains enough continuous coefficient information for inverse-FFT reconstruction.

The claim should not be overstated as physical quantum teleportation. The current program is a simulated complex oscillator transport and decoding experiment. It uses tensor-network and quantum-inspired language because the state representation is complex-valued, phase-bearing, and folded through interval/tunnel geometry, but the validated claim is computational:

> D-LinOSS can operate as a recoverable folded-state transport channel in a simulated quantum-like complex state space.

## Mathematical Interpretation

The recovered patch is produced by:

```math
\hat{p}=\operatorname{Re}\left(\mathcal{F}^{-1}\hat{F}\right)
```

where the retained sparse field is:

```math
\hat{F} = \hat{F}_{core} + \hat{F}_{residual}
```

AQ-FFT-7 shows that adding residual coefficients increases the recoverable reconstruction quality:

```math
\operatorname{PSNR}(\hat{p}_{K=10}) >
\operatorname{PSNR}(\hat{p}_{K=8})
```

with fixed:

- source patch set
- seed
- depth
- noise level
- controller family
- semantic/header machinery

The high energy-weighted phase cosine means:

```math
\frac{\sum_k |F_k|^2 \cos(\phi_k-\hat{\phi}_k)}
{\sum_k |F_k|^2}
\approx 0.9992
```

Therefore the phase information that materially affects reconstruction is preserved even though raw unweighted phase metrics remain weak.

## Reproducibility Notes

Minimal reproduction run:

```bash
python3 -u emergent_quantum_geometries/program_ap_tunnel_eigen_recovery.py \
  --require-tpu \
  --aq-fft-7 \
  --out-dir /home/cityz/program_ap/program_ap_results
```

Expected gate:

- `fft_patch_motif_recovery = 1.0000`
- `fft_patch_psnr > 26 dB`
- `fft_oracle_psnr > 31 dB`
- `fft_phase_cos_energy_weighted > 0.99`
- `fft_top1_frequency_accuracy = 1.0000`

Primary comparison:

- AQ-FFT-4/5/6 `complex_pair_norm`, residual_K=2
- AQ-FFT-7 `complex_pair_norm`, residual_K=4

The result should be treated as one-seed clean-ceiling evidence. The next reproducibility step is to repeat the residual_K=4 codec across additional seeds and then add noise.

## Next Step

AQ-FFT-8 should test robustness rather than more clean-ceiling capacity:

- residual_K=4
- noise: `0.30`
- controllers: `free`, `alpha065`, `shape_charge_control`

At the clean ceiling, `alpha065` has repeatedly underperformed or tied `free`. Under noisy transport, however, controller stabilization may become meaningful.
