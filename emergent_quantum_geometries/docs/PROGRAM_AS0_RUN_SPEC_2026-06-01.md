# Program AS-0 Run Spec - Hierarchical Instruction Genome Transport

## Purpose

AS-0 tests whether the AQ-FFT-1 instruction genome should be transported as a folded tensor state rather than as a flat sequence of coefficient packets.

The current bottleneck is stable:

- semantic/header recovery is solved
- codec capacity is high (`oracle_psnr ~25.05 dB`)
- realized reconstruction is stuck near `17.0 dB`
- residual coefficient instruction realization is the unresolved piece

AS-0 asks whether multi-scale folding helps that residual instruction body survive and unfold.

## Core Representation

AS-0 keeps the AQ-FFT-1 payload:

- semantic header
- fixed k-space core
- sparse residual instructions
- phase-safe `sin(phi), cos(phi)` channels
- checksum

But instead of feeding those packets as a flat sequence, it arranges them as TTN leaves:

```text
coefficient/residual packets
  -> binary TTN fold
  -> root tensor + selected internal bonds + charge sectors
  -> D-LinOSS transport
  -> unfolded coefficient field
  -> inverse FFT patch
```

The transported object is therefore:

```text
root tensor + internal bond messages + charge sectors
```

not every token independently.

## Charge Structure

Each internal bond carries:

| charge | meaning |
| --- | --- |
| `Q_DC` | luminance/DC charge |
| `Q_lowfreq_energy` | low-frequency energy |
| `Q_phase_coherence` | phase coherence |
| `Q_residual_support` | sparse residual support |
| `Q_motif_consistency` | semantic motif consistency |
| `Q_checksum_parity` | checksum parity |

The TTN fold tests approximate conservation:

```math
Q_{parent} \approx Q_{left} + Q_{right}
```

## Launch Mode

```bash
--as-0
```

Expected active banner:

```text
AS-0 ACTIVE:
  payload = AQ-FFT-1 instruction packets
  geometry = binary TTN over coefficient/residual packets
  transported_object = root tensor + selected internal bonds + charge sectors
  baseline = flat AQ-FFT-1 free path
  controller = free
```

If this banner does not appear, the run is not testing AS-0.

## Runtime Envelope

- noise: `0.00`
- seed: `11`
- controller: `free`
- depth: `3`
- D-LinOSS steps: compact gate, capped at `16`
- patch grid: `4 x 4`
- patch size: `4 x 4`
- fixed core: `K=6`
- residual: `residual_K=2`
- no alpha sweep
- no noise sweep
- no hard controls
- no larger payload

## Baseline

AS-0 compares against the flat AQ-FFT-1 free path:

- flat realized PSNR: `~17.00 dB`
- flat oracle PSNR: `~25.05 dB`

The first target is modest:

```text
TTN realized PSNR > 17.0 dB
```

Even `18-19 dB` would be meaningful because it would show that hierarchical folding improves coefficient realization without changing codec capacity.

## Metrics

Primary:

- `fft_patch_psnr`
- `fft_oracle_psnr`
- `as_root_tensor_cosine`
- `as_charge_conservation_error`
- `as_charge_cosine`
- `as_unfolded_coeff_psnr`
- `as_ttn_lift_over_flat_psnr`
- `fft_coefficient_cosine`
- `fft_coeff_mag_cosine`
- `fft_coeff_real_cosine`
- `fft_coeff_imag_cosine`
- `fft_phase_recovery`
- `fft_topk_energy_recall`
- semantic header / motif recovery

## Decision Tree

If TTN PSNR exceeds the flat baseline:

- hierarchical folding is helping the residual instruction body
- next run can compare TTN against a tiny MERA/disentangler variant

If root/charge recovery is strong but PSNR stays flat:

- the folded state survives
- the unfold decoder is probably too weak
- improve the unfold path before changing controllers

If root/charge recovery fails:

- the TTN representation is too compressed or poorly aligned with D-LinOSS
- repair the fold geometry before trying MERA, noise, or alpha tuning

If semantic header recovery drops below `1.0000`:

- AS-0 broke the stable semantic channel
- revert to the flat header path and fold only the coefficient body

## AS-0b Correction

AS-0 completed with a useful negative result:

- root tensor recovery was strong
- charge recovery was very strong
- realized PSNR did not improve over the flat AQ-FFT-1 baseline

The implementation issue is that AS-0 measured the recovered TTN state, but coefficient reconstruction still used the direct signature-to-coefficient decoder:

```text
noisy D-LinOSS signature -> sparse FFT coefficients
```

That meant the hierarchy could survive without being used to realize the coefficient field.

AS-0b corrects this by changing the decoder path:

```text
noisy D-LinOSS signature
  -> recovered TTN root/bonds/charges
  -> TTN unfold decoder
  -> sparse FFT coefficients
  -> inverse FFT patch
```

Mode:

```bash
--as-0b
```

Runtime is the same compact gate as AS-0:

- noise: `0.00`
- seed: `11`
- controller: `free`
- fixed core: `K=6`
- residual: `residual_K=2`
- depth: `3`

New readout field:

```text
decoder=ttn_unfold
```

AS-0b passes only if the TTN unfold path improves realized PSNR or coefficient recovery over the flat baseline. If it does not, the hierarchy is not carrying enough slotwise coefficient detail, and the next correction should target the coefficient-body representation directly, especially the weak imaginary/residual channel.

## AS-0 Result

AS-0 completed cleanly and produced a useful negative result.

Observed readout:

| metric | value |
| --- | ---: |
| `ifft_psnr` | `16.9995 dB` |
| `oracle_psnr` | `25.0480 dB` |
| `as_ttn_lift_over_flat_psnr` | `~0.0000 dB` |
| `as_root_tensor_cosine` | `0.9518` |
| `as_charge_cosine` | `0.9962` |
| `as_charge_conservation_error` | `0.1877` |
| `coeff_mag_cos` | `0.8551` |
| `coeff_real_cos` | `0.8613` |
| `coeff_imag_cos` | `0.6284` |
| `phase_cos` | `0.8930` |
| `freq_idx_overlap` | `0.7578` |
| `top1_freq_acc` | `1.0000` |
| `topK_energy_recall` | `0.8738` |
| motif/header recovery | `1.0000` |

What worked:

- the TTN root tensor survived transport well
- the charge summary survived transport extremely well
- the semantic/header channel stayed intact

What did not work:

- realized reconstruction did not improve over the flat AQ-FFT-1 baseline
- coefficient-support and residual realization metrics did not move materially
- the weakest channel remained the slotwise imaginary/residual detail

Methodological read:

- the folded representation is recoverable as a global summary
- but that summary is not yet the scarce object
- the missing information still lives in fine coefficient-body realization rather than in global shape/charge recovery

This means AS-0 did answer the main question: adding TTN folding alone was not enough to close the gap between the `~25 dB` oracle ceiling and the `~17 dB` realized ceiling.

## AS-0b Result

AS-0b fixed the implementation issue in AS-0 by forcing reconstruction to actually use the recovered TTN state:

```text
noisy D-LinOSS signature
  -> recovered TTN root/bonds/charges
  -> TTN unfold decoder
  -> sparse FFT coefficients
  -> inverse FFT patch
```

Observed readout:

| metric | value |
| --- | ---: |
| `ifft_psnr` | `16.9932 dB` |
| `oracle_psnr` | `25.0480 dB` |
| `flat_psnr` | `16.9995 dB` |
| `ttn_lift` | `-0.0063 dB` |
| `root_cos` | `0.9518` |
| `charge_cos` | `0.9962` |
| `charge_err` | `0.1877` |
| `coeff_mag_cos` | `0.8551` |
| `coeff_real_cos` | `0.8613` |
| `coeff_imag_cos` | `0.6280` |
| `phase_cos` | `0.8931` |
| `freq_idx_overlap` | `0.7578` |
| `top1_freq_acc` | `1.0000` |
| `topK_energy_recall` | `0.8738` |
| `decoder` | `ttn_unfold` |
| motif/header recovery | `1.0000` |

What worked:

- the corrected path ran cleanly and used the TTN unfold decoder as intended
- the TTN state remained strongly recoverable
- semantic/header transport remained solved

What did not work:

- reconstruction still did not improve; it became slightly worse than the flat baseline
- the unfold decoder did not recover more slotwise coefficient detail than the direct flat decoder
- imaginary/residual detail remained the limiting channel

Why:

- TTN folding preserves global and mesoscale organization, which is why `root_cos` and `charge_cos` stay high
- but the AQ-FFT plateau is now driven by missing fine-grained residual instruction realization
- the TTN state is therefore recoverable without being more informative for pixel/patch reconstruction than the direct flat signature path

AS-0b is therefore the stronger negative result. It shows that the lack of PSNR lift is not just a decoder wiring mistake. Even when the recovered TTN state is actually used, the hierarchy still does not improve coefficient-field realization.

## Updated Conclusion

Program AS produced a clean representational test.

What worked:

- semantic/header transport
- TTN root-state recovery
- TTN charge-summary recovery
- phase-safe coefficient transport at the same level as AQ-FFT-1

What did not work:

- TTN folding did not improve realized reconstruction
- TTN unfolding did not improve coefficient-body fidelity
- the oracle-to-realized gap remains dominated by fine residual/imaginary-slot realization

The practical conclusion is:

> hierarchical folding is transport-compatible, but it is not the main missing ingredient for reconstruction fidelity in the current AQ-FFT regime.

So the next correction should not be more TTN/MERA complexity by default. It should target the coefficient-body representation and decoder path that handles weak slotwise complex detail.

## Scientific Read

AS-0 is not a robustness run. It is a noise-free ceiling gate for the representation.

The claim it can support is:

> A hierarchical folded instruction genome can transport a recoverable coefficient field better than a flat packet stream.

The claim it cannot yet support is:

> The hierarchy is robust under noise or adversarial perturbation.

Noise and controller tests should wait until AS-0 beats or clarifies the flat baseline.
