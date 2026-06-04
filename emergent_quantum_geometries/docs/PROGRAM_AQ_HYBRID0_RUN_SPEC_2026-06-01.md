# Program AQ-HYBRID-0 Run Spec - 2026-06-01

## Purpose

AQ-DNA-0 showed that discrete coefficient packets survive transport:

- instruction packet exact match: `0.9062`
- magnitude/phase/residual/support token accuracy: about `0.95` to `0.97`
- motif/header recovery: `1.0000`

But it did not improve the media-side bottleneck:

- realized PSNR stayed at `16.9995 dB`
- oracle PSNR stayed at `25.0480 dB`
- MI gaps over shuffled/null controls stayed near `0` bits

The interpretation is:

> hard tokenization is recoverable, but it is not yet structurally useful.

AQ-HYBRID-0 tests the correction suggested by the latest review note:

> keep continuous D-LinOSS/k-space geometry as the primary payload, and use tokens only as local checksums or constraints.

This avoids forcing the analog amplitude/phase structure into hard bins before transport.

## Hypothesis

The D-LinOSS transport channel is better aligned with continuous oscillator geometry than with hard symbolic bins.

So the payload should remain:

$$
z_i =
[
\operatorname{Re}(F_i),
\operatorname{Im}(F_i),
|F_i|,
\cos \phi_i,
\sin \phi_i,
\mathrm{residual\ geometry}_i
]
$$

while the token grammar acts as a consistency check:

$$
c_i =
[
\mathrm{mag\_state}_i,
\mathrm{phase\_state}_i,
\mathrm{residual\_state}_i,
\mathrm{support\_rank}_i
]
$$

The decoder should reconstruct from continuous geometry first:

$$
\hat{p} =
\Re\left(\mathcal{F}^{-1}_2(\hat{F})\right)
$$

Then test whether the recovered continuous coefficient agrees with the token checksum:

$$
\mathrm{checksum\_agreement}
=
\Pr[q(\hat{F}_i) = c_i]
$$

## Run shape

- mode: `--aq-hybrid-0`
- noise: `0.00`
- seed: `11`
- controller: `free`
- patch grid: `4 x 4`
- patch size: `4 x 4`
- fixed core: `K=6`
- residual: `residual_K=2`
- hard controls: off
- alpha sweep: off
- noise sweep: off

This should stay in the same runtime envelope as AQ-DNA-0.

## Encoding

Use the AQ-FFT-1 continuous body as primary:

```text
continuous_body =
  sparse magnitude
  phase cos/sin
  continuous residual coefficient geometry
  semantic motif header
```

Add token side-channel only as a checksum:

```text
token_checksum =
  magnitude_state
  phase_state
  residual_state
  support_rank_state
```

The token checksum should not replace the continuous body in `source_features` or `graph_features`.

## Local packet windows

AQ-DNA-0 mostly measured per-slot token recovery. AQ-HYBRID-0 should measure packet neighborhoods.

For each retained slot:

$$
W_i = [i_{-1}, i, i_{+1}]
$$

where neighbors come from fixed k-space zigzag/local order.

Report:

- `window_token_consistency`
- `neighborhood_phase_consistency`
- `local_residual_support_consistency`
- `window_mi_gap`

This asks whether nearby coefficient packets carry recoverable structure jointly, not merely whether isolated labels decode.

## Metrics

Core reconstruction:

- `ifft_psnr`
- `oracle_psnr`
- `coeff_mag_cos`
- `coeff_real_cos`
- `coeff_imag_cos`
- `phase_cos`
- `freq_idx_overlap`
- `topK_energy_recall`

Token checksum:

- `checksum_agreement`
- `mag_state_acc`
- `phase_state_acc`
- `residual_state_acc`
- `support_rank_acc`
- `packet_exact_match`

Continuous-vs-token dependency diagnostics:

- `MI(header; continuous_residual_geometry)`
- `MI(header; residual_token)`
- `MI(local_phase_field; patch_class)`
- `MI(phase_token; patch_class)`
- `MI(coefficient_window; reconstruction_error)`
- `MI(window_tokens; reconstruction_error)`

Report the gaps:

$$
\Delta MI_{\mathrm{continuous}}
=
MI_{\mathrm{continuous}} - MI_{\mathrm{continuous,null}}
$$

$$
\Delta MI_{\mathrm{token}}
=
MI_{\mathrm{token}} - MI_{\mathrm{token,null}}
$$

## Decision rule

AQ-HYBRID-0 is promising if:

- semantic header remains `1.0000`
- realized PSNR exceeds `17.0 dB`
- continuous MI gap is greater than token MI gap
- window consistency exceeds isolated packet consistency
- token checksum agreement remains high

If continuous MI beats token MI but PSNR still does not move, keep the hybrid direction and improve the decoder/window model.

If neither continuous MI nor window consistency beats null, the locality structure is still not active enough.

## Interpretation

AQ-DNA-0 says:

> labels are recoverable.

AQ-HYBRID-0 asks:

> does continuous geometry carry stronger recoverable dependency than hard token labels?

This is closer to the D-LinOSS/tensor-network intuition: the transported object is an analog folded geometry, with discrete symbols acting as constraints rather than replacing the geometry itself.
