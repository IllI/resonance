# Program AQ-1c Source Recovery and Sequential Signaling Methodology - 2026-05-30

## TRC-safe handling

This note was written from TPU stdout inspected over SSH. No TPU result artifacts, arrays, checkpoints, directories, or summary JSON files were downloaded.

Run context:

- project: `time-emission`
- node: `program-aq1c-eu-node-v7`
- zone: `europe-west4-a`
- mode: `--aq-1c`
- backend: JAX TPU
- controller: `free`
- noise: `0.00`
- seed: `11`

## Corrected AQ-1c payload

AQ-1c was corrected so the transported payload contains the real source motif:

```python
payload_seqs.append(seqs[src_idx])
```

The pre-run sanity check passed:

```text
payload_seqs sample: [0, 1, 2, 3, 0, 1, 2, 3]
source motif sample: [0, 1, 2, 3, 0, 1, 2, 3]
payload/source match: True
```

This matters because the earlier AQ-1c branch accidentally transported marker sequences instead of semantic motif sequences. The corrected run therefore tests semantic source transport again, not marker recovery.

## Result

| metric | value |
| --- | ---: |
| worst-control margin | `+0.3036` |
| folded recovery score | `0.5050` |
| noisy graph cosine | `0.9106` |
| block lift | `+0.2453` |
| topology lift | `+0.1191` |
| untransported sanity | `0.6043` |
| L1 occupation recovery | `0.9850` |
| L2 transition recovery | `0.9106` |
| L3 FFT/phase recovery | `0.9445` |
| L4 recurrence recovery | `0.9217` |
| L5 operator recovery cosine | `0.5272` |
| inter-layer leakage proxy | `0.1231` |
| source recovery | `1.0000` |
| operator recovery | `0.2500` |
| derived target accuracy | `0.2500` |
| target direct leakage | `0.1875` |

The corrected AQ-1c run recovered source identity perfectly across the `16` source/operator test states. The decoded readout showed every true source matched its predicted source:

| row | true source | predicted source | true operator | predicted operator | derived ok |
| ---: | --- | --- | --- | --- | --- |
| 0 | `sparse` | `sparse` | `SWAP_ATTRACTOR` | `SYMBOL_SHIFT` | `False` |
| 1 | `fractal_nested` | `fractal_nested` | `SYMBOL_SHIFT` | `SYMBOL_SHIFT` | `True` |
| 2 | `palindrome` | `palindrome` | `MIRROR_TIME` | `SYMBOL_SHIFT` | `False` |
| 3 | `cyclic` | `cyclic` | `MIRROR_TIME` | `SYMBOL_SHIFT` | `False` |
| 4 | `sparse` | `sparse` | `SYMBOL_SHIFT` | `SYMBOL_SHIFT` | `True` |
| 5 | `cyclic` | `cyclic` | `ROTATE_PHASE` | `SYMBOL_SHIFT` | `False` |
| 6 | `cyclic` | `cyclic` | `SWAP_ATTRACTOR` | `SYMBOL_SHIFT` | `False` |
| 7 | `cyclic` | `cyclic` | `SYMBOL_SHIFT` | `SYMBOL_SHIFT` | `True` |
| 8 | `palindrome` | `palindrome` | `SWAP_ATTRACTOR` | `SYMBOL_SHIFT` | `False` |
| 9 | `sparse` | `sparse` | `ROTATE_PHASE` | `SYMBOL_SHIFT` | `False` |
| 10 | `palindrome` | `palindrome` | `SYMBOL_SHIFT` | `SYMBOL_SHIFT` | `True` |
| 11 | `sparse` | `sparse` | `MIRROR_TIME` | `SYMBOL_SHIFT` | `False` |
| 12 | `fractal_nested` | `fractal_nested` | `MIRROR_TIME` | `SYMBOL_SHIFT` | `False` |
| 13 | `palindrome` | `palindrome` | `ROTATE_PHASE` | `SYMBOL_SHIFT` | `False` |
| 14 | `fractal_nested` | `fractal_nested` | `ROTATE_PHASE` | `SYMBOL_SHIFT` | `False` |
| 15 | `fractal_nested` | `fractal_nested` | `SWAP_ATTRACTOR` | `SYMBOL_SHIFT` | `False` |

## Interpretation

The corrected run separates two claims cleanly:

1. Source semantic state recovery works in the noise-free AQ-1c gate.
2. Operator derivation does not yet work.

The source result is the important semantic-transport result:

$$
s \xrightarrow{E} z \xrightarrow{\mathrm{transport}} \tilde{z} \xrightarrow{D} \hat{s}
$$

with:

$$
\Pr[\hat{s} = s] = 1.0000.
$$

Layer recovery was also strong:

$$
L_1 = 0.9850,\quad
L_2 = 0.9106,\quad
L_3 = 0.9445,\quad
L_4 = 0.9217.
$$

That supports the narrow AQ claim:

> A folded semantic source state can be encoded, transported, and decoded with high layer-wise fidelity in the noise-free gate.

The operator branch failed because the decoder collapsed almost every operator to `SYMBOL_SHIFT`:

$$
\Pr[\hat{T}=T] = 0.2500.
$$

This is chance for a four-operator family. Therefore:

$$
\Pr[\hat{T}(\hat{s}) = T(s)] = 0.2500.
$$

The obstacle is not source transport. The obstacle is operator fidelity.

## Operator Fidelity Obstacle

The current operator setup treats operators as class labels attached to source states:

$$
z = E(s; C_T)
$$

where \(C_T\) is a compact operator code. This did not make the operators identifiable from the transported state. Since the source sequence itself contains no operator-specific waveform action, the recovered state supports source decoding but not operator decoding.

The collapsed operator table says:

$$
\hat{T} \approx \texttt{SYMBOL\_SHIFT}
\quad
\forall T.
$$

That means the model is not distinguishing the operator basis. It is recovering the semantic state, then defaulting to one operator prototype.

This is not a failure of semantic transport. It is a mismatch between the operator abstraction and the current transport geometry.

For operator derivation to become meaningful, the operator family should be waveform-like and layer-action based, not arbitrary symbolic remapping. Examples:

- null subtraction:
  \( \mathcal{O}(s) = s - P_{\mathrm{null}}s \)
- nodal subtraction:
  \( \mathcal{O}(s) = s - P_{\mathrm{node}}s \)
- harmonic amplification:
  \( \mathcal{O}(s) = F^{-1}G_\omega F s \)
- graph diffusion:
  \( \mathcal{O}(s) = e^{-\tau L}s \)
- recurrence filtering:
  \( \mathcal{O}(s) = R_\lambda s \)

Those operators act on measurable semantic layers rather than on arbitrary source/target labels.

## Sequential Semantic Signaling

The more natural next step is not arbitrary operator derivation. It is sequential semantic signaling.

Define a semantic state at time \(t\):

$$
S_t =
\left[
L_1(s_t),
L_2(s_t),
L_3(s_t),
L_4(s_t)
\right].
$$

Transport and recover each folded state:

$$
s_t
\xrightarrow{E}
z_t
\xrightarrow{\mathrm{transport}}
\tilde{z}_t
\xrightarrow{D}
\hat{S}_t.
$$

Then ask whether the recovered sequence preserves the temporal increment:

$$
\Delta S_t = S_{t+1} - S_t,
$$

and:

$$
\Delta \hat{S}_t = \hat{S}_{t+1} - \hat{S}_t.
$$

Primary sequential metric:

$$
\mathrm{SeqFid}
=
\cos(\Delta \hat{S}_t,\Delta S_t).
$$

A successful sequential run does not need to infer an arbitrary operator label. It only needs to show that the transported/recovered semantic states preserve ordered changes in the folded semantic field.

That is a better bridge to operator math:

$$
\hat{S}_{t+1}
\approx
\mathcal{O}(\hat{S}_t),
$$

where \(\mathcal{O}\) is estimated from lawful waveform dynamics after source recovery is already reliable.

## AQ-SEQ-0 Follow-Up Result

AQ-SEQ-0 was implemented as a compact sequential gate after the source-recovery result. The transported payload remained the real source motif:

```python
payload_seqs.append(seqs[src_idx])
```

The second gate was encoded in feature geometry rather than in the transported sequence. The successful version used a transition-shape code:

```text
L5_transform = target_state_layers + delta_layers
```

This means the tensor network still simulated transport of the semantic source state, while a separate gate carried subsequent-state structure for reconstruction.

Configuration:

- noise: `0.00`
- depth: `3`
- seed: `11`
- controller: `free`
- payload: real motif sequence only
- sequence length: `8`
- sources: `cyclic`, `fractal_nested`, `palindrome`, `sparse`
- ordered transitions: adjacent motif states or controlled waveform perturbations

Final observed AQ-SEQ-0 metrics:

| metric | value |
| --- | ---: |
| source recovery | `1.0000` |
| derived next accuracy | `1.0000` |
| transition class recovery | `1.0000` |
| L5 layer recovery | `0.8045` |
| direct target recovery | `0.7500` |
| exact direct match | `0.7500` |
| leakage proxy | `0.0460` |

Decoded readout:

| true source | predicted source | true next | direct predicted next | derived next |
| --- | --- | --- | --- | --- |
| `cyclic` | `cyclic` | `fractal_nested` | `palindrome` | `fractal_nested` |
| `fractal_nested` | `fractal_nested` | `palindrome` | `palindrome` | `palindrome` |
| `sparse` | `sparse` | `cyclic` | `cyclic` | `cyclic` |
| `palindrome` | `palindrome` | `sparse` | `sparse` | `sparse` |

The important distinction is:

$$
\mathrm{Acc}_{direct\ next} = 0.7500
$$

but:

$$
\mathrm{Acc}_{derived\ next} = 1.0000.
$$

The direct next-state decoder still misread one target, but the recovered source plus recovered transition gate derived the correct next semantic state on all four transitions. This supports the idea that the useful reconstruction path is:

$$
\hat{S}_{t+1}
\approx
G_{\mathrm{seq}}(\hat{S}_t),
$$

where \(G_{\mathrm{seq}}\) is a second semantic gate informed by subsequent-state structure.

## Practical Next Step

The next useful engineering step is to save compact recovered-signal outputs from successful source/sequence gates, then run decoding heuristics separately.

The TPU-side job should stay focused on:

- folded source-state transport
- recovered state vectors
- layer-wise semantic signatures
- transition-gate signatures
- minimal metadata needed to reproduce the run

The offline or separate decoder layer should handle:

- reconstruction heuristics
- semantic-to-media mapping
- operator-space detection
- waveform/layer operator fitting
- larger examples such as video-frame and audio-channel semantic spaces

This keeps the tensor network claim appropriately narrow:

> The transport simulation recovers the folded semantic shape.

The later reconstruction claim is separate:

> A decoder can map recovered semantic shapes back into meaningful media/state representations.

Concrete media examples, such as encoding a movie frame plus audio channels and reconstructing it after transport, will require a larger semantic encoder, richer modality-specific layers, and a reconstruction model outside the current AQ-SEQ-0 gate. AQ-SEQ-0 is evidence that the recovered semantic state and its sequential relation can be made available to such a decoder, not that full media reconstruction is solved.

## AQ-IMG-0 Bridge

The first concrete media-adjacent bridge is `AQ-IMG-0`, a tiny image patch semantic codec.

It keeps the tensor-network/D-LinOSS responsibility narrow:

```text
recover the folded semantic patch state
```

and puts reconstruction into a decoder metric:

```text
recovered semantic state -> grayscale patch pixels
```

Configuration:

- patch grid: `4 x 4`
- patch size: `4 x 4` grayscale pixels
- patch states: `16`
- image motif vocabulary: `8`
- payload: nearest image-domain motif sequence
- residual: scalar correction per patch
- decoder: ridge-style pixel reconstruction from recovered features

Image-domain motifs:

| motif | purpose |
| --- | --- |
| `flat` | uniform region |
| `h_edge` | horizontal edge |
| `v_edge` | vertical edge |
| `diagonal` | diagonal gradient |
| `checkerboard` | high-frequency texture |
| `gradient` | smooth ramp |
| `periodic` | periodic texture |
| `noise_burst` | sparse high-energy patch |

The encoded patch form is:

$$
p_i \rightarrow (m_i, r_i),
$$

where \(p_i\) is a patch, \(m_i\) is the nearest semantic patch motif, and \(r_i\) is a scalar residual.

TPU-side transport remains:

$$
z_i = E(m_i).
$$

The residual and patch position are carried as compact feature geometry:

$$
L_5 = [r_i, \mathrm{pos}_i].
$$

Decoder metrics:

- image patch motif recovery
- ridge pixel PSNR
- motif-catalog-plus-residual PSNR
- residual MAE
- patch luminance MAE
- per-patch true/predicted motif readout

This is not yet video reconstruction. It is the smallest test of:

```text
patch -> semantic motif + residual -> transport -> recovered features -> reconstructed pixels
```

## AQ-IMG-0 First Result

AQ-IMG-0 completed on `2026-05-30`.

TRC-safe handling:

- result was read from TPU stdout only
- no result artifact, checkpoint, array, directory, or summary JSON was downloaded

Run shape:

- node: `program-aqimg-eu-node-v1`
- zone: `europe-west4-a`
- mode: `--aq-img-0`
- noise: `0.00`
- seed: `11`
- controller: `free`
- sequence length: `16`
- interval nodes: `136`

Core result:

| metric | value |
| --- | ---: |
| worst-control margin | `+0.3213` |
| folded score | `0.5022` |
| noisy graph cosine | `0.7429` |
| block lift | `+0.1818` |
| topology lift | `+0.3737` |
| untransported sanity | `0.9082` |
| L1 occupation recovery | `0.9789` |
| L2 transition recovery | `0.9454` |
| L3 FFT/phase recovery | `0.8269` |
| L4 recurrence recovery | `0.9588` |
| L5 residual/position recovery | `0.6592` |
| leakage proxy | `0.0789` |
| motif recovery | `1.0000` |
| target recovery | `1.0000` |
| class recovery | `1.0000` |
| pixel PSNR | `16.8927 dB` |
| catalog PSNR | `35.9693 dB` |
| residual MAE | `0.0150` |
| luma MAE | `0.0252` |

Patch readout:

| row | true motif | predicted motif | target luma | predicted luma |
| ---: | --- | --- | ---: | ---: |
| 0 | `flat` | `flat` | `0.0000` | `0.0050` |
| 1 | `periodic` | `periodic` | `0.3733` | `0.4217` |
| 2 | `gradient` | `gradient` | `0.5005` | `0.4604` |
| 3 | `flat` | `flat` | `0.0000` | `0.0050` |
| 4 | `checkerboard` | `checkerboard` | `0.5200` | `0.5092` |
| 5 | `gradient` | `gradient` | `0.4854` | `0.4604` |
| 6 | `v_edge` | `v_edge` | `0.5050` | `0.4774` |
| 7 | `h_edge` | `h_edge` | `0.5150` | `0.5150` |
| 8 | `h_edge` | `h_edge` | `0.5100` | `0.5150` |
| 9 | `periodic` | `periodic` | `0.3027` | `0.4217` |
| 10 | `diagonal` | `diagonal` | `0.5884` | `0.5644` |
| 11 | `noise_burst` | `noise_burst` | `0.4656` | `0.4546` |
| 12 | `diagonal` | `diagonal` | `0.5955` | `0.5644` |
| 13 | `v_edge` | `v_edge` | `0.5150` | `0.4774` |
| 14 | `checkerboard` | `checkerboard` | `0.5100` | `0.5092` |
| 15 | `noise_burst` | `noise_burst` | `0.4412` | `0.4546` |

Interpretation:

- the patch semantic motif channel recovered perfectly at `16/16`
- the image-domain vocabulary is compatible with the folded semantic transport pipeline
- layer recovery remained high enough to show that the transported state retains structured semantic content, not merely a class label
- reconstruction quality is currently decoder/encoding limited: `pixel_psnr = 16.8927 dB` is much weaker than the perfect motif recovery
- the high `catalog_psnr = 35.9693 dB` and low residual/luma errors suggest the semantic patch dictionary is useful, but the current scalar residual is too low-dimensional for fine patch reconstruction

The key result is therefore:

$$
\Pr[\hat{m}_i=m_i] = 1.0000
$$

while:

$$
\mathrm{PSNR}_{pixel} = 16.8927\ \mathrm{dB}.
$$

That gap should be interpreted as an encoding/decoder obstacle:

$$
p_i \mapsto (m_i,r_i)
$$

recovers the semantic class \(m_i\), but a single residual \(r_i\) does not preserve enough within-class detail:

$$
I(\tilde{z};m_i) \gg I(\tilde{z};p_i \mid m_i).
$$

Recommended next AQ-IMG step:

- keep the same tiny image grid
- keep noise at `0.00`
- do not move to video or larger frames yet
- replace scalar residual with a compact residual vector for mean, contrast, phase/alignment, and a low-frequency DCT-like coefficient
- measure whether pixel PSNR improves while motif recovery remains `1.0000`

Standalone run record:

- `emergent_quantum_geometries/docs/PROGRAM_AQ_IMG0_REMOTE_ANALYSIS_2026-05-30.md`
