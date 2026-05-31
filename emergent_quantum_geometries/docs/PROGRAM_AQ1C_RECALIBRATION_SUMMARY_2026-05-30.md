# Program AQ-1c Recalibration Summary - 2026-05-30

## TRC-safe handling

This summary was written from remote stdout tables inspected over SSH. No TPU result artifacts, arrays, checkpoints, directories, or summary JSON files were downloaded.

The live TPU VM `program-aq1c-eu-node-v6` in `europe-west4-a` was deleted after the final AQ-1c probe because the experiment branch was no longer improving the operator-recovery objective.

## Objective

AQ-1c was intended to test whether a compact semantic payload can carry a reusable operator before returning to noisy transport.

The target compositional claim was:

$$
z = E(s, T),
$$

where:

- \(s\) is the source semantic motif,
- \(T\) is a reusable semantic operator,
- \(E\) is the folded semantic encoder,
- \(z\) is the transported compressed state.

Bob should recover:

$$
\hat{s}, \hat{T} = D(z)
$$

and then derive:

$$
\hat{t} = \hat{T}(\hat{s}).
$$

The important metric is therefore not direct target recovery, but derived target recovery:

$$
\mathrm{Acc}_{derived}
= \Pr\left[\hat{T}(\hat{s}) = t\right].
$$

AQ-1c should pass only if:

$$
\mathrm{Acc}_{source} \ge 0.75,
\quad
\mathrm{Acc}_{operator} \ge 0.75,
\quad
\mathrm{Acc}_{derived} \ge 0.75.
$$

## Methodology

AQ-1c used a small factorial operator-identifiability gate:

- sources: \(4\)
- operators: \(4\)
- combinations: \(4 \times 4 = 16\)
- noise: \(0.00\)
- depth: \(3\)
- controller: `free`
- seed: `11`

The semantic layers were:

$$
\phi(s) =
\left[
L_1(s),
L_2(s),
L_3(s),
L_4(s)
\right],
$$

where:

- \(L_1\): occupation statistics
- \(L_2\): transition graph
- \(L_3\): FFT / phase signature
- \(L_4\): recurrence topology

The operator layer was intended to encode an action over the semantic layers:

$$
L_5(T)
\approx
\mathrm{compress}
\left(
\Delta L_1,
\Delta L_2,
\Delta L_3,
\Delta L_4
\right),
$$

with:

$$
\Delta L_k = L_k(T(s)) - L_k(s).
$$

The ideal behavior is that \(L_5\) identifies the operator \(T\), while \(L_1\)-\(L_4\) preserve the source information needed to apply it.

## Results

### AQ-1c baseline

| metric | value |
| --- | ---: |
| source recovery | `0.5625` |
| operator recovery | `0.4375` |
| derived target accuracy | `0.2500` |
| target leakage | `0.1875` |
| L5 recovery | `0.6242` |

Interpretation:

- source was weak
- operator was weak
- derived target was chance-level
- leakage stayed low
- \(L_5\) was measurable but not operator-identifiable

### Source-marker repair

The next patch separated source and operator positions more clearly and gave source identity an explicit marker channel.

| metric | value |
| --- | ---: |
| source recovery | `0.6875` |
| operator recovery | `0.6875` |
| derived target accuracy | `0.5625` |
| target leakage | `0.1875` |
| L5 recovery | `0.7290` |

Interpretation:

- source recovery improved
- operator recovery improved
- derived target improved
- leakage stayed controlled
- still did not pass the \(0.75\) gate

This was the best balanced AQ-1c result.

### Operator-suffix strengthening

The final patch kept the explicit source marker but strengthened the operator suffix/checksum.

| metric | value |
| --- | ---: |
| source recovery | `0.3750` |
| operator recovery | `0.3750` |
| derived target accuracy | `0.2500` |
| target leakage | `0.1875` |
| L5 recovery | `0.6040` |

Interpretation:

- the run regressed sharply
- both source and operator recovery collapsed
- derived target returned to chance
- leakage stayed low, but only because the representation stopped carrying enough usable structure

## Why Recovery Got Worse

The failure is not transport noise. This was a `noise=0.00`, `free` controller run. The degradation came from an architectural bug in the AQ-1c payload construction.

The root cause was that AQ-1c stopped transporting source motifs.

The broken payload was:

```python
payload_seqs.append(
    jnp.concatenate([source_marker_full[:split], marker_full[split:]], axis=0)
)
```

That means the transported object was not:

$$
z = E(s, T),
$$

but closer to:

$$
z = E(m_s, m_T),
$$

where \(m_s\) and \(m_T\) are hand-crafted source/operator markers. The actual semantic sequence \(s\) was not present in `payload_seqs`.

This matters because `evaluate_semantic_recovery` builds the interval entropy table from the sequences it receives:

```python
clean_entropy = motif_interval_entropy_table(seqs, nodes)
```

In AQ-1c, `seqs` had become marker sequences. Therefore D-LinOSS was processing marker interval statistics while the decoder was being asked to recover \(L_1\)-\(L_4\) source features computed from different motif sequences.

The model had no legitimate semantic source signal to transport.

The earlier AQ-1a-TF payload did include real motif content:

```python
left = seqs[src_idx, :halfway]
right = seqs[tgt_idx, halfway:]
pair_seqs = jnp.concatenate([left, right], axis=1)
```

That explains why AQ-1a-TF showed stronger signal: D-LinOSS saw real source/target motif structure. But because the target half was included directly, the result could be endpoint association rather than reusable operator recovery.

AQ-1c attempted to remove that endpoint leak, but overcorrected by replacing the whole transported semantic sequence with markers. That plugged the target leak and accidentally removed the source channel too.

The partial AQ-1c recoveries were therefore likely ridge/decoder correlations over a small \(4 \times 4\) design, not robust semantic transport.

The current AQ-1c design asks the recovery system to learn three things at once:

$$
z \mapsto \hat{s},
\quad
z \mapsto \hat{T},
\quad
(\hat{s}, \hat{T}) \mapsto \hat{t}.
$$

But the compact payload is only length `8`, split between a source channel and an operator channel. When the operator marker was made stronger, it did not simply make \(T\) easier to decode. It changed the geometry of the whole folded payload.

In effect, the representation was forced to optimize marker separability rather than semantic compositionality:

$$
\max I(z; m_T)
\not\Rightarrow
\max I(z; T \mid s),
$$

where \(m_T\) is the operator marker. A stronger marker can dominate the compressed state while still failing to define a reusable action. More importantly, markers are not a substitute for transporting \(s\).

The source-marker repair helped only because it provided a more stable proxy for source identity:

$$
I(z; s) \uparrow
\quad\Rightarrow\quad
\mathrm{Acc}_{source} \uparrow.
$$

But this was not true source semantic recovery, because \(s\) itself was still absent from the payload. The operator-suffix strengthening shifted the marker geometry again:

$$
I(z; m_T) \uparrow
\quad\text{while}\quad
I(z; s) \downarrow
\quad\text{and}\quad
I(z; T \mid s) \downarrow.
$$

That explains the observed collapse:

- source recovery fell from `0.6875` to `0.3750`
- operator recovery fell from `0.6875` to `0.3750`
- derived target fell from `0.5625` to `0.2500`

This suggests the model was not learning a stable semantic operator. It was learning fragile correlations between source markers, operator markers, and endpoint labels. When those marker correlations changed, the apparent recovery degraded instead of becoming more robust.

The most concerning pattern is:

$$
L_5 \text{ measurable}
\quad\text{but}\quad
\mathrm{Acc}_{operator} \text{ weak}.
$$

That means the \(L_5\) channel contains transformation-like signal, but not a reusable operator basis. In plainer terms: the run can sense that "something changed," but cannot reliably recover the rule that caused the change.

## Diagnosis

AQ-1c is currently invalid as an operator-identifiability test because its transported payload did not contain the source motif.

The run is likely training against the wrong surrogate target:

- marker transport instead of semantic source transport
- marker recovery instead of operator recovery
- endpoint association instead of compositional action
- low-dimensional leakage suppression instead of semantic sufficiency
- pair-state classification instead of reusable \(T(s)\) application

The balanced source-marker run showed that the decoder can exploit structured proxies, but not that a semantic source/operator payload is valid.

## Code Fix

AQ-1c should transport the actual source motif:

```python
payload_seqs.append(seqs[src_idx])
```

The operator should be encoded outside the transported sequence, in the feature/action side:

```python
pair_features = [source_L1_L4_features, operator_code]
L5_transform = operator_code
```

The corrected conceptual payload is:

$$
z = E(s; C_T),
$$

where \(s\) is the real motif sequence and \(C_T\) is an operator action code available to the recovery geometry/features, not a replacement for \(s\).

Before any future TPU run, AQ-1c must print:

```python
print("payload_seqs sample:", payload_seqs[0])
print("source motif sample:", seqs[source_idx_list[0]])
print("payload/source match:", payload_seqs[0] == seqs[source_idx_list[0]])
```

If this check fails, AQ-1c is not transporting semantic content.

## Decision

Stop the current AQ-1c branch.

Do not proceed to noisy transport, alpha tuning, hard controls, or larger seed sweeps.

The next design must separate three concepts more cleanly:

1. source identity
2. operator identity
3. operator action on semantic layers

The next AQ-1c should prove that the operator is identifiable as an action map before testing transport robustness.

## Recalibration Plan

Use the best balanced result as the last useful baseline:

| metric | best balanced value |
| --- | ---: |
| source recovery | `0.6875` |
| operator recovery | `0.6875` |
| derived target accuracy | `0.5625` |
| target leakage | `0.1875` |
| L5 recovery | `0.7290` |

Next implementation should avoid stronger ad hoc markers and instead define explicit layer-action operators:

$$
T =
\left[
T_{L_1},
T_{L_2},
T_{L_3},
T_{L_4}
\right].
$$

Then:

$$
L_k(t)
\approx
T_{L_k}(L_k(s))
\quad
\forall k \in \{1,2,3,4\}.
$$

The next operator code should be a low-rank action map:

$$
C_T = U_T V_T^\top,
$$

not a marker-only or endpoint-delta code.

The next pass should report:

- source confusion matrix
- operator confusion matrix
- composition table
- operator invariance score
- direct target leakage with the operator channel masked
- derived target accuracy from \(\hat{s}\) and \(\hat{T}\)

Pass criteria remain:

$$
\mathrm{Acc}_{source} \ge 0.75,
\quad
\mathrm{Acc}_{operator} \ge 0.75,
\quad
\mathrm{Acc}_{derived} \ge 0.75,
\quad
\mathrm{Leak}_{target} \approx 0.25.
$$

If this does not pass at `noise=0.00`, AQ should stay in grammar repair and should not consume TPU time on noisy transport.
