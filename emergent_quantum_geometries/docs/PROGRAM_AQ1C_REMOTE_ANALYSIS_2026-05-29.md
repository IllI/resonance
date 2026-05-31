# Program AQ-1c Remote Analysis - 2026-05-29

## TRC-safe data handling

This record was created from final stdout tables inspected on the TPU VM. No result artifact, summary JSON, checkpoints, arrays, or remote directories were downloaded.

Run context:

- project: `time-emission`
- node: `program-aq1b-eu-node-v2`
- zone: `europe-west4-a`
- mode: `--aq-1c`
- backend: JAX TPU
- devices visible: `4`

## Configuration

- goal: operator identifiability gate before noisy transport
- sources: `4`
- operators: `4`
- source/operator combinations: `16`
- noise: `0.00`
- depth: `3`
- seed: `11`
- controller: `free`
- hard controls: disabled as experiment arms
- recovery states: one-hot source/operator pair states

Operators:

- `MIRROR_TIME`
- `ROTATE_PHASE`
- `DENSIFY_TRANSITIONS`
- `SPARSIFY_TRANSITIONS`

## Semantic recovery table

| noise | seed | controller | margin | score | noisy cosine | block lift | topology lift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `0.00` | `11` | `free` | `+0.2375` | `0.4427` | `0.6282` | `+0.1649` | `+0.1344` |

## Layer table

| noise | seed | controller | sanity | L1 | L2 | L3 | L4 | L5 | leak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | `11` | `free` | `0.7865` | `0.9504` | `0.7689` | `0.8963` | `0.8638` | `0.6242` | `0.0694` |

## Operator gate table

| noise | seed | controller | L5 | operator recovery | derived target accuracy | target leakage | source recovery | application error | operator swap | source swap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | `11` | `free` | `0.6242` | `0.4375` | `0.2500` | `0.1875` | `0.5625` | `0.2967` | `1.0000` | `1.0000` |

## Interpretation

AQ-1c failed the operator identifiability gate.

The useful part is that this failure happened at `noise=0.00` with `free` transport only. That means the next blocker is not TPU noise, hard controls, or tunnel-eigen alpha. The operator grammar itself is not yet recoverable enough.

Gate results:

- source recovery target: `>= 0.75`; observed `0.5625`
- operator recovery target: `>= 0.75`; observed `0.4375`
- derived target target: `>= 0.75`; observed `0.2500`
- target leakage stayed low at `0.1875`, which is good but not enough to rescue the composition result

Layer readout:

- `L1` remained strong at `0.9504`
- `L3` and `L4` were usable at `0.8963` and `0.8638`
- `L2` was only moderate at `0.7689`
- `L5` was weak at `0.6242`

The current source/operator payload is therefore still under-identified. Bob is not reliably recovering the source or operator, so derived target accuracy is not yet interpretable as a transport result.

## Decision

Do not run AQ-1d/noisy transport yet.

Next implementation step should repair the operator grammar in the noise-free setting:

- strengthen source separability before operator recovery
- make operator codes more orthogonal in `L5`
- add explicit source and operator confusion tables to stdout
- keep AQ-1c small and noise-free until source and operator recovery both pass `0.75`

Only after AQ-1c passes should AQ return to `noise=0.30` with `free`, `alpha055`, and `alpha065`.
