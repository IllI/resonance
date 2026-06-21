# Classification and Evidence Schema

## Unit of analysis

The primary unit is an **experiment family**, not a commit. A family may contain a plan, implementation, smoke run, production run, correction, analysis, and closeout across several commits. A commit may touch several families.

Stable family IDs should use `YYYY-MM-DD_slug`, based on the earliest evidenced start date. Individual runs retain their native names and seeds.

## Required family record

Each family should record:

| Field | Meaning |
|---|---|
| `family_id` | Stable archive identifier |
| `aliases` | Program letters, script stems, informal names |
| `date_start`, `date_end` | Evidence-backed interval |
| `question` | Scientific or engineering question |
| `trigger` | Prior result, failure, conjecture, or external motivation |
| `method` | Model, tensor-network form, data, hardware, controls |
| `run_sequence` | Ordered smoke/production/correction/replication runs |
| `inputs` | Data, checkpoints, configs, and upstream families |
| `outputs` | Result files, figures, logs, and documents |
| `claims` | What contemporaneous records say occurred |
| `assessment` | What the available artifacts justify |
| `downstream` | Later families that consumed the result or method |
| `status` | Lifecycle status below |
| `evidence_grade` | Evidence grade below |
| `interest_score` | Triage score, never a publication verdict |
| `open_questions` | Missing controls, provenance, or interpretation |

## Lifecycle status

- `planned`: specification exists, no run evidence located.
- `implemented`: executable/config exists, no completed result located.
- `smoke_only`: local or reduced test only.
- `partial`: production run incomplete or only some gates evaluated.
- `complete_unverified`: completed artifact exists but has not been audited.
- `supported`: claim survives an artifact/method/control audit.
- `null`: valid run did not support the target effect.
- `falsified`: a stated conjecture failed a designed test.
- `superseded`: later correction invalidated or replaced the interpretation.
- `infrastructure`: operational work, not itself scientific evidence.
- `indeterminate`: evidence is missing, contradictory, or non-comparable.

## Evidence grade

- `E0 claim-only`: commit subject or prose claim without inspectable output.
- `E1 artifact-present`: output exists, provenance or completeness unclear.
- `E2 reproducible-run`: code/config/output and environment are sufficiently linked.
- `E3 controlled`: baselines, nulls, ablations, or positive controls are present.
- `E4 replicated`: independent seed/device/data/time replication is present.
- `E5 publication-audited`: statistics, leakage, multiple comparisons, provenance, and domain assumptions have been reviewed.

Negative results can receive high evidence grades. Exciting language cannot.

## Role in program lineage

Assign one or more roles:

- `foundation`: introduced a model, representation, dataset, or observable.
- `capability`: demonstrated an implementation or scaling capability.
- `diagnostic`: isolated a failure mode or measurement limitation.
- `control`: baseline, null, positive control, ablation, or adversarial test.
- `bridge`: connected two previously separate program strands.
- `pivot`: caused a material change in direction.
- `replication`: repeated an earlier result under changed conditions.
- `closeout`: bounded claims and documented why a path stopped.
- `infrastructure`: TPU provisioning, JAX compilation, checkpointing, transfer, or orchestration.

## Interest triage

Score each axis 0–3 and retain the component scores:

- `novelty`: known demonstration → potentially new phenomenon/method.
- `evidence`: maps approximately to E0–E5, capped at 3.
- `scientific_relevance`: internal engineering → externally meaningful question.
- `surprise`: expected confirmation → robust anomaly or falsification.
- `reusability`: one-off artifact → general method/dataset/benchmark.

Apply penalties of 0–3 each for leakage risk, missing controls, post-hoc selection, irreproducible provenance, and mismatch between metric and scientific claim. Use the total only to prioritize audits. Do not label a “diamond” until an E4/E5 review.

## Provenance rules

1. Cite commit hashes and repository-relative paths for every historical statement.
2. Mark working-tree evidence as `LIVE` with filesystem modification time; never backdate it from a filename.
3. Preserve corrections in sequence. Do not overwrite the earlier belief with the later one.
4. Separate computational success (TPU/JAX run completed) from scientific success (hypothesis supported).
5. Treat filenames such as `final`, `certified`, or `positive_control` as labels, not conclusions.
