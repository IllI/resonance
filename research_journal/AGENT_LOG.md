# Running Agent Log

## 2026-06-21 — archive initialized

### Objective

Reconstruct the experimental journey of `quantum-teleportation-results`, map commits to artifacts and documentation, group runs into causal experiment families, and triage scientifically interesting results without laundering contemporaneous claims into facts.

### Repository state at start

- Branch: `quantum-teleportation-results` tracking `origin/quantum-teleportation-results`.
- Branch history: 218 commits; 185 in May 2026 and 23 in June 2026.
- Branch tip inventory: 744 tracked files, including 94 Markdown files.
- Working tree: heavily modified/untracked, including June Chronos, Page-Wootters, temporal-gauge, TRAPPIST/JWST, MPS, and Program AK–AR artifacts.
- Safety rule: do not clean, reset, stage, or modify pre-existing experiment files.

### Architecture decisions

- Use experiment families as the human unit; retain commits and runs as ordered evidence.
- Keep committed, documentary, and live evidence layers distinct.
- Generate mechanical inventories reproducibly; reserve interpretation for reviewed narrative records.
- Grade evidence independently of novelty or positive/negative outcome.

### Parallel archaeology assignments

- `019eeb32-f01f-7763-8e90-eb0077d4de63` (Socrates): 2026-05-08 through 2026-05-12.
- `019eeb32-c77f-73c3-b83f-328fb17be1e9` (Kant): 2026-05-13 through 2026-05-18.
- `019eeb33-0407-7100-a8af-8334260e7f51` (Franklin): 2026-05-19 through 2026-05-31.
- `019eeb32-dc43-7dc0-828a-45f01e5ba70d` (Herschel): June committed history plus live working-tree families.

### Next work queue

1. Audit top-ranked candidates against raw result files and controls, beginning with IBM Txx, Program O, Program F, representation fronts, SO(3), and the MPS run-1 claim.
2. Add coverage checks: every experiment-like commit and result artifact must map to a family or an explicit `unclassified` queue.
3. Split large live families into run-level records with seeds/configs/checksums.
4. Locate any later Program P result and resolve the G/G2 0.719-versus-0.770 conflict.
5. Preserve raw evidence for AE, AK–AN, AO/AP, and production AQ; explicitly retain missing/empty status where recovery is impossible.

### Completed archaeology report

- May 8–12 completed by Socrates (`019eeb32-f01f-7763-8e90-eb0077d4de63`) and distilled to `reports/2026-05-08_12_FOUNDATION_TO_IBM.md`. The key correction chain runs from an invalid OAT fidelity shortcut to an IBM validation of one PTM witness component, not teleportation.
- May 13–18 completed by Kant (`019eeb32-c77f-73c3-b83f-328fb17be1e9`) and distilled to `reports/2026-05-13_18_OPERATOR_TRANSPORT_TO_PARTIAL_INFORMATION.md`. It separates scientific families from a long launcher-debug lineage and records Program O's 57/60 subject versus 51/60 artifact discrepancy.
- May 19–31 completed by Franklin (`019eeb33-0407-7100-a8af-8334260e7f51`) and distilled to `reports/2026-05-19_31_RECOVERY_TO_SEMANTIC_TRANSPORT.md`. Key record-integrity flags include inactive and saturated controllers, an empty AE summary, AL's broken-looking recurrence metric, and AQ's payload bug.
- June committed history plus live working-tree frontier completed by Herschel (`019eeb32-dc43-7dc0-828a-45f01e5ba70d`) and distilled to `reports/2026-06_JUNE_AND_LIVE.md`.
- The audit identifies temporal-gauge 0c, Page–Wootters June 17 extensions, MPS run 1, real-JWST closeouts, DREAMS cadence claims, and all GLIMPSE physical interpretations as explicit validation targets.

### Handoff protocol

The next agent should first read `README.md`, `SCHEMA.md`, this file, and `TIMELINE.md`; then rebuild the generated inventory. Continue from the first unchecked item in the next work queue. Append dated entries here, including exact commands, decisions, unresolved ambiguity, and files changed. Never silently upgrade a claim's evidence grade.
