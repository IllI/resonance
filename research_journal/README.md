# TRC TPU Experimental Research Journal

This directory is the durable map of the `quantum-teleportation-results` research program. It is designed to answer five questions without relying on memory:

1. What was attempted, and when?
2. Why did that experiment follow the previous one?
3. What evidence was actually produced?
4. What failed, was corrected, or remained unresolved?
5. Which results deserve deeper scientific review?

## Evidence boundaries

The archive keeps three evidence layers separate:

- **Committed history**: immutable commit/file relationships on the named branch.
- **Documented interpretation**: claims in plans, findings, closeouts, and commit messages.
- **Live working tree**: modified or untracked artifacts that may postdate the last commit.

A claim is not treated as a result merely because it appears in a commit subject or Markdown document. Promotion requires a result artifact, enough methodological detail to reproduce it, and appropriate controls.

## Navigation

- `SCHEMA.md` — classification, evidence, lineage, and significance rules.
- `TIMELINE.md` — human-readable chronological synthesis (built iteratively).
- `FAMILY_CATALOG.csv` — family-level status, evidence, lineage, and audit priority.
- `TRIAGE.md` — candidate discoveries, negative results, and plausible paper clusters.
- `AGENT_LOG.md` — running state and handoff instructions.
- `scripts/build_inventory.py` — reproducible Git and working-tree inventory.
- `generated/commits.csv` — one row per commit, chronological.
- `generated/commit_files.csv` — commit-to-file change map.
- `generated/markdown_documents.csv` — tracked Markdown provenance and headings.
- `generated/working_tree_artifacts.csv` — modified/untracked live evidence.

## Rebuild

From the repository root:

```powershell
python research_journal/scripts/build_inventory.py --branch quantum-teleportation-results
```

The generated files are indexes, not scientific conclusions. Human/agent synthesis belongs in `TIMELINE.md` and future experiment-family records.
