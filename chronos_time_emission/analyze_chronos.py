#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from chronos_time_emission.conditions import analyze_condition, condition_spec, write_spectral_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CHRONOS latency CSV locally.")
    parser.add_argument("--csv", required=True, help="Path to latencies.csv")
    parser.add_argument("--out", default=None, help="Output directory; defaults next to CSV")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out) if args.out else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", newline="") as fh:
        rows = list(csv.DictReader(fh))

    results = {}
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = []
        for row in rows:
            if row["condition"] != condition:
                continue
            parsed = dict(row)
            for key in ("block", "trial", "trial_global", "label", "actual_label", "latency_ns", "timestamp_start_ns", "timestamp_end_ns"):
                parsed[key] = int(parsed[key])
            condition_rows.append(parsed)
        results[condition] = analyze_condition(condition, condition_spec(condition), condition_rows, seed=args.seed)

    write_spectral_csv(out_dir / "spectral_features.csv", results)
    compact = {name: {k: v for k, v in result.items() if k not in {"records", "spectral_features"}} for name, result in results.items()}
    (out_dir / "analysis.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'analysis.json'}")
    print(f"Wrote {out_dir / 'spectral_features.csv'}")


if __name__ == "__main__":
    main()

