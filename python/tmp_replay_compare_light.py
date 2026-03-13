from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from time_dilation_dlinoss_learner import TimeMomentStreamingService, make_stream_key, stream_label
from time_dilation_stream_dataset import iter_time_stream_dataset_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=0.01)
    parser.add_argument("--stream-granularity", default="stream_id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = TimeMomentStreamingService(
        iter_time_stream_dataset_records(args.input),
        bin_width=args.bin_width,
        source_mode="stream_dataset",
    )
    stream_presence: Counter[str] = Counter()
    active_hist: Counter[int] = Counter()
    moment_count = 0
    record_count = 0
    first_mid = None
    last_mid = None
    for moment in service.iter_moments():
        host_mid = float(moment["host_time_mid"])
        if first_mid is None:
            first_mid = host_mid
        last_mid = host_mid
        counts: dict[str, int] = {}
        for record in moment["records"]:
            label = stream_label(make_stream_key(record, args.stream_granularity))
            counts[label] = counts.get(label, 0) + 1
        active_hist[len(counts)] += 1
        for label in counts:
            stream_presence[label] += 1
        record_count += len(moment["records"])
        moment_count += 1
    summary = {
        "input_path": str(args.input),
        "source_mode": "stream_dataset",
        "stream_granularity": args.stream_granularity,
        "bin_width": float(args.bin_width),
        "moment_count": int(moment_count),
        "record_count": int(record_count),
        "unique_stream_count": int(len(stream_presence)),
        "max_active_stream_count": int(max(active_hist) if active_hist else 0),
        "stream_presence_by_moment": dict(sorted(stream_presence.items())),
        "active_stream_count_histogram": dict(sorted(active_hist.items())),
        "first_host_time_mid": first_mid,
        "last_host_time_mid": last_mid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

