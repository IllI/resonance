from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from time_dilation_dlinoss_learner import (
    STREAM_GRANULARITY_CHOICES,
    TimeMomentStreamingService,
    load_packet_records,
    make_stream_key,
    stream_label,
)
from time_dilation_stream_dataset import is_time_stream_dataset_path, iter_time_stream_dataset_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay packet records into ordered time moments and summarize concurrent now-stream fan-out.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a stream-native dataset directory or legacy packet .npy file.")
    parser.add_argument("--output-dir", type=Path, default=Path("python/resonance_results/time_dilation_stream_replay"))
    parser.add_argument("--bin-width", type=float, default=0.01)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-moments", type=int, default=0)
    parser.add_argument("--stream-granularity", choices=STREAM_GRANULARITY_CHOICES, default="stream_id")
    parser.add_argument("--replay-scale", type=float, default=0.0, help="0 disables sleeping; 1.0 is real-time; values >1 accelerate replay.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if is_time_stream_dataset_path(args.input):
        records_source = iter_time_stream_dataset_records(args.input, max_records=args.max_records)
        source_mode = "stream_dataset"
    else:
        records = load_packet_records(args.input)
        if args.max_records > 0:
            records = records[:args.max_records]
        records_source = records
        source_mode = "raw_records"
    service = TimeMomentStreamingService(records_source, bin_width=args.bin_width, source_mode=source_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    moments: list[dict[str, object]] = []
    stream_presence = Counter()
    total_records = 0
    previous_mid: float | None = None
    for moment_index, moment in enumerate(service.iter_moments()):
        if args.max_moments > 0 and moment_index >= args.max_moments:
            break
        current_mid = float(moment["host_time_mid"])
        if args.replay_scale > 0 and previous_mid is not None:
            delay = max(current_mid - previous_mid, 0.0) / max(float(args.replay_scale), 1e-6)
            if delay > 0:
                time.sleep(min(delay, 2.0))
        previous_mid = current_mid
        counts: dict[str, int] = {}
        for record in moment["records"]:
            label = stream_label(make_stream_key(record, args.stream_granularity))
            counts[label] = counts.get(label, 0) + 1
        for label in counts:
            stream_presence[label] += 1
        total_records += int(len(moment["records"]))
        moments.append({
            "moment_index": int(moment_index),
            "time_bin": int(moment["time_bin"]),
            "host_time_mid": current_mid,
            "record_count": int(len(moment["records"])),
            "active_stream_count": int(len(counts)),
            "stream_packet_counts": counts,
        })
    summary = {
        "input_path": str(args.input),
        "source_mode": source_mode,
        "stream_granularity": args.stream_granularity,
        "bin_width": float(args.bin_width),
        "moment_count": int(len(moments)),
        "record_count": int(total_records),
        "unique_stream_count": int(len(stream_presence)),
        "max_active_stream_count": int(max((moment["active_stream_count"] for moment in moments), default=0)),
        "stream_presence_by_moment": dict(sorted(stream_presence.items())),
        "first_host_time_mid": None if not moments else float(moments[0]["host_time_mid"]),
        "last_host_time_mid": None if not moments else float(moments[-1]["host_time_mid"]),
        "moment_trace_path": str(args.output_dir / "moment_replay_trace.json"),
    }
    (args.output_dir / "moment_replay_trace.json").write_text(json.dumps(moments, indent=2), encoding="utf-8")
    (args.output_dir / "moment_replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary_path": str(args.output_dir / "moment_replay_summary.json"),
        "source_mode": source_mode,
        "stream_granularity": args.stream_granularity,
        "moment_count": int(len(moments)),
        "record_count": int(total_records),
        "unique_stream_count": summary["unique_stream_count"],
        "max_active_stream_count": summary["max_active_stream_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())