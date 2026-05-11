"""
BOB COLAB SESSION — Passive Phase-Shift Capture
================================================
Paste this into a Colab cell and run simultaneously with Alice's session.
Bob does NO injection — he just measures his own timing jitter and
detects phase-shift events (his own scrambling moments).

Post-hoc analysis: correlate Bob's event timestamps with Alice's.
"""

import time
import json
import numpy as np

# ── CONFIG (must match Alice's PHASE_WINDOW and PHASE_THRESHOLD) ──────────
DURATION_SECONDS  = 2200
MATRIX_SIZE       = 512
PHASE_WINDOW      = 64
PHASE_THRESHOLD   = 2.8
USE_GCS           = False
SHARED_GCS_PATH   = "gs://YOUR_BUCKET/scramble_experiment/"
SAVE_PATH         = "/tmp/bob_passive_events.json"
# ───────────────────────────────────────────────────────────────────────────

def is_phase_shift(window):
    if len(window) < PHASE_WINDOW:
        return False
    w = np.array(window[-PHASE_WINDOW:])
    denom = np.mean(np.abs(w)) + 1e-9
    return (np.std(w) / denom) > PHASE_THRESHOLD

# ── CAPTURE LOOP ────────────────────────────────────────────────────────────
events = []
window = []
t0 = time.time()
B_ref = np.random.randn(MATRIX_SIZE, MATRIX_SIZE)
n_packets = 0

print(f"Bob starting. Duration={DURATION_SECONDS}s  Matrix={MATRIX_SIZE}x{MATRIX_SIZE}")
print(f"Phase threshold={PHASE_THRESHOLD}  No injection — passive capture only")
print()

while time.time() - t0 < DURATION_SECONDS:
    t1 = time.perf_counter()
    A_mat = np.random.randn(MATRIX_SIZE, MATRIX_SIZE)
    _ = A_mat @ B_ref
    t2 = time.perf_counter()
    dt_ms = (t2 - t1) * 1000.0

    window.append(dt_ms)
    n_packets += 1

    if is_phase_shift(window):
        event_t = time.time()
        elapsed = event_t - t0
        events.append({
            "t": event_t,
            "elapsed": elapsed,
            "dt_ms": dt_ms,
            "n_packet": n_packets,
        })
        window.clear()  # reset after event

        if len(events) % 5 == 0:
            print(f"  [{elapsed:6.1f}s] Bob event #{len(events):3d}  dt={dt_ms:.2f}ms")

    time.sleep(0.005)

# ── SUMMARY ──────────────────────────────────────────────────────────────
elapsed_total = time.time() - t0
print()
print(f"Bob done. Total time: {elapsed_total:.1f}s")
print(f"  Packets captured: {n_packets}")
print(f"  Phase-shift events: {len(events)}")
if events:
    gaps = np.diff([e["elapsed"] for e in events])
    print(f"  Mean inter-event gap: {gaps.mean():.1f}s  (std={gaps.std():.1f}s)")

output = {
    "role": "bob",
    "t0": t0,
    "duration": elapsed_total,
    "n_packets": n_packets,
    "events": events,
    "config": {
        "matrix_size": MATRIX_SIZE,
        "phase_window": PHASE_WINDOW,
        "phase_threshold": PHASE_THRESHOLD,
    },
}

with open(SAVE_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved: {SAVE_PATH}")

if USE_GCS:
    import subprocess
    subprocess.run(["gcloud", "storage", "cp", SAVE_PATH,
                    SHARED_GCS_PATH + "bob_passive_events.json"])
else:
    try:
        from google.colab import files
        files.download(SAVE_PATH)
        print("Download started in browser.")
    except Exception:
        print(f"Manual download: {SAVE_PATH}")
