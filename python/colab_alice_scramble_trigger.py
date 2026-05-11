"""
ALICE COLAB SESSION — Scrambling-Triggered Injection
=====================================================
Paste this entire script into a single Colab cell and run it.
Requires: numpy (pre-installed), google-colab auth

Steps:
1. Set SHARED_GCS_PATH to a GCS bucket you can write to from both accounts
   (e.g. a public bucket, or sign in both accounts to the same project)
   OR set USE_GCS=False to download the JSON at the end manually.
2. Run in your first Gmail Colab session.
3. Start Bob's notebook simultaneously in your second Gmail Colab session.
"""

import time
import json
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────
DURATION_SECONDS  = 2200        # ~36 min (leave buffer for setup)
MATRIX_SIZE       = 512         # matrix multiply size
BURST_SCALE       = 6.0         # injection amplitude at scrambling event
BURST_DURATION_S  = 8.0         # seconds of burst per event
PHASE_WINDOW      = 64          # rolling window for phase detection
PHASE_THRESHOLD   = 2.8         # std/mean ratio to declare phase shift
LORENZ_DT         = 0.01
USE_GCS           = False       # set True if you have a shared GCS bucket
SHARED_GCS_PATH   = "gs://YOUR_BUCKET/scramble_experiment/"
SAVE_PATH         = "/tmp/alice_scramble_events.json"
# ───────────────────────────────────────────────────────────────────────────

def lorenz_step(x, y, z, dt=LORENZ_DT, sigma=10, rho=28, beta=8/3):
    return (
        x + sigma * (y - x) * dt,
        y + (x * (rho - z) - y) * dt,
        z + (x * y - beta * z) * dt,
    )

def is_phase_shift(window):
    if len(window) < PHASE_WINDOW:
        return False
    w = np.array(window[-PHASE_WINDOW:])
    denom = np.mean(np.abs(w)) + 1e-9
    return (np.std(w) / denom) > PHASE_THRESHOLD

# ── CAPTURE LOOP ────────────────────────────────────────────────────────────
events = []
timing_log = []
lx, ly, lz = 0.1, 0.0, 0.0
inj_scale = 1.0
window = []
t0 = time.time()
B_ref = np.random.randn(MATRIX_SIZE, MATRIX_SIZE)   # fixed right matrix
burst_end = 0.0
in_burst = False
n_packets = 0

print(f"Alice starting. Duration={DURATION_SECONDS}s  Matrix={MATRIX_SIZE}x{MATRIX_SIZE}")
print(f"Phase threshold={PHASE_THRESHOLD}  Burst scale={BURST_SCALE}x  Burst duration={BURST_DURATION_S}s")
print()

while time.time() - t0 < DURATION_SECONDS:
    now = time.time()

    # Exit burst if done
    if in_burst and now >= burst_end:
        in_burst = False
        inj_scale = 1.0
        window.clear()  # reset variance window after burst

    # Lorenz step
    lx, ly, lz = lorenz_step(lx, ly, lz)
    A_val = np.sqrt(lx**2 + ly**2 + lz**2) / 10.0

    # Matrix compute (timing measurement)
    t1 = time.perf_counter()
    A_mat = np.random.randn(MATRIX_SIZE, MATRIX_SIZE) * inj_scale * (1.0 + A_val)
    _ = A_mat @ B_ref
    t2 = time.perf_counter()
    dt_ms = (t2 - t1) * 1000.0

    window.append(dt_ms)
    n_packets += 1

    # Detect phase shift (scrambling moment)
    if not in_burst and is_phase_shift(window):
        event_t = time.time()
        elapsed = event_t - t0
        events.append({
            "t": event_t,
            "elapsed": elapsed,
            "lorenz_A": float(A_val),
            "dt_ms": dt_ms,
            "n_packet": n_packets,
        })
        # INJECT: burst at scrambling moment (U_AB fires)
        inj_scale = BURST_SCALE * (1.0 + A_val)
        burst_end = event_t + BURST_DURATION_S
        in_burst = True

        if len(events) % 5 == 0:
            print(f"  [{elapsed:6.1f}s] Event #{len(events):3d}  A={A_val:.3f}  dt={dt_ms:.2f}ms  inj={inj_scale:.2f}")

    time.sleep(0.005)   # ~200 Hz loop

# ── SUMMARY ──────────────────────────────────────────────────────────────
elapsed_total = time.time() - t0
print()
print(f"Alice done. Total time: {elapsed_total:.1f}s")
print(f"  Packets captured: {n_packets}")
print(f"  Scrambling events: {len(events)}")
if events:
    gaps = np.diff([e["elapsed"] for e in events])
    print(f"  Mean inter-event gap: {gaps.mean():.1f}s  (std={gaps.std():.1f}s)")

output = {
    "role": "alice",
    "t0": t0,
    "duration": elapsed_total,
    "n_packets": n_packets,
    "events": events,
    "config": {
        "matrix_size": MATRIX_SIZE,
        "burst_scale": BURST_SCALE,
        "burst_duration": BURST_DURATION_S,
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
                    SHARED_GCS_PATH + "alice_scramble_events.json"])
    print(f"Uploaded to {SHARED_GCS_PATH}")
else:
    # Download from Colab
    try:
        from google.colab import files
        files.download(SAVE_PATH)
        print("Download started in browser.")
    except Exception:
        print(f"Manual download: {SAVE_PATH}")
