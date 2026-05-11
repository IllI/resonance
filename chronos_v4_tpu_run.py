"""
chronos_v4_tpu_run.py
=====================
Standalone TPU VM script for Chronos v4 experiment.
Runs on a Google Cloud TPU VM (no Colab required).

Usage:
    python3 chronos_v4_tpu_run.py --role Alice_Scramble
    python3 chronos_v4_tpu_run.py --role Bob_Passive

Both roles run the same D-LinOSS learner on-device, consuming their own
timing jitter as the input signal. The learner's "now-state" predictions
are saved alongside the raw covariance data for cross-machine analysis.
"""
import argparse, json, pathlib, shutil, time, threading, sys
import numpy as np
from scipy import stats

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--role',        default='Bob_Passive',
                    choices=['Alice_Scramble', 'Bob_Passive'])
parser.add_argument('--duration',    type=float, default=900.0,   help='seconds')
parser.add_argument('--chunk-s',     type=float, default=5.0,     help='chunk duration')
parser.add_argument('--window',      type=int,   default=32,      help='rolling window packets')
parser.add_argument('--n-streams',   type=int,   default=4)
parser.add_argument('--sigma',       type=float, default=1.0)
parser.add_argument('--ramp-up',     type=float, default=120.0)
parser.add_argument('--on-s',        type=float, default=120.0)
parser.add_argument('--ramp-down',   type=float, default=60.0)
parser.add_argument('--off-s',       type=float, default=120.0)
parser.add_argument('--mat-size',    type=int,   default=256)
parser.add_argument('--target-hz',   type=float, default=10.0)
parser.add_argument('--dlinoss-dim', type=int,   default=16,      help='D-LinOSS state dim')
args = parser.parse_args()

ROLE         = args.role
N_STREAMS    = args.n_streams
CYCLE_S      = args.ramp_up + args.on_s + args.ramp_down + args.off_s
OUT_DIR      = pathlib.Path(f'chronos_v4_{ROLE.lower()}')
OUT_DIR.mkdir(exist_ok=True)

print(f'[Chronos v4 TPU] Role={ROLE}  Duration={args.duration}s  Target={args.target_hz}Hz')
print(f'Cycle={CYCLE_S}s  Chunk={args.chunk_s}s  Window={args.window}  DLinOSS-dim={args.dlinoss_dim}')

# ── JAX Setup ─────────────────────────────────────────────────────────────────
try:
    import jax, jax.numpy as jnp
    from jax.experimental import io_callback
    USE_JAX = True
    print(f'JAX {jax.__version__}  backend={jax.default_backend()}  devices={len(jax.devices())}')
    for d in jax.devices(): print(f'  {d}')
except Exception as e:
    USE_JAX = False
    print(f'JAX unavailable ({e}), numpy fallback.')

# ── Infrastructure Fingerprint ────────────────────────────────────────────────
import socket, urllib.request, json as _json
infra = {}
def _gcp(path):
    try:
        req = urllib.request.Request(
            f'http://metadata.google.internal/computeMetadata/v1/{path}',
            headers={'Metadata-Flavor': 'Google'})
        return urllib.request.urlopen(req, timeout=3).read().decode().strip()
    except: return None

zone_full = _gcp('instance/zone')
infra['gcp_zone']    = zone_full.split('/')[-1] if zone_full else None
infra['gcp_machine'] = (_gcp('instance/machine-type') or '').split('/')[-1] or None
infra['hostname']    = socket.gethostname()
try:
    geo = _json.loads(urllib.request.urlopen('https://ipinfo.io/json', timeout=5).read())
    infra.update({k: geo.get(k) for k in ['ip','city','region','country','timezone']})
except: pass
print(f'Fingerprint: zone={infra.get("gcp_zone")}  ip={infra.get("ip")}  tz={infra.get("timezone")}')

# ── Lambda schedule ───────────────────────────────────────────────────────────
def compute_lambda(elapsed):
    ph = elapsed % CYCLE_S
    if ph < args.ramp_up:   return 0.5*(1-np.cos(np.pi*ph/args.ramp_up))
    elif ph < args.ramp_up+args.on_s: return 1.0
    elif ph < args.ramp_up+args.on_s+args.ramp_down:
        p = (ph-args.ramp_up-args.on_s)/args.ramp_down
        return 0.5*(1+np.cos(np.pi*p))
    return 0.0

stream_mu      = (N_STREAMS-1)/2.0
STREAM_PROFILE = np.array([np.exp(-0.5*((k-stream_mu)/args.sigma)**2) for k in range(N_STREAMS)], dtype=np.float32)
STREAM_PROFILE /= STREAM_PROFILE.max()

# ── JAX Timed Kernel ──────────────────────────────────────────────────────────
_ts_lock  = threading.Lock()
_ts_store = {k: [] for k in range(N_STREAMS)}

def _record(k, t):
    with _ts_lock: _ts_store[int(k)].append(float(t))

_kern_cache = {}

if USE_JAX:
    def make_kern(k, size):
        @jax.jit
        def _kern(key):
            A = jax.random.normal(key, (size, size), dtype=jnp.float32)
            B = jax.random.normal(jax.random.fold_in(key,1), (size, size), dtype=jnp.float32)
            C = A @ B
            t = io_callback(lambda: np.float32(time.perf_counter()),
                            jax.ShapeDtypeStruct((), jnp.float32))
            io_callback(lambda t: _record(k, t), None, t)
            return C.mean()
        return _kern

    def get_kern(k, size):
        if (k,size) not in _kern_cache:
            kern = make_kern(k, size)
            kern(jax.random.PRNGKey(0))   # warmup JIT
            _kern_cache[(k,size)] = kern
        return _kern_cache[(k,size)]

def run_packet(size, shared_seed, lam, packet_n):
    """Execute all streams; return list of absolute timestamps."""
    # Build shared A, B (Gaussian phase rotation for Alice)
    rng = np.random.default_rng(int(shared_seed))
    A   = rng.standard_normal((size, size)).astype(np.float32)
    B   = rng.standard_normal((size, size)).astype(np.float32)
    if ROLE == 'Alice_Scramble' and lam > 0.01:
        mu = (size-1)/2.0
        G  = np.exp(-0.5*((np.arange(size)-mu)/(args.sigma*size/4))**2).astype(np.float32)
        G /= G.max()
        for k in range(N_STREAMS):
            g_k = G * STREAM_PROFILE[k] * lam
            B  += g_k[:,None] * np.outer(G, B[int(mu),:])

    t_abs = []
    if USE_JAX:
        for k in range(N_STREAMS):
            get_kern(k, size)(jax.random.PRNGKey(packet_n*N_STREAMS+k))
        jax.effects_barrier()
        for k in range(N_STREAMS):
            with _ts_lock:
                t_abs.append(_ts_store[k].pop(0) if _ts_store[k] else 0.0)
    else:
        for k in range(N_STREAMS):
            t1 = time.perf_counter()
            _ = A @ B
            t_abs.append(time.perf_counter())
    return t_abs

# ── D-LinOSS Learner (JAX-native, runs in-situ on same TPU) ─────────────────
# Simplified IMEX D-LinOSS: hidden state H evolves as:
#   H_t = (1 - gamma*dt) * H_{t-1} + W_in * x_t
#   y_t = W_out * H_t
# gamma (damping) is learned. W_in, W_out are fixed random projections.
# The "now-state" is the mean of H across the dlinoss_dim dimensions.
DIM   = args.dlinoss_dim
W_IN  = np.random.default_rng(777).standard_normal((DIM, N_STREAMS)).astype(np.float32) * 0.1
W_OUT = np.random.default_rng(888).standard_normal((N_STREAMS, DIM)).astype(np.float32) * 0.1
GAMMA = np.full(DIM, 0.05, dtype=np.float32)   # damping — slow decay = long memory

dlinoss_state = np.zeros(DIM, dtype=np.float32)
dlinoss_history = []   # list of (t, now_state_mean, prediction_error)

def dlinoss_step(x_t, target_t, dt=1.0):
    """One IMEX D-LinOSS step. x_t: (N_STREAMS,) timing vector. Returns now_state."""
    global dlinoss_state
    # Normalise input
    x_norm = (x_t - x_t.mean()) / (x_t.std() + 1e-6)
    # IMEX update: implicit damping, explicit input
    dlinoss_state = (1.0 - GAMMA * dt) * dlinoss_state + W_IN @ x_norm
    # Output projection
    y_t = W_OUT @ dlinoss_state     # (N_STREAMS,)
    # Prediction error (how well did we predict the next input)
    err = float(np.mean((y_t - x_norm)**2))
    now_state = float(dlinoss_state.mean())
    return now_state, err

def principal_eigvec(cov):
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, -1]

def eig_entropy(cov):
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 1e-12]; ev /= ev.sum()
    return float(-np.sum(ev * np.log(ev + 1e-12)))

# ── Natural Frequency Discovery ───────────────────────────────────────────────
print('\nPhase 1: Natural frequency sweep (30s)...')
print('  Warming up JIT...')
_ = run_packet(args.mat_size, 12345, 0.0, 0)
print('  JIT ready.')

SWEEP_RATES = [1.0, 2.0, 5.0, 10.0, 20.0]
sweep_res = []
for rate in SWEEP_RATES:
    interval = 1.0/rate
    bufs = [[] for _ in range(N_STREAMS)]
    t_sw = time.time()
    for pn in range(20):
        t_p = time.time()
        row = run_packet(args.mat_size, 99999, 0.0, pn)
        for k in range(N_STREAMS): bufs[k].append(row[k])
        el = time.time()-t_p
        if el < interval: time.sleep(interval-el)
    dts  = np.array([np.diff(np.array(bufs[k]))*1000 for k in range(N_STREAMS)])
    cov  = np.cov(dts)
    ev   = np.linalg.eigvalsh(cov)
    ev   = ev[ev>1e-12]; ev /= ev.sum()
    H    = float(-np.sum(ev*np.log(ev+1e-12)))
    actual = 20/(time.time()-t_sw)
    sweep_res.append({'rate': rate, 'H': H, 'actual_hz': actual})
    print(f'  {rate:5.1f}Hz (actual {actual:5.1f}Hz): H={H:.4f}')

best_r     = min(sweep_res, key=lambda x: x['H'])
NATURAL_HZ = best_r['rate']
INTERVAL   = 1.0/NATURAL_HZ
print(f'\n>> NATURAL FREQUENCY: {NATURAL_HZ}Hz  (H={best_r["H"]:.4f})')

# ── Calibration baseline (30s at natural rate) ────────────────────────────────
print(f'\nPhase 2: Baseline calibration at {NATURAL_HZ}Hz...')
cal_bufs = [[] for _ in range(N_STREAMS)]
t_cal = time.time()
pn_cal = 0
while time.time()-t_cal < 30.0:
    t_p = time.time()
    shared_seed = int(t_p//10)*10
    row = run_packet(args.mat_size, shared_seed, 0.0, pn_cal)
    for k in range(N_STREAMS): cal_bufs[k].append(row[k])
    el = time.time()-t_p
    if el < INTERVAL: time.sleep(INTERVAL-el)
    pn_cal += 1
cal_dts = np.array([np.diff(np.array(cal_bufs[k]))*1000 for k in range(N_STREAMS)])
BASELINE_COV  = np.cov(cal_dts).astype(np.float32)
BASELINE_MEAN = np.array([np.diff(np.array(cal_bufs[k]))*1000 for k in range(N_STREAMS)]).mean()
NULL_ALIGN    = float(np.dot(principal_eigvec(BASELINE_COV), principal_eigvec(BASELINE_COV))**2)
print(f'Baseline loop rate: {pn_cal/30.0:.1f}Hz  Null alignment: {NULL_ALIGN:.4f}')

# ── Main Capture + D-LinOSS Loop ──────────────────────────────────────────────
print(f'\nPhase 3: Main capture + D-LinOSS ({args.duration}s)...')
print(f'Started at {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}')

abs_buf   = [[] for _ in range(N_STREAMS)]
chunk_manifest = []
chunks_written = 0
t0         = time.time()
chunk_start = t0
packet_n   = 0

while time.time()-t0 < args.duration:
    t_pkt    = time.time()
    elapsed  = t_pkt - t0
    lam      = compute_lambda(elapsed)
    shared_seed = int(t_pkt//10)*10

    row = run_packet(args.mat_size, shared_seed, lam, packet_n)
    for k in range(N_STREAMS): abs_buf[k].append(row[k])

    if time.time()-chunk_start >= args.chunk_s:
        chunk_mid = (chunk_start+time.time())/2.0
        min_len   = min(len(b) for b in abs_buf)
        if min_len >= args.window+1:
            dts     = np.array([np.diff(abs_buf[k][-(args.window+1):])*1000
                                 for k in range(N_STREAMS)])
            cov     = np.cov(dts).astype(np.float32)
            pv      = principal_eigvec(cov).astype(np.float32)
            H       = eig_entropy(cov)
            lam_mid = float(compute_lambda(chunk_mid-t0))

            # D-LinOSS step on the mean timing vector
            x_t = np.array([np.mean(np.diff(abs_buf[k][-args.window:]))*1000
                             for k in range(N_STREAMS)], dtype=np.float32)
            now_state, pred_err = dlinoss_step(x_t, chunk_mid)

            fname = f'chunk_{chunks_written:05d}.npz'
            np.savez_compressed(
                OUT_DIR/fname,
                host_time_mid     = np.float64(chunk_mid),
                stream_timing     = dts.astype(np.float32),
                covariance_matrix = cov,
                principal_eigvec  = pv,
                eigenvalue_entropy= np.float32(H),
                lambda_val        = np.float32(lam_mid),
                dlinoss_now_state = np.float32(now_state),
                dlinoss_pred_err  = np.float32(pred_err),
                dlinoss_hidden    = dlinoss_state.copy().astype(np.float32),
                shared_seed       = np.int64(int(chunk_mid//10)*10),
                natural_hz        = np.float32(NATURAL_HZ),
                use_jax           = np.bool_(USE_JAX),
            )
            dlinoss_history.append({
                't': float(chunk_mid),
                'now_state': float(now_state),
                'pred_err':  float(pred_err),
                'lambda':    lam_mid,
                'eig_H':     float(H),
            })
            chunk_manifest.append({
                'chunk': fname, 'host_time_mid': float(chunk_mid),
                'lambda_val': lam_mid, 'eig_entropy': float(H),
                'dlinoss_now_state': float(now_state),
            })
            chunks_written += 1
            for k in range(N_STREAMS): abs_buf[k] = abs_buf[k][-(args.window+1):]
            chunk_start = time.time()

            if chunks_written % 12 == 0:
                phase = 'ON' if lam_mid>0.9 else ('RAMP' if lam_mid>0.01 else 'OFF')
                print(f'  [{elapsed:5.0f}s] chunk={chunks_written:3d}  '
                      f'lam={lam_mid:.2f}({phase})  H={H:.3f}  '
                      f'now={now_state:.4f}  err={pred_err:.4f}')

    dt_pkt = time.time()-t_pkt
    if dt_pkt < INTERVAL: time.sleep(INTERVAL-dt_pkt)
    packet_n += 1

elapsed_total = time.time()-t0
print(f'\nCapture done: {elapsed_total:.1f}s  chunks={chunks_written}  '
      f'packets={packet_n}  rate={packet_n/elapsed_total:.2f}Hz')

# ── Save manifest ─────────────────────────────────────────────────────────────
manifest = {
    'schema_version': '4.1',
    'dataset_type':   'shared_coherent_state_v4_dlinoss',
    'role': ROLE,
    't0_utc': t0, 'duration_s': elapsed_total,
    'n_chunks': chunks_written, 'n_streams': N_STREAMS,
    'chunk_seconds': args.chunk_s, 'window_packets': args.window,
    'natural_hz': NATURAL_HZ, 'base_mat_size': args.mat_size,
    'sigma_gaussian': args.sigma, 'stream_profile': STREAM_PROFILE.tolist(),
    'cycle_s': CYCLE_S, 'ramp_up_s': args.ramp_up, 'on_s': args.on_s,
    'ramp_down_s': args.ramp_down, 'off_s': args.off_s,
    'dlinoss_dim': DIM,
    'null_alignment': NULL_ALIGN,
    'baseline_mean_ms': float(BASELINE_MEAN),
    'sweep_results': sweep_res,
    'dlinoss_history': dlinoss_history,
    'infrastructure': infra,
    'chunk_files': [c['chunk'] for c in chunk_manifest],
    'chunks': chunk_manifest,
}
(OUT_DIR/'manifest.json').write_text(json.dumps(manifest, indent=2))

archive = f'chronos_v4_{ROLE.lower()}'
shutil.make_archive(archive, 'zip', str(OUT_DIR))
print(f'Archive: {archive}.zip')
print(f'Download: gcloud compute tpus tpu-vm scp {socket.gethostname()}:{archive}.zip .')
