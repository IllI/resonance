#!/usr/bin/env python3
"""Four-arm MPS spectral-capacity test on the locked synthetic shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import init_observer, run_observer


ARMS = ("mps_dlinoss", "mps_linear_ssm", "dlinoss", "linear_ssm")


def smooth_residual(x, width=9):
    pad = width // 2
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    smooth = sum(padded[:, :, i : i + x.shape[2]] for i in range(width)) / width
    return x - smooth


def corr(a, b):
    a = a.reshape(-1) - jnp.mean(a)
    b = b.reshape(-1) - jnp.mean(b)
    return jnp.sum(a * b) / jnp.maximum(jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b)), 1e-12)


def prepare(npz):
    with np.load(npz, allow_pickle=False) as ar:
        raw = {key: ar[key] for key in ar.files}
    scale = float(np.std(smooth_residual(raw["train_residual"].astype(np.float32)))) + 1e-8
    data = {"window_id": jnp.asarray(raw["window_id"])}
    for split in ("train", "validation", "test"):
        data[f"{split}_x"] = jnp.asarray(smooth_residual(raw[f"{split}_x"].astype(np.float32)) / scale)
        data[f"{split}_target"] = jnp.asarray(smooth_residual(raw[f"{split}_residual"].astype(np.float32)) / scale)
        data[f"{split}_mask"] = jnp.asarray(raw[f"{split}_mask"].astype(np.float32))
        data[f"{split}_z"] = jnp.asarray(raw[f"{split}_z"].astype(np.float32))
    data["physical_periods"] = np.asarray(raw["physical_periods_days"], np.float32)
    return data, scale


def batch_forward(params, arm, x, mask, window_id, omega, dt):
    use_dlinoss = arm in ("mps_dlinoss", "dlinoss")
    return jax.vmap(lambda a, b: run_observer(params, a, b, window_id, omega, dt, dlinoss=use_dlinoss))(x, mask)


def loss_fn(params, arm, x, mask, target, z, window_id, omega, dt):
    out = batch_forward(params, arm, x, mask, window_id, omega, dt)
    w = out["loss_weight"][:, :, None]
    denom_r = jnp.maximum(jnp.sum(w) * target.shape[-1], 1.0)
    denom_z = jnp.maximum(jnp.sum(w) * z.shape[-1], 1.0)
    residual = jnp.sum(w * jnp.square(out["residual_hat"] - target)) / denom_r
    latent = jnp.sum(w * jnp.square(out["latent_hat"] - z)) / denom_z
    return residual + latent


def train(seed, arm, hidden, steps, data, dt, learning_rate=1e-3):
    use_mps = arm.startswith("mps_")
    params = init_observer(jax.random.PRNGKey(seed), 96, hidden, 6, use_mps=use_mps)
    m = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, params, is_leaf=lambda x: x is None)
    v = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, params, is_leaf=lambda x: x is None)
    omega = jnp.resize(2 * jnp.pi / jnp.asarray(data["physical_periods"]), hidden)
    value_grad = jax.value_and_grad(loss_fn)
    batch_size = 12

    def update(index, state):
        current, first, second = state
        idx = (jnp.arange(batch_size) + (index * batch_size + seed) % data["train_x"].shape[0]) % data["train_x"].shape[0]
        value, grads = value_grad(current, arm, data["train_x"][idx], data["train_mask"][idx], data["train_target"][idx], data["train_z"][idx], data["window_id"], omega, dt)
        count = index.astype(jnp.float32) + 1.0
        first = jax.tree_util.tree_map(lambda a, g: 0.9 * a + 0.1 * g, first, grads)
        second = jax.tree_util.tree_map(lambda a, g: 0.999 * a + 0.001 * g * g, second, grads)
        current = jax.tree_util.tree_map(
            lambda p, a, b: p - learning_rate * (a / (1 - 0.9**count)) / (jnp.sqrt(b / (1 - 0.999**count)) + 1e-8),
            current, first, second,
        )
        return current, first, second

    params, _, _ = jax.lax.fori_loop(0, steps, update, (params, m, v))
    val = loss_fn(params, arm, data["validation_x"], data["validation_mask"], data["validation_target"], data["validation_z"], data["window_id"], omega, dt)
    return params, omega, val


def evaluate(params, omega, arm, seed, data, dt, held_indices):
    x, mask, target, z = data["test_x"], data["test_mask"], data["test_target"], data["test_z"]
    warm = int(np.ceil(0.30 * x.shape[1]))
    sl = (slice(None), slice(warm, None))
    real = batch_forward(params, arm, x, mask, data["window_id"], omega, dt)
    key = jax.random.PRNGKey(seed + 7001)
    mask_order = jax.random.permutation(key, x.shape[1])
    wave_order = jax.random.permutation(jax.random.fold_in(key, 1), x.shape[2])
    mask_null = batch_forward(params, arm, x, mask[:, mask_order], data["window_id"], omega, dt)
    wave_null = batch_forward(params, arm, x[:, :, wave_order], mask[:, :, wave_order], data["window_id"], omega, dt)
    held = data["window_id"] == 2
    leave_mask = mask * (1.0 - held[None, None, :])
    leave = batch_forward(params, arm, x, leave_mask, data["window_id"], omega, dt)
    residual_corr = corr(real["residual_hat"][sl], target[sl])
    mask_corr = corr(mask_null["residual_hat"][sl], target[sl])
    wave_corr = corr(wave_null["residual_hat"][sl], target[sl])
    leave_corr = corr(
        jnp.take(leave["residual_hat"][:, warm:], held_indices, axis=-1),
        jnp.take(target[:, warm:], held_indices, axis=-1),
    )
    perm = jax.random.permutation(jax.random.fold_in(key, 2), target.reshape(-1)).reshape(target.shape)
    false_corr = jnp.abs(corr(real["residual_hat"][sl], perm[sl]))
    scale = jnp.maximum(jnp.abs(residual_corr), 1e-12)
    return jnp.asarray((
        residual_corr,
        corr(real["latent_hat"][sl], z[sl]),
        (residual_corr - mask_corr) / scale,
        (residual_corr - wave_corr) / scale,
        leave_corr,
        false_corr,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(800, 2000))
    parser.add_argument("--hidden", type=int, nargs="+", default=(20, 48))
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 23, 31))
    parser.add_argument("--dt-days", type=float, default=0.25)
    parser.add_argument("--require-eight-devices", action="store_true")
    args = parser.parse_args()
    if args.require_eight_devices and jax.device_count() < 8:
        raise RuntimeError(f"eight TPU devices required, found {jax.device_count()}")
    data, scale = prepare(args.npz)
    held_indices = np.where(np.asarray(data["window_id"]) == 2)[0]
    rows = []
    for arm in ARMS:
        candidates = []
        for hidden in args.hidden:
            for steps in args.steps:
                values = []
                if jax.device_count() >= 8:
                    padded = list(args.seeds) + [args.seeds[-1]] * (8 - len(args.seeds))

                    def train_and_evaluate(seed):
                        params, omega, val = train(seed, arm, hidden, steps, data, args.dt_days)
                        return val, evaluate(params, omega, arm, seed, data, args.dt_days, held_indices)

                    validation_batch, metric_batch = jax.pmap(train_and_evaluate)(jnp.asarray(padded, jnp.int32))
                    run_values = zip(args.seeds, np.asarray(validation_batch)[: len(args.seeds)], np.asarray(metric_batch)[: len(args.seeds)])
                else:
                    serial = []
                    for seed in args.seeds:
                        params, omega, val = train(seed, arm, hidden, steps, data, args.dt_days)
                        serial.append((seed, np.asarray(val), np.asarray(evaluate(params, omega, arm, seed, data, args.dt_days, held_indices))))
                    run_values = serial
                for seed, val, metrics in run_values:
                    row = {"arm": arm, "hidden_dim": hidden, "steps": steps, "seed": seed, "validation_loss": float(val)}
                    row.update(dict(zip(("residual_corr", "latent_corr", "mask_shuffle_degradation", "spectral_window_shuffle_degradation", "leave_window_out_corr", "false_positive_corr"), map(float, metrics))))
                    rows.append(row)
                    values.append(row)
                candidates.append((np.mean([v["validation_loss"] for v in values]), hidden, steps))
        _, selected_hidden, selected_steps = min(candidates)
        for row in rows:
            if row["arm"] == arm:
                row["selected"] = row["hidden_dim"] == selected_hidden and row["steps"] == selected_steps

    summaries = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm and r.get("selected")]
        summaries[arm] = {key: float(np.mean([r[key] for r in selected])) for key in (
            "residual_corr", "latent_corr", "mask_shuffle_degradation", "spectral_window_shuffle_degradation", "leave_window_out_corr", "false_positive_corr"
        )}
        summaries[arm]["hidden_dim"] = selected[0]["hidden_dim"]
        summaries[arm]["steps"] = selected[0]["steps"]

    mps_d = summaries["mps_dlinoss"]
    linear = summaries["linear_ssm"]
    gain = (mps_d["residual_corr"] - linear["residual_corr"]) / max(abs(linear["residual_corr"]), 1e-9)
    gates = {
        "leave_window_out_ge_0p75": mps_d["leave_window_out_corr"] >= 0.75,
        "residual_corr_ge_0p88": mps_d["residual_corr"] >= 0.88,
        "beats_linear_ssm_ge_5pct": gain >= 0.05,
        "mask_shuffle_ge_0p20": mps_d["mask_shuffle_degradation"] >= 0.20,
        "spectral_window_shuffle_ge_0p20": mps_d["spectral_window_shuffle_degradation"] >= 0.20,
        "false_positive_le_0p05": mps_d["false_positive_corr"] <= 0.05,
    }
    if all(gates.values()):
        verdict = "PROMOTE MPS_DLINOSS_SPECTRAL_CAPACITY"
    elif summaries["mps_linear_ssm"]["leave_window_out_corr"] >= 0.75 and summaries["mps_linear_ssm"]["residual_corr"] >= 0.88:
        verdict = "PROMOTE MPS_LINEARSSM_ONLY"
    elif linear["residual_corr"] >= mps_d["residual_corr"]:
        verdict = "PROMOTE LINEAR_SSM_BASELINE_ONLY"
    else:
        verdict = "DO_NOT_PROMOTE"
    result = {
        "run": "AQ-DLINOSS-MERA-SPECTRAL-CAPACITY-TEST-0",
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "input_shape": list(data["train_x"].shape),
        "target_scale": scale,
        "architecture": "masked local features -> open-boundary MPS contraction -> spectral state -> temporal observer",
        "rows": rows,
        "summaries": summaries,
        "mps_dlinoss_gain_over_linear_ssm": gain,
        "gates": gates,
        "verdict": verdict,
        "astronomy_authorized": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summaries, "gates": gates, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
