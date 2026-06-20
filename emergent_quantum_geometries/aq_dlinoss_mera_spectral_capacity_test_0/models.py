"""JAX models for the MPS spectral-capacity comparison."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class MPSParams(NamedTuple):
    cores: jax.Array
    left: jax.Array
    projection: jax.Array
    bias: jax.Array


class ObserverParams(NamedTuple):
    encoder: MPSParams | None
    gain: jax.Array
    code_decoder: jax.Array
    output_decoder: jax.Array
    latent_decoder: jax.Array
    transition_mix: jax.Array
    log_decay: jax.Array
    log_gamma: jax.Array
    gate: jax.Array
    gate_bias: jax.Array


def init_mps(key: jax.Array, wavelengths: int, block_size: int, bond_dim: int, spectral_dim: int) -> MPSParams:
    if wavelengths % block_size:
        raise ValueError("wavelength count must be divisible by block_size")
    sites = wavelengths // block_size
    k1, k2 = jax.random.split(key)
    cores = jax.random.normal(k1, (sites, bond_dim, 5, bond_dim)) / jnp.sqrt(5.0 * bond_dim)
    left = jnp.zeros((bond_dim,)).at[0].set(1.0)
    projection = jax.random.normal(k2, (bond_dim, spectral_dim)) / jnp.sqrt(float(bond_dim))
    return MPSParams(cores, left, projection, jnp.zeros((spectral_dim,)))


def mps_encode(params: MPSParams, spectrum: jax.Array, mask: jax.Array, block_size: int) -> jax.Array:
    """Contract masked local spectral features through an open-boundary MPS."""
    sites = spectrum.shape[-1] // block_size
    x = spectrum.reshape(sites, block_size)
    m = mask.reshape(sites, block_size)
    count = jnp.maximum(jnp.sum(m, axis=-1), 1.0)
    mean = jnp.sum(x * m, axis=-1) / count
    rms = jnp.sqrt(jnp.sum(jnp.square(x) * m, axis=-1) / count + 1e-8)
    slope = jnp.sum(x * m * jnp.linspace(-1.0, 1.0, block_size), axis=-1) / count
    local = jnp.stack((jnp.ones_like(mean), mean, rms, jnp.mean(m, axis=-1), slope), axis=-1)

    def contract(state, inputs):
        core, feature = inputs
        next_state = jnp.einsum("i,ipj,p->j", state, core, feature)
        next_state = jnp.tanh(next_state)
        next_state = next_state / jnp.maximum(jnp.linalg.norm(next_state), 1.0)
        return next_state, next_state

    final, _ = jax.lax.scan(contract, params.left, (params.cores, local))
    return final @ params.projection + params.bias


def init_observer(
    key: jax.Array,
    wavelengths: int,
    hidden_dim: int,
    latent_dim: int,
    *,
    use_mps: bool,
    block_size: int = 8,
    bond_dim: int = 12,
    spectral_dim: int = 24,
) -> ObserverParams:
    keys = jax.random.split(key, 9)
    code_dim = spectral_dim if use_mps else wavelengths
    encoder = init_mps(keys[0], wavelengths, block_size, bond_dim, spectral_dim) if use_mps else None
    return ObserverParams(
        encoder=encoder,
        gain=jax.random.normal(keys[1], (code_dim, hidden_dim)) / jnp.sqrt(float(code_dim)),
        code_decoder=jax.random.normal(keys[2], (hidden_dim, code_dim)) / jnp.sqrt(float(hidden_dim)),
        output_decoder=jax.random.normal(keys[3], (hidden_dim, wavelengths)) / jnp.sqrt(float(hidden_dim)),
        latent_decoder=jax.random.normal(keys[4], (hidden_dim, latent_dim)) / jnp.sqrt(float(hidden_dim)),
        transition_mix=jax.random.normal(keys[5], (hidden_dim, hidden_dim)) * 0.02,
        log_decay=jnp.full((hidden_dim,), 1.5),
        log_gamma=jnp.full((hidden_dim,), -1.5),
        gate=jax.random.normal(keys[6], (3, hidden_dim)) * 0.04,
        gate_bias=jnp.full((hidden_dim,), -1.0),
    )


def _encode_sequence(params: ObserverParams, x: jax.Array, mask: jax.Array, block_size: int) -> jax.Array:
    if params.encoder is None:
        return x * mask
    return jax.vmap(lambda a, b: mps_encode(params.encoder, a, b, block_size))(x, mask)


def run_observer(
    params: ObserverParams,
    x: jax.Array,
    mask: jax.Array,
    window_id: jax.Array,
    omega_rad_day: jax.Array,
    dt_days: float,
    *,
    dlinoss: bool,
    block_size: int = 8,
    warmup_fraction: float = 0.30,
) -> dict[str, jax.Array]:
    code = _encode_sequence(params, x, mask, block_size)
    groups = jax.nn.one_hot(window_id, 3, dtype=x.dtype)
    summaries = mask @ groups / jnp.maximum(jnp.sum(groups, axis=0), 1.0)
    hidden_dim = params.gain.shape[-1]
    warmup = jnp.asarray(jnp.ceil(x.shape[0] * warmup_fraction), jnp.int32)
    if dlinoss:
        gamma = jax.nn.softplus(params.log_gamma)
        pole = jnp.exp((-gamma + 1j * omega_rad_day) * dt_days)
        initial = jnp.zeros((hidden_dim,), jnp.complex64)
    else:
        pole = None
        initial = jnp.zeros((hidden_dim,), jnp.float32)

    def step(previous, inputs):
        index, observed_code, summary = inputs
        if dlinoss:
            predicted = pole * previous
            real_predicted = jnp.real(predicted)
        else:
            stable = 0.90 * jnp.eye(hidden_dim) + 0.04 * jnp.tanh(params.transition_mix) / jnp.sqrt(float(hidden_dim))
            real_predicted = previous @ stable
            predicted = real_predicted
        code_prediction = real_predicted @ params.code_decoder
        innovation = observed_code - code_prediction
        alpha = 0.05 + 0.40 * jax.nn.sigmoid(summary @ params.gate + params.gate_bias)
        update = alpha * (innovation @ params.gain)
        posterior = predicted + update
        real_posterior = jnp.real(posterior)
        return posterior, (
            real_posterior @ params.output_decoder,
            real_posterior @ params.latent_decoder,
            real_posterior,
            (index >= warmup).astype(x.dtype),
        )

    indices = jnp.arange(x.shape[0], dtype=jnp.int32)
    final, (residual_hat, latent_hat, states, loss_weight) = jax.lax.scan(
        step, initial, (indices, code, summaries)
    )
    return {
        "residual_hat": residual_hat,
        "latent_hat": latent_hat,
        "states": states,
        "loss_weight": loss_weight,
        "final_state": final,
    }
