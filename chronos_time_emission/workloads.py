from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def setup_devices():
    backend = jax.default_backend()
    devices = jax.devices()
    if len(devices) < 2:
        raise RuntimeError(f"Need at least two devices; got {len(devices)} on backend={backend}")
    split = len(devices) // 2
    alice_devs = devices[:split]
    bob_devs = devices[split:]
    return backend, devices, alice_devs, bob_devs


def build_inputs(alice_devs, bob_devs, d: int, seed: int):
    key = jax.random.PRNGKey(int(seed))
    k1, k2 = jax.random.split(key)
    # Let pmap own the sharding of the leading axis so the inputs stay portable
    # across TPU backends and avoid rank-specific PositionalSharding constraints.
    alice_input = jax.random.normal(k1, (len(alice_devs), d, d), dtype=jnp.float32)
    bob_input = jax.device_put(jax.random.normal(k2, (d,), dtype=jnp.float32), bob_devs[0])
    return alice_input, bob_input


def build_workloads(alice_devs, bob_devs, d: int, sparse_block: int, layers: int):
    if d % sparse_block != 0:
        raise ValueError(f"d={d} must be divisible by sparse_block={sparse_block}")
    n_blocks = d // sparse_block

    @partial(jax.pmap, axis_name="alice", devices=alice_devs)
    def alice_dense(x):
        for _ in range(layers):
            x = jnp.dot(x, x.T)
            x = x / (jnp.linalg.norm(x) + 1e-12)
        return lax.psum(x, axis_name="alice")

    @partial(jax.pmap, axis_name="alice", devices=alice_devs)
    def alice_block_sparse(x):
        block = x[:sparse_block, :sparse_block]
        for _ in range(layers):
            block = jnp.dot(block, block.T)
            block = block / (jnp.linalg.norm(block) + 1e-12)
        out = jnp.zeros_like(x).at[:sparse_block, :sparse_block].set(block)
        return lax.psum(out, axis_name="alice")

    @partial(jax.pmap, axis_name="alice", devices=alice_devs)
    def alice_equal_flop_tiled(x):
        blocks = x.reshape(n_blocks, sparse_block, n_blocks, sparse_block)
        blocks = jnp.swapaxes(blocks, 1, 2).reshape(n_blocks * n_blocks, sparse_block, sparse_block)
        for _ in range(layers):
            blocks = jnp.matmul(blocks, jnp.swapaxes(blocks, -1, -2))
            blocks = blocks / (jnp.linalg.norm(blocks, axis=(-2, -1), keepdims=True) + 1e-12)
        out = blocks.reshape(n_blocks, n_blocks, sparse_block, sparse_block)
        out = jnp.swapaxes(out, 1, 2).reshape(d, d)
        return lax.psum(out, axis_name="alice")

    @partial(jax.pmap, axis_name="alice", devices=alice_devs)
    def alice_equal_flop_permuted(x):
        blocks = x.reshape(n_blocks, sparse_block, n_blocks, sparse_block)
        blocks = jnp.swapaxes(blocks, 1, 2).reshape(n_blocks * n_blocks, sparse_block, sparse_block)
        for _ in range(layers):
            mixed = 0.5 * (blocks + jnp.roll(blocks, shift=1, axis=0))
            blocks = jnp.matmul(mixed, jnp.swapaxes(mixed, -1, -2))
            blocks = blocks / (jnp.linalg.norm(blocks, axis=(-2, -1), keepdims=True) + 1e-12)
        out = blocks.reshape(n_blocks, n_blocks, sparse_block, sparse_block)
        out = jnp.swapaxes(out, 1, 2).reshape(d, d)
        return lax.psum(out, axis_name="alice")

    @partial(jax.pmap, axis_name="alice", devices=alice_devs)
    def alice_noop(x):
        return lax.psum(jnp.zeros_like(x), axis_name="alice")

    @partial(jax.jit, device=bob_devs[0])
    def bob_probe(x):
        return jnp.sum(jnp.tanh(x))

    return {
        "dense_full": alice_dense,
        "block_sparse": alice_block_sparse,
        "equal_flop_tiled": alice_equal_flop_tiled,
        "equal_flop_permuted": alice_equal_flop_permuted,
        "noop": alice_noop,
    }, bob_probe
