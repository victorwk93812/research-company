"""Benchmark harness: wall-time and gradient-quality for QR vs SVD projectors."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass
class TimingResult:
    name: str
    forward_ms: float
    backward_ms: float
    total_ms: float


def time_forward_backward(loss_fn: Callable[[jax.Array], jax.Array], A: jax.Array, name: str,
                          n_repeats: int = 3) -> TimingResult:
    """Measure (forward-only, full backward, total) wall time."""
    # Warmup (JIT compile).
    _ = loss_fn(A).block_until_ready()
    g_fn = jax.value_and_grad(loss_fn)
    v0, g0 = g_fn(A)
    v0.block_until_ready(); g0.block_until_ready()

    fwd_times = []
    bwd_times = []
    for _ in range(n_repeats):
        t0 = time.time()
        v = loss_fn(A).block_until_ready()
        t1 = time.time()
        _, g = g_fn(A)
        g.block_until_ready()
        t2 = time.time()
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
    f_ms = min(fwd_times) * 1000.0
    b_ms = min(bwd_times) * 1000.0
    return TimingResult(name=name, forward_ms=f_ms, backward_ms=b_ms, total_ms=f_ms + b_ms)


def finite_diff_check(loss_fn: Callable[[jax.Array], jax.Array], A: jax.Array,
                      h: float = 1e-5) -> tuple[float, float]:
    """Check AD gradient vs central finite differences on a random direction.

    Returns (ad_directional, fd_directional).
    """
    g = jax.grad(loss_fn)(A)
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, A.shape, dtype=A.dtype)
    v = v / jnp.linalg.norm(v)
    ad_dir = float(jnp.real(jnp.vdot(g, v)))
    fd_dir = float((loss_fn(A + h * v) - loss_fn(A - h * v)) / (2 * h))
    return ad_dir, fd_dir
