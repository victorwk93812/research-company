"""Gradient-descent optimisation wrappers for iPEPS tensors."""
from __future__ import annotations

import time
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax

from ad_pipeline import unrolled_observable, implicit_observable


def _loss_wrapper(mode: str, projector: str, chi: int, n_steps: int) -> Callable:
    """Produce an (A -> scalar) loss."""
    if mode == "unrolled":
        def L(A):
            return unrolled_observable(A, chi, n_steps, projector=projector)
    elif mode == "implicit":
        def L(A):
            return implicit_observable(A, chi, n_steps, projector=projector)
    else:
        raise ValueError(mode)
    return L


def adam_optimize(
    A0: jax.Array,
    chi: int,
    n_ctm_steps: int,
    mode: str = "unrolled",
    projector: str = "qr",
    n_opt_steps: int = 50,
    learning_rate: float = 1e-3,
) -> tuple[jax.Array, list[float], list[float], float]:
    """Run Adam optimisation; return (A_final, loss_history, grad_norm_history, wall_time_s)."""
    L = _loss_wrapper(mode, projector, chi, n_ctm_steps)
    value_and_grad = jax.value_and_grad(L)

    opt = optax.adam(learning_rate)
    opt_state = opt.init(A0)
    A = A0
    losses = []
    grad_norms = []
    t0 = time.time()
    for _ in range(n_opt_steps):
        loss, g = value_and_grad(A)
        g_tree = g
        updates, opt_state = opt.update(g_tree, opt_state)
        A = optax.apply_updates(A, updates)
        losses.append(float(loss))
        grad_norms.append(float(jnp.linalg.norm(g)))
    t1 = time.time()
    return A, losses, grad_norms, t1 - t0
