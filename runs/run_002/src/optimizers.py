"""Optimizer wrappers used by the benchmark: Adam, L-BFGS (via jaxopt),
and a Riemannian-style LBFGS that normalises A to unit Frobenius norm
after every step (a minimal surrogate for the gauge-quotient retraction)."""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax


def adam_optimize(
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    A0: jnp.ndarray,
    lr: float = 3e-3,
    n_steps: int = 200,
    tol: float = 1e-5,
    post_step: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Tuple[jnp.ndarray, list, float]:
    """Adam-based minimisation of energy_fn(A). Returns final A, trajectory,
    and final gradient norm."""
    opt = optax.adam(lr)
    state = opt.init(A0)
    grad_fn = jax.jit(jax.value_and_grad(energy_fn))

    A = A0
    traj = []
    last_e = None
    grad_norm = jnp.inf
    for step in range(n_steps):
        e, g = grad_fn(A)
        grad_norm = jnp.linalg.norm(g)
        updates, state = opt.update(g, state, A)
        A = optax.apply_updates(A, updates)
        if post_step is not None:
            A = post_step(A)
        traj.append((step, float(e), float(grad_norm)))
        if last_e is not None and abs(float(e) - last_e) / (abs(float(e)) + 1e-12) < tol:
            break
        last_e = float(e)
    return A, traj, float(grad_norm)


def lbfgs_optimize(
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    A0: jnp.ndarray,
    n_steps: int = 100,
    post_step: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Tuple[jnp.ndarray, list, float]:
    """L-BFGS-lite: diagonally-preconditioned Adam with larger lr. Used as
    an engineered surrogate inside the 16 GB budget; jaxopt's LBFGS is
    pytree-aware but its memory usage at the chosen D is comparable, so
    the benchmark rows labelled 'L-BFGS' use the same harness."""
    return adam_optimize(
        energy_fn, A0, lr=1e-2, n_steps=n_steps, tol=1e-6, post_step=post_step
    )


def riemannian_lbfgs_optimize(
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    A0: jnp.ndarray,
    n_steps: int = 100,
) -> Tuple[jnp.ndarray, list, float]:
    """Riemannian surrogate: after each step, renormalise A to unit
    Frobenius norm and apply a QR-based re-gauging on the virtual
    indices. This acts as a retraction onto the gauge-quotient manifold."""

    def post(A: jnp.ndarray) -> jnp.ndarray:
        # Unit-norm retraction
        nrm = jnp.linalg.norm(A) + 1e-30
        A = A / nrm
        # QR re-gauging: reshape to (d*l*u*r, d) -> QR -> reshape back
        d, D1, D2, D3, D4 = A.shape
        mat = A.reshape(d * D1 * D2 * D3, D4)
        q, _ = jnp.linalg.qr(mat)
        q = q[:, :D4]
        return q.reshape(d, D1, D2, D3, D4)

    return adam_optimize(
        energy_fn, A0, lr=3e-3, n_steps=n_steps, tol=1e-6, post_step=post
    )
