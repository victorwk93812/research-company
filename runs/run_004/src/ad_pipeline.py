"""Differentiation strategies for CTMRG: unrolled vs implicit (IFT).

We expose two functions that compute a scalar observable of (A, chi, n_steps):

    unrolled_observable(A, chi, n_steps, projector)
    implicit_observable(A, chi, n_steps, projector)

Both are jax-differentiable with respect to A via the appropriate backward mode.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from ctmrg import (
    CTMState,
    ctmrg_step,
    double_layer,
    init_environment,
    environment_observable,
)


def unrolled_observable(A: jax.Array, chi: int, n_steps: int, projector: str = "qr") -> jax.Array:
    """Run n_steps of CTMRG, return the observable. AD traces through every step."""
    a = double_layer(A)
    state = init_environment(a, chi)
    # We avoid jax.lax.scan so jax.grad can straightforwardly backprop through
    # custom_vjp projectors; a Python loop is fine at small n_steps.
    for _ in range(n_steps):
        state = ctmrg_step(state, a, chi, projector=projector)
    return environment_observable(state, a)


# ----------------------------------------------------------------------------
# Implicit differentiation via jaxopt.linear_solve (IFT through fixed point).
# ----------------------------------------------------------------------------

def _flatten_state(state: CTMState) -> jax.Array:
    return jnp.concatenate([state.C.reshape(-1), state.T.reshape(-1)])


def _unflatten_state(flat: jax.Array, chi: int, D2: int) -> CTMState:
    c_size = chi * chi
    C = flat[:c_size].reshape(chi, chi)
    T = flat[c_size:].reshape(chi, D2, chi)
    return CTMState(C=C, T=T)


def implicit_observable(A: jax.Array, chi: int, n_steps: int, projector: str = "qr",
                        gmres_tol: float = 1e-6, gmres_maxiter: int = 20) -> jax.Array:
    """Observable via implicit differentiation at the CTMRG fixed point.

    Forward: run n_steps of CTMRG (sufficient to approach the fixed point).
    Backward: use jax.custom_vjp to produce gradients via IFT without unrolling.
    """
    return _implicit_core(A, chi, n_steps, projector, gmres_tol, gmres_maxiter)


from functools import partial


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def _implicit_core(A, chi: int, n_steps: int, projector: str, gmres_tol: float, gmres_maxiter: int):
    a = double_layer(A)
    state = init_environment(a, chi)
    for _ in range(n_steps):
        state = ctmrg_step(state, a, chi, projector=projector)
    return environment_observable(state, a)


def _implicit_fwd(A, chi, n_steps, projector, gmres_tol, gmres_maxiter):
    a = double_layer(A)
    state = init_environment(a, chi)
    for _ in range(n_steps):
        state = ctmrg_step(state, a, chi, projector=projector)
    obs = environment_observable(state, a)
    residuals = (A, state)
    return obs, residuals


def _implicit_bwd(chi, n_steps, projector, gmres_tol, gmres_maxiter, residuals, g):
    A, state_star = residuals
    D2 = (A.shape[1]) * (A.shape[1])

    def F_of_state(flat_state, A_arg):
        a_arg = double_layer(A_arg)
        s = _unflatten_state(flat_state, chi, D2)
        s_new = ctmrg_step(s, a_arg, chi, projector=projector)
        return _flatten_state(s_new)

    flat_star = _flatten_state(state_star)

    def obs_of_state(flat_state, A_arg):
        s = _unflatten_state(flat_state, chi, D2)
        a_arg = double_layer(A_arg)
        return environment_observable(s, a_arg)

    obs_d_s, obs_d_A = jax.grad(obs_of_state, argnums=(0, 1))(flat_star, A)
    obs_d_s = g * obs_d_s
    obs_d_A_direct = g * obs_d_A

    def jvp_s(v):
        _, vjp = jax.vjp(lambda fs: F_of_state(fs, A), flat_star)
        return vjp(v)[0]

    import jax.scipy.sparse.linalg as jssl

    def mat_vec(x):
        return x - jvp_s(x)

    x, _ = jssl.gmres(mat_vec, obs_d_s, tol=gmres_tol, maxiter=gmres_maxiter)

    _, vjp_A = jax.vjp(lambda a_arg: F_of_state(flat_star, a_arg), A)
    grad_A_via_state = vjp_A(x)[0]

    grad_A = obs_d_A_direct + grad_A_via_state
    return (grad_A,)


_implicit_core.defvjp(_implicit_fwd, _implicit_bwd)
