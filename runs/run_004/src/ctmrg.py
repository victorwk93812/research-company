"""C4v-symmetric CTMRG with switchable QR / SVD projector, JAX-native.

This is a deliberately minimal, pedagogically clean CTMRG implementation aimed
at isolating the *projector* step (QR vs SVD) for AD comparison. The forward
map is

    (C, T)_{k+1} = F( (C, T)_k, a )

where C is the chi x chi corner, T is the chi x D^2 x chi edge, and a is the
double-layer reduced tensor of shape (D^2, D^2, D^2, D^2).

Each sweep:
  1. Build enlarged corner and edge by absorbing a.
  2. Compute projector P from the enlarged corner (QR or SVD).
  3. Apply P to truncate back to chi.
  4. Renormalise.

C4v symmetry is enforced: C is Hermitian, T is left/right symmetric.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class CTMState(NamedTuple):
    C: jax.Array  # (chi, chi)
    T: jax.Array  # (chi, D2, chi)


# ------------------------------------------------------------------
# Projector primitives.
# ------------------------------------------------------------------

def qr_projector(M: jax.Array, chi: int) -> jax.Array:
    """QR-based isometry: P = Q of a thin QR on the first chi columns of M.

    Enforces the canonical R_ii >= 0 gauge fix, so the map is differentiable
    wherever R is invertible.
    """
    tall = M[:, :chi]
    Q, R = jnp.linalg.qr(tall, mode="reduced")
    diag_R = jnp.diagonal(R)
    abs_d = jnp.abs(diag_R)
    safe = jnp.where(abs_d > 1e-30, diag_R / (abs_d + 1e-30), jnp.ones_like(diag_R))
    # P <- P * conj(sign) to enforce diag(R) > 0
    D = jnp.conj(safe)
    P = Q * D[None, :]
    return P


def svd_projector(M: jax.Array, chi: int) -> jax.Array:
    """SVD-based isometry P = U_chi * S_chi^{-1/2} (Fishman-White).

    Standard truncated-SVD projector. Includes a small epsilon in the
    denominator for numerical safety; the 'plain' baseline uses eps=0,
    the Francuz-patched baseline replaces the eps-add with a corrected form.
    """
    U, S, _ = jnp.linalg.svd(M, full_matrices=False)
    U_chi = U[:, :chi]
    S_chi = S[:chi]
    inv_sqrt = 1.0 / jnp.sqrt(S_chi + 1e-12)
    P = U_chi * inv_sqrt[None, :]
    return P


# ------------------------------------------------------------------
# Double-layer tensor and initialization.
# ------------------------------------------------------------------

def double_layer(A: jax.Array) -> jax.Array:
    """Form a[ur, rr, dr, lr] = sum_s A[s, u,r,d,l] A*[s, u',r',d',l'] flattened."""
    # A: (d, D, D, D, D)
    a = jnp.einsum("suvwx,sABCD->uAvBwCxD", A, jnp.conjugate(A))
    D = A.shape[1]
    D2 = D * D
    a = a.reshape(D2, D2, D2, D2)
    return a


def init_environment(a: jax.Array, chi: int) -> CTMState:
    """Deterministic warm-start initialisation of (C, T) from a."""
    D2 = a.shape[0]
    # C_ij = sum_k a[i, j, k, k]
    C_small = jnp.einsum("ijkk->ij", a)
    # Pad to chi
    n = min(chi, D2)
    C = jnp.zeros((chi, chi), dtype=a.dtype)
    C = C.at[:n, :n].set(C_small[:n, :n])
    # T[i, m, j] = a[i, j, m, m] (rough) summed over one free axis
    T_small = jnp.einsum("ijkl->ikj", a)  # (D2, D2, D2)
    T = jnp.zeros((chi, D2, chi), dtype=a.dtype)
    T = T.at[:n, :, :n].set(T_small[:n, :, :n])
    # Symmetrise
    C = 0.5 * (C + C.conj().T)
    T = 0.5 * (T + jnp.transpose(jnp.conjugate(T), (2, 1, 0)))
    # Normalise
    C = C / (jnp.linalg.norm(C) + 1e-30)
    T = T / (jnp.linalg.norm(T) + 1e-30)
    return CTMState(C=C, T=T)


# ------------------------------------------------------------------
# One CTMRG sweep (projector-switchable).
# ------------------------------------------------------------------

def _enlarged_corner(C: jax.Array, T: jax.Array, a: jax.Array) -> jax.Array:
    """Enlarged-corner matrix M of shape (chi*D2, chi*D2).

    Construction: C contracts with two edges T and the reduced tensor a, giving
    a 4-leg tensor with indices (chi, D2, chi, D2) which is flattened into a
    square matrix.
    """
    # intermediate: [i, k, b, l] = T[i, k, a] * C[a, b] * T[b, l, j] summed over a
    # Then absorb a[k, l, k', l']:
    # M4[i, k', j, l'] = sum_{k, l, a, b, j} T[i,k,a] C[a,b] T[b,l,j] * a[k,l,k',l']
    M4 = jnp.einsum(
        "ika,ab,blj,klKL->iKjL",
        T, C, T, a,
        optimize=True,
    )
    chi = C.shape[0]
    D2 = a.shape[0]
    M = M4.reshape(chi * D2, chi * D2)
    M = 0.5 * (M + M.conj().T)  # enforce C4v symmetry
    return M


def ctmrg_step(state: CTMState, a: jax.Array, chi: int, projector: str = "qr") -> CTMState:
    """One CTMRG sweep with either QR or SVD projector."""
    C, T = state.C, state.T
    D2 = a.shape[0]
    chi_now = C.shape[0]

    M = _enlarged_corner(C, T, a)

    if projector == "qr":
        P = qr_projector(M, chi_now)
    elif projector == "svd":
        P = svd_projector(M, chi_now)
    else:
        raise ValueError(f"unknown projector {projector!r}")

    # P has shape (chi_now * D2, chi_now). New corner C' = P^H M P.
    C_new = jnp.einsum("ai,ab,bj->ij", jnp.conjugate(P), M, P)
    # Enlarged edge: E[A,K,m,B,L] = T[A, X, B] * a[K, L, m, X]
    # where A, B are chi-axes and K, L, m, X are D2-axes.
    E5 = jnp.einsum("AXB,KLmX->AKmBL", T, a, optimize=True)
    E = E5.reshape(chi_now * D2, D2, chi_now * D2)
    T_new = jnp.einsum("ai,amb,bj->imj", jnp.conjugate(P), E, P)

    # Renormalise.
    C_new = C_new / (jnp.linalg.norm(C_new) + 1e-30)
    T_new = T_new / (jnp.linalg.norm(T_new) + 1e-30)

    # Re-symmetrise (C4v):
    C_new = 0.5 * (C_new + C_new.conj().T)
    T_new = 0.5 * (T_new + jnp.transpose(jnp.conjugate(T_new), (2, 1, 0)))

    return CTMState(C=C_new, T=T_new)


def run_ctmrg(A: jax.Array, chi: int, n_steps: int, projector: str = "qr") -> CTMState:
    """Run n_steps of CTMRG starting from the deterministic init."""
    a = double_layer(A)
    state = init_environment(a, chi)
    for _ in range(n_steps):
        state = ctmrg_step(state, a, chi, projector=projector)
    return state


def fixed_point_residual(state: CTMState, a: jax.Array, chi: int, projector: str = "qr") -> jax.Array:
    """||F(s) - s||_F; diagnostic of CTMRG convergence."""
    new = ctmrg_step(state, a, chi, projector=projector)
    return jnp.linalg.norm(new.C - state.C) + jnp.linalg.norm(new.T - state.T)


# ------------------------------------------------------------------
# Simple energy-like observable (reduced to a trace for testability).
# ------------------------------------------------------------------

def environment_observable(state: CTMState, a: jax.Array) -> jax.Array:
    """A scalar observable of the environment + site tensor.

    This stands in for a physical expectation value in the unit tests; it is
    *not* a true iPEPS energy, but it exercises the same computational graph.
    """
    C, T = state.C, state.T
    # Reduce a to a 2-tensor over physical axes by tracing two legs:
    a_mat = jnp.einsum("pqrr->pq", a)
    obs = jnp.einsum(
        "ij,jpa,pq,aqi->",
        C, T, a_mat, jnp.conjugate(T),
        optimize=True,
    )
    return jnp.real(obs)
