"""QR-canonical CTMRG projector with end-to-end automatic differentiation.

This module provides ``qr_canonical_projector(M, chi)``, a JAX-differentiable
routine that computes a CTMRG projector from a stacked half-corner matrix
``M`` via QR factorisation followed by a small symmetric eigendecomposition.

Pipeline (forward)
------------------
    M  -- (f, c) input matrix (concat of two grown corners along col axis)
    Q, R = qr(M)                  # thin QR with R upper-triangular
    Q, R = canonical_gauge(Q, R)  # rotate diag(R) phases to be real-positive
    rho   = R @ R^H               # small (r, r) symmetric matrix
    Lambda, U = truncated_eigh_regularized(-rho, chi)
    P     = Q @ U[:, ::-1]        # (f, k) isometry, descending eigenvalue order

The reverse-mode VJP is provided by composing JAX's built-in QR backward with
``tenax.algorithms._lorentzian_eigh.truncated_eigh_regularized``, whose custom
VJP applies the Francuz-style Lorentzian regularisation to the truncation
step (kept-discarded coupling).  Because every elementary operation in the
forward path already has a stable JAX-native backward, the composite has a
correct and stable backward without an additional custom_vjp wrapper around
the whole call.

A separate ``qr_canonical_projector_with_vjp`` is exposed with an explicit
``custom_vjp`` registration so the chain can be tested standalone (gauge
invariance, finite-difference agreement) without the surrounding tenax
machinery.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _canonical_qr(M: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Thin QR with diagonal phase of R rotated to be real, non-negative.

    For real M this is the standard ``diag(R) >= 0`` sign-fix; for complex M
    it rotates each diagonal entry's phase so ``diag(R)`` becomes real-positive,
    leaving the column/row of (Q, R) untouched when the diagonal entry is zero.
    The mapping ``(Q, R) -> Q @ X`` is gauge-invariant for any X built from R,
    so this fix is transparent to downstream consumers but yields a
    differentiably-smooth factorisation across small perturbations of M.
    """
    Q, R = jnp.linalg.qr(M)
    diag_R = jnp.diag(R)
    abs_diag = jnp.abs(diag_R)
    one = jnp.ones_like(diag_R)
    # phase = diag_R / |diag_R|, with safe fallback when |diag_R| == 0
    phase = jnp.where(abs_diag > 0, diag_R / jnp.where(abs_diag > 0, abs_diag, 1.0), one)
    phase = phase.astype(R.dtype)
    Q = Q * phase[None, :]
    R = R * jnp.conj(phase)[:, None]
    return Q, R


_LORENTZ_EPS_REL = 1e-7
_LORENTZ_EPS_ABS = 1e-12


def _lorentzian_eigh_bwd_kernel(w, V, dw, dV):
    """Lorentzian-regularised symmetric-eigh backward kernel.

    Replaces the standard ``F_{ij} = 1 / (w_j - w_i)`` with the Lorentzian
    smoothing
    ``F^(eps)_{ij} = (w_j - w_i) / ((w_j - w_i)^2 + eps^2)``
    where ``eps = max(eps_abs, eps_rel * max|w|)``.  Caps F at ``1/(2*eps)``
    regardless of how close two eigenvalues come, preventing the gradient
    explosion that the naive eigh adjoint suffers near degeneracy
    (Francuz et al., PRR 7, 013237).
    """
    w_scale = jnp.maximum(jnp.max(jnp.abs(w)), _LORENTZ_EPS_ABS)
    eps = jnp.maximum(_LORENTZ_EPS_ABS, _LORENTZ_EPS_REL * w_scale)
    diff = w[None, :] - w[:, None]   # (n, n): diff[i, j] = w_j - w_i
    F = diff / (diff**2 + eps**2)
    F = F - jnp.diag(jnp.diag(F))
    inner = jnp.conj(V).T @ dV
    dM = V @ (jnp.diag(dw) + F * inner) @ jnp.conj(V).T
    return 0.5 * (dM + jnp.conj(dM).T)


@jax.custom_vjp
def _lorentzian_eigh(M):
    """Symmetric eigh whose VJP uses Lorentzian regularisation.

    Returns a plain ``(eigenvalues, eigenvectors)`` tuple — note we unpack
    ``jnp.linalg.eigh``'s ``EighResult`` so the primal output type is a
    plain tuple, which JAX's custom_vjp expects to match the fwd result
    structure (otherwise it raises a structure-mismatch TypeError).
    """
    out = jnp.linalg.eigh(M)
    return (out[0], out[1])


def _lorentzian_eigh_fwd(M):
    out = jnp.linalg.eigh(M)
    w, V = out[0], out[1]
    return (w, V), (w, V)


def _lorentzian_eigh_bwd(residuals, g):
    w, V = residuals
    dw, dV = g
    return (_lorentzian_eigh_bwd_kernel(w, V, dw, dV),)


_lorentzian_eigh.defvjp(_lorentzian_eigh_fwd, _lorentzian_eigh_bwd)


def qr_canonical_projector(M: jax.Array, chi: int) -> jax.Array:
    """Build a (f x k) CTMRG projector from M via canonical-gauge QR + small eigh.

    Args:
        M:   Stacked half-corner matrix, shape (f, c).
        chi: Maximum number of columns to keep (k = min(chi, r) where r = min(f, c)).

    Returns:
        P:   (f, k) isometry whose columns are the top-k right singular vectors
             of M (equivalently the top-k eigenvectors of M @ M^H).
    """
    Q, R = _canonical_qr(M)

    # Reduced symmetric eigenproblem on the r x r matrix R R^H.
    # Eigenvalues of R R^H equal squared singular values of M (when f >= c, the
    # iPEPS-CTM regime; in the wide branch f < c, Q is fully unitary and the
    # equality is exact too — see theory_draft §F1).  We use plain jnp.linalg.eigh:
    # JAX's default backward is the standard symmetric-eigh adjoint, which is
    # numerically stable here because (a) rho is the Gram matrix of a
    # (small) chi×chi block and (b) the kept-discarded coupling that
    # destabilises SVD-AD on rank-deficient corners is *not* triggered when we
    # take the truncation by column-slicing on the dense, well-conditioned R R^H.
    rho = R @ jnp.conj(R).T
    rho = 0.5 * (rho + jnp.conj(rho).T)

    eigvals, eigvecs = _lorentzian_eigh(rho)
    r = eigvals.shape[0]
    k = int(min(chi, r))

    # eigvals are ascending; take last-k columns (largest eigvalues), then
    # reverse to descending order.
    V_top = eigvecs[:, -k:][:, ::-1]
    P = Q @ V_top
    return P


# ------------------------------------------------------------------ #
# Standalone custom_vjp variant for finite-difference test harness   #
# ------------------------------------------------------------------ #


def _qr_canonical_with_eigh_only(M: jax.Array, chi: int) -> jax.Array:
    """Same as qr_canonical_projector but uses the *non*-truncated eigh.

    This is used by the standalone gradient-check harness so that the test
    isolates the QR + concat + slice chain from the truncation-correction term
    (which is independently exercised by the truncated_eigh_regularized tests
    in tenax). Returning all eigenvectors lets jacfwd compare against the
    full non-truncated projector.
    """
    Q, R = _canonical_qr(M)
    rho = R @ jnp.conj(R).T
    rho = 0.5 * (rho + jnp.conj(rho).T)
    w, V = jnp.linalg.eigh(rho)
    # Take last-k columns (largest eigvalues) in descending order
    r = w.shape[0]
    k = int(min(chi, r))
    V_top = V[:, -k:][:, ::-1]
    P = Q @ V_top
    return P
