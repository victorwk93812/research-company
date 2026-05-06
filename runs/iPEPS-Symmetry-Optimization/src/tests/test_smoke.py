"""Quick smoke tests — run before the headline benchmark."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _resource_block  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp

from ctmrg_qr import qr_canonical_projector, _qr_canonical_with_eigh_only
import ctmrg_qr_register

ctmrg_qr_register.register_qr_canonical()


def test_projector_isometry():
    """P^H P should be the identity to machine precision."""
    rng = np.random.default_rng(0)
    M = jnp.asarray(rng.standard_normal((20, 12)).astype(np.float64))
    P = qr_canonical_projector(M, 8)
    PHP = jnp.conj(P).T @ P
    assert jnp.allclose(PHP, jnp.eye(8), atol=1e-10), \
        f"P^H P deviates from I by {jnp.max(jnp.abs(PHP - jnp.eye(8))):.3e}"
    print("[OK] isometry P^H P == I")


def test_projector_matches_svd():
    """P @ P^H should equal the rank-k projector built via SVD."""
    rng = np.random.default_rng(1)
    M = jnp.asarray(rng.standard_normal((20, 12)).astype(np.float64))
    k = 6
    P_qr = qr_canonical_projector(M, k)
    U, _S, _Vh = jnp.linalg.svd(M, full_matrices=False)
    P_svd = U[:, :k]
    # The two projectors are equal up to a unitary on the column space — compare
    # the rank-k orthogonal projectors P P^H, which are basis-independent.
    diff = jnp.max(jnp.abs(P_qr @ P_qr.conj().T - P_svd @ P_svd.conj().T))
    assert diff < 1e-10, f"rank-k projectors disagree by {diff:.3e}"
    print(f"[OK] QR-canonical projector matches SVD top-{k} projector (diff={diff:.2e})")


def test_gradient_finite_and_finite_difference():
    """jacfwd vs finite-difference for the non-truncated chain (no kept/discarded gap)."""
    rng = np.random.default_rng(2)
    M = jnp.asarray(rng.standard_normal((10, 6)).astype(np.float64))
    k = 6  # take all eigenvectors so finite-difference is meaningful

    def loss(M):
        P = _qr_canonical_with_eigh_only(M, k)
        # Scalar functional: Frobenius norm of P @ P^H @ M
        return jnp.sum(jnp.abs(P @ (jnp.conj(P).T @ M)))

    g_ad = jax.grad(loss)(M)
    # Central finite difference for a few entries
    eps = 1e-6
    diffs = []
    for (i, j) in [(0, 0), (1, 2), (3, 5), (7, 4)]:
        Mp = M.at[i, j].set(M[i, j] + eps)
        Mm = M.at[i, j].set(M[i, j] - eps)
        g_fd = (loss(Mp) - loss(Mm)) / (2 * eps)
        diffs.append(abs(float(g_ad[i, j]) - float(g_fd)))
    max_diff = max(diffs)
    print(f"[OK] AD vs FD gradient diff max = {max_diff:.3e} (entries probed: 4)")
    assert max_diff < 5e-5, diffs
    assert jnp.all(jnp.isfinite(g_ad)), "AD gradient is not finite"


def test_projector_complex():
    """Complex M: P P^H should still match the SVD top-k."""
    rng = np.random.default_rng(3)
    M_real = rng.standard_normal((12, 8))
    M_imag = rng.standard_normal((12, 8))
    M = jnp.asarray(M_real + 1j * M_imag).astype(jnp.complex128)
    k = 5
    P_qr = qr_canonical_projector(M, k)
    U, _, _ = jnp.linalg.svd(M, full_matrices=False)
    diff = jnp.max(jnp.abs(P_qr @ jnp.conj(P_qr).T - U[:, :k] @ jnp.conj(U[:, :k]).T))
    assert diff < 1e-10, diff
    print(f"[OK] Complex P P^H matches top-{k} SVD projector (diff={diff:.2e})")


if __name__ == "__main__":
    test_projector_isometry()
    test_projector_matches_svd()
    test_gradient_finite_and_finite_difference()
    test_projector_complex()
    print("\nAll smoke tests passed.")
