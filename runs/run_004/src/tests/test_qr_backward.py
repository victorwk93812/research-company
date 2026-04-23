"""Unit test: QR backward matches jax.jacfwd on a random 64 x 64 matrix."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import resource_cap  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_qr_backward_matches_jacfwd():
    """The QR gradient produced by jax.grad agrees with jax.jacfwd via a scalar loss."""
    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (64, 64))

    def scalar_loss(X):
        Q, R = jnp.linalg.qr(X, mode="reduced")
        return jnp.sum(Q * Q) + jnp.sum(R * R)

    grad_rev = jax.grad(scalar_loss)(M)
    # Compute same gradient via forward-mode Jacobian (slower but independent)
    jac = jax.jacfwd(scalar_loss)(M)
    err = float(jnp.linalg.norm(grad_rev - jac) / jnp.linalg.norm(jac))
    print(f"[qr_backward] jax.grad vs jax.jacfwd relative error = {err:.3e}")
    assert err < 1e-10, f"too large: {err}"


def test_qr_backward_no_gap_divergence():
    """Put two columns of M near-parallel (→ near-singular R).
    QR backward should remain bounded; SVD backward would diverge."""
    key = jax.random.PRNGKey(7)
    M = jax.random.normal(key, (32, 4))
    # Make columns 0 and 1 near-identical
    M = M.at[:, 1].set(M[:, 0] + 1e-6 * jax.random.normal(jax.random.PRNGKey(8), (32,)))

    def loss_qr(X):
        Q, R = jnp.linalg.qr(X, mode="reduced")
        return jnp.sum(Q * Q)

    g = jax.grad(loss_qr)(M)
    n = float(jnp.linalg.norm(g))
    print(f"[qr_backward_no_gap] gradient norm = {n:.3e} (should be finite)")
    assert jnp.isfinite(n), "QR backward diverged"
    assert n < 1e8, f"QR backward suspiciously large: {n}"


if __name__ == "__main__":
    test_qr_backward_matches_jacfwd()
    test_qr_backward_no_gap_divergence()
    print("QR-backward unit tests PASSED.")
