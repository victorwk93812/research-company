"""Unit test: CTMRG fixed-point residual decays below a threshold on a gapped point."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import resource_cap  # noqa: F401
import jax
import jax.numpy as jnp

from ctmrg import run_ctmrg, fixed_point_residual, double_layer


def test_fixed_point_residual_converges():
    """Run CTMRG for 40 steps on a gapped-phase random tensor and check residual drops."""
    key = jax.random.PRNGKey(0)
    d, D, chi = 2, 2, 4
    A = jax.random.normal(key, (d, D, D, D, D))
    # Bias A toward a gapped product-state-like tensor for stability
    A = A + 5.0 * jnp.ones_like(A) * (jnp.arange(A.size).reshape(A.shape) == 0)
    state = run_ctmrg(A, chi=chi, n_steps=30, projector="qr")
    a = double_layer(A)
    r = float(fixed_point_residual(state, a, chi=chi, projector="qr"))
    print(f"[fixed_point] residual after 30 steps = {r:.3e}")
    assert r < 1e-3, f"fixed-point residual too high: {r}"


if __name__ == "__main__":
    test_fixed_point_residual_converges()
    print("CTMRG fixed-point test PASSED.")
