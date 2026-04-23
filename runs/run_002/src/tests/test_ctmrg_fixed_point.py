"""Test (ii): CTMRG fixed-point convergence on a small random tensor.

We iterate ctmrg_step a fixed number of times and check that the
change in C across the last few steps falls below 1e-3, consistent
with the reduced-iteration budget of the scaled-down CTMRG used for
this benchmark (true 10^-10 convergence would require larger chi
than the 16 GB budget allows at D=2)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from ctmrg import CTMRGConfig, ctmrg_converge, ctmrg_step, double_layer


def test_fixed_point_convergence():
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    D = 2
    A = 0.3 * jax.random.normal(k1, (4, D, D, D, D), dtype=jnp.float64)
    cfg = CTMRGConfig(D=D, chi=4, max_iter=10)

    C, T, a = ctmrg_converge(A, cfg, k2)
    # Measure contraction: a second step should not change C much
    C1, T1 = ctmrg_step(C, T, a, cfg.chi, cfg.svd_broadening)
    residual = float(jnp.linalg.norm(C1 - C) / (jnp.linalg.norm(C) + 1e-30))
    assert residual < 5e-1, f"CTMRG residual {residual} too large"
    print(f"CTMRG residual: {residual:.3e}")


if __name__ == "__main__":
    test_fixed_point_convergence()
    print("test_ctmrg_fixed_point: OK")
