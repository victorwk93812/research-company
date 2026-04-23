"""AD pipeline: expose the energy functional as a differentiable function
of the site tensor, with explicit selectors for diff mode / SVD backward.

The full B-ICTMRG variant is implemented as the combination:
    diff_mode = "implicit" + svd_backward = "lorentzian-per-block" +
    optimizer = "riemannian-lbfgs" (see optimizers.py).

For the scope of this benchmark we keep the energy functional dense but
parameterise the site tensor in a way that reflects the U(1)_c
charge structure of the Hubbard model.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from ctmrg import CTMRGConfig, two_site_energy


def make_energy_fn(
    H_bond: jnp.ndarray, cfg: CTMRGConfig, key: jax.Array
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a jit-compiled energy(A) function bound to a fixed H and cfg."""

    @jax.jit
    def energy(A: jnp.ndarray) -> jnp.ndarray:
        return two_site_energy(A, H_bond, cfg, key)

    return energy


def project_symmetric(A: jnp.ndarray, charges_leg: jnp.ndarray) -> jnp.ndarray:
    """Zero out entries that violate U(1)_c conservation on the site tensor.

    charges_leg is a 1D array of length `D` listing the charges on each
    virtual index (the physical leg charges are fixed: (0,1,1,2)). An
    entry A[s,l,u,r,d] is kept iff q_phys[s] = q[l]+q[u]+q[r]+q[d] mod
    irrelevant sign conventions (we just require equality of signed sums).
    """
    q_phys = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
    l, u, r, d = jnp.meshgrid(
        charges_leg, charges_leg, charges_leg, charges_leg, indexing="ij"
    )
    virt_sum = l + u + r + d  # (D,D,D,D)
    phys_expand = q_phys[:, None, None, None, None]
    virt_expand = virt_sum[None]
    mask = (phys_expand == virt_expand).astype(A.dtype)
    return A * mask


def project_c4v(A: jnp.ndarray) -> jnp.ndarray:
    """Symmetrise A over the C4 rotations on the virtual legs.

    90 degree rotation: (phys, l, u, r, d) -> (phys, u, r, d, l).
    """
    r0 = A
    r1 = jnp.transpose(A, (0, 2, 3, 4, 1))
    r2 = jnp.transpose(r1, (0, 2, 3, 4, 1))
    r3 = jnp.transpose(r2, (0, 2, 3, 4, 1))
    # Reflection about the horizontal axis: swap u <-> d
    refl = jnp.transpose(A, (0, 1, 4, 3, 2))
    return 0.2 * (r0 + r1 + r2 + r3 + refl)
