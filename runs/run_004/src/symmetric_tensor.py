"""C4v symmetrization utilities for iPEPS site tensors."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def symmetrize_c4v(A: jax.Array) -> jax.Array:
    """Project a rank-5 iPEPS tensor A[phys, up, right, down, left] onto the C4v-symmetric subspace.

    C4v has 8 elements: 4 rotations + 4 reflections. We enforce
    A to be invariant under all of them.
    """
    # Rotation by 90 deg: (u, r, d, l) -> (r, d, l, u)
    # Reflection (horizontal): (u, r, d, l) -> (u, l, d, r)
    # Our convention: axis order (phys, u, r, d, l).
    def rot(t):  # 90 CCW
        return jnp.transpose(t, (0, 2, 3, 4, 1))

    def refl(t):
        return jnp.transpose(t, (0, 1, 4, 3, 2))

    out = A
    # Add all 4 rotations
    r1 = rot(A)
    r2 = rot(r1)
    r3 = rot(r2)
    out = (A + r1 + r2 + r3) / 4.0
    # Symmetrize with reflection
    out = (out + refl(out)) / 2.0
    return out
