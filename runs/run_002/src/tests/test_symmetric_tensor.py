"""Test (i): symmetric-tensor AD correctness vs dense AD on a small example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from symmetric_tensor import (
    SymTensor,
    SymTensorMeta,
    random_symmetric,
    symmetry_fidelity,
)


def test_pytree_grad_and_dense_agreement():
    meta = SymTensorMeta(
        leg_sign=(+1, +1),
        total_charge=0,
        leg_charge_sectors=((0, 1), (0, 1)),
        leg_sector_dims=((1, 1), (1, 1)),
    )
    key = jax.random.PRNGKey(0)
    t = random_symmetric(meta, key, scale=1.0)
    # loss = ||t||^2
    def loss(t: SymTensor) -> jnp.ndarray:
        return t.norm() ** 2

    g = jax.grad(loss)(t)
    assert isinstance(g, SymTensor), "grad preserves pytree"
    # Dense consistency:
    dense = t.dense()
    expected = 2 * dense
    # Reconstruct grad dense from blocks and compare the Frobenius norm
    norm_blockwise = float(
        sum(jnp.sum(g.blocks[k] ** 2) for k in g.blocks)
    )
    norm_dense = float(jnp.sum(expected * expected))
    assert abs(norm_blockwise - norm_dense) < 1e-10, (norm_blockwise, norm_dense)


def test_symmetry_fidelity_one_for_clean():
    meta = SymTensorMeta(
        leg_sign=(+1, +1),
        total_charge=0,
        leg_charge_sectors=((0, 1), (0, 1)),
        leg_sector_dims=((2, 2), (2, 2)),
    )
    key = jax.random.PRNGKey(1)
    t = random_symmetric(meta, key)
    assert abs(symmetry_fidelity(t) - 1.0) < 1e-12


if __name__ == "__main__":
    test_pytree_grad_and_dense_agreement()
    test_symmetry_fidelity_one_for_clean()
    print("test_symmetric_tensor: OK")
