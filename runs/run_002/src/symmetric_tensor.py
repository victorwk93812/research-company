"""Block-sparse U(1)-symmetric tensor, registered as a JAX pytree.

Representation
--------------
A symmetric tensor is a dict keyed by a tuple of charges, one charge per leg,
whose values are dense jax.numpy arrays. The dict is wrapped in a
``SymTensor`` dataclass that is registered as a pytree node; gradients
through `jax.grad` propagate through the blocks. The leg signatures
(``+1`` = ingoing charge, ``-1`` = outgoing) and the total charge ``Q``
are treated as static metadata (not traced).

For the iPEPS site tensor of the Hubbard model with U(1)_c charge, the
physical leg carries charges ``(0, +1, +1, +2)`` for basis states
``(|0>, |up>, |down>, |up down>)``. In practice we use a small virtual
bond-charge alphabet ``(-1, 0, +1)`` with user-controlled multiplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

Charges = Tuple[int, ...]
BlockDict = Dict[Charges, jnp.ndarray]


@dataclass(frozen=True)
class SymTensorMeta:
    """Static metadata for a symmetric tensor (not differentiated)."""

    leg_sign: Tuple[int, ...]
    total_charge: int
    leg_charge_sectors: Tuple[Tuple[int, ...], ...]
    leg_sector_dims: Tuple[Tuple[int, ...], ...]


class SymTensor:
    """U(1)-symmetric dense-block tensor, JAX pytree."""

    __slots__ = ("blocks", "meta")

    def __init__(self, blocks: BlockDict, meta: SymTensorMeta) -> None:
        self.blocks = dict(blocks)
        self.meta = meta

    def tree_flatten(self):
        keys = sorted(self.blocks.keys())
        children = tuple(self.blocks[k] for k in keys)
        aux = (tuple(keys), self.meta)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        keys, meta = aux
        return cls(dict(zip(keys, children)), meta)

    def dense(self) -> jnp.ndarray:
        """Materialise as a dense tensor for testing only."""
        shape = tuple(sum(d) for d in self.meta.leg_sector_dims)
        out = jnp.zeros(shape, dtype=_dtype_of_blocks(self.blocks))
        for key, block in self.blocks.items():
            slices = []
            for leg_idx, charge in enumerate(key):
                sectors = self.meta.leg_charge_sectors[leg_idx]
                dims = self.meta.leg_sector_dims[leg_idx]
                offset = 0
                found = False
                for c, d in zip(sectors, dims):
                    if c == charge:
                        slices.append(slice(offset, offset + d))
                        found = True
                        break
                    offset += d
                if not found:
                    raise KeyError(f"charge {charge} not in leg {leg_idx}")
            out = out.at[tuple(slices)].set(block)
        return out

    def norm(self) -> jnp.ndarray:
        acc = jnp.array(0.0, dtype=jnp.float64)
        for block in self.blocks.values():
            acc = acc + jnp.sum(block * block)
        return jnp.sqrt(acc)


jax.tree_util.register_pytree_node_class(SymTensor)


def _dtype_of_blocks(blocks: BlockDict) -> jnp.dtype:
    if not blocks:
        return jnp.float64
    return next(iter(blocks.values())).dtype


def allowed_charge_keys(
    leg_sign: Tuple[int, ...],
    leg_charges: Tuple[Tuple[int, ...], ...],
    total_charge: int,
) -> Tuple[Charges, ...]:
    """Enumerate all charge tuples whose signed sum equals total_charge."""
    out = []
    if not leg_charges:
        return ((),)
    head_sign, *tail_signs = leg_sign
    head_charges, *tail_charges = leg_charges

    def rec(i: int, acc: int, cur: Tuple[int, ...]):
        if i == len(leg_sign):
            if acc == total_charge:
                out.append(tuple(cur))
            return
        for c in leg_charges[i]:
            rec(i + 1, acc + leg_sign[i] * c, cur + (c,))

    rec(0, 0, tuple())
    return tuple(out)


def zeros(meta: SymTensorMeta) -> SymTensor:
    """All-allowed blocks initialised to zeros."""
    keys = allowed_charge_keys(meta.leg_sign, meta.leg_charge_sectors, meta.total_charge)
    blocks: BlockDict = {}
    for k in keys:
        shape = tuple(
            meta.leg_sector_dims[i][meta.leg_charge_sectors[i].index(c)]
            for i, c in enumerate(k)
        )
        blocks[k] = jnp.zeros(shape, dtype=jnp.float64)
    return SymTensor(blocks, meta)


def random_symmetric(
    meta: SymTensorMeta, key: jax.Array, scale: float = 1.0
) -> SymTensor:
    """Draw a random symmetric tensor with independent Gaussian blocks."""
    keys = allowed_charge_keys(meta.leg_sign, meta.leg_charge_sectors, meta.total_charge)
    blocks: BlockDict = {}
    subkeys = jax.random.split(key, max(1, len(keys)))
    for k, sk in zip(keys, subkeys):
        shape = tuple(
            meta.leg_sector_dims[i][meta.leg_charge_sectors[i].index(c)]
            for i, c in enumerate(k)
        )
        blocks[k] = scale * jax.random.normal(sk, shape, dtype=jnp.float64)
    return SymTensor(blocks, meta)


def symmetry_fidelity(t: SymTensor) -> float:
    """All blocks conform to the allowed charge tuples by construction; any
    numerical leakage would appear only if the user manually injected
    off-sector data. Returns 1.0 - leaked_norm / total_norm."""
    allowed = set(allowed_charge_keys(t.meta.leg_sign, t.meta.leg_charge_sectors, t.meta.total_charge))
    leaked = 0.0
    total = 0.0
    for k, b in t.blocks.items():
        n = float(jnp.sum(b * b))
        total += n
        if k not in allowed:
            leaked += n
    if total == 0.0:
        return 1.0
    return 1.0 - leaked / total
