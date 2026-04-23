"""Hamiltonian definitions for TFIM, Heisenberg, and J1-J2 on the square lattice.

Each Hamiltonian exposes nearest-neighbour (and, for J1-J2, next-nearest-neighbour)
two-site operators. Energy evaluation contracts them against an iPEPS environment
in ctmrg.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Pauli and spin matrices
_sx = jnp.array([[0.0, 1.0], [1.0, 0.0]])
_sy = jnp.array([[0.0, -1.0j], [1.0j, 0.0]])
_sz = jnp.array([[1.0, 0.0], [0.0, -1.0]])
_id = jnp.eye(2)

# Spin-1/2 operators
_Sx = 0.5 * _sx
_Sy = 0.5 * _sy
_Sz = 0.5 * _sz


@dataclass
class NNHamiltonian:
    """A Hamiltonian with nearest- and (optionally) next-nearest-neighbour 2-site terms.

    Fields
    ------
    h_nn : (4, 4) complex — 2-site nearest-neighbour operator (on horizontal + vertical bonds)
    h_nnn : (4, 4) complex or None — 2-site next-nearest-neighbour operator (diagonals)
    h_onsite : (2, 2) real — optional on-site term (only used for TFIM transverse field)
    """
    h_nn: jax.Array
    h_nnn: jax.Array | None
    h_onsite: jax.Array | None


def tfim(h_over_j: float = 3.04, J: float = 1.0) -> NNHamiltonian:
    """Transverse-field Ising model H = -J sz sz - h sx."""
    h_nn = -J * jnp.kron(_sz, _sz)  # (4,4) complex promotion below
    return NNHamiltonian(
        h_nn=h_nn.astype(jnp.complex128),
        h_nnn=None,
        h_onsite=(-h_over_j * J * _sx).astype(jnp.complex128),
    )


def heisenberg(J: float = 1.0) -> NNHamiltonian:
    """Isotropic Heisenberg antiferromagnet J S.S."""
    h_nn = J * (jnp.kron(_Sx, _Sx) + jnp.kron(_Sy, _Sy) + jnp.kron(_Sz, _Sz))
    return NNHamiltonian(
        h_nn=h_nn.astype(jnp.complex128),
        h_nnn=None,
        h_onsite=None,
    )


def j1j2(j1: float = 1.0, j2_over_j1: float = 0.5) -> NNHamiltonian:
    """J1-J2 Heisenberg. NN bonds carry J1 SS; diagonals carry J2 SS."""
    h_s = jnp.kron(_Sx, _Sx) + jnp.kron(_Sy, _Sy) + jnp.kron(_Sz, _Sz)
    return NNHamiltonian(
        h_nn=(j1 * h_s).astype(jnp.complex128),
        h_nnn=(j2_over_j1 * j1 * h_s).astype(jnp.complex128),
        h_onsite=None,
    )
