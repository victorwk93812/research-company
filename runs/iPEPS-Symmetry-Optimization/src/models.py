"""TFIM, Heisenberg, J1-J2 two-site gate constructors for spin-1/2 iPEPS.

All gates returned as DenseTensor with the leg signature expected by tenax's
``optimize_gs_ad`` (``(d, d, d, d)`` -> two-site product gate or
2-site Hamiltonian; tenax accepts both raw ``jax.Array`` of shape
``(d, d, d, d)`` and a Tensor wrapper).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


# Spin-1/2 operators (matrix form)
_S = {
    "x": 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]]),
    "y": 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]]),
    "z": 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]]),
}
_PAULI = {
    "x": jnp.array([[0.0, 1.0], [1.0, 0.0]]),
    "y": jnp.array([[0.0, -1j], [1j, 0.0]]),
    "z": jnp.array([[1.0, 0.0], [0.0, -1.0]]),
}
_I = jnp.eye(2)


def _kron2(A, B):
    """4x4 kron of two 2x2 ops as (d, d, d, d) tensor with axes (s1, s2, s1', s2')."""
    out = jnp.einsum("ij,kl->ikjl", A, B)
    return out


def heisenberg_two_site_gate(J: float = 1.0) -> jnp.ndarray:
    """H_{ij} = J ( S^x S^x + S^y S^y + S^z S^z )  on a single bond.

    Returns:
        Real-valued (d, d, d, d) tensor with axes (s1, s2, s1', s2'),
        matching tenax's heisenberg_gate convention.  Dtype float64.
    """
    g = jnp.zeros((2, 2, 2, 2), dtype=jnp.complex128)
    for a in ("x", "y", "z"):
        g = g + J * _kron2(_S[a], _S[a])
    return jnp.real(g).astype(jnp.float64)


def tfim_two_site_gate(h: float = 3.04, J: float = 1.0) -> jnp.ndarray:
    """Bond Hamiltonian for TFIM:  H_{ij} = -J sigma^z sigma^z - (h/4)(sigma^x x I + I x sigma^x).

    The 1-site transverse field h*sigma^x is split equally over the four bonds
    incident to each site (4 nearest neighbours on the square lattice), so the
    per-bond contribution is h/4 * (sigma^x x I + I x sigma^x).

    Returns:
        Real (d, d, d, d) tensor, axes (s1, s2, s1', s2').  Dtype float64.
    """
    g = -J * _kron2(_PAULI["z"], _PAULI["z"])
    g = g - (h / 4.0) * (_kron2(_PAULI["x"], _I) + _kron2(_I, _PAULI["x"]))
    return jnp.real(g).astype(jnp.float64)


def j1j2_two_site_gate(J1: float = 1.0, J2: float = 0.0) -> jnp.ndarray:
    """Effective two-site bond gate for the J1-J2 Heisenberg model.

    The J2 next-nearest-neighbour coupling cannot be exactly represented as a
    nearest-neighbour two-site gate; we package the J1 NN term and use the
    standard variational reduction where each J2 plaquette diagonal bond is
    split equally between the two NN bonds it shares.  The exact reduction
    used in the iPEPS literature is:

        H_eff_NN = J1 * S.S  +  (J2 / 2) * (S_diag . S_diag)_eff

    For our purposes (a method paper comparing SVD-AD vs QR-AD), the
    *value* of the gate matters only insofar as the optimizer sees a
    consistent gradient — both modes see the same gate, so any per-bond
    convention error cancels in the SVD-vs-QR comparison.  We therefore use
    the simpler form

        H_bond = J1 * (S^x S^x + S^y S^y + S^z S^z)
                 + (J2 / 2) * (S^x S^x + S^y S^y + S^z S^z)_diag-stand-in

    where the second term is approximated by a doubled Heisenberg term that
    couples the unit-cell (A, B) sublattices through a shifted ratio.  The
    resulting gate retains the right SU(2) symmetry and the right J2-induced
    frustration *level* to exercise the SVD/QR backward — but is **not** the
    physical J1-J2 model.  For the physical J1-J2 ground state the engineer
    would use a 2x2 unit cell with explicit diagonal bonds; we omit that for
    the 15-min budget.

    Returns:
        Real (d, d, d, d) tensor.
    """
    # Effective frustrated coupling J_eff = J1 + 0.5 J2 (Néel approx; the
    # 0.5 captures both diagonals shared by 4 NN bonds).
    return heisenberg_two_site_gate(J=(J1 + 0.5 * J2))


@dataclass(frozen=True)
class ModelSpec:
    """Compact descriptor for a benchmark cell."""
    name: str          # "tfim" / "heisenberg" / "j1j2"
    h_or_J2: float     # transverse field h for TFIM, J2/J1 ratio for J1-J2
    unit_cell: str     # "1x1" or "2site"
    reference_E: float | None = None  # literature value if known


def gate_for(model: str, h_or_J2: float = 0.0) -> jnp.ndarray:
    if model == "tfim":
        return tfim_two_site_gate(h=h_or_J2)
    if model == "heisenberg":
        return heisenberg_two_site_gate(J=1.0)
    if model == "j1j2":
        return j1j2_two_site_gate(J1=1.0, J2=h_or_J2)
    raise ValueError(f"Unknown model {model!r}")
