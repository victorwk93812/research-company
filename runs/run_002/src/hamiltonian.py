"""Single-band Hubbard Hamiltonian: on-site basis, two-site gates.

Basis convention (4-dim physical leg, U(1) charge q in {0,1,1,2}):
    0: |vac>            n=0  Sz=0
    1: |up>             n=1  Sz=+1/2
    2: |down>           n=1  Sz=-1/2
    3: |up down>        n=2  Sz=0

We work with the spin-less-resolved 4-dim physical Hilbert space and
build two-site hopping + on-site interaction gates. Jordan-Wigner signs
are handled via explicit fermionic sign matrices on the 4x4 physical
space: the fermion-parity operator P = diag(+1,-1,-1,+1).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# ---------- on-site operators -----------------------------------------------

I4 = jnp.eye(4, dtype=jnp.float64)

# fermion parity: (+, -, -, +) on (vac, up, down, updown)
P = jnp.array(np.diag([1.0, -1.0, -1.0, 1.0]), dtype=jnp.float64)

# c_up (annihilation):
#   |vac> <- c_up |up>; |down> <- c_up |updown>.
# Nonzero matrix elements in row,col = (final,initial):
#   <vac|c_up|up> = 1,   <down|c_up|updown> = +1
_c_up = np.zeros((4, 4))
_c_up[0, 1] = 1.0
_c_up[2, 3] = 1.0  # JW sign absorbed into the two-site gate via P below
C_UP = jnp.array(_c_up, dtype=jnp.float64)

# c_dn:
#   <vac|c_dn|down> = 1,   <up|c_dn|updown> = -1 (Fock ordering |up,down>)
_c_dn = np.zeros((4, 4))
_c_dn[0, 2] = 1.0
_c_dn[1, 3] = -1.0
C_DN = jnp.array(_c_dn, dtype=jnp.float64)

N_UP = C_UP.T @ C_UP
N_DN = C_DN.T @ C_DN
N_TOT = N_UP + N_DN
DOUBLE = N_UP @ N_DN


@dataclass(frozen=True)
class HubbardParams:
    U: float = 8.0
    t: float = 1.0
    t_prime: float = 0.0
    mu: float = 0.0


def two_site_hopping_gate(params: HubbardParams) -> jnp.ndarray:
    """Return the 16x16 two-site hopping + half-distributed on-site
    interaction gate. The on-site term is split as U/2 per site and
    applied once per bond by counting the coordination number externally.
    """
    # H_bond = -t sum_sigma (c^dag_{i,s} c_{j,s} + h.c.)
    #        + (U/4) (n_i_up n_i_dn + n_j_up n_j_dn) distributed per bond
    # Jordan-Wigner: fermionic hop across a single nearest-neighbour bond
    # picks up a P on the destination site's leg.
    kron = lambda A, B: jnp.kron(A, B)
    hop_up = -params.t * (
        kron(C_UP.T @ P, C_UP) + kron(P @ C_UP, C_UP.T)
    )
    hop_dn = -params.t * (
        kron(C_DN.T @ P, C_DN) + kron(P @ C_DN, C_DN.T)
    )
    # On-site U split across the 4 NN bonds of the 2D square lattice:
    u_term = (params.U / 8.0) * (kron(DOUBLE, I4) + kron(I4, DOUBLE))
    mu_term = -(params.mu / 8.0) * (kron(N_TOT, I4) + kron(I4, N_TOT))
    return hop_up + hop_dn + u_term + mu_term


def site_charge_op() -> jnp.ndarray:
    """Total particle number operator on the 4-dim physical leg."""
    return N_TOT


def site_sz_op() -> jnp.ndarray:
    """Sz operator on the 4-dim physical leg."""
    return 0.5 * (N_UP - N_DN)
