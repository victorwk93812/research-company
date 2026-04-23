"""Ladder-graph topology for the 2026 NCCU challenge QPU.

The ladder has length ``L`` (the number of rungs' columns, i.e. the top-leg
chain has ``L+1`` qubits ``e0, u_1, ..., u_{L-1}, e_1``). ``N = L - 1`` is the
number of interior top-leg qubits.

We use a canonical global qubit layout:

    index 0          -> e_0
    index 1 .. L-1   -> u_1 .. u_{L-1}
    index L          -> e_1
    index L+1 .. 2L+1 -> v_0 .. v_L

and expose a dictionary ``qidx(L)`` mapping logical names to integer indices.
Every two-qubit gate used by a protocol must operate on a pair of integer
indices that corresponds to an allowed edge in :func:`allowed_edges`.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple


def qubit_names(L: int) -> List[str]:
    if L < 1:
        raise ValueError("L must be >= 1")
    top = ["e0"] + [f"u{i}" for i in range(1, L)] + ["e1"]
    bot = [f"v{j}" for j in range(L + 1)]
    return top + bot


def qidx(L: int) -> Dict[str, int]:
    return {n: i for i, n in enumerate(qubit_names(L))}


def n_qubits(L: int) -> int:
    return 2 * L + 2


def allowed_edges(L: int) -> FrozenSet[Tuple[int, int]]:
    """Return the set of allowed 2Q-gate edges as unordered pairs of indices.

    - Top leg: (e_0, u_1), (u_i, u_{i+1}) for i=1..L-2, (u_{L-1}, e_1).
    - Bottom leg: (v_j, v_{j+1}) for j=0..L-1.
    - Rungs: (u_i, v_i) for i=1..L-1.
    """
    idx = qidx(L)

    edges: List[Tuple[int, int]] = []

    # Top leg
    if L == 1:
        edges.append((idx["e0"], idx["e1"]))
    else:
        edges.append((idx["e0"], idx["u1"]))
        for i in range(1, L - 1):
            edges.append((idx[f"u{i}"], idx[f"u{i + 1}"]))
        edges.append((idx[f"u{L - 1}"], idx["e1"]))

    # Bottom leg
    for j in range(L):
        edges.append((idx[f"v{j}"], idx[f"v{j + 1}"]))

    # Rungs: only at inner columns i=1..L-1
    for i in range(1, L):
        edges.append((idx[f"u{i}"], idx[f"v{i}"]))

    return frozenset(tuple(sorted(e)) for e in edges)


def top_leg_chain(L: int) -> List[int]:
    """Ordered list of top-leg qubit indices e_0, u_1, ..., u_{L-1}, e_1."""
    ix = qidx(L)
    names = ["e0"] + [f"u{i}" for i in range(1, L)] + ["e1"]
    return [ix[n] for n in names]
