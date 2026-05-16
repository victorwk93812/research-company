"""Render circuit diagrams for the three baseline protocols at L=5.

Outputs (in ``analysis/figures/``):

  * ``circuit_swap_chain_L5.pdf``    — 3-CNOT SWAP-chain teleportation.
  * ``circuit_cat_chain_L5.pdf``     — GHZ-cat chain + X-basis disentangle.
  * ``circuit_cluster_ladder_L5.pdf`` — 1D cluster state on the top leg
    + X-basis measurements.

Each circuit is rebuilt on a top-leg-only register of size ``L+1`` so the
figure shows only ``e_0, u_1, ..., u_{L-1}, e_1`` (no idle bottom-leg
wires). The gate sequences mirror the corresponding ``build_circuit``
functions in ``swap_chain.py``, ``cat_chain.py``, and ``cluster_ladder.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.visualization import circuit_drawer


def _topleg_swap_chain(L: int) -> QuantumCircuit:
    """Top-leg-only redraw of ``swap_chain.build_circuit(L)``."""
    qr = QuantumRegister(L + 1, "q")
    cr = ClassicalRegister(1, "m")
    qc = QuantumCircuit(qr, cr)
    chain = list(range(L + 1))
    e0, e1 = chain[0], chain[-1]
    N = L - 1

    if N == 0:
        qc.h(e0)
        qc.cx(e0, e1)
        return qc

    qc.h(e0)
    qc.cx(e0, chain[1])
    qc.barrier()
    for k in range(1, N + 1):
        a, b = chain[k], chain[k + 1]
        qc.cx(a, b)
        qc.cx(b, a)
        qc.cx(a, b)
        qc.barrier()
    return qc


def _topleg_cat_chain(L: int) -> QuantumCircuit:
    """Top-leg-only redraw of ``cat_chain.build_circuit(L)``."""
    qr = QuantumRegister(L + 1, "q")
    N = L - 1
    cr = ClassicalRegister(max(1, N), "m")
    qc = QuantumCircuit(qr, cr)
    chain = list(range(L + 1))
    e0, e1 = chain[0], chain[-1]

    if N == 0:
        qc.h(e0)
        qc.cx(e0, e1)
        return qc

    qc.h(e0)
    for k in range(L):
        qc.cx(chain[k], chain[k + 1])
    qc.barrier()

    for j in range(1, N + 1):
        qc.h(chain[j])
        qc.measure(chain[j], cr[j - 1])
    qc.barrier()

    for j in range(1, N + 1):
        with qc.if_test((cr[j - 1], 1)):
            qc.z(e1)
    return qc


def _topleg_cluster_ladder(L: int) -> QuantumCircuit:
    """Top-leg-only redraw of ``cluster_ladder.build_circuit(L)``."""
    qr = QuantumRegister(L + 1, "q")
    N = L - 1
    cr = ClassicalRegister(max(1, N), "m")
    qc = QuantumCircuit(qr, cr)
    chain = list(range(L + 1))
    e1 = chain[-1]

    if N == 0:
        qc.h(chain[0])
        qc.cx(chain[0], chain[1])
        return qc

    for q in chain:
        qc.h(q)
    for k in range(L):
        qc.cz(chain[k], chain[k + 1])
    qc.barrier()

    for j in range(1, N + 1):
        qc.h(chain[j])
        qc.measure(chain[j], cr[j - 1])
    qc.barrier()

    for j in range(1, N + 1):
        with qc.if_test((cr[j - 1], 1)):
            qc.z(e1)
    return qc


def _save(qc: QuantumCircuit, path: Path, *, scale: float = 0.85) -> None:
    fig = circuit_drawer(
        qc,
        output="mpl",
        style={"backgroundcolor": "#FFFFFF"},
        fold=-1,
        idle_wires=False,
        scale=scale,
        plot_barriers=True,
    )
    fig.savefig(path, bbox_inches="tight", format="pdf")


def main() -> None:
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)

    L = 5
    pairs = [
        ("swap_chain", _topleg_swap_chain(L), 0.70),
        ("cat_chain", _topleg_cat_chain(L), 0.80),
        ("cluster_ladder", _topleg_cluster_ladder(L), 0.80),
    ]
    for name, qc, scale in pairs:
        target = out / f"circuit_{name}_L{L}.pdf"
        _save(qc, target, scale=scale)
        print("wrote", target)


if __name__ == "__main__":
    main()
