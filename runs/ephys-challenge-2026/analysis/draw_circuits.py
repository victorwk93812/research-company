"""Render report-quality circuit diagrams for both Bell-preparation protocols.

Outputs (in ``analysis/figures/``):

  * ``circuit_swap_L5.pdf``  — the v1 entanglement-swap protocol at the
    challenge example ``L=5`` (``N=4`` interior top-leg qubits).
  * ``circuit_mocu_L5.pdf``  — the v2 measurement-free MOCU protocol at the
    same ``L=5`` example.

We draw the circuits using qiskit's matplotlib backend with ``idle_wires=False``
so the unused bottom-leg qubits do not clutter the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer

from entanglement_swap import _top_chain
from ladder_graph import qidx
import mocu


def _topleg_swap_L5() -> QuantumCircuit:
    """Self-contained top-leg-only redraw of the v1 protocol at L=5 (N=4).

    We rebuild the circuit on a 6-qubit register labelled (e0,u1,u2,u3,u4,e1)
    so the figure shows only the qubits that participate in the protocol.
    The gate sequence reproduces ``entanglement_swap.build_circuit(5)``
    after restriction to the top leg.
    """
    qr = QuantumRegister(6, "q")
    cr = ClassicalRegister(4, "m")
    qc = QuantumCircuit(qr, cr)
    e0, u1, u2, u3, u4, e1 = range(6)

    # Step 1: three Bell pairs on alternating links.
    qc.h(e0); qc.h(u2); qc.h(u4)
    qc.cx(e0, u1); qc.cx(u2, u3); qc.cx(u4, e1)
    qc.barrier()

    # Step 2: Bell-measurement unitaries on the two inner pairs.
    qc.cx(u1, u2); qc.cx(u3, u4)
    qc.h(u1); qc.h(u3)
    qc.barrier()

    # Step 3: Z-basis measurements on the four interior qubits.
    qc.measure(u1, cr[0])
    qc.measure(u2, cr[1])
    qc.measure(u3, cr[2])
    qc.measure(u4, cr[3])
    qc.barrier()

    # Step 4: feed-forward Pauli correction X^{m2 XOR m4} Z^{m1 XOR m3} on e1.
    with qc.if_test((cr[0], 1)):
        qc.z(e1)
    with qc.if_test((cr[1], 1)):
        qc.x(e1)
    with qc.if_test((cr[2], 1)):
        qc.z(e1)
    with qc.if_test((cr[3], 1)):
        qc.x(e1)

    return qc


def _topleg_mocu(L: int) -> QuantumCircuit:
    """Top-leg-only redraw of MOCU at length ``L``.

    Mirrors :func:`mocu.build_circuit` but on a register of size ``L+1`` so the
    figure only shows the active qubits ``q0=e0, q1, ..., q_{L-1}, qL=e1``.
    """
    qr = QuantumRegister(L + 1, "q")
    qc = QuantumCircuit(qr)
    chain = list(range(L + 1))

    if L == 1:
        qc.h(chain[0])
        qc.cx(chain[0], chain[1])
        return qc

    m = L // 2
    qc.h(chain[m])

    left_n, right_n = m, L - m
    left_filled, right_filled = 0, 0
    if right_n >= left_n:
        qc.cx(chain[m], chain[m + 1]); right_filled = 1
    else:
        qc.cx(chain[m], chain[m - 1]); left_filled = 1

    while left_filled < left_n or right_filled < right_n:
        if left_filled < left_n:
            qc.cx(chain[m - left_filled], chain[m - left_filled - 1])
            left_filled += 1
        if right_filled < right_n:
            qc.cx(chain[m + right_filled], chain[m + right_filled + 1])
            right_filled += 1

    qc.barrier()

    qc.cx(chain[m - 1], chain[m])
    s = 1
    while True:
        left_ok = (m - s >= 1) and (m - s - 1 >= 0)
        right_ok = (m + s <= L - 1) and (m + s + 1 <= L)
        if not left_ok and not right_ok:
            break
        if left_ok:
            qc.cx(chain[m - s - 1], chain[m - s])
        if right_ok:
            qc.cx(chain[m + s + 1], chain[m + s])
        s += 1

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

    swap = _topleg_swap_L5()
    _save(swap, out / "circuit_swap_L5.pdf", scale=0.75)

    mocu_qc = _topleg_mocu(5)
    _save(mocu_qc, out / "circuit_mocu_L5.pdf", scale=0.85)

    print("wrote", out / "circuit_swap_L5.pdf")
    print("wrote", out / "circuit_mocu_L5.pdf")


if __name__ == "__main__":
    main()
