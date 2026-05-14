"""Slides-only depth-vs-L plot, counting *quantum-gate depth only*.

The default ``QuantumCircuit.depth()`` in Qiskit counts measurement
instructions and classically-controlled (feed-forward) Pauli corrections
as separate circuit layers. For the entanglement-swap protocol this
makes the plotted depth grow linearly with the number of branches even
though the *quantum* part — the only part that matters for coherence-time
budgets and Lieb-Robinson arguments — is constant.

This script reuses the protocol builders from ``scaling_benchmark.py``
but counts depth with a filter that excludes:

  * ``measure`` instructions, and
  * classical control-flow ops (``IfElseOp``, ``SwitchCaseOp``, etc.).

It writes a fresh figure to ``figures/scaling_quantum.pdf``. The original
``figures/scaling.pdf`` is *not* modified.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from qiskit.circuit.controlflow import ControlFlowOp

from scaling_benchmark import PROTOCOLS  # protocol list + colours


_QUANTUM_EXCLUDED = {"measure", "reset"}


def _is_quantum_layer(instr) -> bool:
    """Return True iff ``instr`` should count toward circuit depth.

    Quantum-only depth excludes measurements, resets, barriers, and any
    classical control-flow block (e.g. ``if_test`` feed-forward corrections).
    """
    op = instr.operation
    name = op.name
    if name == "barrier":
        return False
    if name in _QUANTUM_EXCLUDED:
        return False
    if isinstance(op, ControlFlowOp):
        return False
    return True


def quantum_depth(qc) -> int:
    return qc.depth(filter_function=_is_quantum_layer)


def two_qubit_count(qc) -> int:
    return sum(
        1 for instr in qc.data
        if instr.operation.name in {"cx", "cz", "swap"}
    )


def main() -> None:
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)

    L_range = list(range(1, 11))
    qdepth = {p.name: [] for p in PROTOCOLS}
    twoq = {p.name: [] for p in PROTOCOLS}

    print(f"{'protocol':<20} {'L':>3} {'q-depth':>8} {'2Q':>5}")
    print("-" * 40)
    for p in PROTOCOLS:
        for L in L_range:
            qc = p.build_circuit(L)
            d = quantum_depth(qc)
            n2 = two_qubit_count(qc)
            qdepth[p.name].append(d)
            twoq[p.name].append(n2)
            print(f"{p.name:<20} {L:>3} {d:>8} {n2:>5}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for p in PROTOCOLS:
        axes[0].plot(L_range, qdepth[p.name], marker="o", label=p.name, color=p.colour)
        axes[1].plot(L_range, twoq[p.name], marker="o", label=p.name, color=p.colour)

    axes[0].set_xlabel("L (ladder length)")
    axes[0].set_ylabel("Quantum-gate depth (measurements excluded)")
    axes[0].set_title("Depth vs L  —  quantum part only")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("L (ladder length)")
    axes[1].set_ylabel("Two-qubit gates")
    axes[1].set_title("Two-qubit gate count vs L")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(outdir, "scaling_quantum.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
