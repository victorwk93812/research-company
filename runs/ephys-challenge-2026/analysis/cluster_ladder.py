"""Stretch variant: 1D cluster-state along the top leg, then X-basis-measure
all intermediate qubits.

A 1D cluster state on qubits (e_0, u_1, ..., u_{L-1}, e_1) is prepared by:
  1. Apply H to every qubit.
  2. Apply CZ on every adjacent pair.

Then X-basis-measuring every intermediate qubit teleports the |+> on e_0
through to e_1 via the cluster, producing a Bell-like state after
appropriate Pauli corrections (which are identical to the cat-chain
baseline: a single Z parity on e_1).

This variant uses only top-leg edges — same connectivity profile as our
protocol, different depth/gate tradeoff.
"""

from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from entanglement_swap import _top_chain
from ladder_graph import n_qubits


def build_circuit(L: int) -> QuantumCircuit:
    if L < 1:
        raise ValueError("L must be >= 1")
    N = L - 1

    qr = QuantumRegister(n_qubits(L), "q")
    cr = ClassicalRegister(max(1, N + 1), "m")
    qc = QuantumCircuit(qr, cr, name=f"cluster_ladder_L{L}")

    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]

    if N == 0:
        qc.h(e0)
        qc.cx(e0, e1)
        return qc

    # Cluster-state prep: H on every top-leg qubit, then CZ on every
    # adjacent top-leg edge.
    for q in chain:
        qc.h(q)
    for k in range(L):
        qc.cz(chain[k], chain[k + 1])

    qc.barrier()

    # X-basis measurement on e_0 (to initialise it as part of the teleportation
    # input), plus all intermediate qubits. Actually: to get a Bell state on
    # (e_0, e_1), we only measure the intermediates. The "input state" to the
    # cluster is |+> on e_0 (which we prepared with H). Then measuring the
    # intermediates teleports |+> to e_1; combined with the residual on e_0,
    # we get a Bell-state-like output.
    for j in range(1, N + 1):
        qc.h(chain[j])
        qc.measure(chain[j], cr[j - 1])

    qc.barrier()

    # Correction: Z^{parity} on e_1, similar logic to cat-chain.
    for j in range(1, N + 1):
        with qc.if_test((cr[j - 1], 1)):
            qc.z(e1)

    return qc
