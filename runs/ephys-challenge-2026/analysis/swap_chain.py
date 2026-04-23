"""SWAP-chain baseline: prepare Bell pair on (e_0, u_1), then SWAP it to e_1.

Depth O(N), 3N + 1 CNOTs (1 for the initial Bell pair, 3 per SWAP).
No measurements, no feed-forward.
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
    cr = ClassicalRegister(1, "m")  # unused, but needed for uniform output
    qc = QuantumCircuit(qr, cr, name=f"swap_chain_L{L}")

    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]

    if N == 0:
        qc.h(e0)
        qc.cx(e0, e1)
        return qc

    # Prepare Bell pair on (e_0, u_1) = (chain[0], chain[1]).
    qc.h(e0)
    qc.cx(e0, chain[1])

    # SWAP u_1 <-> u_2 <-> ... <-> u_N <-> e_1. Each SWAP is 3 CNOTs:
    # SWAP(a, b) = CNOT(a, b); CNOT(b, a); CNOT(a, b).
    for k in range(1, N + 1):
        a = chain[k]
        b = chain[k + 1]
        qc.cx(a, b)
        qc.cx(b, a)
        qc.cx(a, b)

    return qc
