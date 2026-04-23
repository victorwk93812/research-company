"""GHZ-cat chain baseline: build a chain-GHZ on the whole top leg, then
disentangle the intermediate qubits via X-basis measurement.

  H(e_0); CNOT(e_0 -> u_1); CNOT(u_1 -> u_2); ...; CNOT(u_{L-1} -> e_1)
  H on each inner qubit; Z-basis measurement on each inner qubit (= X-basis
  measurement on that qubit); if XOR of all outcomes is 1, apply Z on e_1.

Depth O(L), N + 1 CNOTs (all sequential because each shares a qubit).
One classical bit for the final correction (the parity XOR).
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
    cr = ClassicalRegister(max(1, N), "m")
    qc = QuantumCircuit(qr, cr, name=f"cat_chain_L{L}")

    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]

    if N == 0:
        qc.h(e0)
        qc.cx(e0, e1)
        return qc

    # Cat-chain: H(e_0), then sequential CNOTs.
    qc.h(e0)
    for k in range(L):
        qc.cx(chain[k], chain[k + 1])

    qc.barrier()

    # X-basis measurement on each inner qubit u_1, ..., u_N.
    for j in range(1, N + 1):
        qc.h(chain[j])
        qc.measure(chain[j], cr[j - 1])

    qc.barrier()

    # Correction: Z^{XOR of all m_j} on e_1. Implement via N if_tests each
    # applying Z (because Z^k = I if k even, Z if k odd).
    for j in range(1, N + 1):
        with qc.if_test((cr[j - 1], 1)):
            qc.z(e1)

    return qc
