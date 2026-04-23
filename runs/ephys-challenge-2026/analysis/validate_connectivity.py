"""Connectivity validator: assert every 2-qubit gate is on an allowed edge."""

from __future__ import annotations

from typing import FrozenSet, Tuple

from qiskit import QuantumCircuit

from ladder_graph import allowed_edges


def validate_connectivity(qc: QuantumCircuit, L: int) -> None:
    """Iterate every 2Q instruction and assert its qubit pair is allowed.

    Raises ConnectivityViolation with a descriptive message on failure.
    """
    edges: FrozenSet[Tuple[int, int]] = allowed_edges(L)
    for k, instr in enumerate(qc.data):
        if instr.operation.num_qubits == 2:
            q0 = qc.find_bit(instr.qubits[0]).index
            q1 = qc.find_bit(instr.qubits[1]).index
            pair = tuple(sorted((q0, q1)))
            if pair not in edges:
                raise ConnectivityViolation(
                    f"2Q gate #{k} ({instr.operation.name}) on qubits "
                    f"({q0}, {q1}) is NOT an allowed ladder edge for L={L}"
                )


class ConnectivityViolation(RuntimeError):
    pass
