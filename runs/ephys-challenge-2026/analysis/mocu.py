"""Middle-Out Cat-Uncompute (MOCU): a measurement-free Bell-state preparation
on the ladder QPU's top leg.

See `../theory_draft.md` for the stabiliser-formalism proof.

Forward sweep (Phase A + Phase B): H on the middle qubit, then a balanced
expansion of CNOTs outward in both directions, building the (L+1)-qubit
GHZ state on the top leg.

Reverse sweep (Phase C): a solo CNOT(q_{m-1}, q_m) disentangles q_m, then
parallel pairs of CNOTs (one from each side) disentangle the next inner
qubits outward, until only q_0 = e_0 and q_L = e_1 remain in the active
GHZ — leaving them in |Phi+>.

Total depth: L + 2. Total CNOTs: 2L - 1. Mid-circuit measurements: 0.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister

from entanglement_swap import _top_chain
from ladder_graph import n_qubits


def build_circuit(L: int) -> QuantumCircuit:
    if L < 1:
        raise ValueError("L must be >= 1")

    qr = QuantumRegister(n_qubits(L), "q")
    qc = QuantumCircuit(qr, name=f"mocu_L{L}")

    chain = _top_chain(L)

    if L == 1:
        # Trivial base case: just |Phi+> on adjacent qubits.
        qc.h(chain[0])
        qc.cx(chain[0], chain[1])
        return qc

    m = L // 2

    # === Phase A + B: build GHZ_{L+1} from the middle. ===
    qc.h(chain[m])

    left_n = m            # sites still |0> on the left of m
    right_n = L - m       # sites still |0> on the right of m
    left_filled = 0
    right_filled = 0

    # Solo first CNOT on the longer side. If equal, default to right.
    if right_n >= left_n:
        qc.cx(chain[m], chain[m + 1])
        right_filled = 1
    else:
        qc.cx(chain[m], chain[m - 1])
        left_filled = 1

    # Parallel expansion from both frontiers outward.
    while left_filled < left_n or right_filled < right_n:
        if left_filled < left_n:
            ctrl = chain[m - left_filled]
            tgt = chain[m - left_filled - 1]
            qc.cx(ctrl, tgt)
            left_filled += 1
        if right_filled < right_n:
            ctrl = chain[m + right_filled]
            tgt = chain[m + right_filled + 1]
            qc.cx(ctrl, tgt)
            right_filled += 1

    qc.barrier()

    # === Phase C: shrink the GHZ from the middle. ===
    # Solo CNOT disentangles q_m using its left neighbour as control.
    qc.cx(chain[m - 1], chain[m])

    # Parallel pairs: layer s disentangles q_{m-s} (left) and q_{m+s} (right)
    # using their next-outer neighbours as controls. Stop when no more inner
    # sites remain on either side (boundary qubits q_0, q_L are preserved).
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
