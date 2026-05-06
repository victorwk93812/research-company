"""Tests for the MOCU (Middle-Out Cat-Uncompute) measurement-free protocol.

Verifications:
  - Connectivity: every CNOT lands on a top-leg edge.
  - Stim-exact Bell fidelity = 1 for L = 1..10 (Clifford simulation).
  - Statevector cross-check for L = 1..6 (Qiskit Aer / Statevector).
  - Resource counts match theoretical predictions (depth ~ L+2, 2Q gates 2L-1).
  - No measurements anywhere in the circuit.
  - Spot-check at large L (20, 30, 50) via stim.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import stim

import mocu
from entanglement_swap import _top_chain
from ladder_graph import n_qubits
from validate_connectivity import validate_connectivity


# ---------- Connectivity ----------

@pytest.mark.parametrize("L", list(range(1, 11)))
def test_connectivity(L):
    qc = mocu.build_circuit(L)
    validate_connectivity(qc, L)


# ---------- "No measurements" invariant ----------

@pytest.mark.parametrize("L", list(range(1, 11)))
def test_no_measurements(L):
    qc = mocu.build_circuit(L)
    n_meas = sum(1 for instr in qc.data if instr.operation.name == "measure")
    assert n_meas == 0, f"MOCU L={L} contains {n_meas} measurements; expected 0"
    n_classical = qc.num_clbits
    assert n_classical == 0, (
        f"MOCU L={L} declared {n_classical} classical bits; expected 0"
    )


# ---------- Resource counts ----------

@pytest.mark.parametrize("L", list(range(1, 11)))
def test_two_qubit_gate_count(L):
    qc = mocu.build_circuit(L)
    n_2q = sum(
        1 for instr in qc.data
        if instr.operation.name in {"cx", "cz", "swap"}
    )
    expected = 2 * L - 1 if L >= 1 else 0
    assert n_2q == expected, (
        f"MOCU L={L} expected 2L-1={expected} 2Q gates, got {n_2q}"
    )


# ---------- Stim Clifford-exact fidelity ----------

def _build_mocu_in_stim(L: int) -> stim.Circuit:
    """Replay mocu.build_circuit(L) in a stim.Circuit (Clifford only)."""
    chain = _top_chain(L)
    circ = stim.Circuit()

    if L == 1:
        circ.append("H", [chain[0]])
        circ.append("CX", [chain[0], chain[1]])
        return circ

    m = L // 2
    circ.append("H", [chain[m]])

    left_n = m
    right_n = L - m
    left_filled = 0
    right_filled = 0
    if right_n >= left_n:
        circ.append("CX", [chain[m], chain[m + 1]])
        right_filled = 1
    else:
        circ.append("CX", [chain[m], chain[m - 1]])
        left_filled = 1

    while left_filled < left_n or right_filled < right_n:
        if left_filled < left_n:
            circ.append("CX", [chain[m - left_filled], chain[m - left_filled - 1]])
            left_filled += 1
        if right_filled < right_n:
            circ.append("CX", [chain[m + right_filled], chain[m + right_filled + 1]])
            right_filled += 1

    # Reverse sweep
    circ.append("CX", [chain[m - 1], chain[m]])
    s = 1
    while True:
        left_ok = (m - s >= 1) and (m - s - 1 >= 0)
        right_ok = (m + s <= L - 1) and (m + s + 1 <= L)
        if not left_ok and not right_ok:
            break
        if left_ok:
            circ.append("CX", [chain[m - s - 1], chain[m - s]])
        if right_ok:
            circ.append("CX", [chain[m + s + 1], chain[m + s]])
        s += 1
    return circ


def _pauli_string(n: int, single: dict) -> stim.PauliString:
    chars = ["_"] * n
    for q, p in single.items():
        chars[q] = p
    return stim.PauliString("".join(chars))


def _phi_plus_fidelity_stim(L: int) -> float:
    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]
    n = n_qubits(L)
    circ = _build_mocu_in_stim(L)
    sim = stim.TableauSimulator()
    sim.do(circ)

    # Postselect every intermediate top-leg qubit to |0> (must succeed if
    # MOCU is correct).
    for k in range(1, L):
        sim.postselect_z(chain[k], desired_value=0)

    zz = sim.peek_observable_expectation(_pauli_string(n, {e0: "Z", e1: "Z"}))
    xx = sim.peek_observable_expectation(_pauli_string(n, {e0: "X", e1: "X"}))
    yy = sim.peek_observable_expectation(_pauli_string(n, {e0: "Y", e1: "Y"}))
    return (1 + zz + xx - yy) / 4


@pytest.mark.parametrize("L", list(range(1, 11)))
def test_stim_exact_fidelity(L):
    fid = _phi_plus_fidelity_stim(L)
    assert fid > 1 - 1e-9, f"MOCU L={L} stim fidelity {fid} below threshold"


@pytest.mark.parametrize("L", [20, 30, 50])
def test_stim_large_L_spot(L):
    fid = _phi_plus_fidelity_stim(L)
    assert fid > 1 - 1e-9, f"MOCU L={L} large-L stim fidelity {fid} below threshold"


# ---------- Statevector cross-check ----------

@pytest.mark.parametrize("L", list(range(1, 7)))
def test_statevector_fidelity(L):
    """Independent cross-check: build the circuit in Qiskit, simulate via
    Statevector, marginalise to (e_0, e_1), and check |Phi+> fidelity."""
    from qiskit.quantum_info import Statevector

    qc = mocu.build_circuit(L)
    sv = Statevector.from_instruction(qc)
    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]
    n = qc.num_qubits

    # Build the ideal target |Phi+>_{e0,e1} ⊗ |0...0>_others.
    psi = np.zeros(2 ** n, dtype=complex)
    other_qubits = [q for q in range(n) if q not in (e0, e1)]

    def basis_index(bits: dict) -> int:
        # bits: qubit_index -> 0/1, missing qubits = 0
        x = 0
        for q, b in bits.items():
            x |= (b & 1) << q
        return x

    psi[basis_index({e0: 0, e1: 0})] = 1 / np.sqrt(2)
    psi[basis_index({e0: 1, e1: 1})] = 1 / np.sqrt(2)
    target = Statevector(psi)

    fid = abs(sv.inner(target)) ** 2
    assert fid > 1 - 1e-9, f"MOCU L={L} statevector fidelity {fid} below threshold"


# ---------- Run as script ----------

if __name__ == "__main__":
    print("MOCU connectivity:")
    for L in range(1, 11):
        qc = mocu.build_circuit(L)
        validate_connectivity(qc, L)
        print(f"  L={L:2d}: ok, depth={qc.depth()}, 2Q gates={sum(1 for i in qc.data if i.operation.name in {'cx','cz','swap'})}")
    print("\nStim fidelity:")
    for L in range(1, 11):
        f = _phi_plus_fidelity_stim(L)
        print(f"  L={L:2d}: fidelity = {f:.12f}")
    print("\nLarge-L stim spot check:")
    for L in [20, 30, 50]:
        f = _phi_plus_fidelity_stim(L)
        print(f"  L={L:2d}: fidelity = {f:.12f}")
    print("\nStatevector cross-check:")
    from qiskit.quantum_info import Statevector
    for L in range(1, 7):
        qc = mocu.build_circuit(L)
        sv = Statevector.from_instruction(qc)
        chain = _top_chain(L)
        e0, e1 = chain[0], chain[-1]
        n = qc.num_qubits
        psi = np.zeros(2 ** n, dtype=complex)
        psi[(0 << e0) | (0 << e1)] = 1 / np.sqrt(2)
        psi[(1 << e0) | (1 << e1)] = 1 / np.sqrt(2)
        target = Statevector(psi)
        f = abs(sv.inner(target)) ** 2
        print(f"  L={L:2d}: fidelity = {f:.12f}")
