"""Spot-check the stabiliser structure of the N=4 protocol matches §2 of the theory draft.

Specifically: after Step B (Bell-measurement unitaries) but before measurement,
the six stabilisers should be exactly
  S_1 = X_e0 Z_1 X_2,
  S_2 = Z_e0 X_1,
  S_3 = X_2 Z_3 X_4,
  S_4 = X_1 Z_2 X_3,
  S_5 = X_4 X_e1,
  S_6 = X_3 Z_4 Z_e1.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import stim

from ladder_graph import qidx, n_qubits


def build_n4_pre_measurement_circuit() -> stim.Circuit:
    """N=4 pre-measurement circuit: Step A + Step B (no measurement)."""
    L = 5
    idx = qidx(L)
    e0, u1, u2, u3, u4, e1 = (
        idx["e0"], idx["u1"], idx["u2"], idx["u3"], idx["u4"], idx["e1"],
    )

    c = stim.Circuit()
    # Step A: Bell-pair preparation on (e_0, u_1), (u_2, u_3), (u_4, e_1)
    c.append("H", [e0])
    c.append("CX", [e0, u1])
    c.append("H", [u2])
    c.append("CX", [u2, u3])
    c.append("H", [u4])
    c.append("CX", [u4, e1])
    # Step B: Bell-measurement unitaries on (u_1, u_2), (u_3, u_4)
    c.append("CX", [u1, u2])
    c.append("CX", [u3, u4])
    c.append("H", [u1])
    c.append("H", [u3])
    return c


def _pauli_str(n: int, single: dict[int, str]) -> stim.PauliString:
    chars = ["_"] * n
    for q, p in single.items():
        chars[q] = p
    return stim.PauliString("".join(chars))


def test_n4_stabilisers_after_step_B():
    L = 5
    idx = qidx(L)
    e0, u1, u2, u3, u4, e1 = (
        idx["e0"], idx["u1"], idx["u2"], idx["u3"], idx["u4"], idx["e1"],
    )
    n = n_qubits(L)

    sim = stim.TableauSimulator()
    sim.do(build_n4_pre_measurement_circuit())

    expected = [
        ("S_1", _pauli_str(n, {e0: "X", u1: "Z", u2: "X"})),
        ("S_2", _pauli_str(n, {e0: "Z", u1: "X"})),
        ("S_3", _pauli_str(n, {u2: "X", u3: "Z", u4: "X"})),
        ("S_4", _pauli_str(n, {u1: "X", u2: "Z", u3: "X"})),
        ("S_5", _pauli_str(n, {u4: "X", e1: "X"})),
        ("S_6", _pauli_str(n, {u3: "X", u4: "Z", e1: "Z"})),
    ]

    # Each expected operator should be a stabiliser of the state, i.e.
    # <S_i> = +1.
    for name, op in expected:
        expectation = sim.peek_observable_expectation(op)
        assert expectation == 1, f"{name} has expectation {expectation}, expected +1"


def test_n4_product_P_ZZ_becomes_ZZ_tensor():
    """P_ZZ = S_2 · S_4 · S_6 = Z_e0 Z_2 Z_4 Z_e1.

    After Z-basis measurement of u_2, u_4 with all-zero outcomes, this projects
    down to +Z_e0 Z_e1 on the endpoints, confirming the Bell-state ZZ stabiliser.
    """
    L = 5
    idx = qidx(L)
    e0, u2, u4, e1 = idx["e0"], idx["u2"], idx["u4"], idx["e1"]
    n = n_qubits(L)

    sim = stim.TableauSimulator()
    sim.do(build_n4_pre_measurement_circuit())

    P_zz = _pauli_str(n, {e0: "Z", u2: "Z", u4: "Z", e1: "Z"})
    P_xx = _pauli_str(n, {e0: "X", idx["u1"]: "Z", idx["u3"]: "Z", e1: "X"})
    assert sim.peek_observable_expectation(P_zz) == 1
    assert sim.peek_observable_expectation(P_xx) == 1


if __name__ == "__main__":
    test_n4_stabilisers_after_step_B()
    test_n4_product_P_ZZ_becomes_ZZ_tensor()
    print("Stabiliser structure matches theory draft §2.")
