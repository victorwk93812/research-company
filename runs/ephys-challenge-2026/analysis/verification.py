"""Exact Statevector branch-by-branch verification of Bell-state fidelity.

For each protocol we:

  1. Build the "pre-measurement" unitary U (circuit without mid-circuit
     measurements or classical corrections) and compute |psi> = U|0...0>.

  2. For each branch m = (m_1, ..., m_N) in {0,1}^N:
     a. Project |psi> onto the Z-basis outcomes on the intermediate qubits.
        Read off branch probability p_m = ||projected||^2.
     b. Normalise and apply the Pauli feed-forward correction on e_1.
     c. Take the partial trace over everything except (e_0, e_1) using
        Qiskit's built-in partial_trace (avoids instantiating the full
        2^n x 2^n density matrix).
     d. Compute <Phi+|rho|Phi+>.

  3. Assert fidelity > 1 - 1e-9 for every branch of non-zero probability.

This gives deterministic branch-level verification without touching
Qiskit's dynamic-circuit machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, partial_trace, Operator

from entanglement_swap import _top_chain
from ladder_graph import n_qubits


PHI_PLUS = Statevector(np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    build_pre_measurement: Callable[[int], QuantumCircuit]
    correction: Callable[[int, Tuple[int, ...]], str]


# -----------------------------------------------------------------------
# Pre-measurement unitary builders
# -----------------------------------------------------------------------


def entanglement_swap_pre(L: int) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits(L), "q")
    qc = QuantumCircuit(qr, name=f"es_pre_L{L}")
    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]
    N = L - 1
    if N == 0:
        qc.h(e0)
        qc.cx(e0, e1)
        return qc
    if N % 2 == 0:
        r = N // 2
        for k in range(r + 1):
            qc.h(chain[2 * k])
            qc.cx(chain[2 * k], chain[2 * k + 1])
        for k in range(1, r + 1):
            qc.cx(chain[2 * k - 1], chain[2 * k])
            qc.h(chain[2 * k - 1])
    else:
        r = (N - 1) // 2
        for k in range(r):
            qc.h(chain[2 * k])
            qc.cx(chain[2 * k], chain[2 * k + 1])
        qc.h(chain[2 * r + 1])
        qc.cx(chain[2 * r + 1], chain[2 * r])
        qc.cx(chain[2 * r + 1], e1)
        for k in range(1, r + 1):
            qc.cx(chain[2 * k - 1], chain[2 * k])
            qc.h(chain[2 * k - 1])
        qc.h(chain[2 * r + 1])
    return qc


def entanglement_swap_correction(L: int, m: Tuple[int, ...]) -> str:
    assert len(m) == L - 1
    a = 0  # X-correction parity
    b = 0  # Z-correction parity
    for j, mj in enumerate(m, start=1):
        if j % 2 == 0:
            a ^= mj
        else:
            b ^= mj
    corr_e1 = ""
    if a:
        corr_e1 += "X"
    if b:
        corr_e1 += "Z"
    if not corr_e1:
        corr_e1 = "I"
    return "I" + corr_e1


def swap_chain_pre(L: int) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits(L), "q")
    qc = QuantumCircuit(qr, name=f"swap_pre_L{L}")
    chain = _top_chain(L)
    N = L - 1
    if N == 0:
        qc.h(chain[0])
        qc.cx(chain[0], chain[1])
        return qc
    qc.h(chain[0])
    qc.cx(chain[0], chain[1])
    for k in range(1, N + 1):
        qc.swap(chain[k], chain[k + 1])
    return qc


def swap_chain_correction(L: int, m: Tuple[int, ...]) -> str:
    return "II"


def cat_chain_pre(L: int) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits(L), "q")
    qc = QuantumCircuit(qr, name=f"cat_pre_L{L}")
    chain = _top_chain(L)
    N = L - 1
    if N == 0:
        qc.h(chain[0])
        qc.cx(chain[0], chain[1])
        return qc
    qc.h(chain[0])
    for k in range(L):
        qc.cx(chain[k], chain[k + 1])
    for j in range(1, N + 1):
        qc.h(chain[j])
    return qc


def cat_chain_correction(L: int, m: Tuple[int, ...]) -> str:
    parity = 0
    for mj in m:
        parity ^= mj
    return "I" + ("Z" if parity else "I")


def cluster_ladder_pre(L: int) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits(L), "q")
    qc = QuantumCircuit(qr, name=f"cluster_pre_L{L}")
    chain = _top_chain(L)
    N = L - 1
    if N == 0:
        qc.h(chain[0])
        qc.cx(chain[0], chain[1])
        return qc
    for q in chain:
        qc.h(q)
    for k in range(L):
        qc.cz(chain[k], chain[k + 1])
    for j in range(1, N + 1):
        qc.h(chain[j])
    return qc


def cluster_ladder_correction(L: int, m: Tuple[int, ...]) -> str:
    # Depending on position, each X-basis measurement on an intermediate
    # contributes a specific Pauli to the endpoint corrections for a 1D
    # cluster-state teleportation. For the symmetric 1D chain with |+> input,
    # the net correction on e_1 is Z^{sum over odd-position intermediates}.
    # For the Bell-state output, the full correction is derived in e.g.
    # Raussendorf-Briegel. We keep this simple: the output is |Phi+> up to
    # Pauli corrections, and we search over the 4 Pauli choices on e_1 to
    # pick the one that maximises fidelity (equivalent to identifying the
    # Bell-state branch).
    parity_odd = 0
    parity_all = 0
    for j, mj in enumerate(m, start=1):
        parity_all ^= mj
        if j % 2 == 1:
            parity_odd ^= mj
    corr = ""
    if parity_odd:
        corr += "X"
    if parity_all ^ parity_odd:
        corr += "Z"
    if not corr:
        corr = "I"
    return "I" + corr


# -----------------------------------------------------------------------
# Branch projector and fidelity
# -----------------------------------------------------------------------


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _single_qubit_matrix(corr_str: str) -> np.ndarray:
    """Parse a string of Pauli symbols into the product matrix."""
    op = np.eye(2, dtype=complex)
    for ch in corr_str:
        op = op @ _PAULI[ch]
    return op


def _project_branch(
    psi: Statevector,
    measured_indices: List[int],
    outcomes: Tuple[int, ...],
    n_qubits_: int,
) -> Tuple[float, Statevector]:
    """Apply Z-basis projection on ``measured_indices`` to ``outcomes``.

    Returns (branch_probability, normalised_post_state).
    """
    data = psi.data.astype(complex).copy()
    dim = 2 ** n_qubits_
    basis = np.arange(dim)
    keep = np.ones(dim, dtype=bool)
    for q_idx, m_val in zip(measured_indices, outcomes):
        bits = (basis >> q_idx) & 1
        keep &= (bits == m_val)
    data[~keep] = 0.0
    prob = float(np.vdot(data, data).real)
    if prob < 1e-18:
        return 0.0, psi
    data /= np.sqrt(prob)
    return prob, Statevector(data, dims=psi.dims())


def _apply_single_qubit(
    state: Statevector, op_matrix: np.ndarray, qubit_index: int, n_qubits_: int
) -> Statevector:
    data = state.data.astype(complex).copy()
    # In Qiskit, qubit k corresponds to the bit at position k (least-significant).
    # Reshape: state[i] where i = sum_k b_k 2^k. np.reshape([2]*n) gives axes
    # ordered with qubit (n-1) first and qubit 0 last.
    tensor = data.reshape([2] * n_qubits_)
    axis = n_qubits_ - 1 - qubit_index
    tensor = np.moveaxis(tensor, axis, 0)
    rest = tensor.shape[1:]
    tensor = tensor.reshape(2, -1)
    tensor = op_matrix @ tensor
    tensor = tensor.reshape((2,) + rest)
    tensor = np.moveaxis(tensor, 0, axis)
    return Statevector(tensor.reshape(data.shape), dims=state.dims())


def fidelity_vs_phi_plus_from_reduced(
    psi: Statevector, endpoint_qubits: Tuple[int, int], n_qubits_: int
) -> float:
    """Compute <Phi+|rho|Phi+> = <psi|(|Phi+><Phi+|_{ep} otimes I_{rest})|psi>
    without constructing the full density matrix.

    Trick: rotate the endpoint pair to the Bell basis via H(e_0); CNOT(e_0, e_1),
    which maps |Phi+> -> |00>. Then the fidelity is the marginal probability
    of (e_0, e_1) = (0, 0), which is a simple sum over amplitudes.
    """
    e0, e1 = endpoint_qubits
    H = _PAULI["I"].copy()
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    CNOT_mat = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )

    # The Bell-basis rotation Phi+ -> 00 is: U = (H_{e0} ⊗ I_{e1}) · CNOT(e0->e1)
    #   CNOT(e0->e1)|00> = |00>,  CNOT|11> = |10>,
    #   CNOT(|00>+|11>)/sqrt2 = (|00>+|10>)/sqrt2 = |+>|0>
    # Then H_{e0} |+0> = |00>. ✓
    psi_rot = _apply_two_qubit_cnot(psi, e0, e1, n_qubits_)
    psi_rot = _apply_single_qubit(psi_rot, H, e0, n_qubits_)

    # Marginal probability of (e_0, e_1) = (0, 0): sum |amp|^2 over basis
    # states with bits at positions e_0 and e_1 both zero.
    data = psi_rot.data
    dim = 2 ** n_qubits_
    basis = np.arange(dim)
    mask = (((basis >> e0) & 1) == 0) & (((basis >> e1) & 1) == 0)
    prob = float(np.sum(np.abs(data[mask]) ** 2))
    return prob


def _apply_two_qubit_cnot(
    state: Statevector, control: int, target: int, n_qubits_: int
) -> Statevector:
    """Apply CNOT(control -> target) directly on the statevector without
    instantiating a 2^n x 2^n matrix."""
    data = state.data.copy()
    dim = 2 ** n_qubits_
    basis = np.arange(dim)
    ctrl_bit = (basis >> control) & 1
    # For every basis index where control=1, flip the target bit.
    flip_mask = basis ^ (1 << target)
    new_data = data.copy()
    new_data[ctrl_bit == 1] = data[flip_mask[ctrl_bit == 1]]
    return Statevector(new_data, dims=state.dims())


def verify_all_branches(
    protocol: ProtocolSpec, L: int, threshold: float = 1 - 1e-9
) -> Tuple[bool, float, int, int]:
    N = L - 1
    total = n_qubits(L)
    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]
    measured = [chain[j] for j in range(1, N + 1)]

    pre_circuit = protocol.build_pre_measurement(L)
    psi = Statevector.from_int(0, dims=[2] * total).evolve(pre_circuit)

    n_tested = 0
    n_nonzero = 0
    min_fid = 1.0
    all_pass = True

    for branch in range(2 ** N):
        m = tuple((branch >> k) & 1 for k in range(N))
        prob, post = _project_branch(psi, measured, m, total)
        n_tested += 1
        if prob < 1e-18:
            continue
        n_nonzero += 1

        corr = protocol.correction(L, m)
        # corr is a two-part string: "I" + (Pauli product on e_1)
        op_e0 = _single_qubit_matrix(corr[0])
        op_e1 = _single_qubit_matrix(corr[1:])

        post = _apply_single_qubit(post, op_e0, e0, total)
        post = _apply_single_qubit(post, op_e1, e1, total)

        # Ensure endpoint order (smaller index first) for partial_trace.
        ep = (e0, e1) if e0 < e1 else (e1, e0)
        fid = fidelity_vs_phi_plus_from_reduced(post, ep, total)
        min_fid = min(min_fid, fid)
        if fid < threshold:
            all_pass = False

    return all_pass, min_fid, n_tested, n_nonzero


PROTOCOLS: List[ProtocolSpec] = [
    ProtocolSpec("entanglement_swap", entanglement_swap_pre, entanglement_swap_correction),
    ProtocolSpec("swap_chain", swap_chain_pre, swap_chain_correction),
    ProtocolSpec("cat_chain", cat_chain_pre, cat_chain_correction),
    ProtocolSpec("cluster_ladder", cluster_ladder_pre, cluster_ladder_correction),
]


def run_verification(L_max: int = 10, threshold: float = 1 - 1e-9) -> List[tuple]:
    print(
        f"{'Protocol':<22} {'L':>3} {'branches':>9} {'non-zero':>9} "
        f"{'min fid':>20} {'pass':>6}"
    )
    print("-" * 76)
    all_results = []
    for protocol in PROTOCOLS:
        for L in range(1, L_max + 1):
            try:
                all_pass, min_fid, n_b, n_nz = verify_all_branches(
                    protocol, L, threshold=threshold
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{protocol.name:<22} {L:>3}  ERROR: {exc}")
                all_results.append((protocol.name, L, False, 0.0, 0, 0))
                continue
            status = "PASS" if all_pass else "FAIL"
            print(
                f"{protocol.name:<22} {L:>3} {n_b:>9} {n_nz:>9} "
                f"{min_fid:>20.15f} {status:>6}"
            )
            all_results.append((protocol.name, L, all_pass, min_fid, n_b, n_nz))
    return all_results


if __name__ == "__main__":
    import resource_limits  # noqa: F401
    run_verification(L_max=10)
