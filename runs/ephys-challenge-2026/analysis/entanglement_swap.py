"""Our Bell-state preparation protocol: entanglement swapping on the top leg.

Implements both parities of ``N = L - 1`` from Section 3 of the theory draft.

Notation: the top-leg chain is indexed
    chain[0] = e_0,  chain[1] = u_1, ..., chain[N] = u_N, chain[N+1] = e_1.

Let ``M`` be the number of intermediate qubits on the top leg, i.e. M = N.

For each parity:

  **N = 0 (L = 1):**  H(e_0); CNOT(e_0 -> e_1).  Deterministic |Φ+>.

  **N = 2r (even, r >= 1):**
    - Bell pairs on (chain[2k], chain[2k+1]) for k = 0..r, i.e. (e_0, u_1),
      (u_2, u_3), ..., (u_{N-2}, u_{N-1}), (u_N, e_1). Total r+1 pairs.
    - Bell-measurement unitaries on (chain[2k-1], chain[2k]) for k = 1..r,
      i.e. (u_1, u_2), ..., (u_{N-1}, u_N). Total r measurements.
    - Measure chain[1..N] in Z basis, outcomes m_1,...,m_N.
    - Correction on e_1: X^{XOR_{j even} m_j} Z^{XOR_{j odd} m_j}.

  **N = 2r+1 (odd, r >= 0):**
    - Bell pairs on (chain[2k], chain[2k+1]) for k = 0..r-1. For r=0 this is
      empty. Total r pairs.
    - GHZ-3 on (chain[2r], chain[2r+1], chain[2r+2]) = (chain[2r], u_N, e_1).
      Prep: H(chain[2r+1]); CNOT(chain[2r+1], chain[2r]); CNOT(chain[2r+1], e_1).
    - Bell-measurement unitaries on (chain[2k-1], chain[2k]) for k = 1..r.
    - X-basis measurement on chain[2r+1] = u_N.
    - Measure chain[1..N] in Z basis, outcomes m_1,...,m_N.
    - Correction on e_1: same formula as even (m_N now participates in the
      odd-index XOR since N is odd).
"""

from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from ladder_graph import n_qubits, qidx


def _make_registers(L: int) -> tuple[QuantumRegister, ClassicalRegister]:
    N = L - 1
    qr = QuantumRegister(n_qubits(L), "q")
    # Use max(1, N) so the classical register is always non-empty (qiskit
    # requires at least one bit for if_test compatibility).
    cr = ClassicalRegister(max(1, N), "m")
    return qr, cr


def _top_chain(L: int) -> list[int]:
    """Return the ordered top-leg indices [e_0, u_1, ..., u_{L-1}, e_1]."""
    idx = qidx(L)
    chain = [idx["e0"]]
    for i in range(1, L):
        chain.append(idx[f"u{i}"])
    chain.append(idx["e1"])
    return chain


def build_circuit(L: int) -> QuantumCircuit:
    if L < 1:
        raise ValueError("L must be >= 1")

    N = L - 1
    qr, cr = _make_registers(L)
    qc = QuantumCircuit(qr, cr, name=f"entanglement_swap_L{L}")

    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]

    # Special base case: N = 0.
    if N == 0:
        qc.h(e0)
        qc.cx(e0, e1)
        return qc

    if N % 2 == 0:
        _build_even(qc, chain, cr, N)
    else:
        _build_odd(qc, chain, cr, N)

    return qc


def _build_even(qc: QuantumCircuit, chain: list[int], cr: ClassicalRegister, N: int) -> None:
    r = N // 2

    # Step 1: r+1 Bell pairs on alternating links.
    for k in range(r + 1):
        left = chain[2 * k]
        right = chain[2 * k + 1]
        qc.h(left)
        qc.cx(left, right)

    qc.barrier()

    # Step 2: r Bell-measurement unitaries on inner pairs.
    for k in range(1, r + 1):
        left = chain[2 * k - 1]
        right = chain[2 * k]
        qc.cx(left, right)
        qc.h(left)

    qc.barrier()

    # Step 3: measure all N intermediate qubits, in chain order.
    for j in range(1, N + 1):
        qc.measure(chain[j], cr[j - 1])

    qc.barrier()

    # Step 4: feed-forward Pauli correction on e_1.
    _apply_correction(qc, chain, cr, N)


def _build_odd(qc: QuantumCircuit, chain: list[int], cr: ClassicalRegister, N: int) -> None:
    r = (N - 1) // 2
    e1 = chain[-1]

    # Step 1: r Bell pairs on (chain[2k], chain[2k+1]) for k = 0..r-1.
    for k in range(r):
        left = chain[2 * k]
        right = chain[2 * k + 1]
        qc.h(left)
        qc.cx(left, right)

    # Step 2: GHZ-3 on (chain[2r], chain[2r+1], e_1).
    a = chain[2 * r]       # for r=0 this is e_0 itself; for r>=1 an inner qubit
    b = chain[2 * r + 1]   # this is u_N
    qc.h(b)
    qc.cx(b, a)
    qc.cx(b, e1)

    qc.barrier()

    # Step 3: r Bell-measurement unitaries on (chain[2k-1], chain[2k])
    # for k=1..r. For r=0 this loop is empty.
    for k in range(1, r + 1):
        left = chain[2 * k - 1]
        right = chain[2 * k]
        qc.cx(left, right)
        qc.h(left)

    # Step 4: X-basis measurement prep on u_N = chain[2r+1].
    qc.h(chain[2 * r + 1])

    qc.barrier()

    # Step 5: measure all N intermediate qubits.
    for j in range(1, N + 1):
        qc.measure(chain[j], cr[j - 1])

    qc.barrier()

    # Step 6: feed-forward correction (same formula as even).
    _apply_correction(qc, chain, cr, N)


def _apply_correction(qc: QuantumCircuit, chain: list[int], cr: ClassicalRegister, N: int) -> None:
    """Apply X^{XOR_{j even} m_j} Z^{XOR_{j odd} m_j} on e_1.

    We implement the XOR via a chain of ``if_test`` blocks each applying
    X (respectively Z) when the corresponding classical bit is 1. Because
    X^2 = Z^2 = I, this correctly implements the XOR.
    """
    e1 = chain[-1]
    for j in range(1, N + 1):
        i = j - 1  # classical-bit index (0-based)
        with qc.if_test((cr[i], 1)):
            if j % 2 == 0:
                qc.x(e1)
            else:
                qc.z(e1)
