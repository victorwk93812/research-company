"""Clifford-exact verification via ``stim.TableauSimulator``.

Our protocol is purely Clifford, so stim can simulate it in polynomial time
and give exact stabiliser-state expectation values. For each measurement
branch we:

  1. Build the pre-measurement Clifford circuit in stim.
  2. Project each intermediate qubit onto its Z-basis outcome via
     ``TableauSimulator.postselect_z``.
  3. Apply the Pauli feed-forward correction on e_1.
  4. Evaluate the Bell fidelity as
        F = (1 + <ZZ> + <XX> - <YY>) / 4
     on the (e_0, e_1) pair.

For stabiliser states, each Pauli expectation is in ``{+1, -1, 0}`` and is
returned exactly by ``peek_observable_expectation``.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import stim

from entanglement_swap import _top_chain
from ladder_graph import n_qubits


def _pauli_string(n: int, single: dict[int, str]) -> stim.PauliString:
    """Build a stim.PauliString of length n with specified single-qubit Paulis
    at given indices; identity elsewhere."""
    chars = ["_"] * n
    for q, p in single.items():
        chars[q] = p
    return stim.PauliString("".join(chars))


def _entanglement_swap_pre(circ: stim.Circuit, L: int) -> None:
    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]
    N = L - 1
    if N == 0:
        circ.append("H", [e0])
        circ.append("CX", [e0, e1])
        return
    if N % 2 == 0:
        r = N // 2
        for k in range(r + 1):
            circ.append("H", [chain[2 * k]])
            circ.append("CX", [chain[2 * k], chain[2 * k + 1]])
        for k in range(1, r + 1):
            circ.append("CX", [chain[2 * k - 1], chain[2 * k]])
            circ.append("H", [chain[2 * k - 1]])
    else:
        r = (N - 1) // 2
        for k in range(r):
            circ.append("H", [chain[2 * k]])
            circ.append("CX", [chain[2 * k], chain[2 * k + 1]])
        circ.append("H", [chain[2 * r + 1]])
        circ.append("CX", [chain[2 * r + 1], chain[2 * r]])
        circ.append("CX", [chain[2 * r + 1], e1])
        for k in range(1, r + 1):
            circ.append("CX", [chain[2 * k - 1], chain[2 * k]])
            circ.append("H", [chain[2 * k - 1]])
        circ.append("H", [chain[2 * r + 1]])


def _entanglement_swap_correction(L: int, m: Tuple[int, ...]) -> str:
    a = 0
    b = 0
    for j, mj in enumerate(m, start=1):
        if j % 2 == 0:
            a ^= mj
        else:
            b ^= mj
    s = ""
    if a:
        s += "X"
    if b:
        s += "Z"
    return s


def _swap_chain_pre(circ: stim.Circuit, L: int) -> None:
    chain = _top_chain(L)
    N = L - 1
    if N == 0:
        circ.append("H", [chain[0]])
        circ.append("CX", [chain[0], chain[1]])
        return
    circ.append("H", [chain[0]])
    circ.append("CX", [chain[0], chain[1]])
    for k in range(1, N + 1):
        # SWAP via 3 CNOTs
        circ.append("CX", [chain[k], chain[k + 1]])
        circ.append("CX", [chain[k + 1], chain[k]])
        circ.append("CX", [chain[k], chain[k + 1]])


def _swap_chain_correction(L: int, m: Tuple[int, ...]) -> str:
    return ""


def _cat_chain_pre(circ: stim.Circuit, L: int) -> None:
    chain = _top_chain(L)
    N = L - 1
    if N == 0:
        circ.append("H", [chain[0]])
        circ.append("CX", [chain[0], chain[1]])
        return
    circ.append("H", [chain[0]])
    for k in range(L):
        circ.append("CX", [chain[k], chain[k + 1]])
    for j in range(1, N + 1):
        circ.append("H", [chain[j]])


def _cat_chain_correction(L: int, m: Tuple[int, ...]) -> str:
    parity = 0
    for mj in m:
        parity ^= mj
    return "Z" if parity else ""


PROTOCOLS_STIM: List[tuple[str, Callable, Callable]] = [
    ("entanglement_swap", _entanglement_swap_pre, _entanglement_swap_correction),
    ("swap_chain", _swap_chain_pre, _swap_chain_correction),
    ("cat_chain", _cat_chain_pre, _cat_chain_correction),
]


def _fidelity_phi_plus(sim: stim.TableauSimulator, e0: int, e1: int, n: int) -> float:
    zz = sim.peek_observable_expectation(_pauli_string(n, {e0: "Z", e1: "Z"}))
    xx = sim.peek_observable_expectation(_pauli_string(n, {e0: "X", e1: "X"}))
    yy = sim.peek_observable_expectation(_pauli_string(n, {e0: "Y", e1: "Y"}))
    return (1 + zz + xx - yy) / 4


def verify_protocol_exact(
    build_pre: Callable[[stim.Circuit, int], None],
    correction: Callable[[int, Tuple[int, ...]], str],
    L: int,
) -> tuple[bool, float, int]:
    """For every branch of the 2^N measurement outcomes, project and check
    Bell fidelity against |Phi+>. Returns (all_pass, min_fidelity, n_branches)."""
    N = L - 1
    n = n_qubits(L)
    chain = _top_chain(L)
    e0, e1 = chain[0], chain[-1]
    measured = [chain[j] for j in range(1, N + 1)]

    pre_circ = stim.Circuit()
    build_pre(pre_circ, L)

    min_fid = 1.0
    all_pass = True

    for branch in range(2 ** N):
        m = tuple((branch >> k) & 1 for k in range(N))
        sim = stim.TableauSimulator()
        sim.do(pre_circ)

        # Project each intermediate qubit onto its outcome
        valid = True
        for q_idx, m_val in zip(measured, m):
            try:
                sim.postselect_z(q_idx, desired_value=m_val)
            except ValueError:
                valid = False
                break
        if not valid:
            continue

        # Apply correction on e_1
        corr = correction(L, m)
        for ch in corr:
            sim.do(stim.Circuit(f"{ch} {e1}"))

        fid = _fidelity_phi_plus(sim, e0, e1, n)
        min_fid = min(min_fid, fid)
        if fid < 1 - 1e-9:
            all_pass = False

    return all_pass, min_fid, 2 ** N


def run_stim_verification(L_values: List[int] = None) -> list[tuple]:
    if L_values is None:
        L_values = list(range(1, 11))
    print(f"{'Protocol':<22} {'L':>3} {'branches':>9} {'min fid':>20} {'pass':>6}")
    print("-" * 68)
    results = []
    for name, build_pre, correction in PROTOCOLS_STIM:
        for L in L_values:
            all_pass, min_fid, n_b = verify_protocol_exact(build_pre, correction, L)
            status = "PASS" if all_pass else "FAIL"
            print(f"{name:<22} {L:>3} {n_b:>9} {min_fid:>20.15f} {status:>6}", flush=True)
            results.append((name, L, all_pass, min_fid))
    return results


def run_stim_large_L_spot_check(L_values: List[int] = None) -> None:
    """Spot-check our entanglement_swap protocol at large L.

    We verify a single random branch at each L instead of enumerating 2^N."""
    if L_values is None:
        L_values = [20, 30, 50]
    print(f"\n{'Large-L spot check (entanglement_swap, single random branch)':<60}")
    print(f"{'L':>3} {'branch index':>20} {'fidelity':>20} {'pass':>6}")
    print("-" * 55)
    import random
    rng = random.Random(42)
    for L in L_values:
        N = L - 1
        n = n_qubits(L)
        chain = _top_chain(L)
        e0, e1 = chain[0], chain[-1]
        measured = [chain[j] for j in range(1, N + 1)]
        branch = rng.randrange(2 ** N)
        m = tuple((branch >> k) & 1 for k in range(N))

        pre_circ = stim.Circuit()
        _entanglement_swap_pre(pre_circ, L)

        sim = stim.TableauSimulator()
        sim.do(pre_circ)
        for q_idx, m_val in zip(measured, m):
            sim.postselect_z(q_idx, desired_value=m_val)
        corr = _entanglement_swap_correction(L, m)
        for ch in corr:
            sim.do(stim.Circuit(f"{ch} {e1}"))
        fid = _fidelity_phi_plus(sim, e0, e1, n)
        status = "PASS" if fid > 1 - 1e-9 else "FAIL"
        print(f"{L:>3} {branch:>20d} {fid:>20.15f} {status:>6}", flush=True)


if __name__ == "__main__":
    import resource_limits  # noqa: F401
    run_stim_verification()
    run_stim_large_L_spot_check()
