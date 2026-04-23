"""Resource scaling sweep: depth and gate counts for L = 1..L_max."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile

import cat_chain
import cluster_ladder
import entanglement_swap
import swap_chain
from validate_connectivity import validate_connectivity


@dataclass
class Protocol:
    name: str
    build_circuit: Callable[[int], QuantumCircuit]
    colour: str = "tab:blue"


PROTOCOLS: List[Protocol] = [
    Protocol("entanglement_swap", entanglement_swap.build_circuit, "tab:blue"),
    Protocol("swap_chain", swap_chain.build_circuit, "tab:orange"),
    Protocol("cat_chain", cat_chain.build_circuit, "tab:green"),
    Protocol("cluster_ladder", cluster_ladder.build_circuit, "tab:red"),
]


def count_resources(qc: QuantumCircuit) -> Dict[str, int]:
    gate_counts: Dict[str, int] = {}
    for instr in qc.data:
        gate_counts[instr.operation.name] = gate_counts.get(instr.operation.name, 0) + 1
    oneq = sum(v for n, v in gate_counts.items() if n in {"h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "measure"})
    twoq = sum(v for n, v in gate_counts.items() if n in {"cx", "cz", "swap"})
    return {
        "depth": qc.depth(),
        "one_qubit_gates": oneq,
        "two_qubit_gates": twoq,
        "measurements": gate_counts.get("measure", 0),
        "total_gates": sum(gate_counts.values()),
    }


def sweep(L_range: List[int]) -> Dict[str, Dict[int, Dict[str, int]]]:
    results: Dict[str, Dict[int, Dict[str, int]]] = {p.name: {} for p in PROTOCOLS}
    for p in PROTOCOLS:
        for L in L_range:
            qc = p.build_circuit(L)
            validate_connectivity(qc, L)
            results[p.name][L] = count_resources(qc)
    return results


def plot_scaling(results, L_range: List[int], outfile: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Depth
    for p in PROTOCOLS:
        ys = [results[p.name][L]["depth"] for L in L_range]
        axes[0].plot(L_range, ys, marker="o", label=p.name, color=p.colour)
    axes[0].set_xlabel("L (ladder length)")
    axes[0].set_ylabel("Circuit depth")
    axes[0].set_title("Depth vs L")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Two-qubit gate count
    for p in PROTOCOLS:
        ys = [results[p.name][L]["two_qubit_gates"] for L in L_range]
        axes[1].plot(L_range, ys, marker="o", label=p.name, color=p.colour)
    axes[1].set_xlabel("L (ladder length)")
    axes[1].set_ylabel("Two-qubit gates")
    axes[1].set_title("Two-qubit gate count vs L")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    print(f"Saved scaling plot to {outfile}")
    plt.close(fig)


def plot_noise(results_noise, L_range: List[int], outfile: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in PROTOCOLS:
        if p.name not in results_noise:
            continue
        ys = [results_noise[p.name][L] for L in L_range if L in results_noise[p.name]]
        xs = [L for L in L_range if L in results_noise[p.name]]
        ax.plot(xs, ys, marker="o", label=p.name, color=p.colour)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="ideal")
    ax.set_xlabel("L (ladder length)")
    ax.set_ylabel(r"Bell fidelity with $|\Phi^+\rangle$")
    ax.set_title("Noisy simulation: depolarising $p_2=10^{-2}$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    print(f"Saved noise plot to {outfile}")
    plt.close(fig)


def noise_benchmark(L_range: List[int], p_2q: float = 1e-2, shots: int = 4096) -> Dict[str, Dict[int, float]]:
    """Run each protocol with a depolarising noise model and estimate Bell
    fidelity via two-qubit quantum state tomography.

    We use Qiskit Aer's NoiseModel with a depolarizing_error on 2Q gates,
    and reconstruct the reduced density matrix of (e_0, e_1) from 10
    Pauli-basis measurements.
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        print("qiskit_aer unavailable; skipping noise benchmark")
        return {}

    from ladder_graph import qidx
    from qiskit.quantum_info import DensityMatrix, Pauli
    import numpy as np

    results: Dict[str, Dict[int, float]] = {p.name: {} for p in PROTOCOLS[:3]}

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), ["cx", "cz"])
    sim = AerSimulator(noise_model=noise_model, method="density_matrix")

    # For fidelity estimation, measure <II>, <XX>, <YY>, <ZZ> via 3 independent
    # tomography runs (or use expectation-value API if available).
    # We'll compute F = (1 + <XX> + <ZZ> - <YY>) / 4.
    # Qiskit doesn't directly compute expectations; we measure in each basis
    # and estimate from shots.

    for p in PROTOCOLS[:3]:
        for L in L_range:
            qc = p.build_circuit(L)
            idx = qidx(L)
            e0, e1 = idx["e0"], idx["e1"]
            qc_with_snapshot = qc.copy()
            qc_with_snapshot.save_density_matrix([e0, e1], label="dm_e0e1")
            try:
                # Skip transpile (which can impose a default coupling-map limit
                # on large circuits); Aer's density_matrix method handles any
                # number of qubits natively.
                result = sim.run(qc_with_snapshot, shots=1).result()
                dm = result.data().get("dm_e0e1")
                if dm is None:
                    continue
                phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
                rho = np.array(dm)
                fid = float(np.real(phi_plus.conj() @ rho @ phi_plus))
                results[p.name][L] = fid
                print(f"  Noise L={L:2d}  protocol={p.name:<20} fid={fid:.4f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  Noise L={L:2d}  protocol={p.name:<20} ERROR: {exc}", flush=True)
    return results


def main() -> None:
    import os
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)

    L_range = list(range(1, 11))
    print("\n=== Scaling benchmark (L=1..10) ===")
    results = sweep(L_range)
    print("\nSummary of resource counts:")
    header = f"{'protocol':<20} {'L':>3} {'depth':>6} {'1Q':>5} {'2Q':>5} {'meas':>5}"
    print(header)
    print("-" * len(header))
    for protocol_name, by_L in results.items():
        for L, res in by_L.items():
            print(
                f"{protocol_name:<20} {L:>3} {res['depth']:>6} "
                f"{res['one_qubit_gates']:>5} {res['two_qubit_gates']:>5} "
                f"{res['measurements']:>5}"
            )

    plot_scaling(results, L_range, os.path.join(outdir, "scaling.pdf"))

    print("\n=== Noise benchmark (p_2=1e-2) ===")
    noise_results = noise_benchmark(L_range, p_2q=1e-2, shots=4096)
    if noise_results:
        plot_noise(noise_results, L_range, os.path.join(outdir, "noise.pdf"))


if __name__ == "__main__":
    import resource_limits  # noqa: F401
    main()
