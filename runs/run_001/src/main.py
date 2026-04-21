"""Memory-scaling benchmark: dense vs U(1) block-sparse vs SU(2) block-sparse MPS tensors.

Reference derivation: ../theory_draft.md and ../report/main.tex.
Core formulas:
  - Dense:   M = 2 * chi^2
  - U(1):    M = sum_{s, M_L} d_U1[M_L] * d_U1[M_L + s]
  - SU(2):   M = sum_{S_L, S_R in {S_L +/- 1/2}} d_S[S_L] * d_S[S_R]
With chi = sum_S (2S+1) * d_S and d_U1[M] = sum_{S >= |M|} d_S.

CGC tensors are treated as shared group-level data and excluded from the SU(2)
count, following the QSpace convention of arXiv:1202.5664.

Produces:
  - src/results.txt       raw per-chi numbers
  - src/figures/memory_scaling.pdf
  - src/figures/memory_ratio.pdf
"""
import os
import resource

for env_var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[env_var] = "4"
MAX_MEM = 16 * 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (MAX_MEM, MAX_MEM))


import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from symmetry import (
    BondSpectrum,
    build_heisenberg_like_spectrum,
    memory_dense,
    memory_u1,
    memory_su2,
    memory_bytes,
    u1_multiplicities_from_su2,
)


@dataclass(frozen=True)
class BenchRow:
    chi_target: int
    chi_true: int
    s_max_two: int
    mean_mult_dim: float
    mem_dense: int
    mem_u1: int
    mem_su2: int

    def ratio_u1_over_su2(self) -> float:
        return self.mem_u1 / self.mem_su2 if self.mem_su2 else float("nan")

    def ratio_dense_over_su2(self) -> float:
        return self.mem_dense / self.mem_su2 if self.mem_su2 else float("nan")


def run_bench(chis: List[int], s_max_two: int) -> List[BenchRow]:
    rows: List[BenchRow] = []
    for chi in chis:
        spec = build_heisenberg_like_spectrum(chi, s_max_two)
        chi_true = spec.total_bond_dim()
        row = BenchRow(
            chi_target=chi,
            chi_true=chi_true,
            s_max_two=s_max_two,
            mean_mult_dim=spec.mean_multiplet_dim(),
            mem_dense=memory_dense(chi_true),
            mem_u1=memory_u1(spec),
            mem_su2=memory_su2(spec),
        )
        rows.append(row)
    return rows


def write_results(rows: List[BenchRow], path: Path) -> None:
    header = (
        "#chi_target chi_true s_max_two mean_(2S+1)  "
        "M_dense  M_U1  M_SU2  R(U1/SU2)  R(dense/SU2)\n"
    )
    with path.open("w") as f:
        f.write(header)
        for r in rows:
            f.write(
                f"{r.chi_target:>11d} "
                f"{r.chi_true:>8d} "
                f"{r.s_max_two:>9d} "
                f"{r.mean_mult_dim:>11.4f}  "
                f"{r.mem_dense:>8d}  "
                f"{r.mem_u1:>5d}  "
                f"{r.mem_su2:>5d}  "
                f"{r.ratio_u1_over_su2():>9.3f}  "
                f"{r.ratio_dense_over_su2():>12.3f}\n"
            )


def plot_memory_scaling(rows: List[BenchRow], outpath: Path) -> None:
    chi_true = [r.chi_true for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(chi_true, [memory_bytes(r.mem_dense) for r in rows],
              "o-", label="Dense (no symmetry)")
    ax.loglog(chi_true, [memory_bytes(r.mem_u1) for r in rows],
              "s--", label=r"$U(1)$ block-sparse")
    ax.loglog(chi_true, [memory_bytes(r.mem_su2) for r in rows],
              "d-.", label=r"$SU(2)$ block-sparse")
    ax.set_xlabel(r"Total bond dimension $\chi$")
    ax.set_ylabel("Memory per MPS tensor (bytes, float64)")
    ax.set_title("Per-tensor memory vs. bond dimension")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_memory_ratio(rows: List[BenchRow], outpath: Path) -> None:
    chi_true = [r.chi_true for r in rows]
    ratios = [r.ratio_u1_over_su2() for r in rows]
    dense_ratios = [r.ratio_dense_over_su2() for r in rows]
    mean_dims = [r.mean_mult_dim for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogx(chi_true, ratios, "s-", label=r"$\mathcal{M}_{U(1)} / \mathcal{M}_{SU(2)}$")
    ax.semilogx(chi_true, dense_ratios, "o--",
                label=r"$\mathcal{M}_{\mathrm{dense}} / \mathcal{M}_{SU(2)}$")
    ax.semilogx(chi_true, mean_dims, "d:", label=r"$\langle 2S+1 \rangle_{\mathrm{bond}}$")
    ax.set_xlabel(r"Total bond dimension $\chi$")
    ax.set_ylabel("Memory reduction ratio")
    ax.set_title("Memory reduction from abelian to non-abelian blocking")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main() -> int:
    # Report banner
    print("=" * 72)
    print("SU(2) vs U(1) MPS memory benchmark (Heisenberg chain)")
    print("=" * 72)
    print(f"Python: {sys.version.split()[0]}, NumPy: {np.__version__}")
    print()

    # Bond-dimension sweep; s_max covers up to S=4 (2S=8).
    chis = [16, 32, 64, 128, 256, 512]
    s_max_two = 8
    rows = run_bench(chis, s_max_two=s_max_two)

    out_dir = Path(__file__).resolve().parent
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    results_path = out_dir / "results.txt"
    write_results(rows, results_path)

    # Echo rows to stdout as a human-readable table.
    print(f"{'chi':>5} {'<2S+1>':>7}  "
          f"{'M_dense':>10}  {'M_U1':>8}  {'M_SU2':>6}  "
          f"{'U1/SU2':>7}  {'dense/SU2':>10}")
    for r in rows:
        print(f"{r.chi_true:>5d} {r.mean_mult_dim:>7.3f}  "
              f"{r.mem_dense:>10d}  {r.mem_u1:>8d}  {r.mem_su2:>6d}  "
              f"{r.ratio_u1_over_su2():>7.3f}  {r.ratio_dense_over_su2():>10.3f}")
    print()

    # Dump a representative spectrum for the largest chi.
    biggest = chis[-1]
    spec = build_heisenberg_like_spectrum(biggest, s_max_two)
    print(f"Representative bond spectrum at chi_target={biggest}:")
    print(f"  multiplet spectrum (2S -> d_S): {dict(sorted(spec.mult.items()))}")
    print(f"  chi_true = {spec.total_bond_dim()}")
    print(f"  <2S+1>   = {spec.mean_multiplet_dim():.3f}")
    print(f"  U(1) multiplicities (2M -> d_U1): "
          f"{dict(sorted(u1_multiplicities_from_su2(spec).items()))}")
    print()

    plot_memory_scaling(rows, figs_dir / "memory_scaling.pdf")
    plot_memory_ratio(rows, figs_dir / "memory_ratio.pdf")
    print(f"Wrote plots to {figs_dir}/")
    print(f"Wrote results table to {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
