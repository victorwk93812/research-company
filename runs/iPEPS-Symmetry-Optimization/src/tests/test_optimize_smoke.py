"""End-to-end smoke test: 5 AD steps on Heisenberg D=2 with both modes."""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _resource_block  # noqa: F401

from optimizer_harness import run_cell


def main():
    print("=== SVD baseline ===")
    t0 = time.perf_counter()
    r_svd = run_cell(
        model="heisenberg", D=2, chi=8, ctmrg_mode="svd",
        optimizer="lbfgs", seed=0, num_steps=4, metric_precond=False,
    )
    print(f"  E_final = {r_svd.final_energy:+.6f}")
    print(f"  steps logged = {r_svd.n_steps}, wall = {r_svd.total_wall_time:.2f}s")
    print(f"  error = {r_svd.error}")
    print(f"  energies = {[f'{e:+.4f}' for e in r_svd.energies]}")

    print("\n=== QR-canonical ===")
    r_qr = run_cell(
        model="heisenberg", D=2, chi=8, ctmrg_mode="qr_canonical",
        optimizer="lbfgs", seed=0, num_steps=4, metric_precond=False,
    )
    print(f"  E_final = {r_qr.final_energy:+.6f}")
    print(f"  steps logged = {r_qr.n_steps}, wall = {r_qr.total_wall_time:.2f}s")
    print(f"  error = {r_qr.error}")
    print(f"  energies = {[f'{e:+.4f}' for e in r_qr.energies]}")

    print(f"\nTotal wall-clock = {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
