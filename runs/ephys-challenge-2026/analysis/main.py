"""Top-level driver: run verification + benchmarks and log everything.

Usage:
    uv run python main.py > simulation.log 2>&1
"""

from __future__ import annotations

import resource_limits  # noqa: F401  (must be first)

import datetime
import os
import platform
import subprocess
import sys

import numpy
import qiskit
import qiskit_aer
import stim


def log_environment() -> None:
    print("=" * 60)
    print(f"Run started at {datetime.datetime.utcnow().isoformat()}Z")
    print(f"Python: {sys.version.split()[0]} on {platform.platform()}")
    print(f"numpy: {numpy.__version__}")
    print(f"qiskit: {qiskit.__version__}")
    print(f"qiskit-aer: {qiskit_aer.__version__}")
    print(f"stim: {stim.__version__}")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        print(f"git HEAD: {commit}")
    except Exception:  # noqa: BLE001
        print("git HEAD: (not a git repo or git unavailable)")
    print("=" * 60)


def main() -> None:
    log_environment()

    import stim_verification

    print("\n### Stim Clifford-exact verification, L = 1..10 ###\n")
    stim_verification.run_stim_verification(L_values=list(range(1, 11)))
    print("\n### Large-L spot check ###\n")
    stim_verification.run_stim_large_L_spot_check(L_values=[20, 30, 50])

    print("\n### Statevector verification (sanity check, L = 1..6) ###\n")
    import verification
    verification.run_verification(L_max=6)

    print("\n### Scaling benchmark ###\n")
    import scaling_benchmark
    scaling_benchmark.main()

    print("\n=== Run finished. ===")


if __name__ == "__main__":
    main()
