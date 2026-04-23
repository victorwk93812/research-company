"""Resource-limit header required for every executable script (Engineer persona).

Caps memory to 16 GB and pins each BLAS backend to 4 threads.
Must be imported *before* numpy/scipy/qiskit.
"""

from __future__ import annotations

import os
import resource


def apply_resource_limits(max_mem_bytes: int = 16 * 1024 * 1024 * 1024) -> None:
    for env_var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_var] = "4"
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))
    except (ValueError, OSError):
        pass


apply_resource_limits()
