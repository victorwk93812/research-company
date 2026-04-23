"""Resource limitation block.

Import this module FIRST in every executable script.
Enforces 4-thread BLAS and 16 GB virtual address space cap per process.
"""
from __future__ import annotations

import os
import resource


def apply() -> None:
    for env_var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_var] = "4"
    # JAX-specific: 4-thread XLA on CPU
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_fast_math=false "
                          "--xla_force_host_platform_device_count=1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    max_mem = 16 * 1024 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
    except (ValueError, OSError):
        pass  # may fail in some containers; best effort

apply()
