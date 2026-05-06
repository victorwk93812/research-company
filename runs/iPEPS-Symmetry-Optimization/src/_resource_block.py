"""Standard resource limit block — import first in every executable script."""
from __future__ import annotations

import os
import resource

# Restrict C-library thread hijacking
for _env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env, "4")

# CPU only — no CUDA/TPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Hard memory cap = 16 GB
_MAX_MEM = 16 * 1024 * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (_MAX_MEM, _MAX_MEM))
except (ValueError, OSError):
    # Some sandboxes restrict RLIMIT_AS; fall back silently
    pass
