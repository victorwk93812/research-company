"""Resource-cap block imported first by every entrypoint in this project.

Caps BLAS thread usage to 4 across the common libraries and sets the
address-space rlimit to 16 GB. Import this module *before* jax or numpy.
"""

from __future__ import annotations

import os
import resource

for _env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_env] = "4"

_MAX_MEM = 16 * 1024 * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (_MAX_MEM, _MAX_MEM))
except (ValueError, OSError):
    pass

os.environ.setdefault("JAX_PLATFORMS", "cpu")
