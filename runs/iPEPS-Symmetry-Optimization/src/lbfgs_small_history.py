"""Monkey-patch tenax's L-BFGS optimizer factory to use a smaller memory_size.

Phase-6 cells past seed-0 OOMed at 16 GB during L-BFGS state allocation when
``memory_size=10`` (the tenax default).  The L-BFGS state holds ``2 *
memory_size`` history pytrees, each the size of the full iPEPS parameter
pytree; for D=2, chi=16, a 2-site iPEPS this is ~few MB per pytree, but the
JIT'd update step also caches one or two compiled bytecode copies per cell
in a fresh subprocess, so the cumulative footprint blew past the cap.  We
shrink the L-BFGS history to memory_size=4 to halve that footprint while
still keeping enough curvature info to converge in <=12 AD steps.

Design notes
------------
- We do **not** edit any tenax source.  We replace
  ``tenax.algorithms.ipeps_optimize._build_optimizer`` with a wrapper that,
  for the L-BFGS branch, builds the same optax chain but with our smaller
  memory_size.  All other branches (Adam, CG) fall through to the
  original.
- ``patch_lbfgs_memory_size`` is idempotent: calling it twice with
  different values updates the override value but does not stack patches.
"""
from __future__ import annotations

_ORIGINAL = None
_MEMORY_SIZE = 4


def patch_lbfgs_memory_size(memory_size: int = 4) -> None:
    """Replace tenax's ``_build_optimizer`` with a memory_size-overriding wrapper."""
    global _ORIGINAL, _MEMORY_SIZE
    _MEMORY_SIZE = int(memory_size)

    from tenax.algorithms import ipeps_optimize as _io

    if _ORIGINAL is not None:
        return  # already patched
    _ORIGINAL = _io._build_optimizer

    def _patched(config):
        import optax

        name = config.gs_optimizer.lower()
        if name == "lbfgs":
            return optax.chain(
                optax.scale_by_lbfgs(memory_size=_MEMORY_SIZE),
                optax.clip_by_global_norm(config.gs_max_grad_norm),
                optax.scale(-1.0),
            )
        return _ORIGINAL(config)

    _io._build_optimizer = _patched


def is_patched() -> bool:
    return _ORIGINAL is not None
