"""Monkey-patch tenax's ``_compute_projector_tensor`` to dispatch the
``qr_canonical`` mode to our local implementation under the AD-traced path.

Design notes
------------
- We do **not** edit any tenax source.  We replace the symbol
  ``tenax.algorithms._ctm_projector._compute_projector_tensor`` with a wrapper
  that intercepts the QR mode when JAX tracers are detected, calls our
  ``qr_canonical_projector`` (whose backward is the Francuz-style truncated
  eigh of ``R R^H``), and otherwise falls through to the original tenax
  implementation.
- The non-tracer block-sparse path (``_qr_projector_symmetric``) is left
  intact — that's the forward-only fast path that ships in tenax today.
- Calling ``register_qr_canonical()`` is idempotent.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms import _ctm_projector as _ctmp
from tenax.core.tensor import DenseTensor, SymmetricTensor

from ctmrg_qr import qr_canonical_projector

_ORIGINAL = None


def _wrap_dense_as_input(t):
    """Return underlying dense matrix (jax.Array) for either Tensor type."""
    if isinstance(t, DenseTensor):
        return t._data
    if isinstance(t, SymmetricTensor):
        return t.todense()
    return t


def _qr_canonical_dispatch(C1g, C4g, chi, projector_method, base_charges, projector_backward):
    """Compute the projector pair for ``projector_method == "qr_canonical"``.

    The two output projectors are identical (P_1 == P_2 == P) — matching the
    eigh path in tenax.  The block-sparse path is intentionally not used here:
    the AD-traced backward needs a single dense custom_vjp chain.
    """
    C1g_d = _wrap_dense_as_input(C1g)
    C4g_d = _wrap_dense_as_input(C4g)

    # Concatenate along column axis: M = [C1g | C4g]   shape (f, c1 + c4)
    M = jnp.concatenate([C1g_d, C4g_d], axis=1)
    P_dense = qr_canonical_projector(M, chi)
    k = P_dense.shape[1]

    # Wrap back into the same Tensor type as the inputs, keeping the leg
    # structure that tenax's downstream consumer expects.
    fused_pos = C1g.labels().index("fused")
    fused_idx = C1g.indices[fused_pos]
    chi_new_idx = _ctmp._make_chi_new_index(fused_idx, k, base_charges)

    P = _ctmp._wrap_dense_projector(
        P_dense, fused_idx, chi_new_idx, as_symmetric=isinstance(C1g, SymmetricTensor)
    )
    return P, P


def register_qr_canonical() -> None:
    """Replace ``_compute_projector_tensor`` with a wrapper supporting ``qr_canonical``.

    Safe to call multiple times — only patches once.
    """
    global _ORIGINAL
    if _ORIGINAL is not None:
        return
    _ORIGINAL = _ctmp._compute_projector_tensor

    def _patched(
        C1g,
        C4g,
        chi,
        projector_method: str = "eigh",
        base_charges=None,
        projector_backward: str = "auto",
    ):
        if projector_method == "qr_canonical":
            return _qr_canonical_dispatch(
                C1g, C4g, chi, projector_method, base_charges, projector_backward
            )
        return _ORIGINAL(
            C1g,
            C4g,
            chi,
            projector_method=projector_method,
            base_charges=base_charges,
            projector_backward=projector_backward,
        )

    _ctmp._compute_projector_tensor = _patched

    # Other tenax modules imported _compute_projector_tensor by name at
    # module-load time, so they hold their own references that our patch
    # above does NOT update.  Rebind those references too.
    from tenax.algorithms import (
        _ctm_tensor as _t,
        _ctm_tensor_moves as _tm,
        _ctm_tensor_paired_moves as _tp,
        _split_ctm_tensor_moves as _sm,
    )
    for _mod in (_t, _tm, _tp, _sm):
        if hasattr(_mod, "_compute_projector_tensor"):
            _mod._compute_projector_tensor = _patched

    # Extend the JIT round-trip maps in ad_utils so that
    # CTMConfig.projector_method == "qr_canonical" survives the
    # _config_to_tuple / _config_from_tuple round-trip (used by the JIT'd
    # CTM forward and adjoint paths).
    from tenax.algorithms import ad_utils as _ad

    _ad._PM_STR_TO_INT.setdefault("qr_canonical", 3)
    _ad._PM_INT_TO_STR.setdefault(3, "qr_canonical")


def is_registered() -> bool:
    return _ORIGINAL is not None


def unregister_qr_canonical() -> None:
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    _ctmp._compute_projector_tensor = _ORIGINAL
    _ORIGINAL = None
