"""Phase-6 experimental variant: stronger Lorentzian eps schedule.

The small-magnitude run found that ``qr_canonical`` triggers
``std::bad_alloc`` at the J1-J2 frustrated point with the default
Lorentzian eps (1e-7 relative).  This module provides a runtime override
that bumps the relative eps to a much larger value (default 1e-3),
trading some gradient accuracy for boundedness on the GMRES adjoint.

Usage::

    from ctmrg_qr_strong_eps import use_strong_eps
    use_strong_eps(rel=1e-3)  # call BEFORE register_qr_canonical()
"""
from __future__ import annotations

import ctmrg_qr


def use_strong_eps(rel: float = 1e-3, abs_: float = 1e-10) -> None:
    """Override the module-level Lorentzian eps in ctmrg_qr.

    A larger relative eps caps the F-matrix at a smaller value
    (``1/(2*eps)``), preventing the gradient blow-up that triggers
    ``std::bad_alloc`` in the IFT GMRES adjoint at near-degenerate
    iPEPS geometries (J1-J2 at the frustrated point).
    """
    ctmrg_qr._LORENTZ_EPS_REL = rel
    ctmrg_qr._LORENTZ_EPS_ABS = abs_
