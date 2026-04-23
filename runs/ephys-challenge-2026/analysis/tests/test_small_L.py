"""Exact-Clifford fidelity tests for L = 1..10 via stim."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from stim_verification import (
    PROTOCOLS_STIM,
    verify_protocol_exact,
)


@pytest.mark.parametrize("L", list(range(1, 11)))
@pytest.mark.parametrize(
    "name,build_pre,correction",
    PROTOCOLS_STIM,
    ids=[p[0] for p in PROTOCOLS_STIM],
)
def test_bell_fidelity_all_branches(L, name, build_pre, correction):
    all_pass, min_fid, n_b = verify_protocol_exact(build_pre, correction, L)
    assert all_pass, (
        f"{name} L={L} failed: min fidelity = {min_fid} across {n_b} branches"
    )
    assert min_fid > 1 - 1e-9, f"{name} L={L} min fid {min_fid} below threshold"


if __name__ == "__main__":
    for name, build_pre, correction in PROTOCOLS_STIM:
        for L in range(1, 11):
            ok, f, n = verify_protocol_exact(build_pre, correction, L)
            status = "PASS" if ok else "FAIL"
            print(f"  {name:<22} L={L:2d}  min_fid={f:.12f}  {status}  ({n} branches)")
