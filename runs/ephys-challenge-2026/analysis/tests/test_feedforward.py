"""Exercise the closed-form feed-forward correction map of §3 of the theory draft."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from stim_verification import (
    _entanglement_swap_correction,
    _entanglement_swap_pre,
    verify_protocol_exact,
)


def test_correction_even_N4():
    """For N=4, correction on e_1 is X^{m_2 XOR m_4} Z^{m_1 XOR m_3}."""
    L = 5  # N = 4
    assert _entanglement_swap_correction(L, (0, 0, 0, 0)) == ""
    assert _entanglement_swap_correction(L, (1, 0, 0, 0)) == "Z"  # m_1=1 -> Z
    assert _entanglement_swap_correction(L, (0, 1, 0, 0)) == "X"  # m_2=1 -> X
    assert _entanglement_swap_correction(L, (1, 1, 0, 0)) == "XZ"
    assert _entanglement_swap_correction(L, (1, 0, 1, 0)) == ""  # Z XOR Z = I
    assert _entanglement_swap_correction(L, (0, 1, 0, 1)) == ""  # X XOR X = I


def test_correction_odd_N3():
    """For N=3, correction on e_1 is X^{m_2} Z^{m_1 XOR m_3}."""
    L = 4  # N = 3
    assert _entanglement_swap_correction(L, (0, 0, 0)) == ""
    assert _entanglement_swap_correction(L, (1, 0, 0)) == "Z"
    assert _entanglement_swap_correction(L, (0, 1, 0)) == "X"
    assert _entanglement_swap_correction(L, (1, 1, 1)) == "X"  # X from m_2, Z XOR Z = I


@pytest.mark.parametrize("L", list(range(2, 11)))
def test_odd_and_even_both_work(L):
    all_pass, min_fid, n_b = verify_protocol_exact(
        _entanglement_swap_pre, _entanglement_swap_correction, L
    )
    assert all_pass, f"L={L} fails: min_fid={min_fid}"


if __name__ == "__main__":
    test_correction_even_N4()
    test_correction_odd_N3()
    for L in range(2, 11):
        test_odd_and_even_both_work(L)
    print("All feed-forward tests pass.")
