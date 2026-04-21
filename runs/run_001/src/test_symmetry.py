"""Unit tests for symmetry.py.

Run with: uv run python test_symmetry.py
Tests are pure-Python; no pytest required.
"""
from __future__ import annotations

from symmetry import (
    BondSpectrum,
    build_heisenberg_like_spectrum,
    u1_multiplicities_from_su2,
    memory_dense,
    memory_u1,
    memory_su2,
)


def approx_eq(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_total_bond_dim():
    # d_0 = 2, d_1 (S=1/2) = 3. chi = 1*2 + 2*3 = 8.
    spec = BondSpectrum(mult={0: 2, 1: 3})
    assert spec.total_bond_dim() == 8, spec.total_bond_dim()


def test_u1_from_su2_singlet_only():
    # Pure S=0 multiplets: all U(1) weight sits at M=0.
    spec = BondSpectrum(mult={0: 5})
    d_u1 = u1_multiplicities_from_su2(spec)
    assert d_u1 == {0: 5}, d_u1


def test_u1_from_su2_doublet():
    # Pure S=1/2 (2S=1), d=4. U(1) sectors: M=-1/2, M=+1/2 each have d=4.
    spec = BondSpectrum(mult={1: 4})
    d_u1 = u1_multiplicities_from_su2(spec)
    assert d_u1 == {-1: 4, 1: 4}, d_u1


def test_u1_from_su2_triplet():
    # S=1 multiplet (2S=2), d=3. U(1) sectors: M=-1,0,+1 each d=3.
    spec = BondSpectrum(mult={2: 3})
    d_u1 = u1_multiplicities_from_su2(spec)
    assert d_u1 == {-2: 3, 0: 3, 2: 3}, d_u1


def test_u1_from_su2_mixed_parity_separation():
    # Integer and half-integer multiplets contribute to different M-parities.
    # S=0 (2S=0, d=2) + S=1/2 (2S=1, d=3): integer Ms from S=0, half-int from S=1/2.
    spec = BondSpectrum(mult={0: 2, 1: 3})
    d_u1 = u1_multiplicities_from_su2(spec)
    # 2M = -1: only S=1/2 sector -> 3
    # 2M = 0:  only S=0 sector -> 2
    # 2M = +1: only S=1/2 sector -> 3
    assert d_u1 == {-1: 3, 0: 2, 1: 3}, d_u1


def test_memory_dense():
    assert memory_dense(10) == 200


def test_memory_u1_selection_rule():
    # S=0 only: fusion with spin-1/2 takes S=0 -> S=1/2, which has no states.
    # So both U(1) blocks are zero (no d_U1[M + s]).
    spec = BondSpectrum(mult={0: 3})
    # U(1) count: d_U1 = {0: 3}. For s=+1/2 (2s=+1), M_R=+1/2, d_U1[+1]=0. Same for -1/2.
    assert memory_u1(spec) == 0


def test_memory_u1_doublet():
    # Pure S=1/2 bond, d=4. d_U1 = {-1: 4, +1: 4}.
    # s=+1/2: pairs (M_L=-1/2 -> M_R=0) = 4*0=0; (M_L=+1/2 -> M_R=+1) = 4*0=0.
    # Wait, M_R=+1/2+1/2=+1, but d_U1[+2]=0. So 0. Same for s=-1/2.
    # Actually a bond with pure S=1/2 is closed only at singlet sector.
    # This test verifies we respect the selection rule strictly.
    spec = BondSpectrum(mult={1: 4})
    # For s = +1/2 (2s=+1): 2M_L=-1 -> 2M_R=0, d_U1[0]=0.
    #                       2M_L=+1 -> 2M_R=+2, d_U1[+2]=0.
    # So memory_u1 = 0 for this spectrum.
    assert memory_u1(spec) == 0


def test_memory_u1_coupled_spectrum():
    # S=0 (d=2) + S=1/2 (d=3): d_U1 = {-1:3, 0:2, +1:3}.
    # s=+1/2 (2s=+1):
    #   2M_L=-1 -> 2M_R=0: 3*2 = 6
    #   2M_L= 0 -> 2M_R=+1: 2*3 = 6
    #   2M_L=+1 -> 2M_R=+2: 3*0 = 0
    # s=-1/2 (2s=-1) mirrors: 0 + 6 + 6 = 12
    # Total = 24.
    spec = BondSpectrum(mult={0: 2, 1: 3})
    assert memory_u1(spec) == 24, memory_u1(spec)


def test_memory_su2_coupled_spectrum():
    # S=0 (d=2) + S=1/2 (d=3): fusion S=0 <-> S=1/2.
    # Pairs: (S_L=0, S_R=1/2): 2*3=6; (S_L=1/2, S_R=0): 3*2=6; (S_L=1/2, S_R=1): 3*0=0.
    # Total = 12.
    spec = BondSpectrum(mult={0: 2, 1: 3})
    assert memory_su2(spec) == 12, memory_su2(spec)


def test_memory_su2_le_memory_u1():
    # For any spectrum, memory_su2 should be <= memory_u1 (CGC excluded), since
    # SU(2) is block-compressed relative to U(1).
    for chi in (16, 32, 64, 128):
        for smax in (2, 4, 6, 8):
            spec = build_heisenberg_like_spectrum(chi, smax)
            assert memory_su2(spec) <= memory_u1(spec), (chi, smax, memory_su2(spec), memory_u1(spec))


def test_memory_u1_le_memory_dense():
    for chi in (16, 32, 64, 128):
        for smax in (2, 4, 6, 8):
            spec = build_heisenberg_like_spectrum(chi, smax)
            # memory_u1 uses the true spectrum chi; compare to 2 * chi_true^2.
            chi_true = spec.total_bond_dim()
            assert memory_u1(spec) <= memory_dense(chi_true)


def test_heisenberg_like_spectrum_chi():
    # Check chi is close to target.
    for chi in (16, 32, 64, 128, 256):
        spec = build_heisenberg_like_spectrum(chi, s_max_times_two=8)
        assert abs(spec.total_bond_dim() - chi) <= 4, (chi, spec.total_bond_dim())


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
