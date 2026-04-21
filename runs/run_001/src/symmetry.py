"""Symmetry-aware bond-spectrum utilities and memory accounting.

All memory is reported in units of scalar entries (a dense float64 occupies 8 bytes).
CGC tensors are treated as shared group-level data and excluded from the SU(2) count,
per the QSpace convention (arXiv:1202.5664) and the X-symbol approach (arXiv:1910.13736).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


def half_int_range(s_max_times_two: int) -> list[float]:
    """Return [0, 0.5, 1, 1.5, ..., s_max] as a list of half-integers."""
    return [i / 2 for i in range(0, s_max_times_two + 1)]


@dataclass(frozen=True)
class BondSpectrum:
    """Multiplicity spectrum d_S for each SU(2) multiplet S on a bond.

    Internally stored with keys 2*S (so keys are plain ints: 0,1,2,3,...),
    values are multiplicities.
    """
    mult: Dict[int, int]  # key = 2*S (int), value = d_S

    def total_bond_dim(self) -> int:
        """chi = sum_S (2S+1) * d_S."""
        return sum((two_s + 1) * d for two_s, d in self.mult.items())

    def max_two_s(self) -> int:
        return max(self.mult) if self.mult else 0

    def mean_multiplet_dim(self) -> float:
        """Weighted <2S+1> on the bond (weighted by dense-basis count)."""
        chi = self.total_bond_dim()
        if chi == 0:
            return 0.0
        return sum((two_s + 1) ** 2 * d for two_s, d in self.mult.items()) / chi


def build_heisenberg_like_spectrum(chi_target: int, s_max_times_two: int) -> BondSpectrum:
    """Construct a Heisenberg-like multiplet spectrum with total bond dim ~ chi_target.

    Heuristic: in the middle of a spin-1/2 antiferromagnetic chain the multiplet
    distribution on a bond is peaked at small S and falls off. We mimic this with
    weights w_S ~ (2S+1) * exp(-S), which is a rough Curie-type approximation to
    the symmetry-resolved entanglement spectrum. We then choose integer
    multiplicities d_S so that chi = sum (2S+1) d_S matches chi_target as close
    as possible from below, and pad the S=0 sector with any residual.
    """
    import math

    s_list = [i for i in range(0, s_max_times_two + 1)]  # 2S values
    weights = []
    for two_s in s_list:
        s = two_s / 2.0
        w = (2 * s + 1) * math.exp(-s)
        weights.append(w)
    w_sum = sum(weights)

    # Preliminary dimension per sector: w_S / w_sum * chi_target
    # We want d_S such that sum (2S+1) d_S ~ chi_target.
    # Set (2S+1) d_S proportional to w_S: d_S ~ w_S / (2S+1) * chi / w_sum.
    # Use plain rounding (no max(1,.)) so that at small chi, high-S sectors
    # drop out naturally instead of being padded to 1.
    mult: Dict[int, int] = {}
    used = 0
    for two_s, w in zip(s_list, weights):
        share = w / w_sum * chi_target
        d = int(round(share / (two_s + 1)))
        if d > 0:
            mult[two_s] = d
            used += (two_s + 1) * d

    # Adjust S=0 sector so chi matches chi_target exactly (down to >=0 int).
    diff = chi_target - used
    if diff != 0:
        mult[0] = max(0, mult.get(0, 0) + diff)
        if mult[0] == 0:
            del mult[0]

    # Drop any zero-multiplicity sectors.
    mult = {k: v for k, v in mult.items() if v > 0}
    return BondSpectrum(mult=mult)


def u1_multiplicities_from_su2(spec: BondSpectrum) -> Dict[int, int]:
    """d^{U(1)}_M = sum_{S >= |M|} d^{SU(2)}_S.

    Keys are 2M (int, can be negative).
    """
    out: Dict[int, int] = {}
    max_two_s = spec.max_two_s()
    # 2M runs over {-max_two_s, -max_two_s + 2, ..., max_two_s}, parity matches
    # the parity of 2S for accessible states. For a spin-1/2 bond spectrum mixing
    # integer and half-integer S, both parities of M occur (from different multiplets).
    for two_m in range(-max_two_s, max_two_s + 1):
        total = 0
        for two_s, d in spec.mult.items():
            if two_s >= abs(two_m) and (two_s - abs(two_m)) % 2 == 0:
                total += d
        if total > 0:
            out[two_m] = total
    return out


def memory_dense(chi: int) -> int:
    """Dense MPS tensor for one site: 2 physical legs * chi_L * chi_R."""
    return 2 * chi * chi


def memory_u1(spec: BondSpectrum) -> int:
    """Sum over abelian blocks allowed by M_R = M_L + s.

    Assumes left bond and right bond have the same spectrum (uniform MPS,
    interior of the chain). This is a standard simplifying assumption for
    memory-scaling benchmarks.
    """
    d_u1 = u1_multiplicities_from_su2(spec)
    total = 0
    for two_s in (-1, +1):  # 2 * s, s in {+1/2, -1/2}
        for two_ml, d_l in d_u1.items():
            two_mr = two_ml + two_s
            d_r = d_u1.get(two_mr, 0)
            total += d_l * d_r
    return total


def memory_su2(spec: BondSpectrum) -> int:
    """Sum over SU(2) fusion-allowed reduced blocks.

    Fusion rule: S_R in {|S_L - 1/2|, S_L + 1/2}.
    """
    total = 0
    for two_sl, d_l in spec.mult.items():
        # 2S_L + 1 fusion with spin-1/2 gives 2S_R = 2S_L + 1 or 2S_L - 1.
        for two_sr in (two_sl - 1, two_sl + 1):
            if two_sr < 0:
                continue
            d_r = spec.mult.get(two_sr, 0)
            total += d_l * d_r
    return total


def memory_bytes(entries: int, dtype_bytes: int = 8) -> int:
    return entries * dtype_bytes


__all__ = [
    "BondSpectrum",
    "build_heisenberg_like_spectrum",
    "u1_multiplicities_from_su2",
    "memory_dense",
    "memory_u1",
    "memory_su2",
    "memory_bytes",
]
