"""Every built circuit must use only allowed ladder edges."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import cat_chain
import cluster_ladder
import entanglement_swap
import swap_chain
from validate_connectivity import validate_connectivity


BUILDERS = [
    ("entanglement_swap", entanglement_swap.build_circuit),
    ("swap_chain", swap_chain.build_circuit),
    ("cat_chain", cat_chain.build_circuit),
    ("cluster_ladder", cluster_ladder.build_circuit),
]


@pytest.mark.parametrize("L", list(range(1, 11)))
@pytest.mark.parametrize("name,builder", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_connectivity(L, name, builder):
    qc = builder(L)
    validate_connectivity(qc, L)


if __name__ == "__main__":
    for name, b in BUILDERS:
        for L in range(1, 11):
            qc = b(L)
            validate_connectivity(qc, L)
    print("All protocols pass connectivity validation for L = 1..10")
