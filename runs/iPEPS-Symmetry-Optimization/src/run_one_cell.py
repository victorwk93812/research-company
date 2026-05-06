"""Single-cell runner intended for subprocess invocation.

Reads a JSON-encoded cell spec from argv[1], runs ``run_cell``, and writes the
JSON-encoded ``CellResult.to_dict()`` to argv[2].  Designed to be launched by
``benchmark.py`` so that each cell runs in a fresh JAX process — this both
isolates the cell from cross-cell JIT cache leaks (which trigger occasional
``std::bad_alloc`` on small-mem nodes) and provides a clean per-cell wall
clock.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import _resource_block  # noqa: F401  -- must come before any jax import

from optimizer_harness import run_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec_json")
    ap.add_argument("out_path")
    args = ap.parse_args()

    spec = json.loads(args.spec_json)
    out_path = Path(args.out_path)

    try:
        r = run_cell(
            model=spec["model"],
            D=spec["D"],
            chi=spec["chi"],
            ctmrg_mode=spec["ctmrg_mode"],
            optimizer=spec["optimizer"],
            seed=spec["seed"],
            h_or_J2=spec.get("h_or_J2", 0.0),
            num_steps=spec["num_steps"],
            metric_precond=bool(spec.get("metric_precond", False)),
            gs_c4v=bool(spec.get("gs_c4v", False)),
            chi_extrap=tuple(spec.get("chi_extrap", []) or []),
        )
        rec = r.to_dict()
        rec["status"] = "ok" if r.error is None else "error"
    except Exception as exc:  # noqa: BLE001
        rec = {
            "status": "exception",
            "error": repr(exc)[:300],
            "traceback": traceback.format_exc()[-1000:],
            **spec,
        }
    out_path.write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
