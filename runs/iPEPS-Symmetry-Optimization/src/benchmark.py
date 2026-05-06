"""Headline benchmark driver — sweeps a YAML grid in subprocess-isolated cells.

Each cell is spawned via ``run_one_cell.py`` in a fresh Python process, so a
JIT cache leak / std::bad_alloc in one cell does not corrupt later cells.  Per
cell we keep a soft per-cell wall-clock cap (default 90 s) and a global cap
(default 900 s).  Skipped cells are recorded with ``status=skip``.

Usage:
    uv run python benchmark.py configs/small_magnitude.yaml --budget 900
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent / "data"
LOGS_DIR = Path(__file__).parent / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def _short_label(spec: dict) -> str:
    return (
        f"{spec['model']}_{spec.get('h_or_J2', 0.0):.2f}_"
        f"{spec['ctmrg_mode']}_{spec['optimizer']}"
        + ("_metric" if spec.get('metric_precond') else "")
        + f"_seed{spec['seed']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--budget", type=float, default=900.0, help="global wall-clock cap (s)")
    ap.add_argument("--per-cell-budget", type=float, default=90.0, help="per-cell wall-clock cap (s)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    chi = cfg.get("chi_default", 8)
    D = cfg.get("D_default", 2)
    n_steps = cfg.get("num_steps_default", 8)
    label = cfg.get("grid_label", args.config.stem)

    out_path = DATA_DIR / f"{label}.jsonl"
    log_path = LOGS_DIR / f"{label}.log"
    print(f"[benchmark] writing per-cell records to {out_path}")
    print(f"[benchmark] log -> {log_path}")
    print(f"[benchmark] global budget = {args.budget:.0f}s, per-cell = {args.per_cell_budget:.0f}s")

    t_global = time.perf_counter()
    results = []
    log_lines = []

    cell_idx = 0
    here = Path(__file__).parent.resolve()
    runner_path = here / "run_one_cell.py"
    for cell_spec in cfg["cells"]:
        for seed in cell_spec["seeds"]:
            cell_idx += 1
            elapsed = time.perf_counter() - t_global
            if elapsed > args.budget:
                msg = (
                    f"[skip {cell_idx}] global budget exceeded ({elapsed:.0f}s > {args.budget:.0f}s)"
                )
                print(msg, flush=True)
                log_lines.append(msg)
                results.append({"status": "skip", **cell_spec, "seed": seed})
                continue

            spec = dict(cell_spec)
            spec["seed"] = seed
            spec["D"] = D
            spec["chi"] = chi
            spec["num_steps"] = n_steps
            tag = _short_label(spec)
            print(
                f"[cell {cell_idx:>3}] {tag:<60s} (elapsed {elapsed:.0f}s/{args.budget:.0f}s)",
                flush=True,
            )

            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as out_f:
                out_p = Path(out_f.name)
            t_cell = time.perf_counter()
            try:
                proc = subprocess.run(
                    ["uv", "run", "python", str(runner_path), json.dumps(spec), str(out_p)],
                    cwd=here,
                    capture_output=True,
                    text=True,
                    timeout=args.per_cell_budget,
                    env={**os.environ, "JAX_PLATFORMS": "cpu"},
                )
                if proc.returncode != 0 or not out_p.read_text().strip():
                    rec = {
                        "status": "subprocess_error",
                        "error": f"rc={proc.returncode}; stderr={proc.stderr[-500:]!r}",
                        **spec,
                        "tag": tag,
                    }
                else:
                    rec = json.loads(out_p.read_text())
                    rec["tag"] = tag
            except subprocess.TimeoutExpired:
                rec = {"status": "timeout", **spec, "tag": tag, "error": "per-cell timeout"}
            finally:
                out_p.unlink(missing_ok=True)

            rec["cell_wall_time"] = time.perf_counter() - t_cell
            msg = (
                f"  -> status={rec.get('status')}  "
                f"E={rec.get('final_energy', float('nan')):+.5f}  "
                f"steps={rec.get('n_steps', 0)}  "
                f"wall={rec['cell_wall_time']:.1f}s  "
                f"err={rec.get('error')}"
            )
            print(msg, flush=True)
            log_lines.append(f"{tag}: {msg.strip()}")
            results.append(rec)

            with out_path.open("w") as f:
                for r_ in results:
                    f.write(json.dumps(r_) + "\n")

    total = time.perf_counter() - t_global
    msg = f"\n[benchmark] DONE in {total:.0f}s ({total / 60:.1f} min). {len(results)} cells."
    print(msg, flush=True)
    log_lines.append(msg)
    log_path.write_text("\n".join(log_lines))


if __name__ == "__main__":
    main()
