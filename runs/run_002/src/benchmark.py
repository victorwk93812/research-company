"""Benchmark grid driver.

Each cell records: cell id, symmetry, optimizer, diff_mode, D, chi,
final energy, gradient norm, iterations to convergence, wall time,
peak RSS, final symmetry fidelity.

Results are written as JSON lines to results.jsonl and a summary to
results.json. Individual trajectories are logged every 10 steps.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp

from ad_pipeline import make_energy_fn, project_c4v, project_symmetric
from ctmrg import CTMRGConfig
from hamiltonian import HubbardParams, two_site_hopping_gate
from optimizers import adam_optimize, lbfgs_optimize, riemannian_lbfgs_optimize


@dataclass
class Cell:
    cell_id: int
    U: float
    t_prime: float
    delta: float
    symmetry: str  # "Z2" / "U1c" / "U1c+U1s" / "U1c+C4v" / "U1c+U1s+C4v"
    optimizer: str  # "adam" / "lbfgs" / "riem_lbfgs"
    diff_mode: str  # "unrolled" / "implicit"
    D: int
    chi: int
    seed: int = 0


@dataclass
class Result:
    cell: Cell
    final_energy: float
    final_grad_norm: float
    iters: int
    wall_time_s: float
    peak_rss_kb: int
    sym_fidelity: float
    trajectory: List[tuple] = field(default_factory=list)


def _build_A(
    key: jax.Array, D: int, symmetry: str
) -> jnp.ndarray:
    """Initialise A with the appropriate symmetry enforcement."""
    d = 4
    A = 0.3 * jax.random.normal(key, (d, D, D, D, D), dtype=jnp.float64)
    # Small virtual-charge alphabet: {-1, 0, +1} spread across D (or {0,1} for D=2)
    if D == 2:
        charges_leg = jnp.array([0, 1], dtype=jnp.int32)
    elif D == 3:
        charges_leg = jnp.array([0, 1, 2], dtype=jnp.int32)
    else:
        charges_leg = jnp.arange(D, dtype=jnp.int32)

    if "U1c" in symmetry:
        A = project_symmetric(A, charges_leg)
    if "C4v" in symmetry:
        A = project_c4v(A)
    A = A / (jnp.linalg.norm(A) + 1e-30)
    return A


def run_cell(cell: Cell, n_steps: int = 80) -> Result:
    key = jax.random.PRNGKey(cell.seed)
    # For doped cells, mu is bisected externally; we absorb only its effect
    # on the bond gate via a linear shift. For this benchmark we approximate
    # the bisection via mu = U/2 * (2*(1-delta) - 1) (Hartree-like).
    mu = cell.U * 0.5 * (2 * (1 - cell.delta) - 1)
    hparams = HubbardParams(U=cell.U, t=1.0, t_prime=cell.t_prime, mu=mu)
    H_bond = two_site_hopping_gate(hparams)

    ctm_cfg = CTMRGConfig(
        D=cell.D,
        chi=cell.chi,
        max_iter=8 if cell.diff_mode == "unrolled" else 15,
        diff_mode=cell.diff_mode,
    )
    env_key, init_key = jax.random.split(key)
    energy_fn = make_energy_fn(H_bond, ctm_cfg, env_key)
    A0 = _build_A(init_key, cell.D, cell.symmetry)

    t0 = time.perf_counter()
    if cell.optimizer == "adam":
        A_final, traj, grad_norm = adam_optimize(energy_fn, A0, lr=3e-3, n_steps=n_steps)
    elif cell.optimizer == "lbfgs":
        A_final, traj, grad_norm = lbfgs_optimize(energy_fn, A0, n_steps=n_steps)
    elif cell.optimizer == "riem_lbfgs":
        A_final, traj, grad_norm = riemannian_lbfgs_optimize(
            energy_fn, A0, n_steps=n_steps
        )
    else:
        raise ValueError(cell.optimizer)
    wall = time.perf_counter() - t0

    # symmetry fidelity: compare A_final to its re-projection.
    if "U1c" in cell.symmetry:
        charges_leg = jnp.array(
            [0, 1] if cell.D == 2 else [0, 1, 2], dtype=jnp.int32
        )
        A_proj = project_symmetric(A_final, charges_leg)
    else:
        A_proj = A_final
    if "C4v" in cell.symmetry:
        A_proj = project_c4v(A_proj)
    resid = jnp.linalg.norm(A_final - A_proj)
    total = jnp.linalg.norm(A_final) + 1e-30
    sym_fid = float(1.0 - resid / total)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return Result(
        cell=cell,
        final_energy=float(energy_fn(A_final)),
        final_grad_norm=float(grad_norm),
        iters=len(traj),
        wall_time_s=float(wall),
        peak_rss_kb=int(peak_rss),
        sym_fidelity=sym_fid,
        trajectory=[t for i, t in enumerate(traj) if i % 10 == 0],
    )


def make_reduced_grid() -> List[Cell]:
    """Return the three-sub-grid reduced benchmark described in
    theory_draft.md Sec. G1/G2/G3."""
    cells: List[Cell] = []
    cid = 0

    # ---- G1: global compass (reduced for runtime) ----
    for U in (4.0, 8.0, 12.0):
        for tp in (0.0, -0.25):
            for dl in (0.0, 0.125):
                for sym in ("Z2", "U1c", "U1c+U1s"):
                    cells.append(
                        Cell(
                            cell_id=cid,
                            U=U,
                            t_prime=tp,
                            delta=dl,
                            symmetry=sym,
                            optimizer="adam",
                            diff_mode="unrolled",
                            D=2,
                            chi=4,
                        )
                    )
                    cid += 1

    # ---- G2: deep-dive (scaled down: D=2 only, chi in {4}, 4 symmetries, 3 opts) ----
    for sym in ("Z2", "U1c", "U1c+U1s", "U1c+C4v"):
        for opt in ("adam", "lbfgs", "riem_lbfgs"):
            for dl in (0.0, 0.125):
                cells.append(
                    Cell(
                        cell_id=cid,
                        U=8.0,
                        t_prime=0.0,
                        delta=dl,
                        symmetry=sym,
                        optimizer=opt,
                        diff_mode="unrolled",
                        D=2,
                        chi=4,
                    )
                )
                cid += 1

    # ---- G3: ablation (one-at-a-time) ----
    base = dict(U=8.0, t_prime=0.0, delta=0.0, symmetry="U1c", D=2, chi=4)
    # SVD backward handled via CTMRGConfig.svd_broadening; diff mode via diff_mode.
    # Here we just vary diff_mode and optimizer/step-control axes.
    for diff_mode in ("unrolled", "implicit"):
        cells.append(
            Cell(
                cell_id=cid,
                optimizer="adam",
                diff_mode=diff_mode,
                **base,
            )
        )
        cid += 1
    for opt in ("adam", "lbfgs", "riem_lbfgs"):
        cells.append(
            Cell(
                cell_id=cid,
                optimizer=opt,
                diff_mode="unrolled",
                **base,
            )
        )
        cid += 1

    return cells


def run_grid(outdir: str, n_steps: int = 60) -> None:
    cells = make_reduced_grid()
    results = []
    with open(f"{outdir}/results.jsonl", "w") as fh:
        for cell in cells:
            try:
                r = run_cell(cell, n_steps=n_steps)
            except Exception as exc:  # pragma: no cover
                r = Result(
                    cell=cell,
                    final_energy=float("nan"),
                    final_grad_norm=float("nan"),
                    iters=0,
                    wall_time_s=0.0,
                    peak_rss_kb=0,
                    sym_fidelity=0.0,
                )
                print(f"cell {cell.cell_id} FAILED: {exc!r}")
            results.append(r)
            fh.write(
                json.dumps({"cell": asdict(r.cell), "result": {
                    "final_energy": r.final_energy,
                    "final_grad_norm": r.final_grad_norm,
                    "iters": r.iters,
                    "wall_time_s": r.wall_time_s,
                    "peak_rss_kb": r.peak_rss_kb,
                    "sym_fidelity": r.sym_fidelity,
                    "trajectory": r.trajectory,
                }})
                + "\n"
            )
            fh.flush()
            print(
                f"cell {cell.cell_id:3d} | sym={cell.symmetry:12s} opt={cell.optimizer:10s} "
                f"D={cell.D} chi={cell.chi} U={cell.U} delta={cell.delta:.3f} "
                f"-> E={r.final_energy:+.6f} ||g||={r.final_grad_norm:.2e} "
                f"iters={r.iters} wall={r.wall_time_s:.2f}s"
            )

    # Summary JSON
    with open(f"{outdir}/results.json", "w") as fh:
        json.dump(
            {
                "n_cells": len(results),
                "by_symmetry": _agg(results, "symmetry"),
                "by_optimizer": _agg(results, "optimizer"),
            },
            fh,
            indent=2,
        )


def _agg(results: List[Result], key: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Result]] = {}
    for r in results:
        k = getattr(r.cell, key)
        buckets.setdefault(k, []).append(r)
    out = {}
    for k, group in buckets.items():
        es = [g.final_energy for g in group if g.final_energy == g.final_energy]
        wt = [g.wall_time_s for g in group]
        out[k] = {
            "n": len(group),
            "mean_E": sum(es) / max(1, len(es)),
            "min_E": min(es) if es else float("nan"),
            "mean_wall_s": sum(wt) / max(1, len(wt)),
        }
    return out
