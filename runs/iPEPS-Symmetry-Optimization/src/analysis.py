"""Aggregate per-cell JSONL records into summary tables and persuasive plots.

Reads ``data/<grid_label>.jsonl`` and writes:
  * ``data/<grid_label>_summary.csv`` — one row per (model, J/h, ctmrg_mode,
    optimizer, metric, seed) with means/stds.
  * ``figures/<grid_label>_*.pdf`` — energy-trajectory, gradient-stability,
    interaction-effect, and timing plots.

All matplotlib output uses a consistent palette.  Plots are rendered without
emojis and with all legends, axes, and units explicit.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


_PALETTE = {
    "svd": "#0072B2",          # blue
    "qr_canonical": "#D55E00", # orange
    "eigh": "#117733",         # green
}
_LABEL = {
    "svd": "SVD-AD (Francuz baseline)",
    "qr_canonical": "QR-canonical AD (this run)",
    "eigh": "eigh-AD",
}
_MARKER = {
    "svd": "o",
    "qr_canonical": "s",
    "eigh": "^",
}


def _load(grid_label: str) -> list[dict]:
    path = DATA_DIR / f"{grid_label}.jsonl"
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # Dedupe by (model, h_or_J2, ctmrg_mode, optimizer, metric_precond, seed):
    # later entries override earlier (so a re-run completing successfully
    # supersedes an earlier failure, and vice versa).
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (
            r.get("model"),
            round(r.get("h_or_J2", 0.0), 4),
            r.get("ctmrg_mode"),
            r.get("optimizer"),
            bool(r.get("metric_precond")),
            int(r.get("seed", -1)),
        )
        prev = seen.get(key)
        if prev is None or (prev.get("status") != "ok" and r.get("status") == "ok"):
            seen[key] = r
    return list(seen.values())


def _key(r: dict) -> tuple:
    return (
        r.get("model"),
        round(r.get("h_or_J2", 0.0), 4),
        r.get("ctmrg_mode"),
        r.get("optimizer"),
        bool(r.get("metric_precond")),
    )


def write_summary_csv(rows: list[dict], grid_label: str) -> Path:
    """Write one row per (model, h/J, mode, opt, metric) with seed-aggregated stats."""
    grouped = defaultdict(list)
    for r in rows:
        if r.get("status") not in ("ok", "error"):
            continue
        grouped[_key(r)].append(r)

    out_lines = [
        "model,h_or_J2,ctmrg_mode,optimizer,metric_precond,n_seeds,"
        "E_mean,E_std,grad_mean,grad_std,wall_mean,wall_std,n_steps_mean"
    ]
    for k, recs in sorted(grouped.items()):
        E = np.array([r["final_energy"] for r in recs])
        G = np.array([r.get("final_grad_norm", float("nan")) for r in recs])
        W = np.array([r.get("total_wall_time", float("nan")) for r in recs])
        N = np.array([r.get("n_steps", 0) for r in recs])
        out_lines.append(
            f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{len(recs)},"
            f"{np.nanmean(E):+.6e},{np.nanstd(E):.3e},"
            f"{np.nanmean(G):.3e},{np.nanstd(G):.3e},"
            f"{np.nanmean(W):.2f},{np.nanstd(W):.2f},"
            f"{np.nanmean(N):.1f}"
        )

    out_path = DATA_DIR / f"{grid_label}_summary.csv"
    out_path.write_text("\n".join(out_lines))
    return out_path


def plot_energy_trajectories(rows: list[dict], model: str, h_or_J2: float, grid_label: str) -> Path:
    """For (model, h/J) plot energy vs step for both modes, all seeds."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    handled = set()
    for r in rows:
        if r.get("model") != model or round(r.get("h_or_J2", 0.0), 4) != round(h_or_J2, 4):
            continue
        if r.get("optimizer") != "lbfgs" or r.get("metric_precond"):
            continue
        if r.get("status") not in ("ok",):
            continue
        e = r.get("energies") or []
        if not e:
            continue
        mode = r["ctmrg_mode"]
        ax.plot(
            range(1, len(e) + 1),
            e,
            "-" + _MARKER.get(mode, "o"),
            color=_PALETTE.get(mode, "k"),
            label=_LABEL.get(mode, mode) if mode not in handled else None,
            markersize=4,
            alpha=0.85,
        )
        handled.add(mode)
    ax.set_xlabel("AD optimisation step")
    ax.set_ylabel(r"$E_{\mathrm{var}}$")
    title_suffix = (
        f"$h/J={h_or_J2}$" if model == "tfim" else f"$J_2/J_1={h_or_J2}$" if model == "j1j2" else ""
    )
    ax.set_title(f"{model.upper()} energy trajectory  {title_suffix}\n($D=2$, $\\chi=8$, L-BFGS, all seeds)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    out = FIG_DIR / f"{grid_label}_traj_{model}_{h_or_J2:.2f}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_gradient_stability(rows: list[dict], model: str, h_or_J2: float, grid_label: str) -> Path:
    """Plot log10(||∇E||) vs step, both modes, all seeds, plain L-BFGS."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    handled = set()
    for r in rows:
        if r.get("model") != model or round(r.get("h_or_J2", 0.0), 4) != round(h_or_J2, 4):
            continue
        if r.get("optimizer") != "lbfgs" or r.get("metric_precond"):
            continue
        if r.get("status") not in ("ok",):
            continue
        gn = np.array(r.get("grad_norms") or [], dtype=float)
        if gn.size == 0:
            continue
        gn = np.where(gn > 0, gn, 1e-30)
        mode = r["ctmrg_mode"]
        ax.semilogy(
            range(1, len(gn) + 1),
            gn,
            "-" + _MARKER.get(mode, "o"),
            color=_PALETTE.get(mode, "k"),
            label=_LABEL.get(mode, mode) if mode not in handled else None,
            markersize=4,
            alpha=0.85,
        )
        handled.add(mode)
    ax.set_xlabel("AD optimisation step")
    ax.set_ylabel(r"$\|\nabla E\|$")
    title_suffix = (
        f"$h/J={h_or_J2}$" if model == "tfim" else f"$J_2/J_1={h_or_J2}$" if model == "j1j2" else ""
    )
    ax.set_title(f"{model.upper()} gradient norm  {title_suffix}\n($D=2$, $\\chi=8$, L-BFGS)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    out = FIG_DIR / f"{grid_label}_grad_{model}_{h_or_J2:.2f}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_2x2_interaction(rows: list[dict], grid_label: str, model: str = "j1j2", h_or_J2: float = 0.5) -> Path:
    """RQ5: 2x2 design (SVD/QR x plain/metric LBFGS).

    Bar chart of mean final energy with seed-level dots overlaid.
    """
    cells = {}
    for mode in ("svd", "qr_canonical"):
        for metric in (False, True):
            recs = [
                r for r in rows
                if r.get("model") == model
                and round(r.get("h_or_J2", 0.0), 4) == round(h_or_J2, 4)
                and r.get("ctmrg_mode") == mode
                and r.get("optimizer") == "lbfgs"
                and bool(r.get("metric_precond")) == metric
                and r.get("status") == "ok"
            ]
            cells[(mode, metric)] = recs

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    xticks = [0, 1]
    width = 0.32
    label_done = set()
    for i, mode in enumerate(("svd", "qr_canonical")):
        for j, metric in enumerate((False, True)):
            recs = cells[(mode, metric)]
            if not recs:
                continue
            E = np.array([r["final_energy"] for r in recs])
            x = j + (-width / 2 if mode == "svd" else width / 2)
            ax.bar(
                x,
                np.mean(E),
                width=width,
                color=_PALETTE[mode],
                edgecolor="k",
                alpha=0.7,
                label=_LABEL[mode] if mode not in label_done else None,
            )
            label_done.add(mode)
            ax.scatter(np.full_like(E, x), E, color="k", s=14, zorder=3)

    ax.set_xticks(xticks)
    ax.set_xticklabels(["plain L-BFGS", "metric-preconditioned L-BFGS"])
    ax.set_ylabel(r"Final $E_{\mathrm{var}}$ (lower is better)")
    title_suffix = (
        f"$h/J={h_or_J2}$" if model == "tfim" else f"$J_2/J_1={h_or_J2}$" if model == "j1j2" else ""
    )
    ax.set_title(f"RQ5: 2$\\times$2 interaction at {model.upper()} {title_suffix}\n($D=2$, $\\chi=8$)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    out = FIG_DIR / f"{grid_label}_2x2_interaction.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_chi_extrap(rows: list[dict], grid_label: str) -> Path:
    """One line per cell: x = chi, y = E_CTM (re-evaluated on converged tensor).

    Phase-7 sanity plot — if E_CTM is monotonically increasing with chi
    (relaxing toward the variational floor from below at small chi), the
    cell is in the convergent regime and the variational-floor failure
    is resolved.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    floor_drawn = False
    nplotted = 0
    for r in rows:
        if r.get("status") != "ok":
            continue
        extrap = r.get("chi_extrap_results") or []
        if not extrap:
            continue
        chis = [t[0] for t in extrap]
        E = [t[1] for t in extrap]
        mode = r.get("ctmrg_mode", "?")
        model = r.get("model", "?")
        hj = r.get("h_or_J2", 0.0)
        seed = r.get("seed", 0)
        metric = r.get("metric_precond", False)
        label = f"{model} h/J2={hj} {mode} seed{seed}{' metric' if metric else ''}"
        ax.plot(
            chis,
            E,
            "-" + _MARKER.get(mode, "o"),
            color=_PALETTE.get(mode, "k"),
            alpha=0.8,
            markersize=6,
            label=label,
        )
        nplotted += 1
        # Draw QMC floor reference once (Heisenberg)
        if model == "heisenberg" and not floor_drawn:
            ax.axhline(-0.66944, color="k", linestyle="--", linewidth=1, alpha=0.6,
                       label=r"QMC reference $E_0 = -0.66944$ (Heis.)")
            floor_drawn = True
    if nplotted == 0:
        plt.close(fig)
        return FIG_DIR / f"{grid_label}_chi_extrap_EMPTY.pdf"
    ax.set_xlabel(r"environment $\chi$ (re-eval, no AD)")
    ax.set_ylabel(r"$E_{\mathrm{CTM}}$ on converged variational tensor")
    ax.set_title(
        "Phase-7 $\\chi$-extrapolation cross-check\n"
        "(monotone-increasing $\\chi \\to $ convergent variational regime)"
    )
    ax.legend(loc="best", fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)
    out = FIG_DIR / f"{grid_label}_chi_extrap.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_wallclock(rows: list[dict], grid_label: str) -> Path:
    """Bar chart of wall-clock per AD step, broken out by ctmrg_mode."""
    by_key = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("optimizer") != "lbfgs" or r.get("metric_precond"):
            continue
        m = r.get("model")
        n = r.get("n_steps") or 0
        w = r.get("total_wall_time") or 0.0
        if n == 0 or w == 0:
            continue
        by_key[(m, r["ctmrg_mode"])].append(w / n)

    keys = sorted({(m, mode) for (m, mode) in by_key.keys()})
    models = sorted({m for (m, _) in keys})
    modes = ["svd", "qr_canonical"]
    width = 0.34
    x_pos = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for i, mode in enumerate(modes):
        means = [
            np.mean(by_key.get((m, mode), [np.nan])) if by_key.get((m, mode)) else np.nan
            for m in models
        ]
        stds = [
            np.std(by_key.get((m, mode), [np.nan])) if by_key.get((m, mode)) else np.nan
            for m in models
        ]
        ax.bar(
            x_pos + (i - 0.5) * width,
            means,
            yerr=stds,
            width=width,
            color=_PALETTE[mode],
            label=_LABEL[mode],
            alpha=0.8,
            edgecolor="k",
            capsize=3,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("wall-clock per AD step (s)")
    ax.set_title("Per-step wall-clock CPU time\n($D=2$, $\\chi=8$, L-BFGS, plain)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    out = FIG_DIR / f"{grid_label}_wallclock.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_label")
    args = ap.parse_args()
    rows = _load(args.grid_label)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[analysis] loaded {len(rows)} rows; {n_ok} ok")

    summary = write_summary_csv(rows, args.grid_label)
    print(f"[analysis] wrote {summary}")

    figs = []
    for (m, h) in [("heisenberg", 0.0), ("j1j2", 0.0), ("j1j2", 0.5), ("tfim", 2.5), ("tfim", 3.04), ("tfim", 3.5)]:
        try:
            figs.append(plot_energy_trajectories(rows, m, h, args.grid_label))
        except Exception as e:  # noqa: BLE001
            print(f"  skip energy {m} {h}: {e}")
        try:
            figs.append(plot_gradient_stability(rows, m, h, args.grid_label))
        except Exception as e:  # noqa: BLE001
            print(f"  skip grad {m} {h}: {e}")
    figs.append(plot_2x2_interaction(rows, args.grid_label))
    figs.append(plot_wallclock(rows, args.grid_label))
    try:
        figs.append(plot_chi_extrap(rows, args.grid_label))
    except Exception as e:  # noqa: BLE001
        print(f"  skip chi_extrap: {e}")

    print(f"[analysis] wrote {len(figs)} figures to {FIG_DIR}")
    for f in figs:
        print(f"   {f}")


if __name__ == "__main__":
    main()
