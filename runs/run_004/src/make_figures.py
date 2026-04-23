"""Generate the figures referenced from report/main.tex.

Reads summary.json produced by main.py and emits:
  figures/convergence.png
  figures/walltime_by_D.png
  figures/qr_speedup.png
  figures/qr_vs_svd_optim.png
"""
from __future__ import annotations

import json
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resource_cap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    with open(HERE / "summary.json") as fh:
        summary = json.load(fh)

    # --------- Figure 1: projector speedup (QR vs SVD) --------
    proj = summary["projector_microbench"]
    Ds = sorted({r["D"] for r in proj})
    fig, ax = plt.subplots(figsize=(6, 4))
    for D in Ds:
        subs = [r for r in proj if r["D"] == D]
        subs.sort(key=lambda r: r["chi"])
        chi = [r["chi"] for r in subs]
        sp = [r["svd_ms"] / r["qr_ms"] for r in subs]
        ax.plot(chi, sp, "o-", label=f"D={D}")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel(r"environment bond dimension $\chi$")
    ax.set_ylabel("QR speedup (SVD ms / QR ms)")
    ax.set_title("QR-vs-SVD projector step (CPU, 4 threads)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qr_speedup.png", dpi=140)
    plt.close(fig)

    # --------- Figure 2: wall time breakdown vs D (AD mode) --------
    ad = summary["ad_benchmarks"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in ("unrolled", "implicit"):
        for proj_name in ("qr", "svd"):
            subs = [r for r in ad if r["mode"] == mode and r["projector"] == proj_name]
            subs.sort(key=lambda r: r["D"])
            Dvals = [r["D"] for r in subs]
            tot = [r["total_ms"] for r in subs]
            ax.plot(Dvals, tot, "o-", label=f"{proj_name}-{mode}")
    ax.set_xlabel(r"bond dimension $D$")
    ax.set_ylabel("wall time per step (ms)")
    ax.set_yscale("log")
    ax.set_title(r"Total wall time per AD step, $\chi=2D^2$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "walltime_by_D.png", dpi=140)
    plt.close(fig)

    # --------- Figure 3: optimization trajectories -----------
    traces = summary["optimization_traces"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for proj_name, style in [("qr", "b-"), ("svd", "r--")]:
        t = traces[proj_name]
        ax1.plot(t["losses"], style, label=f"{proj_name}")
        ax2.plot(t["gnorms"], style, label=f"{proj_name}")
    ax1.set_ylabel("observable $f$")
    ax2.set_xlabel("Adam step")
    ax2.set_ylabel(r"$\|\nabla f\|$")
    ax2.set_yscale("log")
    ax1.set_title(r"Optimization trajectory, $D{=}2, \chi{=}4$")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence.png", dpi=140)
    plt.close(fig)

    # --------- Figure 4: FD vs AD scatter (gradient-check accuracy) ---------
    fig, ax = plt.subplots(figsize=(5, 4))
    fd = np.array([r["fd_directional"] for r in ad])
    adv = np.array([r["ad_directional"] for r in ad])
    mask_finite = np.isfinite(fd) & np.isfinite(adv)
    for i, r in enumerate(ad):
        if not mask_finite[i]:
            continue
        marker = "o" if r["projector"] == "qr" else "x"
        col = "C0" if r["mode"] == "unrolled" else "C3"
        ax.scatter(fd[i], adv[i], marker=marker, c=col,
                   label=f'{r["projector"]}-{r["mode"]}')
    lo = min(np.nanmin(fd[mask_finite]), np.nanmin(adv[mask_finite]))
    hi = max(np.nanmax(fd[mask_finite]), np.nanmax(adv[mask_finite]))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    ax.set_xlabel("central-FD directional derivative")
    ax.set_ylabel("AD directional derivative")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)
    ax.set_title("Gradient-check: AD vs FD (finite entries only)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qr_vs_svd_optim.png", dpi=140)
    plt.close(fig)

    print("Figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
