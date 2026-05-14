"""Slides-only redraw of the noise plot with overlap-friendly styling.

Under our deterministic density-matrix noise model, ``entanglement_swap``
and ``cat_chain`` produce identical Bell fidelities at every L (both
incur the same number of two-qubit depolarising errors that touch the
endpoint qubits). In the default plot the blue curve is therefore hidden
behind the green one.

This script reruns the same noise sweep (``noise_benchmark`` from
``scaling_benchmark``) and re-plots with:

  * distinct marker shapes for each protocol (o, s, D, ^),
  * a small ±0.10 horizontal jitter on the two coincident curves so
    that their markers sit visibly side-by-side instead of on top of
    each other,
  * a slightly thicker entanglement_swap line so its colour remains
    visible behind cat_chain's overlay.

The figure is written to ``figures/noise_slides.pdf``; the original
``figures/noise.pdf`` is untouched.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from scaling_benchmark import PROTOCOLS, noise_benchmark


# Per-protocol styling for the slides plot.
STYLE = {
    "entanglement_swap": dict(marker="o", linestyle="-", x_jitter=-0.10, linewidth=2.0, markersize=7),
    "cat_chain":         dict(marker="s", linestyle="-", x_jitter=+0.10, linewidth=1.5, markersize=7),
    "mogu":              dict(marker="D", linestyle="-", x_jitter= 0.00, linewidth=1.5, markersize=7),
    "swap_chain":        dict(marker="^", linestyle="-", x_jitter= 0.00, linewidth=1.5, markersize=7),
}


def main() -> None:
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)

    L_range = list(range(1, 11))
    print("=== Noise benchmark (p_2 = 1e-2, density-matrix simulator) ===")
    results = noise_benchmark(L_range, p_2q=1e-2)

    if not results:
        raise SystemExit("noise_benchmark returned no results")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Plot order: lay swap_chain and mogu first so the overlapping pair sits on top.
    plot_order = ["swap_chain", "mogu", "cat_chain", "entanglement_swap"]
    for p in PROTOCOLS:
        if p.name not in results:
            continue
        style = STYLE.get(p.name)
        if style is None:
            continue
        plot_order  # noqa: B018 (silence ruff: list is consumed below)
    for name in plot_order:
        if name not in results:
            continue
        proto = next(p for p in PROTOCOLS if p.name == name)
        style = STYLE[name]
        xs = [L + style["x_jitter"] for L in L_range if L in results[name]]
        ys = [results[name][L] for L in L_range if L in results[name]]
        ax.plot(
            xs, ys,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            color=proto.colour,
            label=name,
        )

    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="ideal")
    ax.set_xlabel("L (ladder length)")
    ax.set_ylabel(r"Bell fidelity with $|\Phi^+\rangle$")
    ax.set_title(r"Noisy simulation: depolarising $p_2 = 10^{-2}$")
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(L_range)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5, -0.03,
        "Markers for entanglement_swap and cat_chain are horizontally offset by $\\pm 0.1$ "
        "for visibility; fidelities coincide at every integer $L$.",
        ha="center", fontsize=8, style="italic",
    )
    plt.tight_layout()
    out = os.path.join(outdir, "noise_slides.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
