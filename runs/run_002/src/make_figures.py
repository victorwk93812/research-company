"""Generate convergence-plot figures from results.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    here = Path(__file__).resolve().parent
    fig_dir = here / "figures"
    fig_dir.mkdir(exist_ok=True)
    results = [json.loads(l) for l in open(here / "results.jsonl")]
    # 1) Convergence of a few representative cells
    reps = [r for r in results if r["cell"]["cell_id"] in (36, 44, 54, 58)]
    plt.figure(figsize=(6, 4))
    for r in reps:
        traj = r["result"]["trajectory"]
        if not traj:
            continue
        steps = [t[0] for t in traj]
        es = [t[1] for t in traj]
        label = f"{r['cell']['symmetry']}/{r['cell']['optimizer']}"
        plt.plot(steps, es, marker="o", label=label)
    plt.xlabel("AD step")
    plt.ylabel("energy (arb. units)")
    plt.title("Convergence: canonical U/t=8, t'=0, $\\delta$=0")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "convergence.png", dpi=150)
    plt.close()

    # 2) Bar chart: mean energy by symmetry
    summary = json.load(open(here / "results.json"))
    syms = list(summary["by_symmetry"].keys())
    means = [summary["by_symmetry"][s]["mean_E"] for s in syms]
    mins = [summary["by_symmetry"][s]["min_E"] for s in syms]
    plt.figure(figsize=(6, 4))
    x = range(len(syms))
    plt.bar([i - 0.2 for i in x], means, width=0.4, label="mean E")
    plt.bar([i + 0.2 for i in x], mins, width=0.4, label="min E")
    plt.xticks(list(x), syms, rotation=20)
    plt.ylabel("energy (arb. units)")
    plt.title("Energy vs symmetry (aggregated across cells)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "energy_by_symmetry.png", dpi=150)
    plt.close()

    print("wrote figures to", fig_dir)


if __name__ == "__main__":
    main()
