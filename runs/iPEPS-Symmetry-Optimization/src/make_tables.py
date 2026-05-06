"""Render LaTeX tables (summary_table.tex, opt_x_mode.tex) from the JSONL data."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "report" / "tables"
TABLES_DIR.mkdir(exist_ok=True, parents=True)


def _load_all(label: str) -> list[dict]:
    rows = []
    candidates = [f"{label}.jsonl", f"{label}_remediation.jsonl", f"{label}_metric.jsonl", f"{label}_full.jsonl"]
    seen_keys = set()
    for fn in candidates:
        p = DATA_DIR / fn
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                # Deduplicate (model, h_or_J2, ctmrg_mode, optimizer, metric_precond, seed)
                key = (
                    r.get("model"),
                    round(r.get("h_or_J2", 0.0), 4),
                    r.get("ctmrg_mode"),
                    r.get("optimizer"),
                    bool(r.get("metric_precond")),
                    int(r.get("seed", -1)),
                )
                if key in seen_keys and r.get("status") != "ok":
                    continue
                if key in seen_keys:
                    # Replace with later occurrence (likely _metric / _full)
                    rows = [x for x in rows if (
                        x.get("model"), round(x.get("h_or_J2", 0.0), 4),
                        x.get("ctmrg_mode"), x.get("optimizer"),
                        bool(x.get("metric_precond")), int(x.get("seed", -1))
                    ) != key]
                seen_keys.add(key)
                rows.append(r)
    return rows


def _key(r):
    return (
        r.get("model"),
        round(r.get("h_or_J2", 0.0), 4),
        r.get("ctmrg_mode"),
        r.get("optimizer"),
        bool(r.get("metric_precond")),
    )


def make_summary(rows: list[dict]) -> str:
    grouped = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        grouped[_key(r)].append(r)

    lines = [
        r"\begin{tabular}{l c l l c c c r r}",
        r"\toprule",
        r"model & $h$ or $J_2/J_1$ & ctmrg mode & optimiser & metric & seeds & $E$ (mean$\pm$std) & wall (s) & steps \\",
        r"\midrule",
    ]
    for k, recs in sorted(grouped.items()):
        model, hj, mode, opt, metric = k
        E = np.array([r["final_energy"] for r in recs])
        W = np.array([r.get("total_wall_time", float("nan")) for r in recs])
        N = np.array([r.get("n_steps", 0) for r in recs])
        e_str = f"${np.nanmean(E):+.4f} \\pm {np.nanstd(E):.4f}$"
        w_str = f"{np.nanmean(W):.1f}"
        n_str = f"{int(np.nanmean(N))}"
        m_str = "metric" if metric else "--"
        # escape mode underscores
        mode_str = mode.replace("_", r"\_")
        lines.append(
            f"{model} & {hj:.2f} & {mode_str} & {opt} & {m_str} & {len(recs)} & {e_str} & {w_str} & {n_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def make_opt_x_mode(rows: list[dict]) -> str:
    """4-cell table for Heisenberg adam vs lbfgs x svd vs qr_canonical."""
    grouped = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("model") != "heisenberg" or r.get("metric_precond"):
            continue
        k = (r["ctmrg_mode"], r["optimizer"])
        grouped.setdefault(k, []).append(r)

    def cell(mode, opt):
        recs = grouped.get((mode, opt), [])
        if not recs:
            return "---"
        E = np.array([r["final_energy"] for r in recs])
        W = np.array([r["total_wall_time"] for r in recs])
        return f"${np.nanmean(E):+.4f}$  (wall {np.nanmean(W):.0f}s)"

    rows_tex = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"& Adam & L-BFGS \\",
        r"\midrule",
        f"SVD-AD       & {cell('svd', 'adam')} & {cell('svd', 'lbfgs')} \\\\",
        f"QR-canonical & {cell('qr_canonical', 'adam')} & {cell('qr_canonical', 'lbfgs')} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(rows_tex) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_label")
    args = ap.parse_args()
    rows = _load_all(args.grid_label)
    print(f"[tables] loaded {len(rows)} rows from {args.grid_label}{{,_remediation}}.jsonl")

    # Phase-7 sentinel: do not overwrite the locked-in summary_table.tex
    # (consumed by report/main.tex from the small-magnitude cycle).  Write
    # phase-labelled files so the Phase-6 outputs remain pristine.
    if args.grid_label.startswith("phase7"):
        s_path = TABLES_DIR / f"summary_table_{args.grid_label}.tex"
        o_path = TABLES_DIR / f"opt_x_mode_{args.grid_label}.tex"
    else:
        s_path = TABLES_DIR / "summary_table.tex"
        o_path = TABLES_DIR / "opt_x_mode.tex"
    s_path.write_text(make_summary(rows))
    print(f"[tables] wrote {s_path}")
    o_path.write_text(make_opt_x_mode(rows))
    print(f"[tables] wrote {o_path}")


if __name__ == "__main__":
    main()
