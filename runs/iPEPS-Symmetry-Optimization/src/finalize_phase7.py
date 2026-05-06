"""Wait for the Phase-7 benchmark to finish, then write phase7_status.md.

Polls for ``benchmark.py configs/phase7_c4v.yaml`` to disappear from the
process table, then runs analysis + tables + writes a Markdown summary
to ``../phase7_status.md`` with the gs_c4v=True corrected harness
results.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
DATA = SRC / "data" / "phase7_c4v.jsonl"
STATUS = ROOT / "phase7_status.md"
QMC_FLOOR = -0.66944


def _benchmark_running() -> bool:
    out = subprocess.run(
        ["pgrep", "-f", "benchmark.py configs/phase7_c4v.yaml"],
        capture_output=True,
        text=True,
    )
    return bool(out.stdout.strip())


def _load() -> list[dict]:
    if not DATA.exists():
        return []
    rows = []
    for line in DATA.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _key(r):
    return (
        r.get("model"),
        round(r.get("h_or_J2", 0.0), 4),
        r.get("ctmrg_mode"),
        r.get("optimizer"),
        bool(r.get("metric_precond")),
    )


def render(rows: list[dict]) -> str:
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_xc = sum(1 for r in rows if r.get("status") not in ("ok",))

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[_key(r)].append(r)

    # Heisenberg variational floor compliance (RQ1 corrected)
    heis_ok = [r for r in rows if r.get("model") == "heisenberg" and r.get("status") == "ok"]
    heis_above = [r for r in heis_ok if not r.get("below_variational_floor", True)]
    heis_below = [r for r in heis_ok if r.get("below_variational_floor", True)]

    lines = [
        "# Phase-7 c4v + chi=16 corrected re-run status",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} (auto-finalizer)_",
        "",
        "## Executive summary",
        "",
        f"Phase-7 applies the three-patch fix from the user pushback in Phase-6: "
        f"`gs_c4v=True` (C4v sublattice-rotation parametrisation), $\\chi=16$ "
        f"(tenax-safe regime), and L-BFGS `memory_size=4` (down from 10 to dodge "
        f"the OOMs that killed Phase-6 seeds 1+).  Each cell additionally runs a "
        f"$\\chi$-extrapolation cross-check on the converged variational tensor "
        f"at $\\chi\\in\\{{8, 16, 24\\}}$. The auto-finalizer collected "
        f"**{n_total} cells: {n_ok} ok, {n_xc} subprocess_error**.",
        "",
        "## Did `gs_c4v=True` + chi=16 fix the variational-floor failure?",
        "",
        f"**Heisenberg cells above QMC floor $-0.66944$**: "
        f"{len(heis_above)} of {len(heis_ok)} successful cells.",
    ]
    if heis_above:
        lines.append("")
        lines.append("Cells above the floor:")
        for r in heis_above:
            lines.append(
                f"- `{r.get('ctmrg_mode')}` {r.get('optimizer')} "
                f"metric={r.get('metric_precond')} seed={r.get('seed')} "
                f"--- $E={r['final_energy']:+.5f}$, "
                f"residual=${r.get('ctm_residual_post', 'N/A'):.2e}$"
            )
    if heis_below:
        lines.append("")
        lines.append("Cells **below** the floor (still non-variational):")
        for r in heis_below:
            lines.append(
                f"- `{r.get('ctmrg_mode')}` {r.get('optimizer')} "
                f"metric={r.get('metric_precond')} seed={r.get('seed')} "
                f"--- $E={r['final_energy']:+.5f}$"
            )
    if len(heis_above) >= len(heis_ok) // 2 and len(heis_above) >= 1:
        lines.append("")
        lines.append("**Verdict: YES, the Phase-7 fix worked** for at least half of Heisenberg cells.")
    elif heis_ok:
        lines.append("")
        lines.append("**Verdict: PARTIAL** --- some cells variational, others still below floor.")
    else:
        lines.append("")
        lines.append("**Verdict: INSUFFICIENT DATA** --- no Heisenberg cells completed.")

    lines += [
        "",
        "## Per-cell summary",
        "",
        "| model | $J_2/J_1$ or $h$ | mode | optimiser | metric | n_ok | n_fail | $E$ mean$\\pm$std | residual |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(by_cfg):
        recs = by_cfg[k]
        ok = [r for r in recs if r.get("status") == "ok"]
        bad = [r for r in recs if r.get("status") != "ok"]
        if ok:
            E = np.array([r["final_energy"] for r in ok])
            R = np.array([r.get("ctm_residual_post", float("nan")) or float("nan") for r in ok])
            e_str = f"${np.nanmean(E):+.4f} \\pm {np.nanstd(E):.4f}$"
            r_str = f"${np.nanmean(R):.2e}$"
        else:
            e_str = "---"
            r_str = "---"
        m_str = "metric" if k[4] else "--"
        lines.append(
            f"| {k[0]} | {k[1]:.2f} | `{k[2]}` | {k[3]} | {m_str} | {len(ok)} | {len(bad)} | {e_str} | {r_str} |"
        )

    # chi-extrapolation: pull each Heisenberg ok cell's chi_extrap_results
    lines += [
        "",
        "## $\\chi$-extrapolation sanity (Heisenberg, ok cells)",
        "",
        "$E_{\\rm CTM}(A_{\\rm opt}, \\chi')$ re-evaluated on the converged "
        "tensor at $\\chi'\\in\\{8, 16, 24\\}$.  A monotone-increasing "
        "sequence (less negative as $\\chi'$ grows) is the variational signature.",
        "",
        "| seed | mode | optimiser | $E(\\chi'=8)$ | $E(\\chi'=16)$ | $E(\\chi'=24)$ | monotone-up? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in heis_ok:
        ext = r.get("chi_extrap_results") or []
        # ext is [(chi, E, iters, residual), ...]
        e_by_chi = {int(t[0]): float(t[1]) for t in ext if len(t) >= 2}
        e8 = e_by_chi.get(8, float("nan"))
        e16 = e_by_chi.get(16, float("nan"))
        e24 = e_by_chi.get(24, float("nan"))
        mono = "yes" if (e8 <= e16 <= e24) else "no"
        lines.append(
            f"| {r.get('seed')} | `{r.get('ctmrg_mode')}` | {r.get('optimizer')} "
            f"| ${e8:+.4f}$ | ${e16:+.4f}$ | ${e24:+.4f}$ | {mono} |"
        )

    # 2x2 interaction at j1j2(0.5)
    cells22 = {}
    for mode in ("svd", "qr_canonical"):
        for metric in (False, True):
            ok_recs = [
                r for r in rows
                if r.get("model") == "j1j2"
                and round(r.get("h_or_J2", 0.0), 4) == 0.5
                and r.get("ctmrg_mode") == mode
                and r.get("optimizer") == "lbfgs"
                and bool(r.get("metric_precond")) == metric
                and r.get("status") == "ok"
            ]
            cells22[(mode, metric)] = ok_recs
    lines += [
        "",
        "## QR-vs-SVD on the corrected harness (RQ5 2$\\times$2)",
        "",
        "| | plain L-BFGS | metric-preconditioned L-BFGS |",
        "|---|---|---|",
    ]
    def cell(mode, metric):
        recs = cells22.get((mode, metric), [])
        if not recs:
            return "---"
        E = np.array([r["final_energy"] for r in recs])
        return f"${np.nanmean(E):+.4f} \\pm {np.nanstd(E):.4f}$ ({len(recs)} seeds)"
    lines.append(f"| **SVD-AD** | {cell('svd', False)} | {cell('svd', True)} |")
    lines.append(f"| **QR-canonical** | {cell('qr_canonical', False)} | {cell('qr_canonical', True)} |")

    # Failures
    fails = [r for r in rows if r.get("status") != "ok"]
    lines += [
        "",
        "## Honest negative results (cells that did NOT complete)",
        "",
    ]
    if not fails:
        lines.append("None --- all attempted cells completed.")
    else:
        for r in fails:
            lines.append(
                f"- `{r.get('model')}` h_or_J2={r.get('h_or_J2')} "
                f"`{r.get('ctmrg_mode')}` {r.get('optimizer')} "
                f"metric={r.get('metric_precond')} seed={r.get('seed')} "
                f"--- status=`{r.get('status')}`"
            )

    lines += [
        "",
        "## What to review first",
        "",
        "1. The variational-floor verdict above. If at least one Heisenberg "
        "cell is above the QMC floor $-0.66944$ and the chi-extrapolation "
        "is monotone-up, the Phase-7 fix is empirically validated and the "
        "QR-vs-SVD comparison can be re-quoted as physical.",
        "2. The chi-extrapolation table: a downward sequence with chi (more "
        "negative as chi grows) indicates the variational tensor is still "
        "exploiting a different truncation bias --- a deeper fix (e.g. "
        "1-site + sublattice-rotated gate) is needed.",
        "3. The QR-vs-SVD energies on the corrected harness vs the "
        "small-magnitude / Phase-6 numbers: the gap should shrink because "
        "both modes are now in the variational regime.",
        "",
        "## Files",
        "",
        f"- Raw per-cell records: `src/data/phase7_c4v.jsonl`",
        f"- Aggregated CSV: `src/data/phase7_c4v_summary.csv`",
        f"- Figures: `src/figures/phase7_c4v_*.pdf`",
        "- Updated tables (Phase-7 only): `report/tables/summary_table_phase7_c4v.tex`",
        "",
    ]
    return "\n".join(lines)


def main():
    print(f"[finalize-phase7] waiting for Phase-7 benchmark to finish...", flush=True)
    while _benchmark_running():
        time.sleep(20)
    print(f"[finalize-phase7] benchmark done; running analysis + tables", flush=True)

    subprocess.run(
        ["uv", "run", "python", "analysis.py", "phase7_c4v"],
        cwd=SRC,
        check=False,
    )
    subprocess.run(
        ["uv", "run", "python", "make_tables.py", "phase7_c4v"],
        cwd=SRC,
        check=False,
    )

    rows = _load()
    STATUS.write_text(render(rows))
    print(f"[finalize-phase7] wrote {STATUS} with {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
