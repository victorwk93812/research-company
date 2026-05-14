"""Wait for the Phase-8 benchmark to finish, then write phase8_status.md.

Polls for ``benchmark.py configs/phase8_new_tenax.yaml`` to disappear from
the process table, then runs analysis + tables + writes a Markdown summary
to ``../phase8_status.md`` covering: per-cell ok/fail count, variational-
floor compliance for Heisenberg / J1J2(0), CTM-residual diagnostic for QR-
canonical (the Phase-7 defect), 2x2 RQ5 cell at J1-J2(0.5), and
chi-extrapolation table.
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
DATA = SRC / "data" / "phase8_new_tenax.jsonl"
STATUS = ROOT / "phase8_status.md"
QMC_FLOOR = -0.66944


def _benchmark_running() -> bool:
    out = subprocess.run(
        ["pgrep", "-f", "benchmark.py configs/phase8_new_tenax.yaml"],
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

    heis_ok = [r for r in rows if r.get("model") == "heisenberg" and r.get("status") == "ok"]
    heis_above = [r for r in heis_ok if not r.get("below_variational_floor", True)]

    lines = [
        "# Phase-8 new-tenax + su_init=True status",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} (auto-finalizer)_",
        "",
        "## Executive summary",
        "",
        f"Phase-8 re-runs the benchmark on the rebased tenax (post-PR #341 / "
        f"#342 / #343 / #346) with `su_init=True` (simple-update warm-start), "
        f"the corrected harness, and `gs_implicit_ad=True` for SVD vs explicit "
        f"AD for QR-canonical (the new tenax implicit-AD validator rejects "
        f"`projector_method != 'svd'`).  $D=2, \\chi=16$, 8 AD steps, 2 seeds. "
        f"Auto-finalizer collected **{n_total} cells: {n_ok} ok, {n_xc} fail**.",
        "",
        "## Heisenberg variational sanity (RQ1)",
        "",
        f"Heisenberg cells with $E_{{\\rm var}} \\ge $ QMC floor $-0.66944$: "
        f"**{len(heis_above)} of {len(heis_ok)} ok cells**.",
    ]
    if heis_ok:
        lines.append("")
        for r in heis_ok:
            tag = f"`{r.get('ctmrg_mode')}` seed={r.get('seed')}"
            E = r.get("final_energy", float("nan"))
            res = r.get("ctm_residual_post")
            res_str = f"{res:.2e}" if isinstance(res, (int, float)) else "n/a"
            ok_floor = "above" if E >= QMC_FLOOR else "BELOW"
            lines.append(
                f"- {tag} --- $E={E:+.5f}$ ({ok_floor} floor), "
                f"CTM residual = ${res_str}$"
            )

    lines += [
        "",
        "## Per-cell summary",
        "",
        "| model | $h$ or $J_2/J_1$ | mode | metric | n_ok | n_fail | $E$ mean$\\pm$std | residual_post |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(by_cfg):
        recs = by_cfg[k]
        ok = [r for r in recs if r.get("status") == "ok"]
        bad = [r for r in recs if r.get("status") != "ok"]
        if ok:
            E = np.array([r["final_energy"] for r in ok])
            R = np.array(
                [r.get("ctm_residual_post") if isinstance(r.get("ctm_residual_post"), (int, float)) else float("nan") for r in ok]
            )
            e_str = f"${np.nanmean(E):+.4f} \\pm {np.nanstd(E):.4f}$"
            r_str = f"${np.nanmean(R):.2e}$" if np.any(~np.isnan(R)) else "---"
        else:
            e_str = "---"
            r_str = "---"
        m_str = "metric" if k[4] else "--"
        lines.append(
            f"| {k[0]} | {k[1]:.2f} | `{k[2]}` | {m_str} | {len(ok)} | {len(bad)} | {e_str} | {r_str} |"
        )

    # chi-extrap table
    lines += [
        "",
        "## $\\chi$-extrapolation cross-check (ok cells)",
        "",
        "| model | $h$/$J_2$ | mode | seed | $E(\\chi'\\!=\\!8)$ | $E(\\chi'\\!=\\!16)$ | $E(\\chi'\\!=\\!24)$ | monotone-up? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in [r for r in rows if r.get("status") == "ok"]:
        ext = r.get("chi_extrap_results") or []
        if not ext:
            continue
        e_by_chi = {int(t[0]): float(t[1]) for t in ext if len(t) >= 2}
        e8 = e_by_chi.get(8, float("nan"))
        e16 = e_by_chi.get(16, float("nan"))
        e24 = e_by_chi.get(24, float("nan"))
        mono = "yes" if (e8 <= e16 <= e24) else "no"
        lines.append(
            f"| {r.get('model')} | {r.get('h_or_J2')} | `{r.get('ctmrg_mode')}` | "
            f"{r.get('seed')} | ${e8:+.4f}$ | ${e16:+.4f}$ | ${e24:+.4f}$ | {mono} |"
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
        "## RQ5 2$\\times$2 interaction at J$_1$-J$_2(0.5)$",
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
        "## Files",
        "",
        "- Raw per-cell records: `src/data/phase8_new_tenax.jsonl`",
        "- Aggregated CSV: `src/data/phase8_new_tenax_summary.csv`",
        "- Figures: `src/figures/phase8_new_tenax_*.pdf`",
        "- Updated tables: `report/tables/summary_table_phase8_new_tenax.tex`",
        "",
    ]
    return "\n".join(lines)


def main():
    print(f"[finalize-phase8] waiting for Phase-8 benchmark to finish...", flush=True)
    while _benchmark_running():
        time.sleep(20)
    print(f"[finalize-phase8] benchmark done; running analysis + tables", flush=True)

    subprocess.run(
        ["uv", "run", "python", "analysis.py", "phase8_new_tenax"],
        cwd=SRC,
        check=False,
    )
    subprocess.run(
        ["uv", "run", "python", "make_tables.py", "phase8_new_tenax"],
        cwd=SRC,
        check=False,
    )

    rows = _load()
    STATUS.write_text(render(rows))
    print(f"[finalize-phase8] wrote {STATUS} with {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
