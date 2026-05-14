"""Wait for the Phase-8b explicit-AD benchmark to finish, then write
phase8b_status.md.  Mirrors finalize_phase8 but for the explicit-AD
follow-on grid.
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
DATA = SRC / "data" / "phase8b_explicit_ad.jsonl"
STATUS = ROOT / "phase8b_status.md"
QMC_FLOOR = -0.66944


def _benchmark_running() -> bool:
    out = subprocess.run(
        ["pgrep", "-f", "benchmark.py configs/phase8b_explicit_ad.yaml"],
        capture_output=True, text=True,
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
        "# Phase-8b explicit-AD status (workaround for new-tenax adjoint NaN)",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} (auto-finalizer)_",
        "",
        "## Executive summary",
        "",
        f"Phase-8b re-runs the SVD vs QR-canonical comparison with "
        f"`gs_implicit_ad=False` for both modes -- the new tenax implicit-AD "
        f"adjoint produces NaN gradients on the 2-site C4v path at "
        f"$\\chi\\geq 12$, so the explicit (unrolled) AD path is the only "
        f"way to get descent at $\\chi=16$.  $D=2, \\chi=16$, 8 AD steps, "
        f"2 seeds, Heisenberg + J1J2 only (TFIM dropped: SVD already worked "
        f"in Phase-8 implicit, QR-canonical fails for separate scope-limit "
        f"reason on 1x1).  Auto-finalizer: **{n_total} cells, "
        f"{n_ok} ok, {n_xc} fail**.",
        "",
        "## Heisenberg variational sanity (RQ1, the headline)",
        "",
        f"Cells with $E_{{\\rm var}} \\ge $ QMC floor $-0.66944$: "
        f"**{len(heis_above)} of {len(heis_ok)} ok**.",
    ]
    if heis_ok:
        lines.append("")
        for r in heis_ok:
            tag = f"`{r.get('ctmrg_mode')}` seed={r.get('seed')}"
            E = r.get("final_energy", float("nan"))
            res = r.get("ctm_residual_post")
            res_str = f"{res:.2e}" if isinstance(res, (int, float)) else "n/a"
            ok_floor = "above" if E >= QMC_FLOOR else "BELOW"
            lines.append(f"- {tag} --- $E={E:+.5f}$ ({ok_floor} floor), CTM residual = ${res_str}$")

    lines += [
        "",
        "## Per-cell summary",
        "",
        "| model | $h$/$J_2$ | mode | metric | n_ok | n_fail | $E$ mean$\\pm$std | residual |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(by_cfg):
        recs = by_cfg[k]
        ok = [r for r in recs if r.get("status") == "ok"]
        bad = [r for r in recs if r.get("status") != "ok"]
        if ok:
            E = np.array([r["final_energy"] for r in ok])
            R = np.array([
                r.get("ctm_residual_post") if isinstance(r.get("ctm_residual_post"), (int, float))
                else float("nan") for r in ok
            ])
            e_str = f"${np.nanmean(E):+.4f} \\pm {np.nanstd(E):.4f}$"
            r_str = f"${np.nanmean(R):.2e}$" if np.any(~np.isnan(R)) else "---"
        else:
            e_str = "---"
            r_str = "---"
        m_str = "metric" if k[4] else "--"
        lines.append(
            f"| {k[0]} | {k[1]:.2f} | `{k[2]}` | {m_str} | {len(ok)} | {len(bad)} | {e_str} | {r_str} |"
        )

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

    fails = [r for r in rows if r.get("status") != "ok"]
    lines += [
        "",
        "## Honest negative results",
        "",
    ]
    if not fails:
        lines.append("None --- all attempted cells completed.")
    else:
        for r in fails:
            lines.append(
                f"- `{r.get('model')}` h_or_J2={r.get('h_or_J2')} "
                f"`{r.get('ctmrg_mode')}` seed={r.get('seed')} "
                f"--- status=`{r.get('status')}`"
            )

    lines += [
        "",
        "## Files",
        "",
        "- Raw per-cell records: `src/data/phase8b_explicit_ad.jsonl`",
        "- Aggregated CSV: `src/data/phase8b_explicit_ad_summary.csv`",
        "- Figures: `src/figures/phase8b_explicit_ad_*.pdf`",
        "",
    ]
    return "\n".join(lines)


def main():
    print("[finalize-phase8b] waiting for Phase-8b benchmark to finish...", flush=True)
    while _benchmark_running():
        time.sleep(20)
    print("[finalize-phase8b] benchmark done; running analysis + tables", flush=True)
    subprocess.run(["uv", "run", "python", "analysis.py", "phase8b_explicit_ad", "--chi", "16"], cwd=SRC, check=False)
    subprocess.run(["uv", "run", "python", "make_tables.py", "phase8b_explicit_ad"], cwd=SRC, check=False)
    rows = _load()
    STATUS.write_text(render(rows))
    print(f"[finalize-phase8b] wrote {STATUS} with {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
