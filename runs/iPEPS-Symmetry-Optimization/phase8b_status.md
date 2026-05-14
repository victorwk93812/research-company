# Phase-8b explicit-AD status (workaround for new-tenax adjoint NaN)

_Generated: 2026-05-11 04:05:08 (auto-finalizer)_

## Executive summary

Phase-8b re-runs the SVD vs QR-canonical comparison with `gs_implicit_ad=False` for both modes -- the new tenax implicit-AD adjoint produces NaN gradients on the 2-site C4v path at $\chi\geq 12$, so the explicit (unrolled) AD path is the only way to get descent at $\chi=16$.  $D=2, \chi=16$, 8 AD steps, 2 seeds, Heisenberg + J1J2 only (TFIM dropped: SVD already worked in Phase-8 implicit, QR-canonical fails for separate scope-limit reason on 1x1).  Auto-finalizer: **12 cells, 12 ok, 0 fail**.

## Heisenberg variational sanity (RQ1, the headline)

Cells with $E_{\rm var} \ge $ QMC floor $-0.66944$: **4 of 4 ok**.

- `svd` seed=0 --- $E=-0.63338$ (above floor), CTM residual = $3.95e-01$
- `svd` seed=1 --- $E=-0.51186$ (above floor), CTM residual = $5.14e-01$
- `qr_canonical` seed=0 --- $E=-0.63338$ (above floor), CTM residual = $3.95e-01$
- `qr_canonical` seed=1 --- $E=-0.51186$ (above floor), CTM residual = $5.14e-01$

## Per-cell summary

| model | $h$/$J_2$ | mode | metric | n_ok | n_fail | $E$ mean$\pm$std | residual |
|---|---|---|---|---|---|---|---|
| heisenberg | 0.00 | `qr_canonical` | -- | 2 | 0 | $-0.5726 \pm 0.0608$ | $4.55e-01$ |
| heisenberg | 0.00 | `svd` | -- | 2 | 0 | $-0.5726 \pm 0.0608$ | $4.55e-01$ |
| j1j2 | 0.00 | `qr_canonical` | -- | 2 | 0 | $-0.5726 \pm 0.0608$ | $4.55e-01$ |
| j1j2 | 0.00 | `svd` | -- | 2 | 0 | $-0.5726 \pm 0.0608$ | $4.55e-01$ |
| j1j2 | 0.50 | `qr_canonical` | -- | 2 | 0 | $-0.7153 \pm 0.0755$ | $4.61e-01$ |
| j1j2 | 0.50 | `svd` | -- | 2 | 0 | $-0.7153 \pm 0.0755$ | $4.61e-01$ |

## $\chi$-extrapolation cross-check (ok cells)

| model | $h$/$J_2$ | mode | seed | $E(\chi'\!=\!8)$ | $E(\chi'\!=\!16)$ | $E(\chi'\!=\!24)$ | monotone-up? |
|---|---|---|---|---|---|---|---|
| heisenberg | 0.0 | `svd` | 0 | $-0.6334$ | $-0.6334$ | $-0.6334$ | no |
| heisenberg | 0.0 | `svd` | 1 | $-0.5544$ | $-0.5119$ | $-0.5119$ | no |
| heisenberg | 0.0 | `qr_canonical` | 0 | $-0.6334$ | $-0.6334$ | $-0.6334$ | no |
| heisenberg | 0.0 | `qr_canonical` | 1 | $-0.5544$ | $-0.5119$ | $-0.5119$ | no |
| j1j2 | 0.0 | `svd` | 0 | $-0.6334$ | $-0.6334$ | $-0.6334$ | no |
| j1j2 | 0.0 | `svd` | 1 | $-0.5544$ | $-0.5119$ | $-0.5119$ | no |
| j1j2 | 0.0 | `qr_canonical` | 0 | $-0.6334$ | $-0.6334$ | $-0.6334$ | no |
| j1j2 | 0.0 | `qr_canonical` | 1 | $-0.5544$ | $-0.5119$ | $-0.5119$ | no |
| j1j2 | 0.5 | `svd` | 0 | $-0.7908$ | $-0.7908$ | $-0.7908$ | no |
| j1j2 | 0.5 | `svd` | 1 | $-0.6930$ | $-0.6398$ | $-0.6398$ | no |
| j1j2 | 0.5 | `qr_canonical` | 0 | $-0.7908$ | $-0.7908$ | $-0.7908$ | no |
| j1j2 | 0.5 | `qr_canonical` | 1 | $-0.6930$ | $-0.6398$ | $-0.6398$ | no |

## Honest negative results

None --- all attempted cells completed.

## Files

- Raw per-cell records: `src/data/phase8b_explicit_ad.jsonl`
- Aggregated CSV: `src/data/phase8b_explicit_ad_summary.csv`
- Figures: `src/figures/phase8b_explicit_ad_*.pdf`
