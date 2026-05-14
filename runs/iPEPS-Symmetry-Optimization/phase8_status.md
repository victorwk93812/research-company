# Phase-8 new-tenax + su_init=True status

_Generated: 2026-05-11 03:48:20 (auto-finalizer)_

## Executive summary

Phase-8 re-runs the benchmark on the rebased tenax (post-PR #341 / #342 / #343 / #346) with `su_init=True` (simple-update warm-start), the corrected harness, and `gs_implicit_ad=True` for SVD vs explicit AD for QR-canonical (the new tenax implicit-AD validator rejects `projector_method != 'svd'`).  $D=2, \chi=16$, 8 AD steps, 2 seeds. Auto-finalizer collected **32 cells: 27 ok, 5 fail**.

## Heisenberg variational sanity (RQ1)

Heisenberg cells with $E_{\rm var} \ge $ QMC floor $-0.66944$: **6 of 6 ok cells**.

- `svd` seed=0 --- $E=-0.51339$ (above floor), CTM residual = $3.35e-01$
- `svd` seed=1 --- $E=-0.52262$ (above floor), CTM residual = $6.05e-01$
- `qr_canonical` seed=0 --- $E=-0.51339$ (above floor), CTM residual = $3.35e-01$
- `qr_canonical` seed=1 --- $E=-0.52262$ (above floor), CTM residual = $6.05e-01$
- `svd` seed=0 --- $E=-0.50127$ (above floor), CTM residual = $4.13e-01$
- `qr_canonical` seed=0 --- $E=-0.50127$ (above floor), CTM residual = $4.13e-01$

## Per-cell summary

| model | $h$ or $J_2/J_1$ | mode | metric | n_ok | n_fail | $E$ mean$\pm$std | residual_post |
|---|---|---|---|---|---|---|---|
| heisenberg | 0.00 | `qr_canonical` | -- | 3 | 0 | $-0.5124 \pm 0.0087$ | $4.51e-01$ |
| heisenberg | 0.00 | `svd` | -- | 3 | 0 | $-0.5124 \pm 0.0087$ | $4.51e-01$ |
| j1j2 | 0.00 | `qr_canonical` | -- | 2 | 0 | $-0.5180 \pm 0.0046$ | $4.70e-01$ |
| j1j2 | 0.00 | `svd` | -- | 2 | 0 | $-0.5180 \pm 0.0046$ | $4.70e-01$ |
| j1j2 | 0.50 | `qr_canonical` | -- | 3 | 0 | $-0.6402 \pm 0.0114$ | $4.51e-01$ |
| j1j2 | 0.50 | `qr_canonical` | metric | 2 | 0 | $-0.6475 \pm 0.0058$ | $4.70e-01$ |
| j1j2 | 0.50 | `svd` | -- | 3 | 0 | $-0.6402 \pm 0.0114$ | $4.51e-01$ |
| j1j2 | 0.50 | `svd` | metric | 2 | 0 | $-0.6475 \pm 0.0058$ | $4.70e-01$ |
| tfim | 2.50 | `qr_canonical` | -- | 1 | 1 | $-1.6493 \pm 0.0000$ | $1.99e+00$ |
| tfim | 2.50 | `svd` | -- | 0 | 2 | --- | --- |
| tfim | 3.04 | `qr_canonical` | -- | 2 | 0 | $-1.0976 \pm 0.9124$ | $1.85e+00$ |
| tfim | 3.04 | `svd` | -- | 1 | 1 | $-3.1099 \pm 0.0000$ | $2.00e+00$ |
| tfim | 3.50 | `qr_canonical` | -- | 2 | 0 | $-0.9371 \pm 0.8616$ | $1.37e+00$ |
| tfim | 3.50 | `svd` | -- | 1 | 1 | $-3.5738 \pm 0.0000$ | $2.00e+00$ |

## $\chi$-extrapolation cross-check (ok cells)

| model | $h$/$J_2$ | mode | seed | $E(\chi'\!=\!8)$ | $E(\chi'\!=\!16)$ | $E(\chi'\!=\!24)$ | monotone-up? |
|---|---|---|---|---|---|---|---|
| heisenberg | 0.0 | `svd` | 0 | $-0.5134$ | $-0.5134$ | $-0.5134$ | no |
| heisenberg | 0.0 | `svd` | 1 | $-0.5226$ | $-0.5226$ | $-0.5226$ | no |
| heisenberg | 0.0 | `qr_canonical` | 0 | $-0.5134$ | $-0.5134$ | $-0.5134$ | no |
| heisenberg | 0.0 | `qr_canonical` | 1 | $-0.5226$ | $-0.5226$ | $-0.5226$ | no |
| j1j2 | 0.0 | `svd` | 0 | $-0.5134$ | $-0.5134$ | $-0.5134$ | no |
| j1j2 | 0.0 | `svd` | 1 | $-0.5226$ | $-0.5226$ | $-0.5226$ | no |
| j1j2 | 0.0 | `qr_canonical` | 0 | $-0.5134$ | $-0.5134$ | $-0.5134$ | no |
| j1j2 | 0.0 | `qr_canonical` | 1 | $-0.5226$ | $-0.5226$ | $-0.5226$ | no |
| j1j2 | 0.5 | `svd` | 0 | $-0.6417$ | $-0.6417$ | $-0.6417$ | no |
| j1j2 | 0.5 | `svd` | 1 | $-0.6533$ | $-0.6533$ | $-0.6533$ | no |
| j1j2 | 0.5 | `qr_canonical` | 0 | $-0.6417$ | $-0.6417$ | $-0.6417$ | no |
| j1j2 | 0.5 | `qr_canonical` | 1 | $-0.6533$ | $-0.6533$ | $-0.6533$ | no |
| j1j2 | 0.5 | `svd` | 0 | $-0.6417$ | $-0.6417$ | $-0.6417$ | no |
| j1j2 | 0.5 | `svd` | 1 | $-0.6533$ | $-0.6533$ | $-0.6533$ | no |
| j1j2 | 0.5 | `qr_canonical` | 0 | $-0.6417$ | $-0.6417$ | $-0.6417$ | no |
| j1j2 | 0.5 | `qr_canonical` | 1 | $-0.6533$ | $-0.6533$ | $-0.6533$ | no |
| tfim | 2.5 | `qr_canonical` | 1 | $-1.6509$ | $-1.6504$ | $-1.6496$ | yes |
| tfim | 3.04 | `svd` | 1 | $-3.0960$ | $-3.0741$ | $-2.9428$ | yes |
| tfim | 3.04 | `qr_canonical` | 0 | $-0.1705$ | $-0.1819$ | $-0.1768$ | no |
| tfim | 3.04 | `qr_canonical` | 1 | $-2.0112$ | $-2.0110$ | $-2.0086$ | yes |
| tfim | 3.5 | `svd` | 1 | $-3.5494$ | $-3.5497$ | $-3.5421$ | no |
| tfim | 3.5 | `qr_canonical` | 0 | $-0.1203$ | $-0.1032$ | $-0.1124$ | no |
| tfim | 3.5 | `qr_canonical` | 1 | $-1.7520$ | $-1.7531$ | $-1.7570$ | no |
| heisenberg | 0.0 | `svd` | 0 | $-0.5013$ | $-0.5013$ | $-0.5013$ | no |
| heisenberg | 0.0 | `qr_canonical` | 0 | $-0.5013$ | $-0.5013$ | $-0.5013$ | no |
| j1j2 | 0.5 | `svd` | 0 | $-0.6256$ | $-0.6256$ | $-0.6256$ | no |
| j1j2 | 0.5 | `qr_canonical` | 0 | $-0.6256$ | $-0.6256$ | $-0.6256$ | no |

## RQ5 2$\times$2 interaction at J$_1$-J$_2(0.5)$

| | plain L-BFGS | metric-preconditioned L-BFGS |
|---|---|---|
| **SVD-AD** | $-0.6402 \pm 0.0114$ (3 seeds) | $-0.6475 \pm 0.0058$ (2 seeds) |
| **QR-canonical** | $-0.6402 \pm 0.0114$ (3 seeds) | $-0.6475 \pm 0.0058$ (2 seeds) |

## Honest negative results (cells that did NOT complete)

- `tfim` h_or_J2=2.5 `svd` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `tfim` h_or_J2=2.5 `svd` lbfgs metric=False seed=1 --- status=`error`
- `tfim` h_or_J2=2.5 `qr_canonical` lbfgs metric=False seed=0 --- status=`error`
- `tfim` h_or_J2=3.04 `svd` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `tfim` h_or_J2=3.5 `svd` lbfgs metric=False seed=0 --- status=`subprocess_error`

## Files

- Raw per-cell records: `src/data/phase8_new_tenax.jsonl`
- Aggregated CSV: `src/data/phase8_new_tenax_summary.csv`
- Figures: `src/figures/phase8_new_tenax_*.pdf`
- Updated tables: `report/tables/summary_table_phase8_new_tenax.tex`
