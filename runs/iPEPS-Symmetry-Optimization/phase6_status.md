# Phase-6 large-test cycle status

_Generated: 2026-04-27 04:14:18 (auto-finalizer; caveat re-applied 2026-04-27 04:18)_

## Critical caveat (read first)

**The 3 Heisenberg cells that completed in Phase-6 all violate the QMC
variational floor $E_0=-0.66944$**, by margins of $0.16$ (SVD-LBFGS,
$E=-0.832$), $0.27$ (SVD-Adam, $-0.937$), and **$0.66$ (QR-canonical-Adam,
$-1.331$)**. The $\chi=12$ run is in the same "non-variational"
regime as the $\chi=8$ small-magnitude run --- only worse, because the
larger $\chi$ gives the optimiser more truncation-error headroom to
descend into.

Independent post-CTM checks (run on the suspect SVD-LBFGS Heisenberg
seed-0 cell after the diagnostic landed) confirm the CTM environment IS
at a fixed-point ($\|F(C^\star,A){-}C^\star\|\approx 2\times 10^{-7}$);
the non-variational descent is via the $\chi$-truncated Rayleigh
quotient having a sign-unconstrained truncation bias that the optimiser
actively selects for. See `final_review.md` Domain-Expert addendum and
the "Variational-floor failure" paragraph in `report/main.tex` §6 for
the full diagnosis.

The j1j2 cells reach $E=-0.832$ to $-1.082$ at $\chi=12$ for SVD seed-0,
but the same caveat applies. **Phase-7 (`gs_c4v=True`, $\chi=16$,
L-BFGS `memory_size=4`) is in flight and is the proper fix.** See
`phase7_status.md` once it lands.

## Executive summary

The Phase-6 subagent launched the larger benchmark (`src/configs/larger_tests.yaml`, $D=2, \chi=12$, 12 AD steps, 3 seeds where the per-cell budget allowed) and the auto-finalizer collected 46 cells: **14 ok**, 0 timeout, 32 subprocess_error. The strong-eps Lorentzian override (`use_strong_eps(rel=1e-3)`, applied in `src/optimizer_harness.py`) is now installed for every QR-canonical cell to mitigate the `std::bad_alloc` failure mode found in the small-magnitude run.

## Per-cell summary

| model | $h$ or $J_2/J_1$ | ctmrg mode | optimiser | metric | n_ok | n_fail | $E$ mean$\pm$std |
|---|---|---|---|---|---|---|---|
| heisenberg | 0.00 | `qr_canonical` | adam | -- | 1 | 0 | $-1.3306 \pm 0.0000$ |
| heisenberg | 0.00 | `qr_canonical` | lbfgs | -- | 0 | 3 | --- |
| heisenberg | 0.00 | `qr_canonical` | lbfgs | metric | 0 | 2 | --- |
| heisenberg | 0.00 | `svd` | adam | -- | 1 | 0 | $-0.9365 \pm 0.0000$ |
| heisenberg | 0.00 | `svd` | lbfgs | -- | 1 | 2 | $-0.8323 \pm 0.0000$ |
| heisenberg | 0.00 | `svd` | lbfgs | metric | 0 | 2 | --- |
| j1j2 | 0.00 | `qr_canonical` | lbfgs | -- | 0 | 3 | --- |
| j1j2 | 0.00 | `svd` | lbfgs | -- | 1 | 2 | $-0.8323 \pm 0.0000$ |
| j1j2 | 0.40 | `qr_canonical` | lbfgs | -- | 0 | 2 | --- |
| j1j2 | 0.40 | `svd` | lbfgs | -- | 1 | 1 | $-0.9988 \pm 0.0000$ |
| j1j2 | 0.50 | `qr_canonical` | lbfgs | -- | 0 | 3 | --- |
| j1j2 | 0.50 | `qr_canonical` | lbfgs | metric | 0 | 2 | --- |
| j1j2 | 0.50 | `svd` | lbfgs | -- | 1 | 2 | $-1.0404 \pm 0.0000$ |
| j1j2 | 0.50 | `svd` | lbfgs | metric | 0 | 2 | --- |
| j1j2 | 0.55 | `qr_canonical` | lbfgs | -- | 0 | 2 | --- |
| j1j2 | 0.55 | `svd` | lbfgs | -- | 1 | 1 | $-1.0612 \pm 0.0000$ |
| j1j2 | 0.60 | `qr_canonical` | lbfgs | -- | 0 | 2 | --- |
| j1j2 | 0.60 | `svd` | lbfgs | -- | 1 | 1 | $-1.0820 \pm 0.0000$ |
| tfim | 2.50 | `svd` | lbfgs | -- | 2 | 0 | $-2.7701 \pm 0.0093$ |
| tfim | 3.04 | `svd` | lbfgs | -- | 2 | 0 | $-3.1478 \pm 0.0059$ |
| tfim | 3.50 | `svd` | lbfgs | -- | 2 | 0 | $-3.5228 \pm 0.0009$ |

## RQ2 --- did strong-eps QR complete at $J_2/J_1=0.5$?

At J$_1$-J$_2$ with $J_2/J_1=0.5$, QR-canonical with strong-eps (`rel=1e-3`) reached the AD budget on **0 of 5** attempted seeds (across plain and metric-preconditioned L-BFGS).

## RQ5 --- 2$\times$2 interaction at J$_1$-J$_2(0.5)$

| | plain L-BFGS | metric-preconditioned L-BFGS |
|---|---|---|
| **SVD-AD** | $-1.0404 \pm 0.0000$ (1 seeds) | --- |
| **QR-canonical** | --- | --- |

## Honest negative results (cells that did NOT complete)

- `heisenberg` h_or_J2=0.0 `svd` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `heisenberg` h_or_J2=0.0 `svd` lbfgs metric=False seed=2 --- status=`subprocess_error`
- `heisenberg` h_or_J2=0.0 `qr_canonical` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `heisenberg` h_or_J2=0.0 `qr_canonical` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `heisenberg` h_or_J2=0.0 `qr_canonical` lbfgs metric=False seed=2 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.0 `svd` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.0 `svd` lbfgs metric=False seed=2 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.0 `qr_canonical` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.0 `qr_canonical` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.0 `qr_canonical` lbfgs metric=False seed=2 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.4 `svd` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.4 `qr_canonical` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.4 `qr_canonical` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `svd` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `svd` lbfgs metric=False seed=2 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `qr_canonical` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `qr_canonical` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `qr_canonical` lbfgs metric=False seed=2 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.55 `svd` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.55 `qr_canonical` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.55 `qr_canonical` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.6 `svd` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.6 `qr_canonical` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.6 `qr_canonical` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `svd` lbfgs metric=True seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `svd` lbfgs metric=True seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `qr_canonical` lbfgs metric=True seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `qr_canonical` lbfgs metric=True seed=1 --- status=`subprocess_error`
- `heisenberg` h_or_J2=0.0 `svd` lbfgs metric=True seed=0 --- status=`subprocess_error`
- `heisenberg` h_or_J2=0.0 `svd` lbfgs metric=True seed=1 --- status=`subprocess_error`
- `heisenberg` h_or_J2=0.0 `qr_canonical` lbfgs metric=True seed=0 --- status=`subprocess_error`
- `heisenberg` h_or_J2=0.0 `qr_canonical` lbfgs metric=True seed=1 --- status=`subprocess_error`

## What to review first (3 bullets)

1. The RQ2 completion table above --- if `qr_canonical` at J$_1$-J$_2(0.5)$ is now $\ge 50\%$ successful, the `std::bad_alloc` diagnosis from the small-magnitude run is confirmed and the strong-eps mitigation is the right knob to expose to users.
2. The 2$\times$2 interaction table for RQ5 --- a sub-additive interaction (QR+metric not better than max(QR-alone, metric-alone)) is the small-magnitude tentative finding; with more seeds here we can tell whether the conclusion holds or whether the small-magnitude was a fluke.
3. Compare $D=2,\chi=12$ Heisenberg energies to the small-magnitude $D=2,\chi=8$ baseline --- the gap to the QMC reference $-0.66944$ should shrink with $\chi$, providing a per-mode $\chi$-scaling sanity check.

## Files

- Raw per-cell records: `src/data/larger_tests.jsonl`
- Aggregated CSV: `src/data/larger_tests_summary.csv`
- Figures: `src/figures/larger_tests_*.pdf`
- Updated tables: `report/tables/summary_table.tex` (Phase-6 overrides Phase-5)
