# Phase-7 c4v + chi=16 corrected re-run status

_Generated: 2026-04-27 04:36:02 (auto-finalizer)_

## Executive summary

Phase-7 applies the three-patch fix from the user pushback in Phase-6: `gs_c4v=True` (C4v sublattice-rotation parametrisation), $\chi=16$ (tenax-safe regime), and L-BFGS `memory_size=4` (down from 10 to dodge the OOMs that killed Phase-6 seeds 1+).  Each cell additionally runs a $\chi$-extrapolation cross-check on the converged variational tensor at $\chi\in\{8, 16, 24\}$. The auto-finalizer collected **16 cells: 9 ok, 7 subprocess_error**.

## Did `gs_c4v=True` + chi=16 fix the variational-floor failure?

**Heisenberg cells above QMC floor $-0.66944$**: 2 of 2 successful cells.

Cells above the floor:
- `svd` lbfgs metric=False seed=0 --- $E=-0.64859$, residual=$1.43e-07$
- `qr_canonical` lbfgs metric=False seed=1 --- $E=-0.62173$, residual=$6.60e-01$

**Verdict: PARTIAL --- and the auto-headline above is misleading.**
The Phase-7 fix works for the SVD baseline (Heisenberg seed-0 SVD-LBFGS gives
$E=-0.6486$ above floor, CTM residual $1.4\times 10^{-7}$, $\chi$-extrapolation
**flat** at $-0.6486$ across $\chi\in\{8,16,24\}$ --- a fully-converged
variational result). It does **not** work for QR-canonical: the QR Heisenberg
seed-1 cell reports $E=-0.6217$ above the floor, but the post-CTM diagnostic
shows the CTM **never reached its fixed point** (residual $\mathbf{0.66}$,
*six orders of magnitude* worse than SVD; CTM hit max\_iter=80 at every
$\chi'$). The $\chi$-extrapolation **slides downward** with $\chi'$
($-0.6288 \to -0.6301 \to -0.6324$), the opposite of the variational
signature. The "above floor" QR result is therefore the AD-step energy,
not a true variational energy --- it sits above the floor only by luck.

**This is a new honest negative result, separate from the small-magnitude
\texttt{std::bad\_alloc} and the Phase-6 OOM**: the QR-canonical projector
at $\chi=16$ produces variational tensors whose CTM contraction is
**non-convergent** in our implementation. It points to a defect (or
mis-formulation) in the QR-canonical projector itself --- e.g. the
single-projector $P_1\!=\!P_2\!=\!P$ form may not be the right object for
fixed-point convergence under the C4v 2-site checkerboard, or the
adaptive Lorentzian eps with the strong-eps override is incompatible with
the C4v geometry. Phase-8 must investigate.

## Per-cell summary

Aggregated by configuration; one row per (model, h/J2, mode, optimiser, metric):

| model | $J_2/J_1$ or $h$ | mode | optimiser | metric | gs_c4v | n_ok | n_fail | $E$ mean$\pm$std | ctm_residual_post | below_var_floor | $E_{\rm extrap}@\chi\!=\!24$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heisenberg | 0.00 | `qr_canonical` | lbfgs | --     | True | 1 | 1 | $-0.6217 \pm 0.0000$ | $6.60\times 10^{-1}$ | False | $-0.6324$ |
| heisenberg | 0.00 | `svd`          | lbfgs | --     | True | 1 | 1 | $-0.6486 \pm 0.0000$ | $1.43\times 10^{-7}$ | False | $-0.6486$ |
| j1j2       | 0.00 | `qr_canonical` | lbfgs | --     | True | 1 | 1 | $-0.6217 \pm 0.0000$ | $6.60\times 10^{-1}$ | n/a   | $-0.6324$ |
| j1j2       | 0.00 | `svd`          | lbfgs | --     | True | 1 | 1 | $-0.6486 \pm 0.0000$ | $1.43\times 10^{-7}$ | n/a   | $-0.6486$ |
| j1j2       | 0.50 | `qr_canonical` | lbfgs | --     | True | 1 | 1 | $-0.8072 \pm 0.0000$ | $1.96\times 10^{0}$  | n/a   | $-0.8096$ |
| j1j2       | 0.50 | `qr_canonical` | lbfgs | metric | True | 2 | 0 | $-0.8109 \pm 0.0039$ | $1.85\times 10^{0}$  | n/a   | $-0.8063$ |
| j1j2       | 0.50 | `svd`          | lbfgs | --     | True | 1 | 1 | $-0.8107 \pm 0.0000$ | $1.43\times 10^{-7}$ | n/a   | $-0.8107$ |
| j1j2       | 0.50 | `svd`          | lbfgs | metric | True | 1 | 1 | $-0.8123 \pm 0.0000$ | $1.17\times 10^{-7}$ | n/a   | $-0.8123$ |

`below_var_floor` is the QMC test ($E_{var}<-0.66944$); only defined for
Heisenberg.  `n/a` for other models.

## $\chi$-extrapolation sanity (Heisenberg, ok cells)

$E_{\rm CTM}(A_{\rm opt}, \chi')$ re-evaluated on the converged tensor at $\chi'\in\{8, 16, 24\}$.  A monotone-increasing sequence (less negative as $\chi'$ grows) is the variational signature.

| seed | mode | optimiser | $E(\chi'=8)$ | $E(\chi'=16)$ | $E(\chi'=24)$ | monotone-up? |
|---|---|---|---|---|---|---|
| 0 | `svd` | lbfgs | $-0.6486$ | $-0.6486$ | $-0.6486$ | yes |
| 1 | `qr_canonical` | lbfgs | $-0.6288$ | $-0.6301$ | $-0.6324$ | no |

## QR-vs-SVD on the corrected harness (RQ5 2$\times$2)

| | plain L-BFGS | metric-preconditioned L-BFGS |
|---|---|---|
| **SVD-AD** | $-0.8107 \pm 0.0000$ (1 seeds) | $-0.8123 \pm 0.0000$ (1 seeds) |
| **QR-canonical** | $-0.8072 \pm 0.0000$ (1 seeds) | $-0.8109 \pm 0.0039$ (2 seeds) |

## Honest negative results (cells that did NOT complete)

- `heisenberg` h_or_J2=0.0 `svd` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `heisenberg` h_or_J2=0.0 `qr_canonical` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.0 `svd` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.0 `qr_canonical` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `svd` lbfgs metric=False seed=1 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `qr_canonical` lbfgs metric=False seed=0 --- status=`subprocess_error`
- `j1j2` h_or_J2=0.5 `svd` lbfgs metric=True seed=1 --- status=`subprocess_error`

## What to review first

1. The variational-floor verdict above. If at least one Heisenberg cell is above the QMC floor $-0.66944$ and the chi-extrapolation is monotone-up, the Phase-7 fix is empirically validated and the QR-vs-SVD comparison can be re-quoted as physical.
2. The chi-extrapolation table: a downward sequence with chi (more negative as chi grows) indicates the variational tensor is still exploiting a different truncation bias --- a deeper fix (e.g. 1-site + sublattice-rotated gate) is needed.
3. The QR-vs-SVD energies on the corrected harness vs the small-magnitude / Phase-6 numbers: the gap should shrink because both modes are now in the variational regime.

## Files

- Raw per-cell records: `src/data/phase7_c4v.jsonl`
- Aggregated CSV: `src/data/phase7_c4v_summary.csv`
- Figures: `src/figures/phase7_c4v_*.pdf`
- Updated tables (Phase-7 only): `report/tables/summary_table_phase7_c4v.tex`
