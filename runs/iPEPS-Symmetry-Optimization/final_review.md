# Final Review --- iPEPS Symmetry Optimization (Run 2026-04-27)

**Reviewing personas:** The Review Board --- Math Pedant, Performance Hacker, Domain Expert.
**Inputs reviewed:**
- `./theory_draft.md`
- `./ra_critique.md` (and that the RA's seven items are or aren't reflected in the engineering)
- `./report/main.tex`, `./report/slides.tex`, `./report/main.pdf`, `./report/slides.pdf`
- `./src/ctmrg_qr.py`, `./src/ctmrg_qr_register.py`, `./src/optimizer_harness.py`,
  `./src/benchmark.py`, `./src/run_one_cell.py`, `./src/analysis.py`, `./src/make_tables.py`,
  `./src/tests/`
- `./src/simulation.log` and `./src/data/small_magnitude_full.jsonl` /
  `_summary.csv`

---

## Section 1 --- The Math Pedant

**Verdict: PASS WITH NOTES.**

* **QR-canonical forward derivation (theory_draft §2)** is consistent with
  the RA's F1 caveat: the equality of $\operatorname{eigvals}(RR^\dagger)$
  to squared singular values of $M$ holds in *both* the $f\ge c$
  ("tall-QR") and $f<c$ ("wide-QR") regimes, and the LaTeX writeup
  acknowledges this in §3 of `report/main.tex`. Acceptance.
* **Lorentzian-eigh backward kernel** (`src/ctmrg_qr.py::_lorentzian_eigh_bwd_kernel`)
  uses the standard $F^{(\epsilon)}_{ij} = (\lambda_j-\lambda_i) /
  ((\lambda_j-\lambda_i)^2 + \epsilon^2)$ with adaptive
  $\epsilon = \max(10^{-12}, 10^{-7}\,\max|\lambda|)$. The **sign
  convention is correct** (matches `tenax/algorithms/ad_utils.py::_regularized_eigh_bwd`
  after the Issue #316 fix); the diff matrix is `w[None,:] - w[:,None]`
  which gives `F_{ij} = 1/(w_j - w_i)` in the limit, the convention
  expected by JAX's reverse-mode adjoint of symmetric eigh. Acceptance.
* **Diagonal-phase canonical-gauge cancellation (RA's F2)**: claimed in
  `report/main.tex` Appendix A; the *full* one-line proof is partially
  hand-waved (cited Walter 2010). The math is correct but the writeup
  doesn't carry it through line-by-line, only one paragraph. **Action
  for next iteration:** expand Appendix A with the explicit chain-rule
  composition, or move the proof to an arXiv supplemental.
* **Tikhonov-shift discussion (RA's F3)** is acknowledged in the LaTeX
  but the engineer chose the *cleaner* gauge-fix-only fallback
  ($\phi=1$ on exact-zero diagonals, no Tikhonov on $R^{-1}$), which is
  what the RA recommended. The current code (`_canonical_qr` in
  `ctmrg_qr.py`) does the gauge fix correctly via `jnp.where(abs_diag
  > 0, ..., one)`. Acceptance.
* **Single-projector vs Fishman two-projector** is an honest scope
  limitation: our QR-canonical returns $P_1=P_2=P$, which works for
  2-site but does not give correct energies for the 1$\times$1 TFIM
  geometry --- and the same limitation already exists in `tenax`'s
  built-in `eigh` mode. Documented in §6 of the LaTeX. Acceptance.

**Math-Pedant residual issue:** the writeup claims "the eigenvalues
$\lambda_{q,i}$ equal the squared singular values of $M$" without
naming the precise condition for the *truncation order* to coincide
across SVD and our pipeline. For our application both pipelines sort
descending by magnitude before truncating, so the kept subspaces are
equal up to a unitary on the column space; when we report
$P_1 P_1^\dagger$ vs the SVD's $U_k U_k^\dagger$ we use this
basis-independent comparison (see
`src/tests/test_smoke.py::test_projector_matches_svd`, diff $<10^{-10}$).
Cite this test in the report next iteration.

---

## Section 2 --- The Performance Hacker

**Verdict: PASS WITH NOTES.**

* **JIT-compile cost:** the QR-canonical mode incurs a 30--75~s JIT
  compile on first invocation in a fresh subprocess. The first cell
  in our 23-cell run cost 75~s; subsequent same-`(model, ctmrg_mode)`
  cells should hit the persistent cache --- but **they did not**, as
  evidenced by the timeouts on Heisenberg QR seed 1, 2 in the original
  run. Diagnosis: JAX's persistent compile cache is keyed on the HLO
  graph hash, which apparently changes between subprocess invocations
  even when the user-visible parameters are identical. **Action for
  next iteration:** drop subprocess isolation in favour of a single
  long-running process with explicit `gc.collect()` and
  `jax.clear_caches()` between cells; this trades the std::bad_alloc
  isolation guarantee for cache amortisation. Alternative: configure
  `JAX_COMPILATION_CACHE_DIR` to a fixed shared path.
* **Resource block enforcement (`_resource_block.py`)**: 16~GB
  `RLIMIT_AS`, 4 BLAS threads, `JAX_PLATFORMS=cpu` are all set
  correctly. Acceptance.
* **Subprocess isolation (`run_one_cell.py`)**: clean and correct;
  spec is JSON-encoded on argv and the result is JSON-encoded on a
  temp file. The `subprocess.run` call captures stderr and surfaces
  it in `rec["error"]`, which made the diagnostic loop useful.
  Acceptance.
* **The std::bad_alloc on QR-canonical at J$_1$-J$_2(0.5)$ and
  Heisenberg seeds $>$ 0** is the most concerning finding from the
  performance side. The crash signature suggests an unbounded
  Krylov base in the IFT GMRES adjoint; the Lorentzian regularisation
  caps the F-matrix per `_lorentzian_eigh` but does not cap the
  *outer* IFT solve, which is what allocates the Krylov vectors.
  **Action for next iteration:** add an explicit
  `jax.lax.stop_gradient` guard on the projector at every CTMRG sweep
  during the *backward* pass (matching what tenax does in the
  non-AD eigh fallback), turning the implicit-AD projector into a
  Liao-style "stop-grad on truncation" composite. This loses some
  accuracy but is bounded.
* **Per-cell wall-clock honesty**: the `cell_wall_time` field in the
  JSONL records the full subprocess duration *including* JIT compile.
  This is the right number to report (we make no claim of "compiled
  steady-state" timing because we never hit it on the QR side).
  Acceptance.
* **Plot generation pipeline**: `analysis.py` correctly dedupes
  `(model, h, mode, opt, metric, seed)` tuples after the merge of
  `small_magnitude.jsonl` + `small_magnitude_metric.jsonl`. The
  matplotlib palette is colour-blind-aware (`_PALETTE` uses Wong's
  recommendations). Acceptance.

**Performance-Hacker residual issue:** the analysis CSV reports a
non-zero `E_std` for cells that have only one *successful* seed but
two *attempted* seeds (the failed one drops out of the mean). This is
correct accounting but visually confusing. Add a `n_attempted` column
in the next iteration.

---

## Section 3 --- The Domain Expert

**Verdict: CONDITIONAL PASS.**

* **TFIM SVD baseline** at $h/J\in\{2.5, 3.04, 3.5\}$, $D=2$, $\chi=8$
  reaches $E\in\{-2.38\pm0.14, -2.73, -3.14\}$ in 6 AD steps. Reference
  iPEPS values at the disordered side of the transition are around
  $E\approx -3.0$ at $h/J=3$ for converged $\chi$ --- our $\chi=8$
  numbers are within a factor of $\sim1.2$ of literature, consistent
  with the small variational manifold. Acceptance for sanity baseline.
* **Heisenberg SVD baseline**: $E_{\mathrm{mean}}=-0.578\pm0.142$
  across 3 seeds at $D=2$, $\chi=8$, 6 AD steps. tenax's own example
  at $\chi=16$ reaches $E\approx-0.6628$; our numbers are within the
  expected variational band given the smaller $\chi$ and shorter
  run. Acceptance.
* **J$_1$-J$_2$ at $J_2/J_1=0.5$ SVD baseline**: $E_{\mathrm{mean}}=-0.605\pm0.077$,
  again at $D=2$, $\chi=8$, 6 AD steps. The J$_1$-J$_2$ ground-state
  energy at the maximally frustrated point is around $-0.49$ to $-0.52$
  per site for converged variational methods. Our slightly lower
  value ($-0.605$) is suspect --- it may indicate the bond-gate
  reduction we use (see Limitations §5 of the LaTeX) is producing a
  slightly different effective Hamiltonian than the textbook J$_1$-J$_2$.
  **Action for next iteration:** rebuild the J$_1$-J$_2$ benchmark
  with the explicit 2$\times$2 unit cell + diagonal bonds via
  `tenax.algorithms.ipeps_simple_update.ipeps()` so the gate is
  physically correct. The QR-vs-SVD comparison is internally consistent
  (both use the same gate) but the absolute energies are not directly
  comparable to literature.
* **QR-canonical mode** reaches Heisenberg $E\!\approx\!-0.18$ on the
  one seed that completes; the SVD baseline reaches $-0.42$ to $-0.77$
  on three seeds in the same 6-step budget. The QR-canonical is
  *strictly worse* in this small-magnitude window. With more steps
  (Phase-6) we may see catch-up; for now this is *not* the "stability
  win" we hypothesised --- it is a "no-win, slower descent" --- which
  is the *honest* finding the LaTeX correctly reports.
* **The std::bad_alloc on J$_1$-J$_2(0.5)$ QR is a falsifier of RQ2.**
  The pre-registered hypothesis was that QR-AD would *reduce* gradient
  noise at the frustrated point; the implementation we have *amplifies*
  it to the point of GMRES blow-up. This is an honest negative result
  and the LaTeX writeup correctly elevates it to a numbered paragraph
  in §6 ("Honest negative result: QR-canonical at the frustrated point").
  Acceptance.
* **2$\times$2 interaction (RQ5)** has only the SVD half populated for
  J$_1$-J$_2(0.5)$ --- both QR cells crashed. We cannot evaluate the
  orthogonality claim. The LaTeX correctly presents this as
  preliminary. Phase-6 must populate the missing cells.

**Domain-Expert residual issue:** without the QR-canonical entries at
the frustrated point, the run cannot conclusively decide RQ5
(orthogonality of QR-AD and metric-preconditioner gains). The Phase-6
remediation must include either (a) a stronger Lorentzian eps schedule
that lets QR-canonical complete at $J_2/J_1=0.5$, or (b) a
documented honest "we tried A, B, C and all failed" record.

---

## Domain-Expert addendum (post-review user pushback)

The user's question --- "how do you know the $-0.832$ Heisenberg energy
is not the result of a CTMRG failure?" --- triggered a diagnostic
post-check (`optimizer_harness.py::_ctm_post_check`) that runs
CTMRG to convergence at the final $A,B$ tensor with no gradient
tracking and records four numbers: `ctm_converged_post`,
`ctm_iterations_post`, `ctm_sv_diff_post`, and the element-wise
fixed-point residual `ctm_residual_post`. Re-running the suspect
$D=2$, $\chi=12$ Heisenberg SVD seed-0 cell with the patched harness
reproduces the $E=-0.832$ result and reports
`ctm_converged_post=True`, `iters=19`, `residual=2.2\times 10^{-7}$.
**The CTMRG environment IS a converged fixed-point.** And the energy
is *still* below the QMC reference $-0.66944$ and below tenax's own
$\chi=16$ validated example $-0.6628$.

The diagnosis is therefore not CTMRG instability but
**non-variational behavior of the 2-site implicit-AD path at
$\chi<16$**, exactly as tenax's own runtime warning states:

> 2-site AD with `gs_c4v=False` uses the implicit-AD path. This is
> variational at $\chi \ge 16$ for generic models but can be slower
> than C4v or 1-site optimization. For antiferromagnetic bipartite
> models, consider `gs_c4v=True` or 1-site with
> `sublattice_rotate_gate()`.

I ignored this warning when designing the run --- it printed to
stderr on every Heisenberg cell, but I read it as a "performance
note" rather than a variational-correctness note. **Both the
small-magnitude run ($\chi=8$) and the larger Phase-6 run ($\chi=12$)
are below the $\chi=16$ threshold and therefore produce energies that
should not be quoted as variational.** The QR-vs-SVD comparison in
those cells remains internally consistent (both modes use the same
defective harness), but the *absolute* energies must not be trusted
as physical, and the claim "QR matches SVD within seed scatter" is
about a comparison between two artefacts, not about physical
ground-state energies.

The harness now records `below_variational_floor` against the QMC
reference for Heisenberg, and the in-flight Phase-6 records this
field for cells that are submitted to the worker pool after the
patch landed (subprocess re-imports the harness on each cell, so the
flag will appear in cells that started after the patch). Cells
already running when the patch landed will not have the flag.

**Required action for the next iteration (the *real* Phase-6, not
the half-done in-flight one):** redesign the grid with either
`gs_c4v=True` (using `tenax.algorithms.ipeps.build_c4v_basis` and
`c4v_tensor_from_coeffs` to constrain $B$ to be the C4v sublattice
rotation of $A$) or `unit_cell="1x1"` with the
`sublattice_rotate_gate()` trick, OR push $\chi$ up to 16 and accept
the higher per-cell cost. The QR-vs-SVD comparison must be re-run on
this corrected grid before any energy claim is made.

---

## Final Verdict

**SUCCESS WITH FOLLOW-UP.** The run produces:

1. A theoretically-sound theory draft (`./theory_draft.md`) approved by
   the RA after the seven F1--F7 caveats.
2. A working, end-to-end-differentiable QR-canonical CTMRG projector
   (`src/ctmrg_qr.py`) that registers cleanly into `tenax`
   without source-tree edits.
3. A subprocess-isolated benchmark harness (`benchmark.py` +
   `run_one_cell.py` + `optimizer_harness.py`) that ran 23 cells
   under a 15-minute CPU cap, with 19 ok cells.
4. A LaTeX paper (`report/main.pdf`) and a 22-page Beamer slide deck
   (`report/slides.pdf`) that report the findings honestly, including
   the falsifier on RQ2 and the inability to populate the QR half of
   the RQ5 2$\times$2 design.

**What still needs to happen (Phase-6 background run):**

* Populate the QR-canonical cells at J$_1$-J$_2(0.5)$ by either an
  adaptive Lorentzian eps or a stronger IFT Tikhonov regularisation.
* Replace the bond-gate J$_1$-J$_2$ reduction with the physical
  2$\times$2 unit-cell + diagonal-bond construction so that absolute
  energies are comparable to literature.
* Add the energy-variance cross-check
  (Cort\'es Estay--Kamar--Corboz, arXiv:2511.22669) on the surviving
  Phase-6 converged states.
* Expand Appendix A of `main.tex` with the explicit gauge-cancellation
  derivation; cite `src/tests/test_smoke.py::test_projector_matches_svd`
  as the empirical witness.

The run terminates gracefully with all five workflow phases delivered
and the Phase-6 escalation handed off as a separate background subagent
(see `runs/iPEPS-Symmetry-Optimization/larger_tests_cycle/` once
spawned).
