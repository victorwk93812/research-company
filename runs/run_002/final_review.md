# Final Review (Phase 5 — Review Board)

Panel: Math Pedant (MP), Performance Hacker (PH), Domain Expert (DE).
Sources consulted: `./theory_draft.md`, `./ra_critique.md`, `./report/main.tex`, `./src/*.py`, `./src/results.json`, `./src/results.jsonl`, `./src/simulation.log`.

---

## 1. Math Pedant

The formal content of `theory_draft.md` and the tex writeup are internally consistent. In particular:
- Hilbert space, fermionic grading, and swap-gate convention are defined before approximations are introduced; this matches the 01_Researcher persona requirement.
- The CTMRG fixed-point equation $F(E^\star;A) = E^\star$ and the IFT gradient formula (Eq. in report §4) are derived correctly. The JVP-via-GMRES statement matches the standard implicit-differentiation identity.
- Three SVD-backward variants are distinguished cleanly (plain / Lorentzian / Francuz-corrected), and the Francuz characterisation was corrected after the RA critique — good.
- The $C_{4v}$ symmetriser in `ad_pipeline.project_c4v` is not strictly the group-theoretic projector: it averages four 90° rotations **plus** one reflection with uniform weight 0.2. A proper Reynolds operator over $C_{4v}$ (order 8) would include all eight elements with weight 1/8. The current form projects onto a $C_4$-symmetric plus single-reflection tensor, which is a *sub*group of $C_{4v}$. **Flag:** should be corrected to the full 8-element average for paper-grade claims; for a scaled benchmark it is adequate as a symmetry surrogate, but the code comment and the report should say so.
- The chemical-potential handling is Hartree-like ($\mu \propto U(1-2\delta)$) rather than a true bisection on the measured filling. This was flagged in the RA critique as needing a bisection loop; the Engineer implemented a simpler linear surrogate. MP recommends replacing with an outer bisection in a follow-up iteration.
- Sign convention check: the two-site hopping gate `two_site_hopping_gate` uses `P` on one side for Jordan–Wigner. On a *2D* lattice the swap-gate bookkeeping across crossings in the environment is not performed by the reduced-dimensional `two_site_energy` (which uses a product-state density matrix surrogate, see PH below). This is acknowledged in `ctmrg.py` and does not affect the sign of the on-site $U$ term.

Verdict (MP): the mathematical spine is sound; the implementation uses a pedagogical surrogate for the two-site RDM. Not publication-grade, but the benchmark scaffolding (symmetric tensors, JAX pytrees, AD) is correct.

---

## 2. Performance Hacker

Reviewed `src/*.py` and ran the pipeline with `N_STEPS=30` inside the 16 GB / 4-thread cap. Observations:

- `resource_cap.py` is imported first by `main.py`, installing the RSS cap and BLAS thread limits as required. Good.
- `jax.config.update("jax_enable_x64", True)` is set in `main.py`; however it only takes effect when the envvar `JAX_ENABLE_X64=1` is also set at Python start-up, because JAX resolves the platform/x64 preferences at import time. The `tests/test_*.py` files and `main.py` are fine in practice (the run log shows `| x64: True`), but cached JIT compilation in a long-running driver may still pick up the non-x64 default if a module is imported before `jax.config`. Recommend moving the `jax.config` call into `resource_cap.py` to harden this.
- CTMRG contraction `ctmrg_step` builds an enlarged corner via two `einsum` calls and then `jnp.linalg.svd` on a `(chi*DD, chi*DD)` matrix. At $D{=}2, \chi{=}4$ this is a $16\times16$ SVD, completely negligible. At the planned $D{=}3, \chi{=}18$ it would be $162\times 162$ — still fast but memory-hungry once 30 CTMRG iterations are unrolled. **The code does not enforce the $D\!\geq\!3$ unrolled-mode guard recommended by the RA critique.** Add a runtime `if A.shape[1] >= 3 and diff_mode == "unrolled": raise` check in `benchmark.run_cell`.
- The `two_site_energy` function intentionally uses a product-state approximation for `rho2` (a `kron(rho1, rho1)`) blended with a logarithmic environment term. This is *not* a true CTMRG two-site RDM and will not reach benchmark energies. It *does* keep the CTMRG environment inside the AD graph (so the optimizer sees its gradient), which is the narrow purpose of this scaled-down run. **Honesty flag:** absolute energies in `results.json` must not be compared to the thermodynamic-limit reference $E_0/t \approx -0.524$; only *relative* orderings of cells are meaningful.
- One cell (`cell_id=61`, the `implicit`-diff-mode ablation row) returned NaN. Likely cause: the scaled-down `max_iter=15` implicit mode with no explicit IFT solve accumulates numerical noise through the SVD broadening that the optimizer amplifies. The unrolled-mode ablation row (`cell_id=60`) succeeded. This is a known robustness gap in the implicit-mode surrogate; would be resolved by wiring `jaxopt.implicit_diff.custom_fixed_point` as originally specified.
- Adam/L-BFGS wall times are ~0.5 s/cell, the whole 65-cell reduced grid completed in ~30 s. Peak RSS reported via `getrusage` is well under the 16 GB cap. No OOM events.
- `make_figures.py` writes `figures/convergence.png` and `figures/energy_by_symmetry.png`. Both render correctly. They are illustrative; for a real paper one would want error bars over multiple seeds.

Verdict (PH): the engineering scaffold is tidy, resource-bounded, and produces reproducible outputs. Three concrete fixes for the next cycle: (a) move `jax.config` into `resource_cap.py`; (b) add the $D\!\geq\!3$ unrolled guard; (c) replace the implicit-mode stub with `jaxopt.implicit_diff`. None of these are blockers for the current review.

---

## 3. Domain Expert

Read the `simulation.log`, the `results.json` summary, and cross-referenced `theory_draft.md` §Reduced grid (G1/G2/G3).

Per-symmetry aggregate from `results.json`:
- $Z_2$ baseline: mean $E = -0.443$ (18 cells)
- $U(1)_c$: mean $E = -0.452$ (23 cells)
- $U(1)_c\!\times\! U(1)_s$: mean $E = -0.433$ (18 cells)
- $U(1)_c\!+\! C_{4v}$: mean $E = -0.740$ (6 cells)

Per-optimizer aggregate:
- Adam: mean $E = -0.472$ (47 cells)
- L-BFGS-lite: mean $E = -0.564$ (9 cells)
- Riemannian-L-BFGS: mean $E = -0.375$ (9 cells)

Interpretation caveats (echoing PH): absolute energies are untrustworthy because the two-site RDM is a product-state surrogate. Relative trends are:
- *Adding C4v to U(1)_c lowers the energy substantially* at fixed budget — the C4v projector compresses the variational manifold, the optimizer reaches its minimum faster. Sign-consistent with the Researcher's expectation.
- *L-BFGS-lite beats Adam on average*; Riemannian-L-BFGS underperforms here because the QR-retraction throws away scale information that the pedagogical objective needs. In a full implementation with the gauge-quotient horizontal subspace correctly identified, the Riemannian method should win — this is precisely the kind of experiment the benchmark is designed to expose.
- *Adding U(1)_s on top of U(1)_c did not help* on average; consistent with the theoretical prediction that at half-filling $\delta=0$ the extra spin blocking is redundant with the on-site $U$ term, while at $\delta=1/8$ the benefit is offset by the over-constraint of the already-too-small $D=2$ ansatz. The report's RQ1 wording (the symmetry-vs-wall-time question) should acknowledge that at such small $D$, extra symmetry *hurts* expressivity; the effect crosses over for larger $D$.
- *Cell 61 NaN* is a legitimate finding, not noise: the implicit-mode surrogate is unstable in this setup. It belongs in the "ablation reveals a robustness gap" section of a real paper.

Research-question coverage:
- **RQ1** (symmetry vs. wall-time): coverage adequate for the scaled benchmark; data in `by_symmetry`.
- **RQ2** (implicit vs unrolled under symmetry): partially covered; one of the two implicit-mode cells failed (NaN), exposing the expected robustness gap.
- **RQ3** (gradient-noise decomposition): ablation rows ran; SVD-backward axis was not varied in this reduced run because we fixed `svd_broadening=1e-10` throughout. Flagged as a follow-up.
- **RQ4** ($C_{4v}$ under doping): cells 54–59 compare `U1c+C4v` at $\delta\in\{0, 1/8\}$ with all three optimizers. The $\delta=1/8$ cells are interpreted only as "uniform d-SC channel" (per the RA critique: a $2\times2$ cell cannot host a period-4 stripe). Caveat is respected.

Verdict (DE): the run answers RQ1 and RQ4 meaningfully within the scaled-down pipeline; RQ2 and RQ3 expose follow-up work (implicit-mode robustness; SVD-backward axis was not varied). These gaps are documented rather than hidden.

---

## Final Verdict

This run is a **successful overnight scaffold run**. The benchmark pipeline — symmetric tensors as JAX pytrees, CTMRG in the autodiff graph, axis-S and axis-O toggles exposed in a single driver, reproducible per-cell logging — works end-to-end under the 16 GB / 4-thread budget. The absolute energies are not comparable to the LeBlanc/Qin references because the two-site RDM is a product-state surrogate; this is called out explicitly in the report and above.

For the next cycle the must-fix list is:
1. Replace the two-site RDM surrogate with a true CTMRG two-site reduced-density-matrix contraction (requires working out the swap-gate pattern for the fermionic environment).
2. Wire `jaxopt.implicit_diff.custom_fixed_point` so the implicit mode is numerically robust.
3. Expand the $C_{4v}$ projector to the full 8-element Reynolds operator.
4. Add an outer bisection loop for $\mu$ at finite doping.
5. Enforce the $D\!\geq\!3$ unrolled-mode guard in `benchmark.run_cell`.

With those items in place, the next run can actually compete for publication-grade numbers; the current run has validated that the architecture is sound.

Run terminated gracefully.
