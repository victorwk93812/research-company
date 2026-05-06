# Objective

**Add QR-based CTMRG with end-to-end automatic differentiation to the `tenax` library, then benchmark it against tenax's existing Lorentzian-regularised SVD-AD baseline across the full set of optimization methods that tenax already ships.**

Two recent algorithmic advances are the immediate motivation:

- Naef–Hauschild *et al.* (`arXiv:2505.00494`, square / C4v) and the honeycomb extension (`arXiv:2509.05090`, C3v) replaced the SVD in CTMRG projector construction with QR decompositions, reporting **up to two orders of magnitude forward-pass speedup on GPU**. Neither paper integrates with AD.
- Francuz *et al.* (*Phys. Rev. Research* **7**, 013237, 2025) identified — and gave a fix for — a fundamental inaccuracy in the standard truncated-SVD backward used inside CTMRG-AD; the fix is **already implemented in tenax** as `truncated_svd_ad` (`src/tenax/algorithms/ad_utils.py`), and it is the baseline this run must compare against.

Independently, tenax also already ships:
- A working AD-iPEPS pipeline with Adam, L-BFGS, and CG optimisers + Hager–Zhang line search (`algorithms/ipeps_optimize.py`).
- The Rader *et al.* (`arXiv:2511.09546`) tangent-space metric preconditioner (`algorithms/_metric_precond.py`).
- Implicit CTM differentiation via Neumann series and GMRES inside `lax.while_loop` (`algorithms/_gmres_lax.py`).
- A working 2-site C4v Heisenberg example (`examples/heisenberg_ipeps_ad.py`) we will use as the sanity baseline.

**Central hypothesis.** Inside a single, fixed substrate (tenax) and on CPU hardware, replacing tenax's SVD-CTMRG-AD with a new QR-CTMRG-AD will:

1. Match SVD-CTMRG energies and CTM fixed-points to within published tolerances on TFIM, Heisenberg, and J1-J2.
2. Produce strictly smoother gradient trajectories near gauge-degenerate points (J1-J2 at $J_2/J_1 \approx 0.5$).
3. Show a CPU forward-pass speedup that is *modest* (the 100× number is GPU-specific and we must not overclaim).
4. Show stability gains that are **robust to optimizer choice** — i.e. QR-AD is preferable under Adam, under L-BFGS, *and* under metric-preconditioned L-BFGS, not just one of them.
5. Provide stability gains that are **at least partly orthogonal** to the metric-preconditioner gains already in tenax (i.e. QR + metric > QR alone > metric alone > plain SVD).

The run's job is to prove or falsify each clause with evidence.

**Framework / library (hard constraint).** JAX-only via the vendored tenax library at `./tenax/`. Install with `uv pip install -e ./tenax`. No PyTorch, no Julia, no other tensor-network libraries. All work goes through tenax's `SymmetricTensor`, `optimize_gs_ad`, `_ctm_projector`, and friends — wrap and extend, don't reimplement.

---

# Scope

- **Algorithm:** AD-compatible QR-CTMRG added as a new projector mode inside `tenax/algorithms/_ctm_projector.py`, alongside the existing SVD/eigh paths. Begin with C4v-symmetric square-lattice ansätze (matching `2505.00494` and tenax's existing 2-site Heisenberg example), then extend to non-symmetric 2×2 unit cells.
- **Differentiation:** route the new QR forward through tenax's existing implicit-fixed-point machinery (Neumann-series default, GMRES fallback) with a **new custom_vjp** for the QR-CTMRG truncation step. Re-use `_gmres_lax.py` and the Arnoldi spectral-radius precheck unchanged.
- **Optimizer comparison (the orthogonal axis added by this run):** every (model, D, χ, ctmrg_mode) cell is re-run under {Adam, L-BFGS, metric-preconditioned L-BFGS}. Stretch goal: stochastic reconfiguration with warm-started SVD (`arXiv:2512.05749`) as a fourth optimizer.
- **Validation models, in order:**
  1. 2D transverse-field Ising (TFIM) — sanity check, smooth phase transition.
  2. 2D Heisenberg antiferromagnet — exercises the existing C4v + 2-site path; reproduces tenax's documented example.
  3. 2D J1-J2 Heisenberg — frustrated, near-degenerate SVD values expected in the columnar/Néel crossover; this is the regime where SVD-backward pathologies hurt and where any QR-AD stability gain should be visible.
- **Hubbard escalation (conditional).** Only if T1–T5 finish with wall-clock budget remaining: build a spinful fermionic gate on top of `algorithms/fermionic_ipeps.py`, validate against 2×2 ED, and run a single benchmark point at U/t=8, δ=0, t'=0, D=4, χ=16 against the Simons AFQMC reference $E_0/t \approx -0.524$.
- **Non-goals.** Non-abelian (SU(2)) symmetry — tenax has only a stub. Non-square lattices (honeycomb / triangular). 3D iPEPS. Production GPU benchmarks (we report CPU honestly; the H100 number is not ours).

---

# Research Questions

1. **Correctness.** Does QR-CTMRG inside tenax converge to the same fixed-point environment as SVD-CTMRG (residual $< 10^{-10}$) on TFIM, Heisenberg, and J1-J2? Do final variational energies agree to within $10^{-5}$ relative on TFIM/Heisenberg?
2. **Gradient stability.** Near $J_2/J_1 \approx 0.5$, does QR-AD reduce the gradient-noise / line-search-restart count vs Lorentzian-SVD-AD? Quantify with $\|\nabla E\|$ variance across the last 50 optimiser steps.
3. **Speed (CPU honesty).** What is the QR-vs-SVD wall-clock ratio on CPU at modest $\chi$ ($\le 32$), separately for forward CTMRG, backward, and full optimisation step? The 100× claim is GPU-specific; report what we actually see.
4. **Optimizer robustness.** Hold ctmrg_mode fixed; sweep optimiser ∈ {Adam, L-BFGS, metric-LBFGS, (SR-warm-start)}. Which optimiser × ctmrg_mode combinations converge fastest / most reliably? Is QR-AD's stability gain visible *for every optimiser* or only some?
5. **Interaction with the metric preconditioner.** Is the QR-AD stability gain orthogonal to the Rader metric-preconditioner gain, or do they share headroom? 2×2 design: {SVD, QR} × {plain L-BFGS, metric-LBFGS}.
6. **Symmetry × QR cost.** When U(1) (or other available abelian symmetry) is enabled, per-block sizes shrink. Does the per-block QR cost shrink the same way per-block SVD cost does, or does the QR-vs-SVD ratio change with symmetry?

---

# Technical Challenges

## C1. QR-CTMRG forward inside tenax

- Add a new projector mode `qr_canonical` to `_ctm_projector.py` next to the existing SVD/eigh paths.
- Use `tenax.linalg.qr` (already block-sparse-aware over `SymmetricTensor`).
- Canonicalise QR with $R_{ii} > 0$ for gauge fixing; verify the canonicalisation is differentiably smooth.
- Address both the C4v symmetric path (matching `2505.00494`) and the generic 2×2 unit-cell path.
- Acceptance: CTM environment fixed-point residual matches SVD-CTM to $< 10^{-10}$ on TFIM at $D=2$, $\chi=16$.

## C2. QR-CTMRG backward (`custom_vjp`)

- Implement the QR-backward following Roberts (1963), Mathias (1996), Hubig–McCulloch (2019) — use tenax's existing `truncated_svd_ad` as the structural template (`algorithms/ad_utils.py`).
- The QR is used to *build the projector for truncation*, not to factor the tensor itself; the adjoint of the truncation-by-projector step must be derived explicitly.
- Integrate with the existing implicit-fixed-point machinery (`_ctm_energy_ad.py`, `_gmres_lax.py`); confirm the spectral-radius pre-check (`_arnoldi.py`) still applies.
- Handle the QR diagonal-phase gauge — the IFT Jacobian is singular in the gauge direction and must be projected out (mirror the existing `forward_gauge="phase"` treatment in tenax).
- Acceptance: QR-AD gradient matches `jax.jacfwd` central finite-difference to $< 10^{-6}$ on a 4-site toy.

## C3. Optimizer harness

- Build a single wrapper `run_cell(model, D, chi, ctmrg_mode, optimizer, seed)` over `optimize_gs_ad()` that takes `ctmrg_mode ∈ {svd_lorentzian, qr_canonical}` and `optimizer ∈ {adam, lbfgs, metric_lbfgs, sr_warm_start}` as orthogonal toggles.
- All four optimisers reuse tenax's existing Hager–Zhang line search and tangent-projection where applicable.
- For `sr_warm_start` (`arXiv:2512.05749`): if implementation cost exceeds 1 day of engineering, **mark as stretch goal**, document the omission, and run only the first three optimisers.
- Per-step logging: energy, $\|\nabla E\|$, line-search step, sweep wall-clock, CTM iterations, projector mode, optimiser. Single CSV per cell.

## C4. CPU-realistic benchmark protocol

- Hard-code `JAX_PLATFORMS=cpu`. Disable any GPU/TPU/Metal code path. Use `OMP_NUM_THREADS=4` (per persona resource block).
- Bond dimensions $D \in \{2, 3, 4, 5\}$. Environment $\chi$ ramped from $4D$ to $\min(8D, 32)$ via tenax's existing `chi_schedule`.
- Each (model, D, χ, J2/J1, ctmrg_mode, optimizer) cell run with **3 seeds**; report mean ± std on energy, gradient norm, wall-clock.
- All other axes — initialisation, max-iterations, tolerance, gauge-fixing, χ-ramp policy, stall-recovery — held identical across cells via a YAML config in `./src/configs/`.
- Sanity gate: reproduce `examples/heisenberg_ipeps_ad.py` energy ($E \approx -0.6628$ at $D=2$, $\chi=16$) within $10^{-4}$ before any tenax modification, and again after the QR mode is added (under `ctmrg_mode=svd_lorentzian` to confirm no regression).

## C5. Symmetry × QR cost study (lightweight)

- For Heisenberg and J1-J2, rerun a single $D=4$ cell with U(1) charge symmetry on (where the model permits) and off; compare per-block QR vs per-block SVD cost.
- This is exploratory, not the headline; budget ≤ 1 day of engineering.

## C6. Variance-extrapolation cross-check

- For each accepted converged state, compute energy variance and apply zero-variance extrapolation (`arXiv:2511.22669`); E_extrapolated should be consistent across (ctmrg_mode, optimizer) for the same physical point. Inconsistency is a red flag for the stability claim.

---

# Variables To Toggle

The grid is deliberately narrow — this is a **method paper** on a specific technical question, not a physics phase diagram.

## A. Algorithmic axis (the subject of the study)

| Variable | Values | Notes |
|---|---|---|
| Projector construction | `svd_lorentzian` (existing, Francuz patch) vs `qr_canonical` (new) | The core comparison |
| Differentiation mode | `implicit (Neumann + GMRES)` | Inherited from tenax; do NOT add an unrolled mode this run |
| QR gauge | `R_ii > 0` (canonical) | Single fixed choice |

## B. Truncation / ansatz axis

| Variable | Values | Physics interpretation | Why toggle | What it helps |
|---|---|---|---|---|
| `D` (bond dimension) | 2, 3, 4, 5 | Variational manifold size | CPU keeps us small; D=5 is the realistic ceiling | Establishes finite-D scaling of the QR-vs-SVD difference |
| `χ` (CTM env) | $4D$, $6D$, $\min(8D, 32)$ via χ-ramp | Environment expressivity | Larger χ exposes more SVD degeneracies | Probes the regime where Lorentzian SVD-AD struggles |
| Unit cell | 1×1 (TFIM), 2×2 C4v (Heisenberg), 2×2 generic (J1-J2 near transition) | AFM-vs-symmetric ansatz | Matches model symmetry | Required for qualitative correctness |
| Symmetry | trivial, optional U(1) for Heisenberg/J1-J2 | Block-sparse decomposition | Probes C5 | Establishes whether QR speedup compounds with symmetry |

## C. Physics axis (just enough to exercise the algorithm)

| Model | Parameter points | Reference |
|---|---|---|
| TFIM | $h/J \in \{2.5, 3.04\,(\text{critical}), 3.5\}$ | Standard iPEPS literature |
| Heisenberg (square) | one point (isotropic) | $E_0/J \approx -0.6694$ (QMC); tenax example reaches $\approx -0.6628$ at $D=2,\chi=16$ |
| J1-J2 | $J_2/J_1 \in \{0.0, 0.4, 0.5, 0.55, 0.6\}$ | Spans Néel → plaquette/columnar; SVD pathologies worst here |

## D. Optimizer axis (the orthogonal axis added by this run)

| Optimiser | Source in tenax | Why include |
|---|---|---|
| `adam` | `optax.adam`, cosine LR decay | Cheap baseline; tests whether QR-AD gain survives in noisy first-order regime |
| `lbfgs` | `optax.scale_by_lbfgs(memory_size=10)` + Hager–Zhang | Standard iPEPS workhorse; the comparison most papers report |
| `metric_lbfgs` | `_metric_precond.py` (Rader 2025) + L-BFGS | Tenax's strongest optimiser; tests RQ5 (orthogonality of QR-AD and metric-preconditioner gains) |
| `sr_warm_start` (stretch) | New, port `arXiv:2512.05749` | Natural-gradient direction; only if budget permits |

Tangent projection, manifold normalisation, stall-recovery (noise / reset) — held at tenax defaults across cells.

---

# Benchmark Metrics

For every (algorithmic variant × truncation × physics point × optimiser × seed) cell, record:

1. **Converged energy** $E_0$ and error vs reference (published iPEPS / QMC values).
2. **Gradient quality:**
   - $\|\nabla E\|$ trajectory vs iteration,
   - variance of $\|\nabla E\|$ across the last 50 optimiser steps (the *stability* metric for RQ2),
   - line-search restart count (proxy for landscape roughness).
3. **Wall-clock time (CPU):** per CTMRG sweep (forward), per AD step (forward + backward), total to convergence. Honest CPU numbers, no GPU extrapolation.
4. **CTM iterations to converge** at each optimiser step — exposes whether one mode requires more inner iterations.
5. **Fixed-point residual** $\|F(\mathcal{C}^\star, A) - \mathcal{C}^\star\|$ at optimiser convergence — confirms CTMRG itself converged.
6. **Energy variance + zero-variance extrapolated $E$** (C6).
7. **Per-cell metadata:** JAX version, tenax git SHA, CPU model, OMP threads, RNG seed, total wall-clock budget consumed.

---

# Deliverables (per CLAUDE.md workflow)

1. **Phase 1 — Researcher → `./theory_draft.md`:**
   - Literature review (≥ 5 papers from the gathered 2024–2026 set, at minimum: `2505.00494`, `2509.05090`, Francuz PRR 7 013237 2025, Liao 2019 PRX 9 031041, Rader `2511.09546`, plus one of `2502.10298` split-CTMRG / `2508.10822` MCF gauge fixing / `2511.22669` variance extrapolation).
   - Audit of tenax's existing AD-CTMRG pipeline (what is the Lorentzian SVD-backward formula tenax actually uses; what is the metric-preconditioner formula; what gauge fixing is in `forward_gauge="phase"`).
   - Rigorous derivation of the QR-CTMRG backward pass including (a) the truncation-by-projector adjoint, (b) the diagonal-phase gauge projection, (c) integration with the implicit-fixed-point IFT solve already in tenax.
   - Explicit hypotheses for RQ1–RQ6.
   - Positioning vs `2505.00494` (forward only, GPU) and Liao 2019 (original AD-CTMRG, SVD only).

2. **Phase 2 — RA Skeptic → `./ra_critique.md`:** independent arxiv check; pedantic review of the QR-backward derivation; verify the gauge-fixing claim is complete; **specifically attack** RQ3 (CPU-vs-GPU honesty about QR speedup) and RQ5 (orthogonality claim — easy to fake by cherry-picking points). Loop with Phase 1 until "APPROVAL GRANTED".

3. **Phase 3 — LaTeX Writer → `./report/main.tex`:** XeLaTeX, `revtex4-2`. Required figures: (a) energy convergence trajectory SVD vs QR at J1-J2 $J_2/J_1=0.5$, broken out per optimiser; (b) wall-clock scaling vs $D$ on CPU, separately for forward / backward / total; (c) gradient-noise stability metric (RQ2) per (ctmrg_mode × optimiser); (d) interaction-effect plot for RQ5 (2×2 SVD/QR × plain/metric-preconditioned); (e) table of converged energies vs literature.

4. **Phase 4 — Python Engineer → `./src/`:** `uv init` JAX project; install tenax editable via `uv pip install -e ../tenax`. Modules:
   - `ctmrg_qr.py` — new QR projector + backward, **upstreamable into tenax** as a follow-up.
   - `ctmrg_qr_register.py` — registers the new projector mode into tenax's `_ctm_projector.py` dispatch.
   - `optimizer_harness.py` — the `run_cell()` wrapper of C3.
   - `models.py` — TFIM, Heisenberg, J1-J2 gates (re-use `tenax.algorithms.ipeps.heisenberg_gate` etc.).
   - `benchmark.py` — top-level sweep driver; YAML configs in `./src/configs/`.
   - `analysis.py` — per-cell CSV → aggregated tables + plots in `./src/plots/`.
   - `tests/`:
     - QR-backward vs `jax.jacfwd` on a random $64\times 64$ matrix ($< 10^{-6}$).
     - CTM fixed-point residual $< 10^{-10}$ for both modes on TFIM at $\chi=16$.
     - SVD-AD and QR-AD gradients agree to $< 10^{-6}$ on an away-from-criticality TFIM point.
     - Reproduce `examples/heisenberg_ipeps_ad.py` energy at $D=2,\chi=16$ within $10^{-4}$.
   - Pipe the headline run to `./src/simulation.log`.
   - Resource-block enforcement: 16 GB cap, OMP threads = 4, `JAX_PLATFORMS=cpu`.

5. **Phase 5 — Review Board → `./final_review.md`:** math-pedant on the QR-backward derivation; performance-hacker on the JAX jit/vmap structure and CPU-honest timing; domain-expert on the J1-J2 energies and convergence behavior; library-citizen check that the new code is upstreamable into tenax (passes `pytest -m core` in tenax with no regressions).

---

# Practical Constraints

- **Hardware:** CPU only. `JAX_PLATFORMS=cpu` set in every script. No CUDA / TPU / Metal.
- **Memory:** 16 GB hard cap per process.
- **BLAS threads:** 4 (set OMP/OpenBLAS/MKL/VecLib/NumExpr env vars).
- **Precision:** `float64` (tenax forces `jax_enable_x64=True` on import — verify in code).
- **Reproducibility:** PRNG seeds, git commits (research-company root + tenax submodule), JAX / optax / tenax versions, CPU model, thread count logged in every CSV.
- **tenax handling:** install editable (`uv pip install -e ../tenax`); do not modify tenax files in-place — write the QR mode in `./src/ctmrg_qr.py` and *register* it via tenax's existing dispatch hook. Submit upstream as a follow-up PR after Phase 5.
- **Sanity gate before headline runs:** `pytest -m core` in tenax must remain green; the existing Heisenberg example must reproduce within $10^{-4}$.

---

# Success Criteria

The run is a success if **by the end of Phase 5** the benchmark answers each Research Question with evidence:

- **RQ1 pass:** converged energies at every $(D,\chi)$ on TFIM and Heisenberg agree between QR-AD and Lorentzian-SVD-AD to better than $10^{-5}$ relative; J1-J2 disagreements (if any) are attributed and explained.
- **RQ2 pass:** a quantitative metric (gradient-norm variance over last 50 steps, or line-search restart count) shows reduced gradient noise for QR-AD at $J_2/J_1 \approx 0.5$, **and** the difference holds under at least 2 of the 3 mandatory optimisers.
- **RQ3 pass:** CPU wall-time numbers reported for every $D$, broken out forward / backward / total. No claim is made about GPU speedup.
- **RQ4 pass:** a single 2D table (ctmrg_mode × optimiser) reports converged $E$, wall-clock, and stability metric; reader can see at a glance which combinations are best.
- **RQ5 pass:** the 2×2 design (SVD/QR) × (plain/metric-preconditioned LBFGS) is fully populated; the interaction term (additive vs sub-additive) is explicitly reported.
- **RQ6 pass:** the U(1)-on / U(1)-off comparison is reported for at least Heisenberg at $D=4$.

Failure modes that still count as a valid run (publishable *negative result*):

- QR-CTMRG-AD is numerically stable but **not faster** on CPU once the backward tape is included → publishable methods note ("stability without speed").
- QR-CTMRG-AD achieves a forward speedup but introduces a **different** backward pathology (e.g. rank-revealing failure on near-singular corners) → publishable cautionary tale.
- The stability gain is **dominated by** the metric preconditioner already in tenax → publishable orthogonality result with practical recommendation.

Either outcome is a method paper. The point of the run is to settle which.

---

# AI / Library Disclosure

- The tenax library is itself partly developed with AI assistance (per its own `CLAUDE.md` notice). All numerical claims must be cross-checked against `examples/heisenberg_ipeps_ad.py` and against published reference values; do not trust tenax results without a reference comparison.
- Disclose Claude Code / agentic-AI usage in the final report per current publication norms.
