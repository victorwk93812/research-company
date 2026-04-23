# Objective

**Derive, implement, and benchmark a QR-based Corner Transfer Matrix Renormalization Group (CTMRG) algorithm with end-to-end automatic differentiation (AD) for variational iPEPS optimization.**

Recent work (Naef–Hauschild, `arXiv:2505.00494`; and the honeycomb extension `arXiv:2509.05090`) replaced the SVD in CTMRG projector construction with QR decompositions, reporting **up to two orders of magnitude speedup on GPUs** (≈1 hour on an H100 for state-of-the-art J1–J2 calculations). Those papers treat only the **forward** contraction and **explicitly do not integrate with AD**.

Meanwhile, the AD-through-CTMRG pipeline (Liao *et al.*, PRX 9, 031041, 2019) has been dogged by two well-known pathologies rooted in SVD backward:

- **Singular-value near-degeneracies** introduce divergences of the form $1/(\sigma_i^2 - \sigma_j^2)$ in the backward pass.
- Francuz *et al.* (*Phys. Rev. Research* **7**, 013237, 2025) recently identified an additional *fundamental inaccuracy* in the standard SVD backward used for truncation in CTMRG-AD, independent of near-degeneracy.

**Central hypothesis:** Because the QR factorization has a smooth backward rule with **no singular-value-gap divergence**, an AD pipeline built on QR-CTMRG can be *simultaneously* faster (from the forward-pass gains of `2505.00494`) and more stable (by eliminating the SVD-backward pathologies) than any existing AD-iPEPS implementation.

The run's job is to prove or falsify that hypothesis with a clean method paper.

**Framework (hard constraint):** JAX only — `jax`, `jax.numpy`, `optax`, `jaxopt`, custom `jax.custom_vjp` primitives where needed. Use `jax.lax.custom_root` or `jaxopt.implicit_diff` for implicit differentiation. No PyTorch, no Julia.

---

# Scope

- **Algorithm:** AD-compatible QR-CTMRG for 2D square-lattice iPEPS. Begin with C4v-symmetric ansätze (matching `2505.00494`), then extend to generic 2×2 unit cells without point-group symmetry.
- **Differentiation strategy:** two branches — (i) unrolled AD through the full QR-CTMRG sweep, (ii) implicit AD at the CTMRG fixed point (IFT with a Jacobian–linear-solve). Both must be implemented and compared.
- **Validation models (order of increasing difficulty):**
  1. 2D transverse-field Ising (TFIM) — sanity check with a smooth phase transition.
  2. 2D Heisenberg antiferromagnet — standard iPEPS benchmark with SU(2) breaking.
  3. 2D J1–J2 Heisenberg — frustrated, near-degenerate SVD values expected in the columnar/Néel crossover; this is where SVD-backward pathologies hurt most.
- **Non-goal:** production-grade fermionic Hubbard at large D. Fermionic extension is scoped as a *discussion* section, not a benchmark target.

---

# Research Questions

1. **Correctness.** Does gradient-based iPEPS optimization with QR-CTMRG converge to the same energies as SVD-CTMRG-AD (with the Francuz patch) at fixed $(D,\chi)$, within a documented tolerance?
2. **Stability.** Near the J1–J2 phase boundary and at critical TFIM, does QR-CTMRG-AD produce smaller gradient-noise and more monotone convergence than SVD-CTMRG-AD? Quantify.
3. **Speed.** What is the wall-time speedup of QR-CTMRG-AD over SVD-CTMRG-AD, (a) forward-only, (b) forward + backward? Is the reported forward speedup of `2505.00494` preserved when the backward tape is added?
4. **Unrolled vs implicit.** For QR-CTMRG, does implicit differentiation still help (as it does for SVD-CTMRG), or does the already-cheap QR backward make unrolling acceptable?
5. **Gauge structure.** QR is only defined up to a diagonal-phase gauge; how does this gauge freedom interact with the iPEPS bond gauge, and does it require an explicit gauge-fix in the backward pass?

---

# Technical Challenges (the researcher must address all of these)

## C1. Derivation of the QR-CTMRG backward pass

- The forward pass of QR-CTMRG replaces the SVD-based projector construction $P = U \Sigma^{-1/2}$ (truncating to $\chi$) with a QR-based isometry $Q$ from a rank-revealing QR of the enlarged corner.
- The backward rule for QR is standard (Roberts 1963; Mathias 1996; Hubig–McCulloch 2019 for TN): write $\bar A = \bar Q R^{-T} + Q \, \mathrm{copyltu}(Q^\dagger \bar Q - \bar R R^{-T}) R^{-T}$ or equivalent. But CTMRG-QR uses the QR to build *projectors for truncation*, not to factor the tensor itself — the researcher must derive the adjoint of this *truncation-by-projector* step explicitly.
- Address both the square-lattice C4v case and the general non-symmetric case.

## C2. Fixed-point (implicit) differentiation of QR-CTMRG

- Formulate CTMRG as $\mathcal{C}^\star = F(\mathcal{C}^\star, A)$ where $A$ is the site tensor and $\mathcal{C}^\star$ the converged environment. Use IFT: $\partial\mathcal{C}^\star/\partial A = (I - \partial_{\mathcal{C}} F)^{-1} \partial_A F$. The linear solve is the only expensive piece; compare `jaxopt.linear_solve` GMRES against `jax.scipy.sparse.linalg.bicgstab`.
- Handle the QR gauge freedom: the fixed point is unique only up to a diagonal-phase gauge; the IFT Jacobian is singular in that gauge direction and must be projected.

## C3. Gauge fixing of the QR factor

- The QR factorization is unique only after fixing $\mathrm{sign}(\mathrm{diag}(R))$ (real case) or the diagonal phase of $R$ (complex case). Within a CTMRG sweep, inconsistent gauge choices across iterations break the fixed-point convergence. Enforce a canonical choice (e.g. $R_{ii} > 0$) and verify it is differentiably smooth.

## C4. Benchmark protocol that isolates the QR vs SVD change

- All other axes — ansatz initialization, optimizer (L-BFGS from `jaxopt`), unit cell size, CTMRG sweep count, convergence tolerance — must be held identical across QR and SVD runs. Hyperparameters frozen in a YAML config.

## C5. Numerical-rank-deficient corners

- QR is not rank-revealing unless augmented (pivoted QR or a randomized QR). The researcher must choose one and justify the choice: does the CTMRG iteration actually produce corners with enough separation of scales to make pivoting unnecessary at the $\chi$ values of interest?

---

# Variables To Toggle

The grid is deliberately narrower than run_003 — this is a **method paper** on a specific technical question, not a physics benchmark.

## A. Algorithmic axis (the subject of the study)

| Variable | Values | Notes |
|---|---|---|
| Projector construction | `SVD` (with Francuz patch) vs `QR` (canonical gauge) vs `pivoted-QR` | The core comparison |
| Differentiation mode | `unrolled` vs `implicit (IFT)` | Both for each projector type |
| SVD-backward variant (baseline only) | `plain`, `Lorentzian`, `Francuz-corrected` | Only to demonstrate the SVD baseline is as strong as possible |

## B. Truncation / ansatz axis

| Variable | Values |
|---|---|
| Bond dimension `D` | 2, 3, 4, 6 |
| Environment dimension `χ` | $D^2$, $2D^2$, $3D^2$ |
| Unit cell | 1×1 (TFIM, Heisenberg), 2×2 (J1–J2 near transition) |

## C. Physics axis (just enough to exercise the algorithm)

| Model | Parameter points |
|---|---|
| TFIM | $h/J \in \{2.5, 3.04\, (\text{critical}), 3.5\}$ |
| Heisenberg | one point (isotropic); compare to reference $E_0/J \approx -0.6694$ |
| J1–J2 | $J_2/J_1 \in \{0.0, 0.4, 0.5, 0.55, 0.6\}$ — spans the Néel → plaquette/columnar crossover, where SVD pathologies are worst |

## D. Optimizer axis (kept fixed to isolate the CTMRG change)

- Fixed choice: `jaxopt.LBFGS` with strong-Wolfe line search, 100 max iters; Adam sanity check at 1000 steps.
- *Not* varied across cells — we want to attribute performance differences to QR vs SVD, not to optimizer interactions.

---

# Benchmark Metrics

For every (algorithmic variant × truncation × physics point) cell, record:

1. **Converged energy** $E_0$ and error vs reference (published iPEPS and/or DMRG-cylinder values).
2. **Gradient quality:**
   - $\|\nabla E\|$ trajectory vs iteration,
   - discrepancy between AD gradient and a central-finite-difference check on a small test case,
   - maximum relative gradient entry where the SVD-backward has near-degenerate singular values (the "pathology probe").
3. **Wall time:** per CTMRG sweep (forward), per AD step (forward + backward), total to convergence. Report on both CPU and GPU if available.
4. **Peak memory** (unrolled mode — where the tape is the memory cost): measured via `jax.profiler.memory_profile`.
5. **Fixed-point residual:** $\|F(\mathcal{C}^\star,A) - \mathcal{C}^\star\|$ at optimizer convergence, to confirm CTMRG itself is converged.

---

# Deliverables (per CLAUDE.md workflow)

1. **Phase 1 — Researcher → `./theory_draft.md`:**
   - Literature review (arxiv MCP; at minimum `2505.00494`, `2509.05090`, `2508.10822`, Liao 2019, Francuz PRR 2025, Fishman–White 2018 on projector construction, Hauschild–Pollmann 2018).
   - Rigorous derivation of the QR-CTMRG backward pass (both unrolled and implicit variants).
   - Explicit treatment of the QR diagonal-gauge fix and of the CTMRG fixed-point gauge.
   - Complexity analysis vs SVD-CTMRG-AD.
   - Positioning statement vs `2505.00494` (their forward-only result) and Liao 2019 (the original AD-CTMRG).

2. **Phase 2 — RA Skeptic → `./ra_critique.md`:** independent arxiv check, pedantic review of the QR-backward and IFT derivations, verify the gauge-fixing claim is complete (no hidden gauge-direction singularity), confirm feasibility within the 16 GB memory cap. Loop with Phase 1 until approval.

3. **Phase 3 — LaTeX Writer → `./report/main.tex`:** XeLaTeX, `revtex4-2`. Main figures: (a) convergence trajectory SVD vs QR at J1–J2 $J_2/J_1 = 0.5$, (b) wall-time scaling forward / backward / total vs $D$, (c) gradient-check accuracy scatter, (d) table of converged energies vs references.

4. **Phase 4 — Python Engineer → `./src/`:** `uv init` JAX project. Modules: `hamiltonian.py`, `ctmrg_svd.py`, `ctmrg_qr.py` (and `ctmrg_qr_pivoted.py`), `backward_svd.py` (with Francuz patch), `backward_qr.py` (new), `implicit_diff.py`, `benchmark.py`, `main.py`. Tests in `./src/tests/`:
   - unit test: QR backward matches `jax.jacfwd` on a random 64×64 matrix,
   - unit test: CTMRG fixed-point residual decays to $<10^{-10}$,
   - unit test: SVD-AD and QR-AD gradients agree to $<10^{-6}$ on an away-from-criticality TFIM point,
   - unit test: gradient-descent on a 2-parameter toy surface recovers the known minimum.
   Enforce the 16 GB / 4-thread resource block. Pipe final run to `./src/simulation.log`.

5. **Phase 5 — Review Board → `./final_review.md`:** math-pedant scrutinizes the QR-backward derivation; performance-hacker verifies the reported speedups reproduce and that the JAX jit/vmap structure is clean; domain-expert confirms the J1–J2 energies and convergence behavior are physically reasonable.

---

# Practical Constraints

- **Memory:** 16 GB hard cap per process (enforced in the Engineer's resource block).
- **BLAS threads:** 4 (OMP/OpenBLAS/MKL/VecLib/NumExpr).
- **Precision:** default `float64` (`jax.config.update("jax_enable_x64", True)`).
- **Reproducibility:** PRNG seeds, git commit, JAX / optax / jaxopt versions logged.
- **GPU optional:** if unavailable, report CPU-only timings but still execute the full grid on J1–J2 at $D \leq 4$.

---

# Success Criteria

The run is a success if **by the end of Phase 5** the benchmark answers each of the five Research Questions with evidence:

- **RQ1 pass:** converged energies at every $(D,\chi)$ agree between QR-AD and SVD-AD (with Francuz patch) to better than $10^{-5}$ relative on TFIM and Heisenberg, and any disagreements on J1–J2 are attributed and explained.
- **RQ2 pass:** a quantitative metric shows reduced gradient noise for QR-AD at $J_2/J_1 \approx 0.5$ — e.g. lower variance of $\|\nabla E\|$ across the last 50 optimizer steps, or fewer line-search restarts.
- **RQ3 pass:** wall-time speedup numbers reported for every $D$ studied.
- **RQ4 pass:** unrolled-vs-implicit tradeoff quantified specifically for QR-CTMRG.
- **RQ5 pass:** the gauge-fixing choice is stated, tested, and shown to give smooth gradients.

Failure modes that still count as a valid run (publishable *negative result*):

- QR-CTMRG-AD is numerically stable but **not faster in practice** once the backward tape is included.
- QR-CTMRG-AD achieves the speedup but introduces a different backward pathology (rank-revealing failure near criticality).

Either outcome is a method paper. The point of the run is to settle which.
