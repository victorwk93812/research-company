# Objective

Design and execute a **full benchmark study of Automatic-Differentiation (AD) iPEPS with CTMRG environments on the 2D single-band Hubbard model**, with the research spine being a head-to-head comparison across two under-systematized axes:

- **Axis S — symmetries enforced on the ansatz** (fermion parity, U(1) charge, U(1) spin, SU(2), C4v point group, and combinations).
- **Axis O — the AD optimizer stack** (differentiation mode, optimizer, gauge handling, SVD-backward stabilization).

The Researcher is free to propose the exact novel angle (e.g., a new gauge-aware Riemannian optimizer, a new symmetric-tensor AD primitive, or a new stabilization scheme for CTMRG-AD under symmetry), but that angle **must be validated inside the benchmark grid below**, not in isolation.

**Framework (hard constraint):** JAX only. Use `jax`, `jax.numpy`, `optax`, `jaxopt`, `equinox`/pytrees; implicit differentiation via `jax.lax.custom_root` or `jaxopt.implicit_diff`. No PyTorch, no Julia.

---

# Scope and Model

**Hamiltonian (2D square lattice):**
$$H = -t\sum_{\langle ij\rangle,\sigma} c^\dagger_{i\sigma} c_{j\sigma} - t'\sum_{\langle\langle ij\rangle\rangle,\sigma} c^\dagger_{i\sigma} c_{j\sigma} + U\sum_i n_{i\uparrow}n_{i\downarrow} - \mu\sum_i n_i.$$

Set $t=1$ as the energy unit. Fermionic iPEPS with explicit parity swap gates. Energy evaluated via CTMRG two-site reduced density matrix on the appropriate unit cell.

**Reference baselines (for anchoring benchmark energies):**
- Simons Collaboration 2D Hubbard benchmarks (AFQMC / DMRG / DMET), LeBlanc *et al.* PRX 5, 041041 (2015) and Qin *et al.* PRX 10, 031016 (2020).
- At $U/t=8$, $t'=0$, $\delta=0$: thermodynamic-limit $E_0/t \approx -0.524$.
- At $U/t=8$, $t'=0$, $\delta=1/8$: stripe vs uniform-d-wave competition known from AFQMC/DMRG.

---

# Research Questions (to be sharpened by Researcher)

1. Which **symmetries**, when enforced at the tensor-block level during AD, give the best **energy-per-wall-time** at fixed effective bond dimension?
2. Does **implicit CTMRG differentiation** (IFT at the fixed point) retain accuracy advantages in the **presence** of imposed symmetries, or does the block structure change the picture?
3. How much of the residual gradient noise in AD-iPEPS is due to (a) SVD near-degeneracy in the CTM, (b) gauge freedom on virtual bonds, (c) optimizer mis-scaling? The benchmark must separate these contributions.
4. Under doping and finite $t'$, does imposing **C4v** accelerate or bias the identification of d-wave SC vs stripe phases?

---

# Variables To Toggle

For every cell in the benchmark grid, record all hyperparameters plus the metrics listed in Section *Benchmark Metrics*.

## A. Physical parameters (phase-diagram axes)

| Variable | Values | Physics | Why toggle | What it helps |
|---|---|---|---|---|
| `U/t` | 2, 4, 6, 8, 12 | weak-coupling SDW → intermediate "pseudogap" → strong-coupling Mott | Different regimes stress different parts of the numerics; AD stability is regime-dependent | Reveals whether conclusions about symmetry/optimizer transfer across phases |
| `t'/t` | 0, -0.2, -0.3 | Geometric frustration; cuprate-relevant | Breaks bipartite symmetry; stabilizes d-SC, suppresses stripes | Probes near-degenerate competing minima that stress optimizers |
| Filling $n = 1-\delta$ | 1.0, 0.875, 0.8 | half-filling (AFM Mott) / underdoped (stripe vs d-SC) / moderate (FL+SC) | Non-trivial filling requires tuning $\mu$; U(1) charge enforcement becomes load-bearing | Diagnoses whether optimizer-level mis-scaling of $\mu$ corrupts doping sweeps |
| Unit cell | 1×1, 2×2, 2×4 | uniform / Néel / stripe | Larger cells allow spontaneous translation-symmetry breaking | Tests interaction between symmetry breaking and AD convergence |

## B. Ansatz / truncation

| Variable | Values | Physics | Why toggle | What it helps |
|---|---|---|---|---|
| Bond dim `D` | 2, 3, 4, 6 (target 6; 8 only in focused deep-dive) | Max entanglement per bond | Primary extrapolation axis for iPEPS | Tests whether AD+symmetry scales to larger $D$ in practice |
| Env dim `χ` | $D^2$, $2D^2$, $3D^2$ | CTMRG truncation → effective correlation length | AD-through-SVD instabilities scale with CTM singular-value gap | Separates CTMRG truncation error from ansatz error from gradient error |

## C. Symmetries enforced on the ansatz (core axis S)

| Variant | Description | Physics | Why | What it helps |
|---|---|---|---|---|
| `Z2` | Fermion parity only (baseline) | Minimum for fermionic PEPS | Always on | Reference point |
| `U1c` | U(1) charge (total $N$) | Exact Hubbard symmetry | Block-sparse tensors → factor $\sim n_\text{sectors}$ cost reduction, eliminates spurious charge fluctuations | Larger effective $D$ at fixed cost; clean doping sweeps |
| `U1c × U1s` | Add U(1) spin ($S^z$) | Exact in $B=0$, no SOC | Further block decomposition | Clean AFM vs FM discrimination |
| `SU2` | Non-abelian spin (multiplets + CG) | Full rotational invariance in spin space | Prevents spurious SU(2) breaking at finite $D$ (known pathology) | Cleaner diagnostic of true vs algorithmic symmetry breaking |
| `C4v` | Lattice point group (90° rotations, reflections) | Square-lattice point symmetry | Halves/quarters independent tensor parameters; enforces s- vs d-wave irrep decomposition | Direct readout of d-wave vs extended-s pairing channel |
| Combinations | `U1c × U1s × C4v`, `SU2 × C4v`, etc. | — | Isolate individual contributions | Ablation study for symmetry stack |

The JAX implementation MUST represent symmetric tensors as pytrees of block-sparse arrays so that `jax.grad` / `jax.jit` traverse them correctly. A rough sketch (to be finalized by the Researcher and Python Engineer) is expected in the theory draft.

## D. AD / optimizer stack (core axis O)

| Variable | Values | Physics / math | Why | What it helps |
|---|---|---|---|---|
| Diff mode | `unrolled` vs `implicit` (IFT) | CTMRG is a fixed-point iteration | Unrolled: $\mathcal{O}(n_\text{iter}\cdot\chi^2 D^4)$ memory; implicit: one linear solve but no unroll tape | Memory–accuracy tradeoff, especially near criticality |
| Optimizer | Adam, L-BFGS (`jaxopt`), Riemannian L-BFGS on gauge-quotient, Stochastic Reconfiguration / Natural-gradient, Trust-region | Euclidean vs manifold geometry; first- vs second-order | iPEPS has $GL(D)$ gauge redundancy per bond → flat directions kill first-order methods | Separates "landscape pathology" from "optimizer weakness" |
| Gauge fixing | none / QR per step / polar decomp per step / Riemannian parameterization | Bond gauge redundancy | Stale L-BFGS memory and ill-calibrated Adam steps are gauge artefacts | Tests whether gauge cleanup is necessary on top of optimizer choice |
| SVD-backward stabilization | plain / Lorentzian broadening (Hasik–Corboz) / truncated-rank (Francuz 2023) | $1/(\sigma_i^2-\sigma_j^2)$ divergence at near-degeneracies | Degeneracies are generic in symmetric sectors and near criticality | Robustness of the benchmark numerics across all other axes |
| Step-size / line-search | fixed, cosine schedule, backtracking, Wolfe | Step control | Non-convex landscapes; naive steps overshoot | Isolates step-control from direction-quality effects |
| Batching of grad noise | deterministic / mini-env-stochastic | — | Stochastic regularization for escaping local minima | Ablation for convergence robustness |

## E. Benchmark metrics (recorded for every cell in the grid)

- **Accuracy:** final energy $E_0$ vs the AFQMC/DMRG reference at matched $(U,t',\delta)$; relative error.
- **Convergence:** iterations to reach $|\Delta E / E| < 10^{-4}$; final $\|\nabla E\|$; trajectory curves.
- **Wall time per optimization step** (report separately on CPU vs GPU if both are used).
- **Peak memory** (unrolled vs implicit, with `jax.profiler` or `resource.getrusage`).
- **Symmetry fidelity:** imposed symmetries preserved to machine precision (within block structure)?
- **Order parameters:** staggered magnetization $m_s$, d-wave SC correlator $\langle \Delta_d^\dagger \Delta_d\rangle$, stripe structure factor $S(\mathbf{q})$ at $\mathbf{q}=(\pi,\pi/4)$ or similar.
- **Correlation length** $\xi$ from the CTM transfer-matrix leading spectral gap.

---

# Benchmark Protocol

1. **Global sweep (breadth):** $(U/t,\, t'/t,\, \delta)$ over the grid in Section A at $D=4$, $\chi=2D^2$, 2×2 unit cell, for each symmetry variant in Section C and each optimizer variant in Section D. Budget: one run per cell, ≤500 AD steps each.
2. **Focused deep-dive (depth):** the canonical point $U/t=8$, $t'=0$ at $\delta=0$ AND $\delta=1/8$, scanning $D \in \{2,3,4,6\}$ and $\chi \in \{D^2,2D^2,3D^2\}$ across **all** symmetry×optimizer combinations. This is the dataset for the main paper figures.
3. **Ablation:** hold $(U,t',\delta,D,\chi)$ fixed at the canonical point; vary one knob at a time from axes C and D to produce ablation tables.
4. Each run logs: git commit, JAX version, PRNG seed, full hyperparameter dict, all metrics above, and convergence trajectory (every 10 steps).

The Researcher is expected to **reduce** this grid if the compute budget demands it, but the reduction must be justified in `theory_draft.md` and preserve the ability to answer the Research Questions.

---

# Deliverables (by phase, per CLAUDE.md workflow)

1. **Phase 1 — Researcher → `./theory_draft.md`:**
   - Literature review (≥5 recent papers via arxiv MCP; Liao 2019 PRX, Hasik/Corboz 2021, Francuz 2023, Ponsioen, Scheb, Corboz, plus Hubbard iPEPS references Corboz 2014/2016).
   - Positioning of the novel contribution.
   - Formal definitions: Hilbert space, fermionic PEPS with swap gates, CTMRG fixed-point equations, each symmetry's tensor-block decomposition, and the AD primitives required.
   - Complexity analysis for each (symmetry × diff-mode × optimizer) combination.
   - Reduced benchmark grid (if any) with justification.

2. **Phase 2 — RA Skeptic → `./ra_critique.md`:** independent literature cross-check, pedantic math review, feasibility check against the 16 GB memory limit and reasonable wall-time budget. Loop with Phase 1 until approval.

3. **Phase 3 — LaTeX Writer → `./report/main.tex`:** XeLaTeX, `revtex4-2`, full benchmark tables, convergence plots (from `./src/figures/`), energy-vs-$D$ extrapolations, ablation tables.

4. **Phase 4 — Python Engineer → `./src/`:** `uv init`-ed JAX project. Modular harness: `hamiltonian.py`, `symmetric_tensor.py`, `ctmrg.py`, `ad_pipeline.py`, `optimizers.py`, `benchmark.py`, `main.py`. Every toggle in Sections A–D exposed via config (YAML or dataclass). Resource block (16 GB cap, 4 threads per BLAS lib) at the top of every entry point. Tests in `./src/tests/` for: (i) symmetric tensor AD correctness vs dense AD on a tiny example, (ii) CTMRG fixed-point convergence, (iii) implicit vs unrolled gradient agreement on a toy model. Execute and pipe logs to `./src/simulation.log`.

5. **Phase 5 — Review Board → `./final_review.md`:** math-pedant, performance-hacker, domain-expert sections; verdict on whether the benchmark answers the Research Questions.

---

# Practical Constraints

- **Memory:** hard 16 GB cap per Python process (enforced via the `resource` block in the Engineer persona).
- **BLAS threads:** capped at 4 (OMP/OpenBLAS/MKL/VecLib/NumExpr).
- **Reproducibility:** every run logs PRNG seed, git commit hash, `jax.__version__`, `optax.__version__`.
- **Numerical precision:** default to `float64` (`jax.config.update("jax_enable_x64", True)`); document any cell run in `float32` explicitly.
- **GPU optional:** if no GPU is available, prioritise the focused deep-dive over the global sweep.
- **Scope discipline:** if a novel contribution is claimed, it must be testable inside the benchmark — no untestable claims.
