# QR-CTMRG with End-to-End AD inside `tenax`: Theory Draft

**Author role:** Lead Theoretical Researcher (Phase 1)
**Run dir:** `/workspace/runs/iPEPS-Symmetry-Optimization`
**Date:** 2026-04-27

---

## Literature Review

I consulted the following papers (arXiv id, title, one-sentence contribution, role here).

| arXiv id | Short title | Contribution | Role |
|---|---|---|---|
| **2505.00494** | Naef–Hauschild / Zhang–Yang–Corboz, *Accelerating 2D tensor-network contractions using QR* (2025) | Replaces the SVD in C4v-CTMRG projector construction with QR; reports up to two orders of magnitude forward-pass speedup on H100 GPU on Heisenberg and J1–J2. **No AD.** | The forward we wrap, but with AD added and on CPU. |
| **2509.05090** | Yang & Corboz, *Efficient iPEPS on honeycomb via QR-CTMRG* (2025; PRB 2026) | Extends the QR-CTMRG idea to C3v honeycomb; reports order-of-magnitude speedup; same accuracy. **No AD.** | Independent confirmation that the QR projector is sound on a non-square geometry. |
| **2311.11894** | Francuz, Schuch, Vanhecke, *Stable and efficient differentiation of tensor-network algorithms*, **PRR 7, 013237 (2025)** | Identifies & fixes a fundamental inaccuracy in the standard truncated-SVD backward used inside CTMRG-AD; introduces a Lorentzian-regularised F-matrix and a kept↔discarded truncation correction; gauge-fixing recipe. | The **baseline**: this fix is what `truncated_svd_ad` and `truncated_eigh_regularized` in tenax implement. We must compare against this — not against the naïve SVD-AD. |
| **2511.09546** | Zhang, Yang, Corboz, Haegeman, Tang, *Accelerating 2D tensor-network optimisation by preconditioning* (Nov 2025, rev. Mar 2026) | Local quantum-geometric-tensor (QGT) preconditioner for L-BFGS / CG iPEPS optimisation on Heisenberg, Kitaev. | The metric preconditioner is in tenax (`_metric_precond.py`); we hold ctmrg fixed and toggle this on the orthogonal axis. |
| **2502.10298** | Naumann, Weerda, Eisert, Rizzi, Schmoll, *Split-CTMRG for large-bond iPEPS* (2025) | Two-projector / Fishman split corner formulation with biorthogonality $P_1^\dagger P_2 = I$. | Sets the convention used by tenax for the SVD-baseline projector pair. |
| **2508.10822** | Tang, Vanderstraeten, Haegeman, *Gauging the variational optimisation of PEPS* (2025) | Demonstrates that gradient-based PEPS optimisation exploits gauge degrees of freedom and biases energies; proposes a gauge-fixed scheme. | Justifies the diagonal-phase gauge we impose on QR ($R_{ii}>0$). |
| **2511.22669** | Cortés Estay, Kamar, Corboz, *Accurate energy variance with iPEPS* (Nov 2025) | Formula and CTMRG protocol for the iPEPS energy variance; enables zero-variance extrapolation. | We use this as the *cross-check*: the extrapolated energy must be (ctmrg_mode, optimiser)-independent. |
| **1903.09650** | Liao, Liu, Wang, Xiang, *Differentiable programming tensor networks*, PRX 9, 031041 (2019) | Original AD-CTMRG; introduces the implicit-fixed-point trick. | The intellectual ancestor — uses standard SVD-AD; we replace SVD with QR and use Francuz's stable backward in the eigh substep. |
| **2512.05749** | Zhou *et al.*, *Stochastic reconfiguration with warm-started SVD* (Dec 2025) | SR with iterative low-rank refinement of the preconditioner, in VMC. | Stretch goal only — the paper is VMC, not iPEPS, so we treat it as a *structural template* for a fourth optimiser and document if not implemented. |

I read the abstracts of all nine via `arxiv.org/abs/...` (the MCP `arxiv` server returned HTTP 429 for the bulk batch; abstracts were cross-checked via the public web mirror). For the load-bearing Francuz, Naef–Hauschild, Liao, and Rader papers I additionally inspected tenax's existing implementations (`ad_utils.py::truncated_svd_ad`, `_lorentzian_eigh.py::truncated_eigh_regularized`, `_metric_precond.py`, `_gmres_lax.py`, `_ctm_projector.py::_qr_projector_symmetric`) to confirm the methodology already in the substrate.

---

## Positioning

The new contribution of this run, *as a method-paper inside the fixed substrate `tenax`*, is the combination

$$
\boxed{\text{QR-CTMRG}_{\text{forward}} \;\oplus\; \text{Francuz-style truncation backward}_{\text{eigh}} \;\oplus\; \text{tenax's implicit-fixed-point IFT}}
$$

evaluated *under three optimisers* (Adam, L-BFGS, metric-preconditioned L-BFGS) on the standard ladder of 2D models (TFIM, Heisenberg, J1–J2). Concretely:

* `arXiv:2505.00494` did the forward QR replacement on GPU and did **not** integrate AD.
* `arXiv:2311.11894` (Francuz) gave a *truncated-SVD* backward; nobody has yet checked whether the same Lorentzian regularisation, applied to the *small-eigh substep that follows QR*, is sufficient to make a QR-CTMRG-AD pipeline numerically stable end-to-end.
* `arXiv:2511.09546` (Rader) gave a metric preconditioner — orthogonal to the projector choice. Whether the QR-AD stability gain *adds to* or *is subsumed by* this preconditioner is an open question (RQ5).

We add nothing exotic to tenax; we register one new projector mode and a single dispatcher hook. The headline experimental claims are *honest CPU* numbers and a *2×2 interaction-effect* table over (SVD/QR) × (plain/metric-preconditioned). We deliberately do **not** reproduce the 100× GPU speedup; that number is hardware-specific.

---

## 1. Setting and notation

We work on the infinite square lattice $\mathbb{Z}^2$ with site Hilbert space $\mathcal{H}_s = \mathbb{C}^d$ ($d=2$ for spin-1/2). The variational ansatz is an iPEPS,

$$
|\Psi(A)\rangle = \mathrm{tr}_{\text{aux}} \prod_{(i,j)\in\mathbb{Z}^2} A^{(s_{ij})}_{u\,d\,l\,r},
\qquad A \in \mathbb{C}^{D\times D \times D \times D \times d},
$$

with bond dimension $D \le 5$ (CPU constraint). For the AFM Heisenberg and J1–J2, we use a 2-site checkerboard $\{A,B\}$ unit cell; for TFIM, $1\times1$.

The expectation value $E(A) = \langle \Psi(A) | H | \Psi(A) \rangle / \langle \Psi(A) | \Psi(A) \rangle$ is approximated via CTMRG of bond dimension $\chi$. The CTMRG fixed-point environment $\mathcal{C}^\star(A)$ (eight tensors $\{C_i, T_i\}_{i=1\dots4}$) satisfies

$$
\mathcal{C}^\star(A) = F(\mathcal{C}^\star(A), A),
\qquad F:\text{one CTMRG sweep}.
$$

Differentiating $E$ through this fixed point requires the implicit-function-theorem (IFT) Jacobian $(I - \partial_\mathcal{C} F)^{-1}$ — solved in tenax by either Neumann series (default) or restarted GMRES inside `lax.while_loop` (`_gmres_lax.py`).

### 1.1 Hamiltonians

* **TFIM** (1×1): $H_{\text{TFIM}} = -J \sum_{\langle ij\rangle} \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i$.  We sweep $h/J\in\{2.5, 3.04_{\text{c}}, 3.5\}$ at $J=1$.
* **Heisenberg** (2-site C4v): $H_{\text{H}} = J\sum_{\langle ij\rangle} \mathbf{S}_i\cdot\mathbf{S}_j$, $J=1$. Reference $E_0 \approx -0.66944$ (QMC; tenax example reaches $-0.6628$ at $D{=}2,\chi{=}16$).
* **J1–J2** (2×2 generic): $H_{\text{J}_1\text{J}_2} = J_1\sum_{\langle ij\rangle}\mathbf{S}_i\cdot\mathbf{S}_j + J_2\sum_{\langle\langle ij\rangle\rangle}\mathbf{S}_i\cdot\mathbf{S}_j$; $J_1=1$, $J_2/J_1\in\{0.0,0.4,0.5,0.55,0.6\}$.

### 1.2 CTM projector — the object being replaced

Inside one CTMRG sweep, the projector pair $(P_1, P_2)$ is built by combining two grown half-corner tensors $C_{1g}, C_{4g}$ along their shared "fused" leg of dimension $f$. The kept dimension is $\chi$. tenax already implements three modes (`_ctm_projector.py`):

* **`eigh`** — form $\rho = C_{1g}C_{1g}^\dagger + C_{4g}C_{4g}^\dagger$, eigh, keep top-$\chi$. AD route uses `regularized_eigh` (Lorentzian).
* **`svd`** (Fishman) — form $M = C_{1g}^\dagger C_{4g}$, SVD; build $P_1 = C_{4g}V S^{-1/2}, P_2 = C_{1g}U S^{-1/2}$. AD route uses `truncated_svd_ad` (Francuz patch — this is the **baseline** of this run).
* **`qr`** (forward only as of today) — block-sparse QR per charge sector; under AD it falls back to a *dense* `regularized_svd(R)`, i.e. the AD path is still SVD-based.

The new mode added by this run is **`qr_canonical`**, distinguished from the existing `qr` by:

1. End-to-end differentiable forward (no SVD substitution under tracers).
2. Custom backward routed through `truncated_eigh_regularized` (the Francuz-style Lorentzian truncation kernel that is already in tenax for the eigh path) rather than through `regularized_svd`.
3. A canonical $R_{ii}>0$ gauge that is differentiably smooth.

### 1.3 What "stability" means here

We adopt three operational metrics for RQ2:

* **Gradient-norm variance** over the last 50 optimiser steps near a converged point: $\text{Var}\bigl(\log \|\nabla E\|_k\bigr)_{k=N-49}^N$.
* **Line-search restart count** (Hager–Zhang) per 100 steps.
* **CTM iterations to converge** at each optimiser step.

Lower variance, fewer restarts, fewer CTM iterations ↔ smoother landscape.

---

## 2. The QR-canonical CTMRG projector — forward

For each fused-leg charge sector $q$ (or globally if no symmetry):

1. **Concatenate** the column-blocks of the two grown corners along their non-fused index:
   $$M_q = \bigl[\, C_{1g,q}\;\;C_{4g,q}\,\bigr]\;\in\;\mathbb{C}^{f_q \times c_q},\qquad c_q = c_{1,q}+c_{4,q}.$$
2. **QR-factor** $M_q$: $M_q = Q_q R_q$ with $Q_q\in\mathbb{C}^{f_q\times r_q}$, $R_q\in\mathbb{C}^{r_q\times c_q}$, $r_q=\min(f_q,c_q)$.
3. **Canonicalise the QR gauge** by rotating the diagonal phase of $R_q$ to be real-positive:
   $$\phi_q = \mathrm{diag}(R_q) / |\mathrm{diag}(R_q)|,\qquad
     Q_q \leftarrow Q_q\,\mathrm{diag}(\phi_q),\;\; R_q \leftarrow \mathrm{diag}(\bar\phi_q)\,R_q.$$
   At zero diagonal we set $\phi=1$ (the gauge is unconstrained there).
4. **Reduced symmetric eigendecomposition** of the small $r_q\times r_q$ matrix:
   $$\rho_q^{(s)} = R_q R_q^\dagger,\qquad \rho_q^{(s)} = U_q\,\Lambda_q\,U_q^\dagger,\qquad \Lambda_q = \mathrm{diag}(\lambda_{q,1}\ge\dots\ge \lambda_{q,r_q}).$$
   The eigenvalues $\lambda_{q,i}$ equal the squared singular values of $M_q$ (since $Q_q$ is an isometry), so global truncation by $\lambda$ is exactly the SVD-based truncation by $\sigma^2$.
5. **Global truncation** to $\chi$ eigenvectors across sectors (matching tenax's existing dispatch).
6. **Lift back**: per kept index, the projector column is
   $$P_{q,i} = Q_q\,U_{q,i}\;\in\;\mathbb{C}^{f_q}.$$
   With both projectors equal ($P_1 = P_2 = P$) for `qr_canonical`, mirroring the eigh path; the Fishman two-projector form is reserved for the SVD baseline.

**Cost.** Per sector: $\mathrm{QR}(f_q,c_q) = O(f_q c_q^2)$ + small eigh $O(c_q^3)$ — versus per-sector dense SVD $O(f_q c_q\,\min(f_q,c_q))$. When $c_q \ll f_q$ (the iPEPS-CTM regime, $c_q = D^2$ vs $f_q = \chi D^2$), the asymptotic ratio favours QR. On CPU the constants matter: BLAS QR uses Householder triangularisation while SVD uses the slower bidiagonal-then-iterative QR; we expect a *modest* speedup.

---

## 3. The QR-canonical CTMRG projector — backward

The full chain is $M \xrightarrow{\text{QR}} (Q, R) \xrightarrow{\rho=RR^\dagger} \rho \xrightarrow{\text{eigh}} (U,\Lambda) \xrightarrow{\text{trunc}} (U_k, \Lambda_k) \xrightarrow{P=QU_k} P$.

The cotangent $\bar P \in \mathbb{C}^{f\times \chi}$ flows back as:

### 3.1 Through the lift $P = Q U_k$

$$\bar Q = \bar P\,U_k^\dagger,\qquad \bar U_k = Q^\dagger\,\bar P. \tag{3.1}$$

### 3.2 Through the truncated eigh of $\rho$

We re-use tenax's existing `truncated_eigh_regularized` (in `_lorentzian_eigh.py`); its custom_vjp uses the Lorentzian-regularised differential

$$F^{(\epsilon)}_{ji} = \frac{\lambda_i-\lambda_j}{(\lambda_i-\lambda_j)^2 + \epsilon^2},\qquad \epsilon = \max(\epsilon_0,\;\epsilon_{\text{rel}}\,|\lambda|_{\max}),$$

which handles **both** the kept↔kept anti-gauge term and the kept↔discarded truncation correction in a single $n\times k$ inner product (`_truncated_eigh_lorentzian_backward`, `_lorentzian_eigh.py:38–78`). The output is a Hermitian $\bar\rho\in\mathbb{C}^{r\times r}$.

### 3.3 Through $\rho = R R^\dagger$

$\rho = R R^\dagger$ is bilinear in $R$, so

$$d\rho = (dR)\,R^\dagger + R\,(dR)^\dagger \;\Rightarrow\; \bar R = (\bar\rho + \bar\rho^\dagger)\,R = 2\,\bar\rho\,R$$

since `truncated_eigh_regularized` already symmetrises $\bar\rho$.

### 3.4 Through the canonical-gauge QR

Let $A\equiv M$ for clarity. The QR factorisation $A=QR$ with $Q^\dagger Q = I$ and $R$ upper-triangular has the well-known reverse-mode rule (Roberts 1963; Walter 2012; Hubig & McCulloch 2019):

$$\bar A = \Bigl[\,\bar Q + Q\;\mathrm{copyltu}\bigl(R\,\bar R^\dagger - \bar Q^\dagger Q\bigr)\,\Bigr]\,(R^{-\dagger}), \tag{3.4}$$

where $\mathrm{copyltu}(X)_{ij} = X_{ij}$ for $i>j$, $X_{ii}/2$ on the diagonal, and the strictly upper triangle filled by Hermitian conjugation of the strictly lower triangle. Two practical points:

1. **Diagonal-phase gauge.** Step (3) of §2 performs $Q\to Q\Phi, R\to\Phi^\dagger R$ with $\Phi=\mathrm{diag}(\phi)$. Inserted into (3.4), the $\Phi$ rotations cancel in $\bar A$ identically when $|\phi|=1$, **provided** the upstream $\bar P$ is gauge-equivariant — which it is, because $P=QU_k$ is built from the same $Q$ that has been rotated. We therefore apply the gauge fix in the forward and let the reverse-mode chain rule absorb it without an explicit Jacobian projection.

2. **Singular $R$.** When $\mathrm{rank}\,R < r$ (encountered near rank-deficient corners at small $D$), $R^{-1}$ does not exist. We use `jnp.linalg.solve_triangular` with a small Tikhonov shift $R\to R + \delta_R\, I$ where $\delta_R = 10^{-12}\,\|R\|_{\!2}$; the gauge fix already maps zero diagonals to $\phi=1$ so the singularity is at most a single mode and the Tikhonov regularisation is bounded.

### 3.5 Through `concat`

The leading `concat` distributes back trivially: $\bar C_{1g,q} = \bar M_q[:,:c_{1,q}]$, $\bar C_{4g,q} = \bar M_q[:,c_{1,q}:]$.

### 3.6 Integration with the implicit-fixed-point IFT

The custom-VJP for the QR-canonical projector is local — i.e. it returns $(\bar C_{1g}, \bar C_{4g})$ given $\bar P$. The outer fixed-point loop $\mathcal{C}^\star = F(\mathcal{C}^\star, A)$ is differentiated by tenax's `_ctm_tensor_converge_fwd/_ctm_tensor_converge_bwd` (`ad_utils.py:1117–1303`), which composes our local projector VJP with the Neumann/GMRES IFT solve and, optionally, the Arnoldi spectral-radius pre-check (`_arnoldi.py`). **Nothing in steps 3.1–3.5 changes the IFT Jacobian assumption** ($\rho(\partial_{\mathcal{C}} F) < 1$ at the fixed point). The QR forward and the eigh backward both leave the spectral radius of the linearisation in the same range as the SVD baseline, so the existing precheck and GMRES solver apply unchanged.

---

## 4. Hypotheses (RQ1–RQ6)

| RQ | Hypothesis | Falsifier |
|---|---|---|
| RQ1 | QR-AD and SVD-AD reach the same fixed-point environment ($\|\Delta\mathcal{C}\|<10^{-10}$) and the same converged energy ($\Delta E_{\text{rel}}<10^{-5}$) on TFIM at $h/J=2.5,3.04,3.5$ and Heisenberg. | Energy gap $>10^{-5}$ or fixed-point disagreement $>10^{-9}$. |
| RQ2 | At J1–J2 with $J_2/J_1=0.5$, QR-AD has lower $\mathrm{Var}(\log\|\nabla E\|)_{50}$ and fewer Hager–Zhang restarts than SVD-AD, *for at least 2 of {Adam, LBFGS, metric-LBFGS}*. | QR-AD does not improve any of the three metrics on $\ge 2$ optimisers. |
| RQ3 | On CPU at $\chi\le 32$, QR-CTMRG forward is $1.2\times{-}3\times$ faster than SVD-CTMRG; backward QR is *roughly equal* to backward SVD because both go through Lorentzian-regularised eigh of comparable size. | QR backward is $>2\times$ slower than SVD backward (would mean our truncation kernel is mis-tuned). |
| RQ4 | The (ctmrg_mode, optimiser) table has metric-LBFGS+QR as the fastest-and-stablest cell; Adam+SVD as the worst. | Adam+QR or LBFGS+QR equals or beats metric-LBFGS+QR (would mean preconditioner is unnecessary). |
| RQ5 | The 2×2 interaction at J1–J2($J_2/J_1=0.5$) is *additive*: gain(QR) + gain(metric) $\approx$ gain(QR+metric) within 30 %. | gain(QR+metric) $<$ max(gain(QR), gain(metric))+10 %, indicating the two are degenerate. |
| RQ6 | With U(1) symmetry on, the QR-vs-SVD ratio on Heisenberg at $D{=}4$ improves (per-sector $c_q$ shrinks faster for SVD because of QR's better cubic constant). | The ratio is unchanged or reverses. |

---

## 5. Engineering / harness summary (handoff to Phase 4)

* **Implementation site:** `./src/ctmrg_qr.py` exports a JAX-`custom_vjp` callable `qr_canonical_projector(M, chi) -> P_dense` and `./src/ctmrg_qr_register.py` monkey-patches `tenax.algorithms._ctm_projector._compute_projector_tensor` to dispatch `projector_method="qr_canonical"` to our routine *only when the AD-traced (dense, tracer) branch is hit* — the non-AD block-sparse `_qr_projector_symmetric` remains unchanged. **No edits to `./tenax/`.**
* **Optimizer harness:** `./src/optimizer_harness.py::run_cell(model, D, chi, ctmrg_mode, optimizer, seed)` wraps `tenax.optimize_gs_ad` with the right `iPEPSConfig` toggles:
  - `ctm.projector_method` ∈ {`"svd"`, `"qr_canonical"`} (eigh kept available as ablation)
  - `ctm.projector_backward = "lorentzian"` for the SVD baseline (Francuz patch)
  - `gs_optimizer` ∈ {`"adam"`, `"lbfgs"`}; `gs_metric_precond` ∈ {`False`, `True`}
  - `chi_ramp = [(50, 4D), (100, 6D), (200, min(8D, χ_max))]`
* **Sanity gate:** `tests/test_qr_vs_svd.py` reproduces `examples/heisenberg_ipeps_ad.py` energy at $D{=}2,\chi{=}16$ within $10^{-4}$ under both modes; QR-AD vs `jax.jacfwd` finite-difference on a random $32\times 32$ matrix to $<10^{-6}$; CTM fixed-point residual $<10^{-10}$ for both modes on TFIM at $h/J=2.5$.
* **CPU honesty:** `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=4`. Per cell: 3 seeds, log mean ± std on (E, ‖∇E‖, wall-clock fwd/bwd/total, CTM iters, line-search restarts, fixed-point residual). YAML configs in `./src/configs/`.

For today's run the **headline 15-min budget** uses a small magnitude grid:

* TFIM, $D=2,\chi\in\{8,16\}$, h ∈ {2.5, 3.04, 3.5} — 1×1 unit cell.
* Heisenberg, $D=2,\chi=16$ — 2-site checkerboard.
* J1–J2, $D=2, \chi=16$, $J_2/J_1\in\{0.0, 0.5\}$ — 2-site generic.
* All four optimisers × both ctmrg modes × 2 seeds, ≤40 AD steps each, with early stopping. The Phase 4 engineer will trim the grid further if time runs over.

The larger-magnitude run (background subagent, Phase 6) escalates to $D\in\{3,4\}, \chi\in\{16,32\}$, full 5-point J1–J2 sweep, 3 seeds.
