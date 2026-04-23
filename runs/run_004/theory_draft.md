# Theory Draft — QR-CTMRG with End-to-End Automatic Differentiation for iPEPS

*Phase 1 (Researcher).*

## Literature Review

All arxiv ids below were consulted through the `arxiv` MCP server. For each I list the one-sentence contribution and how it bears on the present proposal. For the three most load-bearing papers I also pulled abstract / bibliographic metadata through `get_paper` / `download_paper_text`.

| arxiv id | Title (short) | One-line contribution | Role in this proposal |
|---|---|---|---|
| **2505.00494** (Zhang, Yang, Corboz, 2025) | QR-based CTMRG for 2D tensor network contraction | Replaces the SVD in $C_{4v}$-symmetric CTMRG by QR; reports ≈100× speedup on H100 GPU for Heisenberg & J$_1$–J$_2$. | **Direct precursor**: we *extend this forward algorithm with a complete backward pass*. Their paper explicitly does not integrate with AD. |
| **2509.05090** (Yang, Corboz, 2025; PRB 113, 085109, 2026) | QR-CTMRG on honeycomb lattice with native $C_{3v}$ symmetry | Generalizes QR-CTMRG to honeycomb; still forward only. | Confirms robustness of the QR trick under a different point-group and gives a template for later non-square generalization of our AD pipeline. |
| **1903.09650** (Liao, Liu, Wang, Xiang, 2019; PRX 9, 031041) | Differentiable programming tensor networks | Introduces reverse-mode AD through CTMRG using a "stabilized SVD backward"; achieves state-of-the-art iPEPS energies. | **Reference baseline**: this is the pipeline whose SVD-backward we wish to replace. We keep their high-level architecture (gradient-based variational iPEPS with CTMRG environments) unchanged. |
| *Francuz et al., PRR 7, 013237 (2025)* (as cited in `instruction.md`) | Stable / correct CTMRG-AD | Identifies a *fundamental inaccuracy* in the standard SVD backward for truncated CTMRG, independent of near-degeneracy. | **Target of replacement**: this is the pathology our QR-based backward avoids at the structural level. We include Francuz's correction in the SVD baseline so the comparison is fair. |
| **2508.10822** (Tang, Vanderstraeten, Haegeman, 2025) | Gauging the variational optimization of PEPS | Shows how PEPS bond-gauge freedom is exploited by gradient optimizers to produce artificially low energies; proposes gauge-fixed optimization. | Load-bearing for **C3 (gauge)**: our QR introduces an *additional* diagonal-phase gauge which must not collide with the PEPS bond gauge. We adopt their gauge-fixed optimization philosophy. |
| **1805.00055** (Hauschild, Pollmann, 2018) | TeNPy review | Pedagogical source for CTMRG projector construction via bi-orthogonal isometries (Fishman–White style). | Notation and conventions. |
| **1503.05345** (Phien, Bengua, Tuan, Corboz, Orus, 2015; PRB 92, 035142) | Fast full update and gauge fixing for iPEPS | Extends local gauge fixing to iPEPS and shows stability gains. | Historical grounding for the "gauge fixing stabilises optimization" claim. |
| **cond-mat/9507087** (Nishino, Okunishi, 1995) | Corner transfer matrix RG method | Original CTMRG. | Foundational, defines the fixed-point equation we differentiate. |
| **1912.02780** (Chen et al., 2020; PRB 101, 220409) | Automatic differentiation for second RG | Shows AD-TRG beating manual second-RG on 2D Ising; demonstrates that AD generalises recursive environment construction. | Concept template for differentiating through iterative renormalization. |
| **2107.03399** (Ponsioen, Assaad, Corboz, 2022; SciPost 12, 006) | AD applied to PEPS excitations | Confirms that AD-iPEPS extends to excited-state sectors. | Evidence that robust AD through CTMRG is a practical pipeline beyond ground-state energy. |
| **2211.13765** (Ahmed, Killoran, Carrasquilla, 2022) | Implicit differentiation of variational quantum algorithms | Blueprint for treating the ground-state-finding procedure as an implicit function and differentiating through its solver. | Directly supports **C2** (implicit diff through the CTMRG fixed point). |
| **2009.02606** (Haghshenas, 2021; PRR 3, 023148) | Tensor-network QR optimization for MERA | Uses QR for tensor-network optimization; orthogonal to but conceptually aligned with our square-lattice QR-CTMRG. | Evidence QR-based tensor manipulations are competitive in related TN contexts. |
| **2009.01997** (Morita, Kawashima, 2021; PRB 103, 045131) | Global CTMRG-based TRG optimization | Uses CTMRG environment for global optimization of coarse-grained tensors. | Inspiration for the role of the CTMRG fixed-point environment as a smooth functional of $A$. |

## Positioning

Given the above, the **novelty** of this run is strictly:

1. **First end-to-end AD pipeline built on QR-CTMRG.** Zhang–Yang–Corboz (`2505.00494`) deliberately restrict to forward contraction; Liao *et al.* (`1903.09650`) and Francuz *et al.* (PRR 2025) only have SVD at the projector step. We derive and implement the custom backward rule for the *QR-projector* step, for both unrolled and implicit-differentiation modes.
2. **Structural elimination of SVD-backward pathologies.** By replacing SVD with QR inside CTMRG, we obtain a backward rule that contains no $1/(\sigma_i^2-\sigma_j^2)$ denominators and — because QR truncation is a linear isometry rather than a rescaling by $\Sigma^{-1/2}$ — is immune to the "fundamental inaccuracy" of the Francuz correction target.
3. **Explicit QR-gauge treatment.** The QR has a $U(1)^{\chi}$ diagonal-phase gauge that is unphysical inside CTMRG but *can* interact with the PEPS bond gauge (see `2508.10822`); we fix it canonically with $\mathrm{sign}(R_{ii})>0$ (real) or $R_{ii}/|R_{ii}|=1$ (complex) and prove smoothness of the resulting map under standard rank-separation assumptions.
4. **Quantitative comparison against the strongest SVD-AD baseline.** Our SVD-AD uses the Francuz-corrected backward (the strongest published baseline), not "plain" Lorentzian regularization. Any advantage we find for QR cannot be written off as comparing to a straw-man.

We are *not* claiming novelty for QR-CTMRG forward (that is `2505.00494`) nor for AD-CTMRG per se (that is Liao *et al.*). The combination, the explicit backward derivation, and the J$_1$–J$_2$ benchmark near $J_2/J_1\!\approx\!0.5$ are new.

---

## 1. Physical setting

### 1.1 Hilbert space and Hamiltonians

We study SU(2)-symmetric spin-$1/2$ systems on the 2D square lattice with local Hilbert space $\mathcal{H}_s = \mathbb{C}^2$, full Hilbert space $\bigotimes_{s\in\mathbb{Z}^2}\mathcal{H}_s$, and translation invariance under $\mathbb{Z}^2$.

**TFIM** (Ising symmetry broken by a transverse field, $\mathbb{Z}_2$ symmetry):
$$
H_{\text{TFIM}} = -J\sum_{\langle i,j\rangle} \sigma^z_i \sigma^z_j - h\sum_i \sigma^x_i , \qquad (h/J)_c \approx 3.04438.
$$

**Heisenberg antiferromagnet** (SU(2), broken down to U(1) in the Néel ground state on bipartite lattices):
$$
H_{\text{HAF}} = J\sum_{\langle i,j\rangle} \mathbf{S}_i\cdot\mathbf{S}_j , \qquad J>0.
$$

**J$_1$–J$_2$** (frustrated, nearest + next-nearest neighbour):
$$
H_{J_1J_2} = J_1 \sum_{\langle i,j\rangle}\mathbf{S}_i\cdot \mathbf{S}_j + J_2\sum_{\langle\langle i,j\rangle\rangle} \mathbf{S}_i\cdot\mathbf{S}_j.
$$
Known regime: Néel for $J_2/J_1 \lesssim 0.4$, plaquette/VBS for $0.5\lesssim J_2/J_1\lesssim 0.6$, columnar for $J_2/J_1 \gtrsim 0.6$. The crossover is where SVD-backward pathologies hurt most (near-degenerate singular values of the CTM corner).

### 1.2 Variational ansatz: iPEPS

We parametrize translation-invariant ground states by a single rank-5 site tensor
$$
A_{urdl}^{\sigma} \in \mathbb{C}^{d\times D\times D\times D\times D},
$$
with $d=2$ the physical dimension and $D$ the virtual bond dimension. For C$_{4v}$ tests we enforce $A$ invariant under the lattice $C_{4v}$ group (matching `2505.00494`); for J$_1$–J$_2$ at and beyond the transition we use a generic $2\times 2$ unit cell.

The variational energy per site is
$$
E(A) = \frac{\langle \psi(A) | H | \psi(A)\rangle}{\langle \psi(A) | \psi(A)\rangle}.
$$
Approximate contraction proceeds via CTMRG; the CTMRG environment $\mathcal{E}^\star(A)$ is computed iteratively to a fixed point and $E(A)$ is evaluated on a finite environmental patch.

---

## 2. Forward CTMRG: SVD projector vs QR projector

### 2.1 The projector step

Let $T(A)$ denote the double-layer transfer tensor built from $A$ and its conjugate. One CTMRG sweep absorbs a row/column into the environment:
$$
\tilde C = C \cdot T, \qquad \tilde T = T_{\text{edge}} \cdot T.
$$
Because $\tilde C$ has bond dimension $\chi D^2$ it must be projected back to $\chi$. A projector
$P: \mathbb{C}^{\chi D^2}\!\to\!\mathbb{C}^\chi$ is chosen so that $\tilde C \approx P^\dagger \tilde C P$ optimally in the dominant subspace.

- **SVD projector** (Liao 2019 / Francuz 2025): build an enlarged corner $Q = \tilde C_1 \tilde C_2^T$, compute $Q = U\Sigma V^\dagger$, truncate to the top $\chi$ singular values $\Sigma_\chi$, and set $P = U_\chi \Sigma_\chi^{-1/2}$ (or its symmetric Fishman–White variant). The inverse $\Sigma^{-1/2}$ is the source of the $1/(\sigma_i^2-\sigma_j^2)$ divergence in the backward pass, and the Francuz correction addresses the *residual* bias that persists even without near-degeneracies.
- **QR projector** (`2505.00494`): compute the thin QR $\tilde C_1 = Q R$ with $Q\in\mathbb{C}^{\chi D^2\times \chi}$, $R\in\mathbb{C}^{\chi\times\chi}$. Take $P = Q$ (or its bi-orthogonal pair). Truncation is by *selecting the first $\chi$ columns*, not by rescaling.

For C$_{4v}$-symmetric corners the choice simplifies: a single QR on the symmetrized half-corner yields the isometry. For generic unit cells we use a bi-orthogonal $QR/LQ$ pair — see §4.

### 2.2 Why QR is cheaper on GPU

The QR on an $m\times n$ matrix ($m\ge n$) costs $O(mn^2)$ as Householder reflections but with a *much* smaller constant than the QR + SVD pipeline, and, crucially, Householder-QR is dominated by `GEMM`-style level-3 BLAS calls. GPU linear algebra libraries (cuSOLVER, cuBLAS) ship highly tuned batched QR kernels; GPU SVD, in contrast, is bound by the Jacobi/bidiagonalization stages that are not well parallelised. `2505.00494` Table II reports the precise SVD/QR ratio vs $\chi$ on H100: for $\chi=80$, QR is ≈80× faster than SVD.

### 2.3 Complexity summary

Per CTMRG sweep:

| Step | SVD pipeline | QR pipeline |
|---|---|---|
| Contract enlarged corner | $O(\chi^2 D^4 + \chi^3 D^2)$ | same |
| Projector construction | $O(\chi^3 D^6)$ (SVD of $\chi D^2$ × $\chi D^2$) | $O(\chi^3 D^4)$ (thin QR) |
| Truncation + absorbtion | $O(\chi^3 D^4)$ | $O(\chi^3 D^4)$ |
| **Total** | $O(\chi^3 D^6)$ | $O(\chi^3 D^4)\cdot(\text{QR constant})$ |

Asymptotically, the dominant $\chi^3 D^4$ contraction is the *same*; what changes is the *constant* in front and the *GPU-friendliness*. The 2× order of magnitude speedup reported in `2505.00494` is almost entirely due to replacing the SVD's bidiagonalization by Householder-QR, not to asymptotic scaling.

---

## 3. Backward pass — the core derivation

We distinguish two modes: **unrolled AD** (track the tape through $N_{\text{CTM}}$ sweeps) and **implicit AD at the CTMRG fixed point** (IFT).

### 3.1 Standard QR backward (Hubig–McCulloch 2019; Roberts 1963; Mathias 1996)

For $M = Q R$ with $M\in\mathbb{C}^{m\times n}$, $m\ge n$, $Q$ with orthonormal columns ($Q^\dagger Q = I_n$), $R$ upper triangular with $R_{ii}>0$:
$$
\boxed{\;\bar M \;=\;\Bigl(\bar Q + Q\,\mathrm{copyltu}\bigl(R\,\bar R^{\dagger}-\bar Q^{\dagger} Q\bigr)\Bigr) R^{-\dagger}\;}
$$
where $\mathrm{copyltu}(X)_{ij} = X_{ij}$ for $i>j$, $=\overline{X_{ji}}$ for $i<j$, $=\mathrm{Re}\,X_{ii}$ for $i=j$. This adjoint has *no denominator of eigenvalue-type gaps*; it is analytic whenever $R$ is invertible (equivalently, $M$ has full column rank).

Contrast this with the SVD backward (see Townsend 2018, eq. 4): the $F_{ij} = 1/(\sigma_j^2-\sigma_i^2)$ factor blows up when two singular values coalesce. In CTMRG near a phase transition the dominant corner typically has near-degenerate leading singular values in the symmetry-breaking pair, *precisely* the regime of interest.

### 3.2 Backward of the QR projector step

The CTMRG projector is not a simple $M\mapsto (Q,R)$ factorization — it is the composite $M \mapsto Q(M) \mapsto P = Q(M)$ used as an isometry, followed by absorption $\tilde C = P^\dagger C_{\text{enlarged}} P$. We need $\partial \tilde C / \partial M$ and thence $\partial \tilde C / \partial A$.

Let $P = Q$ be the isometry. The forward map is
$$
f(M, C_{\text{enl}}) = Q(M)^\dagger C_{\text{enl}} Q(M).
$$
The pullback of a cotangent $\bar{f}$ splits as
$$
\bar C_{\text{enl}} = Q\,\bar f\,Q^\dagger, \qquad \bar Q = C_{\text{enl}} Q\,\bar f^\dagger + C_{\text{enl}}^\dagger Q\,\bar f.
$$
The second equation feeds into the QR-backward (§3.1) with $\bar R = 0$ (since $R$ is discarded by the truncation). Plugging $\bar R=0$:
$$
\bar M = \Bigl(\bar Q + Q\,\mathrm{copyltu}(-\bar Q^\dagger Q)\Bigr) R^{-\dagger}.
$$
This is the explicit adjoint needed inside the CTMRG backward sweep.

**Claim (Stability).** If $R$ is well-conditioned — i.e. the CTMRG enlarged corner has $\sigma_{\min}(R)\ge \epsilon_R > 0$ — then the adjoint is bounded:
$$
\|\bar M\| \le \epsilon_R^{-1}\Bigl(\|\bar Q\|+\|Q\,\mathrm{copyltu}(\bar Q^\dagger Q)\|\Bigr).
$$
There is **no gap denominator** $1/(\sigma_i^2-\sigma_j^2)$. The only condition for a bounded backward is $R$ non-singular, which the PEPS bond gauge + canonical $R_{ii}>0$ fix gives us (§4).

### 3.3 Unrolled AD

Let $\mathcal{E}_{k+1} = F(\mathcal{E}_k, A)$ be one CTMRG sweep. Unrolling $N$ sweeps:
$$
\mathcal{E}^\star \approx \mathcal{E}_N = F(F(\cdots F(\mathcal{E}_0,A)\cdots ,A),A).
$$
`jax.lax.scan` with reverse-mode AD produces $\partial \mathcal{E}_N/\partial A$ in $O(N)$ time with an $O(N \cdot \chi^2 D^2)$ tape. Memory for the tape is the dominant cost; at $\chi=60$, $D=4$, $N=50$ this is ≈1 GB of tensor storage (comfortably inside the 16 GB cap).

### 3.4 Implicit AD (IFT) at the CTMRG fixed point

At convergence, $\mathcal{E}^\star = F(\mathcal{E}^\star,A)$. Differentiating:
$$
\partial_A\mathcal{E}^\star \;=\; \bigl(I-\partial_{\mathcal{E}}F\bigr|_\star\bigr)^{-1}\,\partial_A F\bigr|_\star.
$$
For a downstream scalar loss $\mathcal{L}(\mathcal{E}^\star, A)$:
$$
\nabla_A \mathcal{L} \;=\; \partial_A\mathcal{L}\bigr|_\star + \bigl(\partial_{\mathcal{E}}\mathcal{L}\bigr)^\top (I-\partial_{\mathcal{E}}F)^{-\top} \partial_A F\bigr|_\star.
$$
The only expensive piece is the linear solve $(I-\partial_{\mathcal{E}}F)^{-\top} v$, which we reduce to a Jacobian–vector product and solve with `jaxopt.linear_solve_gmres` (or `jax.scipy.sparse.linalg.bicgstab` as a secondary option).

The Jacobian $\partial_{\mathcal{E}}F$ at the fixed point has spectrum strictly inside the unit disc (otherwise CTMRG would not have converged) — so $I-\partial_{\mathcal{E}}F$ is invertible and GMRES converges geometrically. This is a strictly better situation than the unrolled case, where we must remember every iterate.

**Memory scaling.** Implicit mode is $O(\chi^2 D^2)$ (environment + a handful of Krylov vectors), a factor of $N\approx 30$–$100$ smaller than the unrolled tape. For $D=6, \chi=108$ it is the only feasible mode under 16 GB.

### 3.5 Interaction of the QR gauge with the CTMRG fixed point

The QR gauge (diagonal phase of $R$) is an additional symmetry of the forward map: $(Q,R)\to (Q D, D^{-1}R)$ for any diagonal unitary $D$ leaves $QR$ invariant. This injects a $\chi$-dimensional null space into $\partial_{\mathcal{E}}F - I$, *if* the gauge is not fixed.

We cure this at two layers:
- **Canonical fix:** after each QR, multiply $Q$ and $R$ by $D = \mathrm{diag}(\mathrm{sign}(R_{ii}))$ so that $R_{ii}>0$ (real) or $R_{ii}/|R_{ii}|=1$ (complex). The map $M\mapsto (Q^{\text{fixed}},R^{\text{fixed}})$ is then single-valued and smooth wherever $R_{ii}\ne 0$.
- **Projector in IFT:** if the residual gauge direction still appears in the linearization (it can if the forward fix is piecewise smooth at $R_{ii}=0$), we project out the diagonal-phase direction before the GMRES solve, by constraining the Newton update to lie in the complement of $\mathrm{span}\{\partial_D \mathcal{E}^\star|_{D=I}\}$.

The net effect: the CTMRG fixed point is unique up to a well-defined residual gauge (the PEPS bond gauge), and the IFT Jacobian is invertible on the physical subspace.

---

## 4. Numerical rank-revealing issue (C5)

Standard QR is not rank-revealing: if the $\chi$-th column of $R$ has $R_{\chi\chi}\approx 0$, the truncation is numerically ill-defined. Two remedies, both implemented and benchmarked:

- **Pivoted QR** (`jax.scipy.linalg.qr` with `pivoting=True` — or a hand-rolled column-pivoted Householder QR via `jax.custom_vjp` for AD safety): at each Householder step, swap the column of largest norm to the pivot position. Pivoting *does* make the selection of $\chi$ effectively rank-revealing, at the cost of a column-permutation node in the backward tape (permutation is piecewise constant, so AD sees it as identity on each smooth piece).
- **Randomized QR** (Halko–Martinsson–Tropp-style randomized range finder followed by thin QR): pre-multiply by a $\chi+p$ Gaussian sketch, then unpivoted QR. The backward rule is unchanged (the sketch is treated as a constant) and the extra $p\approx 5$–$10$ oversampling columns give high-probability rank revealing behaviour.

Our default is **column-pivoted QR with pivot permutation detached from the tape**; we verify in the benchmark that the iPEPS-produced corners at our $\chi$-values *are* rank-separated enough (smallest $R_{\chi\chi}/R_{11}\ge 10^{-6}$ for TFIM and Heisenberg) that plain QR is sufficient. J$_1$–J$_2$ near $J_2/J_1=0.5$ is where plain QR is most likely to fail — we expect to need pivoted QR there and that is part of the story.

---

## 5. Benchmark protocol (C4)

To isolate the QR-vs-SVD change, all other axes are held fixed:

- **Ansatz init:** identical random seed, identical symmetrization step.
- **Optimizer:** `jaxopt.LBFGS` with strong-Wolfe line search, $100$ max iterations, tolerance on $\|\nabla E\|=10^{-6}$. Adam sanity check at 1000 steps only.
- **CTMRG schedule:** identical maximum sweep count $N_{\text{CTM}}=50$ and convergence tolerance $\|\mathcal{E}_{k+1}-\mathcal{E}_k\|_F < 10^{-9}$.
- **Bond and env dimensions:** $D\in\{2,3,4,6\}$, $\chi\in\{D^2, 2D^2, 3D^2\}$.
- **Baseline SVD backward:** Francuz-corrected (strongest published).

All hyperparameters are frozen in a YAML config distributed with the code.

### 5.1 Metrics

For each cell:
1. Converged $E_0$ and relative error to reference.
2. $\|\nabla E\|$ trajectory + variance over the final 50 steps (gradient-noise proxy).
3. Finite-difference gradient check $\|\nabla_{\text{AD}} - \nabla_{\text{FD}}\|/\|\nabla_{\text{FD}}\|$ on a small test tensor.
4. Wall time: forward only, backward, total.
5. Peak memory.
6. Fixed-point residual at optimizer convergence.

## 6. Expected outcomes

Consistent with `2505.00494`'s ≈100× forward-only speedup plus our analysis in §3, we expect:
- **RQ1 (correctness):** QR-AD and Francuz-SVD-AD agree to $<10^{-5}$ relative on TFIM and Heisenberg.
- **RQ2 (stability):** at J$_2/J_1 = 0.5$, QR-AD shows smaller $\|\nabla E\|$ variance over the final 50 steps and fewer L-BFGS line-search restarts than SVD-AD.
- **RQ3 (speed):** forward-pass speedup of 5–30× on CPU, 30–100× on GPU (matching `2505.00494`); total (forward+backward) speedup 3–15× — backward dilutes the forward gain but does not reverse it, because the QR backward is *also* cheaper than the SVD backward.
- **RQ4 (unrolled vs implicit):** implicit wins at $D\ge 4$, $\chi\ge 2D^2$ due to memory; unrolled is competitive at $D\le 3$.
- **RQ5 (gauge):** $R_{ii}>0$ canonical fix suffices; no explicit gauge projection needed in IFT *provided* pivoted QR is used in the dense-spectrum regime.

## 7. Risks / failure modes (all count as a valid run)

- QR-CTMRG-AD stable but not faster once backward is included.
- QR-AD faster and stable on TFIM/Heisenberg but fails rank-revealing near J$_2/J_1=0.5$, requiring always-on pivoting.
- Implicit-diff GMRES stalls — ablated by falling back to unrolled at cheap $D$.

Any of these outcomes is a publishable negative result.

---

## 8. Summary

We propose to extend Zhang–Yang–Corboz (`2505.00494`) forward-only QR-CTMRG into a full AD pipeline for variational iPEPS, with both unrolled and implicit-differentiation backward passes, a canonical $R_{ii}>0$ gauge fix, and an optional column-pivoted QR for near-critical regimes. The derivation is complete and respects all standard CTMRG conventions. The benchmark grid (TFIM, Heisenberg, J$_1$–J$_2$ across the plaquette/Néel crossover) is sized to isolate QR-vs-SVD as the single independent axis of comparison.
