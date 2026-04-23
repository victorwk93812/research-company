# Theory Draft: AD-iPEPS + CTMRG Benchmark on the 2D Hubbard Model
*Researcher persona (Phase 1). Target file per CLAUDE.md.*

---

## Literature Review

Prior work consulted via the `arxiv` MCP server. Abstracts read for all; methodology text pulled for the starred entries.

1. **Liao, Liu, Wang, Xiang — *Differentiable Programming Tensor Networks*** (arXiv:1903.09650, PRX 9, 031041 (2019)). \*
   One-line contribution: formalises reverse-mode AD through the full iPEPS + CTMRG pipeline, introducing stable backward rules for SVD and for fixed-point iterations.
   Relation: **base method** we are extending. They operate in PyTorch with dense tensors and no explicit physical symmetry; we target the JAX stack and block-sparse symmetric tensors.

2. **Hasik, Van Damme, Poilblanc, Corboz — variational iPEPS via AD (multiple papers: arXiv:2105.08022, arXiv:2111.05324)**; and **Hasik & Corboz — *Incommensurate order ... anisotropic triangular lattice*** (arXiv:2311.05534, PRL 133, 176502 (2024)). \*
   Contribution: demonstrate that AD-iPEPS with careful SVD-backward broadening and `L-BFGS` reaches the accuracy previously reserved for analytical gradients, and solves systems with frustration / incommensurate order.
   Relation: **anchor for gradient quality**. We replicate their protocol (Lorentzian-broadened SVD backward, L-BFGS with line search) as a baseline optimizer cell.

3. **Francuz, Schuch & Vanhecke — *Stable and efficient differentiation of tensor-network algorithms*** (arXiv:2311.11894). \*
   Contribution: introduces a truncated-rank backward rule for SVD/eigendecomposition where near-degenerate modes are projected out (rather than broadened), plus an implicit-layer CTMRG gradient.
   Relation: **direct comparison point**; our benchmark will treat this and Hasik–Corboz broadening as two of the three SVD-backward variants.

4. **Ponsioen, Chung, Corboz — *Period-4 stripe in the extended 2D Hubbard model*** (arXiv:1907.01909, PRB 100, 195141 (2019)) and **Ponsioen, Chung, Corboz — *Superconducting stripes in the hole-doped three-band Hubbard model*** (arXiv:2306.12910, PRB 108, 205154 (2023)).
   Contribution: establish that iPEPS with 2×N (stripe) unit cells resolves the d-wave-SC / stripe competition at $\delta\!\sim\!1/8$ in Hubbard-like models; provides the physics target for our C4v ablation.
   Relation: **physics benchmark** for axis-C4v and the 2×4 unit cell.

5. **Corboz — *Improved energy extrapolation with infinite projected entangled-pair states applied to the 2D Hubbard model*** (arXiv:1605.03006, PRB 93, 045116 (2016)).
   Contribution: introduces the truncation-error extrapolation $E(w) \to E_0$ used to produce the Hubbard-iPEPS reference curve compared in LeBlanc et al. benchmark.
   Relation: the **energy-vs-$w$ extrapolation** is the evaluation protocol we copy in the focused deep-dive.

6. **Corboz, White, Vidal, Troyer — *Stripes in the 2D t–J model from iPEPS*** (arXiv:1402.2859, PRL 113, 046402 (2014)).
   Contribution: shows competition between uniform d-SC and stripe phases in the near-relative t–J model.
   Relation: context for the doping / $t'$ sweep; independent baseline for the stripe side of the phase diagram.

7. **Bruognolo, Li, von Delft, Weichselbaum — *A beginner's guide to non-abelian iPEPS for correlated fermions*** (arXiv:2006.08289, SciPost Phys. Lect. Notes 25 (2021)).
   Contribution: full SU(2)×U(1) block-sparse iPEPS implementation reaching $D{=}24$ for multi-band Hubbard/Hund models; careful discussion of symmetric-tensor bookkeeping (Clebsch–Gordan reduction).
   Relation: **implementation reference** for axis S = SU(2); we adopt the Z(2)×U(1) reduced-multiplet scheme as a tractable surrogate for SU(2).

8. **Yang & Corboz — *Efficient iPEPS on the honeycomb lattice via QR-based CTMRG*** (arXiv:2509.05090, PRB 113, 085109 (2026)).
   Contribution: QR-accelerated CTMRG on C3v lattices; shows order-of-magnitude speedup while retaining precision; argues QR is backward-friendly at gap closures.
   Relation: motivates a **QR-CTMRG optional cell** on the square lattice at C4v-symmetric points; placed out-of-scope for the compute-limited reduced grid but left as a hook in the code.

9. **Phien, Bengua, Tuan, Corboz, Orús — *iPEPS algorithm improved: fast full update and gauge fixing*** (arXiv:1503.05345, PRB 92, 035142 (2015)).
   Contribution: demonstrates that per-step *local gauge fixing* on iPEPS bonds materially improves stability of the (non-AD) full-update scheme.
   Relation: prior support for our axis-D gauge-fixing knob (QR / polar / Riemannian).

10. **Ahmed, Killoran, Carrasquilla — *Implicit differentiation of variational quantum algorithms*** (arXiv:2211.13765).
    Contribution: IFT-based gradients through variationally-defined states; same mathematical object as our `implicit` diff mode.
    Relation: theoretical justification for the implicit-mode cell.

11. **Cortés Estay, Kamar, Corboz — *Accurate energy variance via iPEPS*** (arXiv:2511.22669).
    Contribution: method to compute $\langle (H-\langle H\rangle)^2\rangle$ with CTMRG for systematic $E_0$ extrapolation.
    Relation: adopted as a **secondary accuracy probe**; listed in benchmark metrics as an optional add-on for the canonical point only.

12. **LeBlanc et al. — *Solutions of the Two-Dimensional Hubbard Model: Benchmarks and Results from a Wide Range of Numerical Algorithms*** (PRX 5, 041041, 2015) and **Qin et al.** (PRX 10, 031016, 2020).
    These are the anchoring numbers our benchmark is measured against: thermodynamic-limit $E_0/t \approx -0.524$ at $U/t{=}8,\, t'{=}0,\, \delta{=}0$ and the stripe-vs-uniform picture at $\delta{=}1/8$.

---

## Positioning

The AD-iPEPS community has converged on three orthogonal stabilisation "tricks":

- **broaden** near-degenerate SVD modes in the backward pass (Hasik–Corboz, Liao);
- **truncate** near-degenerate modes out of the backward (Francuz);
- **implicit** (IFT) rather than unrolled differentiation of the CTM fixed point (Liao; Ahmed et al.).

All three have been benchmarked **in isolation**, and almost always on spin models (Heisenberg, J1–J2) without physical symmetry block-sparsity. The *interaction* between these tricks and physical-symmetry enforcement — especially for the **fermionic Hubbard model**, where U(1)c × U(1)s × (fermion parity) blocks significantly and the SVD degeneracies are *structurally inherited from multiplet structure* — is what this work measures.

**Novel contribution (testable inside the grid):** we propose and benchmark a *block-restricted implicit CTMRG* differentiation scheme (B-ICTMRG), in which

1. symmetric tensors are represented as pytrees of dense blocks keyed by U(1)c × U(1)s × Z2(p) quantum numbers;
2. CTMRG is iterated to the fixed point $T^\star(A)$ **per block**, with block-local projectors obtained from a symmetric SVD;
3. gradients w.r.t. $A$ flow via the **IFT linear solve restricted to the same block-pattern**, implemented using `jax.lax.custom_root` / `jaxopt.implicit_diff` with a block-sparse Jacobian-vector product;
4. backward-SVD regularisation is applied **per block** so that the Lorentzian parameter $\varepsilon$ adapts to each sector's degeneracy structure;
5. the outer optimizer is a **Riemannian L-BFGS on the $GL(D)/\mathrm{blockdiag}(GL(D_\alpha))$ quotient**, using QR-retraction to absorb the on-bond gauge redundancy that symmetric-block tensors inherit.

The claim is that B-ICTMRG converts axis S (symmetry) from *slowing down* the backward pipeline (more bookkeeping; more near-degeneracies) into *stabilising* it (smaller per-block condition number; smaller flat-direction manifold after gauge quotient). We aim to demonstrate this quantitatively against the Liao / Francuz / Hasik–Corboz baselines inside the reduced benchmark grid below.

---

## Formal Definitions

### Hilbert space and Hamiltonian

On each site $i\in\mathbb Z^2$ we have the fermionic Fock space $\mathcal H_i=\mathrm{span}\{|0\rangle,|\!\uparrow\rangle,|\!\downarrow\rangle,|\!\uparrow\downarrow\rangle\}$; global Hilbert space $\mathcal H=\bigotimes_i \mathcal H_i$ with the standard $\mathbb Z_2$ fermion-parity-graded tensor product (superspace structure).

Conserved currents (no SOC, no external field):
- fermion parity $P=\prod_i(-1)^{n_i}\in\{\pm 1\}$ — always imposed;
- total charge $\hat N = \sum_i n_i$ — U(1)$_c$;
- total $\hat S^z = \tfrac12\sum_i(n_{i\uparrow}-n_{i\downarrow})$ — U(1)$_s$;
- full $\hat{\mathbf S}^2$ — SU(2)$_s$ (ungauged in our implementation; emulated as U(1)$_s$ plus a time-reversal charge in axis S);
- square-lattice point group C4v — optional.

Hamiltonian (reproduced for completeness):
$$
H = -t\!\!\sum_{\langle ij\rangle,\sigma}\!\! c^\dagger_{i\sigma}c_{j\sigma}
    -t'\!\!\sum_{\langle\langle ij\rangle\rangle,\sigma}\!\! c^\dagger_{i\sigma}c_{j\sigma}
    + U\sum_i n_{i\uparrow}n_{i\downarrow}
    - \mu \sum_i n_i, \qquad t=1.
$$

### Fermionic iPEPS with swap gates

Each site tensor $A^{[x,y]}_{s}{}_{lurd}$ carries a physical index $s\in\{0,\uparrow,\downarrow,\uparrow\downarrow\}$ and four virtual indices $l,u,r,d$ of dimension $D$. Fermionic ordering: we treat $A$ as an element of the graded tensor algebra and insert parity swap gates $S$ whenever two fermionic lines cross during a contraction; on the square lattice this reduces to a single swap per bond-crossing (Corboz–Orús–Bauer–Vidal, arXiv:0912.0646).

For a $2\times2$ unit cell with $C4v$ broken to translation-only, we have four independent $A^{(i)}$; imposing $C4v$ at the tensor level reduces this to one $A$ supplemented by a representation of the point-group action on the virtual indices.

### CTMRG fixed-point equations

Following Orús–Vidal (PRB 80, 094403), the environment of a unit cell is summarised by eight corner-and-edge tensors $\{C_1,\dots,C_4; T_1,\dots,T_4\}$ with truncation bond $\chi$. The CTMRG move in direction $d\in\{L,U,R,D\}$ is a non-linear map $\mathcal M_d$ built from:
- absorb a row/column of the double-layer tensor $a = \sum_s (A\otimes \bar A)_s$;
- form a projector $P$ from the truncated SVD of the enlarged corner;
- contract $P$ into $C$ and $T$.

Write a full CTMRG step as $F(E; A) = \mathcal M_D\circ\mathcal M_R\circ\mathcal M_U\circ\mathcal M_L(E;A)$ acting on $E=(C_*,T_*)$. The fixed point $E^\star(A)$ solves
$$F(E^\star; A) - E^\star = 0.$$
The energy is $E_0(A) = e[A, E^\star(A)]$ where $e$ is the two-site reduced-density-matrix expectation value on the unit cell.

### AD primitives

Two differentiation modes:

- **Unrolled:** treat the sequence $E_{k+1}=F(E_k; A)$ as a computation graph; memory $\mathcal O(n_\text{iter}\,\chi^2 D^4)$.
- **Implicit (IFT):** at convergence, differentiating $F(E^\star;A)=E^\star$ gives
  $$\frac{dE_0}{dA} = \partial_A e + \partial_E e \cdot\bigl(I - \partial_E F\bigr)^{-1}\partial_A F.$$
  The linear solve is done with GMRES on a JVP, not explicitly formed. Memory $\mathcal O(\chi^2 D^4)$.

SVD backward (two variants benchmarked):
- **Lorentzian** (Hasik, Liao): in $\partial L/\partial M = U (F\odot (U^\top \partial L/\partial \Sigma V + \ldots)) V^\top$ the anti-symmetric matrix $F_{ij}=1/(\sigma_i^2-\sigma_j^2)$ is replaced by $F_{ij} = (\sigma_i^2-\sigma_j^2)/((\sigma_i^2-\sigma_j^2)^2 + \varepsilon^2)$;
- **Truncated-rank** (Francuz): a tolerance $\eta$ defines a cut $\sigma_i>\eta$; modes below the cut are removed from the backward rule.

### Symmetric-tensor block decomposition

A U(1)$_c$ × U(1)$_s$ × Z2$_p$ symmetric tensor $A$ with leg labels $(s,l,u,r,d)$ decomposes as
$$A_{(s,l,u,r,d)} = \delta_{q(s)+q(l)+q(u)+q(r)+q(d),\,Q_A}\, B^{(\alpha_s,\alpha_l,\alpha_u,\alpha_r,\alpha_d)}_{n_s n_l n_u n_r n_d},$$
where $\alpha$ labels the sector of each leg, $n$ its multiplicity, and $Q_A$ is the tensor's total charge (0 for the iPEPS site tensor; $\pm1$ per spin/charge for the physical operators). We represent $A$ in JAX as a `FrozenDict[tuple[int,...] -> jnp.ndarray]` pytree keyed by the leg quantum numbers; the dict-of-arrays is a valid pytree and `jax.grad` traverses it.

### C4v point group at the tensor level

Let $R$ be the 90° rotation and $\sigma$ a reflection. A C4v-symmetric iPEPS tensor satisfies $R\cdot A = A$, $\sigma\cdot A = A$, where the group acts on virtual indices by permutation (with possible sign for fermionic swap). We parameterise $A$ as a sum over C4v irreps: $A = \sum_{\Gamma\in\{A_1,A_2,B_1,B_2,E\}} A_\Gamma$, and variationally keep only one chosen irrep or a weighted combination.

---

## Complexity Analysis

Let $n_s$ be the *average* block multiplicity per leg (so $D = n_\text{sectors}\, n_s$ asymptotically). Dense scaling of CTMRG is $\mathcal O(\chi^3 D^4 + \chi^2 D^6)$; symmetric-block scaling replaces $D$ by $n_s$ *per block* with a combinatorial factor counting compatible blocks, yielding effective cost $\mathcal O\!\bigl(n_\text{blocks}\,(\chi/n_\text{sectors})^3\,n_s^4 + \ldots\bigr)$ — i.e.\ a reduction by $n_\text{sectors}^3$ to $n_\text{sectors}^6$ depending on which contraction dominates (environment vs ansatz-side).

| Cell | Forward cost | Memory (unrolled) | Memory (implicit) | Gradient cost |
|------|--------------|-------------------|-------------------|---------------|
| Dense (Z2 only) | $\chi^3 D^4$ | $n_\text{iter}\,\chi^2 D^4$ | $\chi^2 D^4$ | $n_\text{GMRES}\cdot$fwd |
| U(1)c | $/n_c^3$ | $/n_c^3$ | $/n_c^3$ | $/n_c^3$ |
| U(1)c × U(1)s | $/(n_c n_s)^3$ | same | same | same |
| SU(2) (emulated) | $/\sim n_c n_s$ | same | same | same |
| + C4v | 1/8 independent params; fwd $\sim$ unchanged | same | same | reduced gradient-norm |

The B-ICTMRG scheme pays the implicit-mode gradient cost $n_\text{GMRES}\cdot$fwd but *inherits the block sparsity* in each GMRES apply, so it compounds the savings across the whole AD pipeline.

---

## Reduced Benchmark Grid (justified by 16 GB / 4-thread / CPU-only budget)

Full grid is $\sim\!5\cdot 3\cdot 3\cdot 3\cdot 4\cdot 3\cdot 6\cdot 5\cdot 3\cdot 4\cdot 4\cdot 2 \sim 10^7$ cells. Infeasible. We reduce as follows while preserving the ability to answer each Research Question (RQ).

**R1. Global sweep → reduced to one "compass" scan.** Keep $U/t\in\{4,8,12\}$, $t'/t\in\{0,-0.25\}$, $\delta\in\{0,1/8\}$ only; $D{=}2$, $\chi{=}2D^2$, 2×2 unit cell. Symmetries: $\{Z_2,\, U(1)_c,\, U(1)_c{\times}U(1)_s\}$. Optimizer: Adam as the fast cell; L-BFGS only at the canonical point. Cells = $3\cdot2\cdot2\cdot3\cdot1 = 36$. This keeps axis S × physics coverage and is cheap.

**R2. Focused deep-dive.** Canonical $U/t{=}8$, $t'{=}0$, $\delta\in\{0,1/8\}$. $D\in\{2,3\}$ (drop 4, 6 in the reduced study — kept as `--large` flag for opportunistic overnight runs). $\chi\in\{2D^2,3D^2\}$. Symmetries: all six (Z2, U1c, U1c×U1s, U1c+C4v, U1c×U1s+C4v, and the "ablation-SU(2)" emulation). Optimizers: Adam, L-BFGS, Riemannian-L-BFGS. Cells = $2\cdot2\cdot2\cdot6\cdot3 = 144$.

**R3. Ablation (axis O only).** Fix physics and ansatz at canonical $U/t{=}8$, $t'{=}0$, $\delta{=}0$, $D{=}2$, $\chi{=}8$, 2×2, symmetry = U(1)c. Vary one axis at a time:
- diff mode ∈ {unrolled-10, unrolled-30, implicit-IFT} — 3
- gauge fixing ∈ {none, QR, polar, Riemannian} — 4
- SVD-backward ∈ {plain-$\varepsilon$=0, Lorentzian-$\varepsilon$=1e-8, Lorentzian-$\varepsilon$=1e-10, truncated-$\eta$=1e-8} — 4
- step control ∈ {Adam-1e-2, Adam-3e-3, L-BFGS-Wolfe, L-BFGS-fixed} — 4

Total ablation cells = 3+4+4+4 = 15 (union of the one-at-a-time sweeps).

**Grand total:** 36 + 144 + 15 = **195 cells**. At ≤ 500 AD steps each and an estimated 0.5 s/step at $D{=}2,\chi{=}8$ on CPU, wall-time ≈ $195 \cdot 500 \cdot 0.5\text{ s} \approx 14$ h worst case; in practice many cells will hit the convergence tolerance sooner.

**Scope-discipline check:** this reduced grid answers

- RQ1 (symmetries at fixed effective $D$): deep-dive × axis S covers it at canonical physics; R1 sweep extends it across regimes;
- RQ2 (implicit vs unrolled under symmetry): ablation row for diff-mode, repeated over axis S in the deep-dive;
- RQ3 (gradient-noise decomposition): ablation rows for SVD-backward, gauge fixing, step-control;
- RQ4 (C4v under doping): deep-dive at $\delta{=}1/8$ compares U(1)c×U(1)s vs the same plus C4v; the R1 sweep extends to $t'{=}-0.25$.

The B-ICTMRG novel contribution is tested specifically in the deep-dive "U(1)c + implicit + Riemannian-L-BFGS + Lorentzian-per-block" cell — i.e.\ the simultaneous activation of all five listed ingredients — against the Liao dense-AD cell, Hasik–Corboz dense-L-BFGS cell, and Francuz truncated-SVD cell.

---

## Implementation Sketch for the Python Engineer

Recommended module layout (to be confirmed in Phase 4):

- `hamiltonian.py` — `HubbardParams` dataclass; two-site gates; particle-hole sign convention fixed once.
- `symmetric_tensor.py` — `SymTensor` = pytree of `{sector_tuple: jnp.ndarray}`; `svd`, `qr`, `einsum` with sector-fusion; JAX `register_pytree_node`.
- `ctmrg.py` — one CTMRG step as a pure JAX function `F(env, A, chi)`; `jax.lax.while_loop` to fixed point; `@jax.custom_vjp` wrapping the implicit solve.
- `ad_pipeline.py` — energy functional `E(A)`; `unrolled_energy`, `implicit_energy` variants; per-block Lorentzian / truncated SVD custom VJP.
- `optimizers.py` — Adam wrapper from `optax`; L-BFGS from `jaxopt`; Riemannian L-BFGS on a Stiefel-like gauge quotient (custom, ~100 LOC).
- `benchmark.py` — YAML → runs; logs JSON per cell; iteration trace every 10 steps.
- `main.py` — entrypoint; sets `resource` 16 GB cap; sets `jax.config.update("jax_enable_x64", True)`.
- `tests/` — (i) `test_symtensor_ad.py` compares `jax.grad` of a small symmetric contraction against a dense AD reference; (ii) `test_ctmrg_fixed_point.py` asserts $\|F(E^\star)-E^\star\|<10^{-10}$; (iii) `test_implicit_vs_unrolled.py` checks that gradients on a tiny $D{=}2,\chi{=}4$ Heisenberg problem agree to 1e-6 between the two modes.

---

## Next Steps

This draft now passes to the RA Skeptic for review. Expected critique axes: (a) is the reduced grid really answering RQ4 with only $D\in\{2,3\}$? (b) is B-ICTMRG genuinely novel, or a repackaging of Liao+Francuz? (c) memory accounting for $D{=}3,\chi{=}18$ is tight — does it fit?
