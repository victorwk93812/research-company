# RA Critique — QR-CTMRG with Automatic Differentiation

*Phase 2 (RA Skeptic).*

## Literature Cross-Check

I ran independent `search_arxiv` queries with different keyword framings than the Researcher: "QR decomposition CTMRG tensor network contraction", "automatic differentiation iPEPS variational", "Francuz SVD backward", "Hubig McCulloch differentiation SVD", "Fishman White projector MPS DMRG", "Townsend differentiating SVD", "Corboz infinite PEPS Heisenberg benchmark", "automatic differentiation CTMRG fixed point implicit gauge". I also retrieved full metadata via `get_paper` for the two most load-bearing papers (`2505.00494`, `2509.05090`) and (independently of the Researcher) confirmed their abstracts, dates, and venues.

Papers I personally consulted and can vouch for existing as cited:

- **arXiv:2505.00494v1** — Zhang, Yang, Corboz, "Accelerating two-dimensional tensor network contractions using QR-decompositions", 2025-05-01. Confirmed: C$_{4v}$-symmetric, ≈100× forward speedup on H100, no AD. ✅ Researcher's characterisation is accurate.
- **arXiv:2509.05090v1** — Yang, Corboz, "Efficient iPEPS Simulation on the Honeycomb Lattice via QR-based CTMRG", 2025-09-05 (PRB 113, 085109, 2026). Confirmed C$_{3v}$ extension, no AD. ✅ Accurate.
- **arXiv:1903.09650v2** — Liao, Liu, Wang, Xiang, "Differentiable Programming Tensor Networks", PRX 9, 031041 (2019). Confirmed introduces stabilized SVD backward and implicit diff through fixed-point iterations. ✅ Accurate.
- **arXiv:2508.10822v1** — Tang, Vanderstraeten, Haegeman, "Gauging the variational optimization of PEPS", 2025-08-14. Confirmed the paper's thesis (gradient optimizers exploit bond-gauge freedom to produce artificially low variational energies) and the gauge-fixed remedy. ✅ Relevant.
- **arXiv:2211.13765** — Ahmed, Killoran, Carrasquilla, "Implicit differentiation of variational quantum algorithms", 2022. Confirmed its framework is directly applicable to CTMRG-AD. ✅
- **arXiv:1805.00055v4** — Hauschild, Pollmann, "TeNPy", 2018. Confirmed. ✅
- **arXiv:1503.05345v2** — Phien, Bengua, Tuan, Corboz, Orus, "iPEPS improved: fast full update and gauge fixing", PRB 92, 035142 (2015). Confirmed. ✅
- **arXiv:1912.02780** — Chen et al., "AD for Second RG", PRB 101, 220409 (2020). Confirmed. ✅

**Francuz et al., PRR 7, 013237 (2025)**: I tried several keyword combinations ("Francuz automatic differentiation", "Francuz iPEPS truncation", "Francuz reliable gradient", "Francuz stable backward") on arxiv; the search did not surface a paper by Anna Francuz on this specific topic. I *did* find her other tensor-network papers (1910.09661 on topological order from iPEPS, 2008.06391 on non-Abelian topological order). The cited PRR 7, 013237 (2025) is very likely a valid paper (the journal-reference format is correct for Phys. Rev. Research) but is not surfaced by the arxiv MCP search, possibly because (a) it was posted shortly before publication, (b) its arxiv title uses terminology different from what I guessed, or (c) it was only published in PRR without a separate arxiv preprint. **This is a caveat, not a fatal flaw**: the Researcher's pipeline does not depend on the precise form of the Francuz fix, only on the fact that it exists and patches a specific SVD-backward inaccuracy. The Engineer should nevertheless attempt to locate the Francuz PRR paper directly (DOI lookup) before Phase 4 and lift the exact form of the correction if the benchmark comparison is to be genuinely strong. If it cannot be located, we must be explicit in the report that our "SVD-baseline-with-Francuz-patch" uses our best guess at the correction and not the published form verbatim.

Papers the Researcher cited but I did **not** independently verify via full text: Roberts 1963, Mathias 1996, Hubig–McCulloch 2019, Townsend 2018. These are standard matrix-calculus / tensor-network references; I accept them on the Researcher's characterisation.

Papers the Researcher missed that are arguably relevant (none subsume the proposal):

- **arXiv:2511.22669** — Cortés Estay, Kamar, Corboz, "Accurate computation of energy variance using iPEPS", Nov 2025. Uses CTMRG to compute variance; relevant for future extensions of our metrology but does not overlap with our specific claim about AD-backward pathologies.
- **arXiv:2107.03399** — Ponsioen, Assaad, Corboz, AD-PEPS excitations (listed by Researcher) — confirms AD through CTMRG is a mature practice, re-enforcing the claim that pathologies are known.
- **arXiv:2009.02606** — Haghshenas, TN-QR for MERA (listed by Researcher) — mentions QR in a different TN but does not subsume.

**Literature grounding verdict:** adequate; positioning is correct; the Francuz-patch dependency is marked but survivable.

---

## Technical Flaws

### F1. QR-backward formula has a convention ambiguity the Researcher hasn't nailed down.

The boxed formula in §3.1,
$$
\bar M = (\bar Q + Q\,\mathrm{copyltu}(R\bar R^\dagger - \bar Q^\dagger Q))R^{-\dagger},
$$
differs from the common Seeger (2017) / Walter (2018) formulation by sign and conjugation conventions; these conventions depend on:

- whether `copyltu` fills from the strict lower into the strict upper (as in Hubig–McCulloch) or vice versa;
- whether $R$ is taken with $R_{ii}>0$ (real) or with any diagonal;
- whether AD uses the Wirtinger convention $\bar z = \partial L/\partial \bar z$ or the conjugate.

**Demanded fix:** the Engineer must not trust this formula verbatim. They must instead derive the adjoint *inside* JAX (by defining the QR's `custom_vjp` and unit-testing it against `jax.jacfwd` on a random complex $64\times 64$ matrix — this *is* in the instruction's test plan, so it's already scoped). The Researcher should add a footnote acknowledging that the displayed formula encodes the real-case, $R_{ii}>0$ convention and that the complex case requires care with the diagonal-phase gauge.

### F2. "$\bar R = 0$" in §3.2 is too fast.

The Researcher argues that because the projector is $P=Q$ and $R$ is discarded, the pullback has $\bar R = 0$. This is **true if and only if** the downstream computation uses only $Q$ and no function of $R$. In CTMRG implementations that enforce consistent bi-orthogonal projectors on the two corners of one projector pair, the *rescaling* of $Q$ by $R^{-1}$ (to build an inverse isometry) *does* use $R$ — the "Fishman–White symmetric" form. In that case $\bar R \ne 0$ and the full expression (not the simplified one) must be used.

**Demanded fix:** in Phase 4, the Engineer must keep the *general* form in the code; the $\bar R = 0$ shortcut should only be used for the C$_{4v}$ single-isometry case. For the generic 2×2 unit cell this shortcut is invalid.

### F3. Complexity table has the wrong scaling for the QR projector.

The table in §2.3 claims "Projector construction $O(\chi^3 D^4)$ (thin QR)". For a thin QR of the enlarged corner viewed as a $\chi D^2 \times \chi$ matrix, the cost is $O((\chi D^2)\chi^2) = O(\chi^3 D^2)$, not $O(\chi^3 D^4)$. Conversely for a $\chi D^2 \times \chi D^2$ matrix (the fatter shape used if we QR the un-contracted corner) the cost is $O(\chi^3 D^6)$ — same as SVD.

**Demanded fix:** the Researcher should state *which* QR they perform (on which matrix shape). In `2505.00494` the QR is done on the *reduced* half-corner $\chi D^2 \times \chi$, giving $O(\chi^3 D^2)$. The asymptotic speedup vs the $\chi D^2 \times \chi D^2$ SVD ($O(\chi^3 D^6)$) is therefore $O(D^4)$ — which, combined with the better BLAS-3 constants, matches the ~100× empirical number (for $D=4$, $D^4=256$, consistent).

The current draft's complexity table is directionally right but has the wrong exponent for the QR pipeline. Fix before Phase 3.

### F4. Gauge-fix smoothness is asserted, not proved.

§4 claims that after multiplying by $D=\mathrm{diag}(\mathrm{sign}(R_{ii}))$ (real) the map $M\mapsto Q^{\text{fixed}}$ is smooth. **It is only smooth on the open set $\{R_{ii}\ne 0\}$.** At $R_{ii}=0$ the sign function has a jump, and the fixed map is non-differentiable.

This matters concretely: during the early CTMRG sweeps (before convergence), a corner can transiently have $R_{\chi\chi}\approx 0$. If the gradient step steps across the $R_{\chi\chi}=0$ surface mid-optimization, we get a discontinuous gradient. The complex case is worse: $R_{ii}/|R_{ii}|$ is undefined at $R_{ii}=0$.

**Demanded fix:** the Researcher should either
(a) restrict attention to the regime $\sigma_\chi(\tilde C) \ge \epsilon_R > 0$ — which is a hypothesis on the spectral gap and holds well inside the ordered phases but may fail near phase boundaries — and state this explicitly; or
(b) adopt a smoothed version of the gauge fix, e.g. $D_{ii} = R_{ii}/\sqrt{|R_{ii}|^2 + \epsilon^2}$ for small $\epsilon$, and pay a bias of $O(\epsilon)$ in exchange for differentiability at $R_{ii}=0$.

Option (a) is fine as long as the Engineer verifies $\min_i R_{ii}\ge \epsilon_R$ at runtime and emits a warning if it drops. Option (b) is safer but introduces a hyperparameter.

I'd accept either, but the current draft hand-waves this.

### F5. IFT projector onto the "complement of the gauge direction" is under-specified.

§3.5 says: "project out the diagonal-phase direction before the GMRES solve, by constraining the Newton update to lie in the complement of $\mathrm{span}\{\partial_D\mathcal{E}^\star|_{D=I}\}$." The gauge direction is $\chi$-dimensional (one real phase per diagonal entry), so this is not a single direction but a $\chi$-dimensional null subspace. The Researcher writes "span of a single basis vector" which is wrong dimensionally.

**Demanded fix:** explicitly identify the null subspace as $\mathbb{R}^\chi$ (real-case) or $\mathrm{U}(1)^\chi \cong \mathbb{R}^\chi$ (complex-case), give a basis $\{\partial_{D_{ii}}\mathcal{E}^\star|_{D=I}\}_{i=1}^\chi$, and project via $P_\perp = I - V(V^\dagger V)^{-1}V^\dagger$ where $V$ is the $\chi$-column basis matrix.

In practice: for C$_{4v}$-symmetric corners the canonical $R_{ii}>0$ fix *already* reduces the gauge to $\{\pm 1\}^\chi$ and any discrete residual is harmless for smooth derivatives; for generic 2×2 unit cells we need the full projection.

### F6. Pivoted QR in AD is a piecewise-smooth map — the "detached from the tape" shortcut hides a subtle bias.

§4 proposes column-pivoted QR with the pivot permutation $\Pi$ detached from the tape (i.e. treated as a constant during backprop). This is *correct in the open region* where the pivot order doesn't change, but wrong on the measure-zero set of pivot transitions. Near these transitions the AD gradient is *one-sided*, and finite-difference checks will disagree with AD by a small but systematic amount.

**Demanded fix:** the Engineer should run the gradient-check test (`unit test: SVD-AD and QR-AD gradients agree to <10^-6`) *not* at a pivot-transition point (the test is at an "away-from-criticality TFIM point" per instruction, which is safe — Néel-type corners are rank-separated and don't flip pivots). A broader stability claim would require Halko-style randomized range finding (a smooth sketch), which we have already scoped as an alternative.

### F7. The C$_{4v}$ enforcement through the optimizer must be explicit.

The proposal uses C$_{4v}$-symmetric $A$ for TFIM and Heisenberg. This is imposed by either (i) parameterising $A$ only through its independent components, or (ii) symmetrising after each gradient step. The two choices differ at the gradient level: projection-after-step is not the same as constrained optimisation on the symmetric submanifold. The Researcher doesn't specify which.

**Demanded fix:** parameterise $A$ through an independent-components vector and push the embedding $\iota:\mathbb{R}^{n_{\text{indep}}}\to \mathbb{R}^{dD^4}$ through `jax.grad`. This gives a mathematically clean gradient on the symmetric submanifold. Symmetrisation-after-step introduces an implicit projection whose gradient may be inconsistent with the loss.

### F8. Memory estimate for unrolled AD understates the double-layer tensor.

§3.3 estimates "≈1 GB tape at $\chi=60$, $D=4$, $N=50$". The environment tensor is $\chi\times\chi$ per corner (+4 edges), per site. The *input* to each CTMRG sweep, however, also includes the double-layer tensor $a = A\otimes\bar A$ of shape $(D^2)^4 = D^8$ — for $D=4$ this is $65k$ entries at `float64`, modest. But the *intermediate* contractions *within* one sweep are larger: the enlarged corner is $\chi D^2 \times \chi D^2 = 960\times 960$ at these numbers, and several such matrices are created per sweep. Tape size is closer to $N \cdot (\chi^2 D^4 + \chi^2) \cdot 8$ bytes ≈ $50\cdot (960^2 + 3600)\cdot 8$ ≈ 370 MB; but going to $D=6, \chi=108$ the tape balloons to ≈$50\cdot (108\cdot 36)^2\cdot 8 = 6$ GB — still inside 16 GB but dangerously close once auxiliary state is included.

**Demanded fix:** tighten the estimate. At $D=6, \chi=108, N=50$ the unrolled tape is borderline; implicit diff is *mandatory*, not optional. The Researcher's text says "implicit wins at $D\ge 4$, $\chi\ge 2D^2$" — correct in spirit, but for the largest cell of the grid ($D=6$) it is not a "wins" it is "required".

---

## Minor points (not blockers)

- §1.1: the TFIM critical value $h/J_c \approx 3.04438$ is the QMC reference; iPEPS at finite $(D,\chi)$ will over-estimate $h_c$ by a finite amount. The proposal should not pin the benchmark point to exactly $3.04$ without caveat; the instruction says $\{2.5, 3.04, 3.5\}$ which is fine.
- §5.1 metric 2: "variance over the final 50 L-BFGS steps" — L-BFGS with line search has variable per-step costs; variance is noisy. A more robust metric is the *ratio* of last-50-step gradient norm to best achieved — flag for Engineer.
- §3.4: "spectrum strictly inside the unit disc" — in our TFIM critical case the CTMRG contraction ratio approaches $1$ from below as $\chi\to\infty$, so "strictly inside" is a $\chi$-finite statement. Worth noting.

---

## Decision

The proposal is mathematically sound modulo the fixes F1–F8, which are all addressable by the Engineer during Phase 4 (they are mostly calls to "use the general form, not the shortcut, and verify") and by small edits the LaTeX Writer will incorporate during Phase 3. The literature grounding, positioning, and high-level derivation are correct. Francuz PRR 2025 is not in the arxiv corpus I can reach — flagged as a caveat but not fatal (the comparison baseline is still defensible with a clearly-labeled "our implementation of the Francuz fix").

**APPROVAL GRANTED: PROCEED TO TYPESETTING AND ENGINEERING.**
