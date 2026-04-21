# RA Skeptic Critique of `theory_draft.md`

## Literature Cross-Check

I independently queried arxiv MCP with the following framings: "SU(2) symmetric tensor network DMRG", "non-abelian symmetry tensor network Heisenberg", "McCulloch non-abelian SU(2) DMRG", "Wigner-Eckart MPS block sparse", "SU(2) DMRG memory scaling multiplet efficiency benchmark". I personally read the following, either via `get_paper` or `download_paper_text`:

- **cond-mat/0012319v3** (McCulloch & Gulácsi, 2001) — methodology text pulled. Verified: the Researcher's characterisation of Eqs. (4)–(8) of that paper (Wigner-Eckart reduction, 9j couplings for tensor-products of block operators) is accurate. Table I of that paper, comparing U(1)×U(1) vs U(1)×SU(2) vs SO(4) at fixed m, is the empirical precedent the Researcher alludes to.
- **1202.5664v2** (Weichselbaum, 2012) — methodology text pulled (Sec. II "MPS implementation of non-abelian symmetries"). Verified: the QSpace data structure indeed factorises a symmetric tensor as `(reduced multiplet tensor) ⊗ (CGC tensor)`, which is precisely the split used in §4 of the draft.
- **1203.2222v2** (Singh, 2012) — abstract verified; thesis explicitly covers U(1) and SU(2) symmetric MPS. Independent baseline confirmed.
- **1910.13736v2** (Weichselbaum, 2019) — abstract verified; X-symbols are tabulated CGT contraction coefficients, allowing CGCs to be sidestepped in contractions. Researcher's characterisation is accurate; correctly marked out-of-scope for the prototype.
- **1501.05504v2** (Hubig et al., 2015) — abstract verified; single-site DMRG with subspace expansion is compatible with non-abelian symmetries, as stated.

No paper was found that already **benchmarks** a pure-Python, minimal memory-scaling comparison between U(1) block-sparse and SU(2) block-sparse MPS tensors in the Heisenberg-chain context; the scope of the prototype is therefore defensible as a pedagogical replication, not a novelty claim. The Researcher is appropriately explicit about this in the `## Positioning` section ("there is no new physics").

No missed load-bearing references. No misrepresentations of prior work.

## Technical Flaws

### 1. (Minor — clarification needed) 6j vs 9j in the two-site contraction

§6 step 1 of the draft writes a 6j-like factor $F(S_L, \tfrac12, S_M, \tfrac12, S_R)$ for the contraction of two neighbouring MPS tensors to form $\Theta$. Each MPS tensor here carries one physical leg ($k=\tfrac12$) and two virtual legs, so the tensor is rank-3. The contraction of two rank-3 tensors sharing one bond and *fusing* the two free physical legs into a two-site combined leg is in general governed by a **9j symbol** (as in McCulloch–Gulácsi Eq. 8). A 6j only appears when one of the coupled legs is trivial or the two physical legs are kept unfused. The draft's claim that "6j symbol collapsing the two internal Clebsch–Gordans" is not quite right — it should be 9j (or, equivalently, a product of two CGCs summed over the intermediate $M$-index that reduces to a 9j by Racah's identity). This does not change the memory accounting in §5, but it needs to be corrected in §6 step 1 so the typesetting phase does not propagate the error.

**Required fix:** say "9j symbol" (or "equivalent product of CGCs"), not "6j symbol."

### 2. (Minor — tighten the inequality) U(1) → SU(2) ratio

The expression $\mathcal R \approx \langle (2S+1) \rangle^{2} / (\text{selection rule factor})$ in §5 is correct as a scaling estimate but under-specified. The precise statement is: writing $D_S \equiv 2S+1$,

$$
\mathcal M_{\mathrm{U(1)}} = \sum_{s=\pm\tfrac12}\sum_{M_L} \Bigl(\sum_{S_L \ge |M_L|} d_{S_L}\Bigr)\Bigl(\sum_{S_R \ge |M_L + s|} d_{S_R}\Bigr)
$$

and

$$
\mathcal M_{\mathrm{SU(2)}} = \sum_{S_L} \sum_{S_R \in \{S_L \pm \tfrac12\}} d_{S_L} d_{S_R}.
$$

The ratio is *not* in general simply $\langle D_S \rangle^2$ because the U(1) sum runs over all $(M_L,s)$ pairs — approximately $2 \sum_S D_S$ of them — while the SU(2) sum runs over ~$2 N_S$ pairs where $N_S$ is the number of occupied multiplets. So the ratio is closer to

$$
\mathcal R \;\sim\; \frac{\sum_S D_S}{N_S} \cdot \langle D_S \rangle \;\sim\; \langle D_S\rangle^2,
$$

which *is* what the draft says — but the draft should show one extra line of algebra making that explicit so the Phase 4 prototype can sanity-check against the formula directly rather than just the asymptotic. **Soft suggestion, not a blocker.**

### 3. (Accounting question — answered but flag for the engineer) CGC storage

§5 treats the Clebsch–Gordan factor as a "global" object and excludes it from the SU(2) memory count. This is defensible given Weichselbaum's QSpace (CGCs are cached once per group, per triple $(S_L,\tfrac12,S_R)$) and is standard practice. The Phase 4 report **must** be explicit about this accounting convention so the reader is not misled: the number reported is "storage per MPS tensor, excluding shared CGC tables." If the prototype is tempted to *actually store* dense CGCs per tensor, the apparent gain will vanish — this is a pitfall the engineer should be warned about.

**Required fix:** Add one sentence in §5 or §7 stating the accounting convention explicitly ("we do not count the CGC tensor, treated as a shared table per the QSpace convention of arXiv:1202.5664").

### 4. (Check — passed) Selection rule for spin-1/2 fusion

$S_R \in \{|S_L - \tfrac12|, S_L + \tfrac12\}$ is correct (two fusion channels, except $S_L=0$ where only $S_R = \tfrac12$ survives). No sign errors, no dropped channels.

### 5. (Check — passed) Heisenberg interaction as a scalar operator

The statement in §6 step 2 that $\hat{\mathbf S}\cdot\hat{\mathbf S}$ is a scalar ($k=0$) under SU(2) and therefore block-diagonal in $S$ is correct. The per-sector matrix-product form for $\hat H_{\rm eff}$ follows from the Wigner-Eckart theorem applied to a rank-0 tensor.

### 6. (Check — passed) No conservation-law violations, no non-commuting-treated-as-commuting errors, no index mismatches in the MPS tensor shapes.

### 7. (Feasibility — passed) Prototype scope

The Phase 4 deliverable (memory-only benchmark with synthetic symmetric tensors, no full DMRG eigensolver) is realistic for the overnight timeline. Refusing to build a full SU(2)-DMRG engine is the right call. The memory ratio is decoupled from the eigensolver and can be measured from the tensor representation alone.

## Verdict

The two **required** fixes (label 6j → 9j in §6 step 1; add one sentence on CGC accounting convention) are trivial and local. They do not invalidate the derivation or the prototype plan. I will make them inline by lightly editing `theory_draft.md` rather than forcing a full rewrite — the rest of the document is sound.

After those edits:

**APPROVAL GRANTED: PROCEED TO TYPESETTING AND ENGINEERING.**
