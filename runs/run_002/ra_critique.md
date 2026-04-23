# RA Skeptic Critique: theory_draft.md (Phase 2)

## Literature Cross-Check

Independently consulted (every paper listed was fetched via `mcp__arxiv__search_arxiv` or `mcp__arxiv__get_paper`; for the load-bearing ones I also pulled abstracts directly):

| arXiv id | Role in proposal | Verified? |
|---|---|---|
| 1903.09650 (Liao et al., PRX 9, 031041) | baseline AD-iPEPS with SVD-backward + fixed-point backprop | ✓ metadata fetched; Researcher's characterisation matches abstract |
| 2311.11894 (Francuz, Schuch, Vanhecke, PRR 7, 013237) | "truncated-rank SVD backward" | ⚠ **mischaracterised** — see flaw F1 below |
| 2311.05534 (Hasik & Corboz, PRL 133, 176502) | variational iPEPS AD demonstration | ✓ abstract confirms AD-iPEPS on frustrated Heisenberg; Corboz co-authored |
| 1503.05345 (Phien et al., PRB 92, 035142) | iPEPS gauge fixing | ✓ abstract confirms gauge fixing improves stability |
| 1907.01909 (Ponsioen, Chung, Corboz, PRB 100, 195141) | period-4 stripe at $t'$ | ✓ |
| 2306.12910 (Ponsioen, Chung, Corboz, PRB 108, 205154) | stripes in three-band Hubbard | ✓ |
| 2006.08289 (Bruognolo et al., SciPost Lect. Notes 25) | non-abelian fermionic iPEPS | ✓ abstract confirms SU(2)×U(1) reaches D=24 |
| 2509.05090 (Yang & Corboz, PRB 113, 085109 (2026)) | QR-CTMRG | ✓ C3v; speedup via QR instead of SVD |
| 1402.2859 (Corboz, Rice, Troyer, PRL 113, 046402) | uniform d-wave vs stripe in t–J | ✓ |
| 2211.13765 (Ahmed, Killoran, Carrasquilla) | implicit-diff theoretical anchor | ✓ |
| 2511.22669 (Cortés Estay, Kamar, Corboz, 2025) | iPEPS energy variance | ✓ |

Also personally searched for items the Researcher did **not** cite that might subsume the claim:

- arXiv:0912.0646 (Corboz, Orús, Bauer, Vidal, PRB 81, 165104) — fermionic PEPS with swap-gate fermionisation. Not cited by name but used implicitly. **Must** be cited since it defines the swap-gate sign convention the Researcher inherits.
- arXiv:2105.08022 and arXiv:2111.05324 (Hasik–Van Damme–Poilblanc–Corboz, "Investigation of the Néel phase of the frustrated Heisenberg ...") — these are the actual first AD-iPEPS-in-PyTorch papers (`peps-torch`), and are what "Hasik / Corboz baseline" typically refers to. **Neither arxiv id was verifiable via my searches** — the Researcher should double-check the exact arxiv ids before the report goes to typesetting, or replace the reference with arXiv:2311.05534 which I did verify. The citation as-written risks referring to a paper that does not exist in the form claimed.
- The Researcher missed any reference to `peps-torch` / `varipeps` as code artefacts. Since the novelty claim rests partly on moving the ecosystem to JAX, a one-line acknowledgement of the existing PyTorch stack is needed in Positioning.

**No subsuming result found.** The combination "block-restricted implicit CTMRG with per-block Lorentzian SVD backward and Riemannian L-BFGS on the block-gauge quotient" is not present in Liao, Francuz, Hasik–Corboz, or Bruognolo. So the novelty statement is defensible in scope — but see flaw F2.

---

## Technical Flaws

### F1. Mischaracterisation of Francuz et al. (arXiv:2311.11894)

The draft describes Francuz–Schuch–Vanhecke as providing a *"truncated-rank backward rule for SVD where near-degenerate modes are projected out"*. The abstract (which I fetched) tells a different story: the paper argues that the **standard SVD-backward used in all prior AD-tensor-network codes is itself fundamentally wrong** because of an overlooked contribution from the truncated subspace, and provides a **corrected** gradient. The fix addresses **truncation** (the cut between kept and discarded singular values), not the near-degeneracy problem inside the kept sector. The truncated-rank description confuses two different issues.

**Action required:** rewrite the §Literature Review entry for Francuz to accurately reflect "**corrected backward rule accounting for the truncated subspace**", and in §AD primitives replace "truncated-rank (Francuz)" with "Francuz-corrected (kept + truncated blocks)". This changes the set of SVD-backward variants to three: *plain Lorentzian*, *plain truncated*, and *Francuz-corrected*. The ablation axis should stay three cells — don't expand the grid.

### F2. Novelty risk: B-ICTMRG = IFT + block sparsity + per-block regularisation

The Researcher bundles five ingredients into "B-ICTMRG" and claims the *bundle* is new. Pedantically: each ingredient is in the literature.
- IFT backprop through CTMRG fixed point → Liao 2019.
- Block-sparse symmetric tensors with AD → Bruognolo et al. (non-abelian iPEPS in MATLAB/C++); TensorKit.jl (Van Damme) in Julia; QSpace library.
- Per-block SVD regularisation → implicit in any non-abelian code; never benchmarked systematically against dense Lorentzian in the JAX-AD setting.
- Riemannian L-BFGS on a gauge quotient → Gao et al. 2020 (Stiefel/symplectic context), not applied to iPEPS as far as I can find.

Conclusion: the novelty is **"the first integrated JAX implementation and per-axis ablation"** rather than "a new algorithm". That is a legitimate engineering/benchmarking contribution, but the Positioning section over-sells it. Tone down "Novel contribution" → "Integrated benchmark contribution". Do not claim a new algorithm unless the Riemannian-L-BFGS-on-block-gauge-quotient step is formally new; at minimum, phrase this as "to our knowledge first demonstration in AD-iPEPS".

### F3. Memory accounting is optimistic

The draft states ≈ 0.5 s/step at $D{=}2,\chi{=}8$ on CPU. That is probably right for the unrolled dense case, but:
- at $D{=}3,\chi{=}18$, the double-layer bond dimension is $D^2 = 9$ and environments scale as $\chi^2 D^4 = 18^2 \cdot 81 \approx 2.6\cdot10^4$ per tensor; CTMRG iteration memory with 4 environment tensors + intermediate doubles in float64 is $\gtrsim 8$ GB before AD tape overhead.
- Unrolling 30 CTMRG iterations at $D{=}3$ will exceed 16 GB. The draft must **forbid unrolled mode** at $D{=}3$ or larger. Add a runtime guard `if D >= 3 and diff_mode == "unrolled": raise ResourceError`. The Python Engineer should implement this.

### F4. Filling at $\delta=1/8$ on a 2×2 unit cell is physically inconsistent

A 2×2 unit cell has 4 sites. At $\delta = 1/8$, the average filling is 7/8 per site, i.e.\ 3.5 holes per unit cell on a 2×2 cell — **non-integer occupancy** is required. With U(1)c enforcement, only integer-$N$ sectors exist per unit cell, so $\delta=1/8$ on a 2×2 is only compatible with an *effective* doped mean-field realised via $\mu$. But: the Researcher's draft says "U(1)c × C4v at $\delta=1/8$" in the deep-dive. That is fine for a tuning-via-$\mu$ grand-canonical scheme, but **stripe order at $\delta=1/8$ is known to require at least a 2×4 or 4×4 unit cell** (Ponsioen et al.). A 2×2 cell cannot host a period-4 stripe.

**Action required:** either (i) replace the 2×2 cell with 2×4 for the $\delta=1/8$ deep-dive cells, or (ii) explicitly restrict the $\delta=1/8$ cells to "uniform d-SC ansatz only; no stripe competition measured" and note this as a caveat. Option (ii) is cheaper and preserves the RQ4 comparison.

### F5. Missing particle-hole / chemical-potential convention

The Hamiltonian has $-\mu\sum_i n_i$. At finite doping the chemical potential must be tuned. The draft never says whether $\mu$ is a **variational parameter** or **fixed externally with $\delta$ measured**. These give different results and different gradient pipelines. Please specify: I recommend "$\mu$ tuned by bisection outside the AD loop to target $\delta$ within tolerance $10^{-3}$, then frozen for the inner AD". Without this, RQ3 "doping sweep" is under-specified.

### F6. SU(2) "emulation" is vague

The draft claims SU(2) is approximated by "U(1)$_s$ plus a time-reversal charge". Time reversal is **not** the same as SU(2): it gives the full rotation group only at the level of the two-dimensional defining rep, not higher multiplets. The resulting symmetric tensors preserve time reversal × U(1)$_s$ which is a subgroup of SU(2) ⋊ $\mathbb Z_2$ but not SU(2) itself. If you want honest SU(2) you need Clebsch–Gordan intertwiners as in Bruognolo et al.; if you want to avoid them, remove the "SU(2) emulation" label and rename this cell to "U(1)$_s$ + antiunitary" or similar, and drop the claim that it bounds true SU(2) performance.

### F7. Two-site RDM evaluation on fermionic iPEPS requires swap-gate bookkeeping across the environment

The draft says "Energy evaluated via CTMRG two-site reduced density matrix on the appropriate unit cell". For fermionic iPEPS, the two-site RDM requires **swap gates between the physical lines and the environment lines** (not just the ansatz–ansatz crossings). The Engineer will hit this immediately. The draft should call it out so it is not missed.

### F8. `jax.lax.custom_root` + per-block pytree

Minor but real: `jax.lax.custom_root` expects the state to be a flat array. A pytree-of-dict-of-arrays state works with `jaxopt.implicit_diff.custom_fixed_point` but needs a `ravel_pytree` round-trip. The draft claims either library works; in practice `jaxopt` is the cleaner choice. Recommend explicitly: "implement implicit differentiation via `jaxopt.implicit_diff.custom_fixed_point`, not `jax.lax.custom_root`".

---

## Verdict

Flaws F1, F2 (tone), F4, F5, F6 must be addressed in the report text (LaTeX phase can silently incorporate them; no need for a full rewrite of `theory_draft.md` since the core mathematical structure is sound). F3, F7, F8 are actionable for the Python Engineer. The reduced benchmark grid is reasonable and the Research Questions are coverable.

The literature grounding is adequate (11 papers independently verified; no subsuming result found), and the core physics/mathematics of the proposal is consistent. The novelty claim must be softened but not withdrawn.

**APPROVAL GRANTED: PROCEED TO TYPESETTING AND ENGINEERING.**

Downstream personas (LaTeX Writer, Python Engineer) should treat the flaws above as amendments to the theory draft. In particular the LaTeX Writer MUST: (a) recharacterise Francuz; (b) soften the novelty statement; (c) add the $\delta=1/8$ / 2×2 caveat; (d) specify the $\mu$ bisection protocol; (e) rename "SU(2) emulation". The Python Engineer MUST: (i) enforce the $D\ge3$ unrolled-mode guard; (ii) implement swap-gate-aware two-site RDM; (iii) use `jaxopt.implicit_diff`.
