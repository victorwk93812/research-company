# RA Critique of `theory_draft.md`

**Reviewer role:** Critical Research Assistant (Phase 2)
**Date:** 2026-04-27

---

## Literature Cross-Check

I ran independent arXiv queries (`automatic differentiation CTMRG iPEPS QR decomposition`, `QR decomposition reverse mode automatic differentiation`, `iPEPS metric preconditioner`) and pulled abstracts directly from the arXiv mirror. I personally consulted:

| arXiv id | Verified | Use to me |
|---|---|---|
| 2505.00494 (Zhang–Yang–Corboz, *Accelerating 2D TN contractions using QR*) | Title, authors, GPU-only speedup claim verified. Two-orders-of-magnitude is on **H100 GPU** for the Heisenberg / J1–J2 forward only — confirmed. | The researcher's "modest CPU speedup" framing is honest; not overclaimed. |
| 2509.05090 (Yang & Corboz, honeycomb QR-CTMRG) | Verified; confirms order-of-magnitude on GPU; **no AD discussed**. | Independent: QR forward sound on a non-square geometry. |
| 2311.11894 (Francuz, Schuch, Vanhecke; PRR 7 013237) | Verified; the four ingredients (SVD-backward inaccuracy, Lorentzian, kept↔discarded truncation correction, gauge fix) are correctly characterised. | Confirms baseline. |
| 2511.09546 (Zhang, Yang, Corboz, Haegeman, Tang; metric preconditioner) | Verified; the QGT-leading-term preconditioner with L-BFGS / CG; Heisenberg + Kitaev. | Confirms RQ5 framing. |
| 2502.10298 (Naumann *et al.*, split-CTMRG) | Verified existence and authors; the Fishman two-projector wording in the draft is supported by tenax's existing implementation comments. | OK. |
| 2511.22669 (Cortés-Estay, Kamar, Corboz; energy variance) | Verified; the variance-extrapolation cross-check claim in C6 is well-motivated. | OK. |
| 1009.6112 (Walter, Lehmann, Lamour, *Higher-order derivatives of QR and symmetric eigh*, 2010) | Found independently. | This is a load-bearing reference for §3.4 of the draft (QR reverse-mode formula). The draft cites Roberts 1963 / Walter 2012 / Hubig–McCulloch 2019 — Walter 2010 (arXiv:1009.6112) is the most pedagogically explicit free version and **should be cited explicitly** in the LaTeX paper. |
| 1903.09650 (Liao, Liu, Wang, Xiang; PRX 9, 031041) | Authors, year, "stable AD for tensor decomposition + fixed-point backprop" all verified. | OK as ancestor. |
| 2512.05749 (Zhou *et al.*, *SR with warm-started SVD*) | Verified — and the abstract is **VMC / atomic-molecular**, not iPEPS. | The draft already correctly downgrades this to a stretch goal; good. |

I did **not** find a paper that already publishes an end-to-end QR-CTMRG-AD pipeline with the metric-preconditioner orthogonality study. The novelty claim of the draft (the *combination* of QR forward + Francuz-style backward + IFT + 2×2 metric × projector design) survives my independent search. **The literature positioning is accurate and the novelty claim is defensible.**

---

## Technical Flaws

I find the draft mostly correct but flag **six items** the Researcher (or, where the issue is purely a typesetting matter, the LaTeX Writer) must address before Phase 4.

### F1. §2 step 4: $\mathrm{eigvals}(RR^\dagger) = \sigma^2(M)$ holds only with a dimension caveat

The draft writes: *"the eigenvalues $\lambda_{q,i}$ equal the squared singular values of $M_q$ (since $Q_q$ is an isometry), so global truncation by $\lambda$ is exactly the SVD-based truncation by $\sigma^2$."* This is true **only in the iPEPS-CTM regime $f_q \ge c_q$**, i.e. when the fused leg is at least as large as the column dimension. Then thin QR gives $r_q = c_q$, $Q_q^\dagger Q_q = I_{c_q}$ and $RR^\dagger$ has the $c_q$ non-zero eigenvalues of $MM^\dagger$ exactly. In the opposite regime ($f_q < c_q$) the QR is "wide", $Q_q$ is fully unitary and the eigh of $R R^\dagger$ has $f_q$ eigenvalues equal to the squared SVs.

For the cells in Variables To Toggle (§Variables.B, instruction.md), $f_q = \chi$ (after fusion), $c_q \le 2 D^2$. With $\chi \ge 4D$ (the smallest entry of the chi-ramp), $f_q \ge 4D \ge 2D^2$ whenever $D \le 2$ — so for $D \in \{2\}$ we are exactly at the boundary and for $D \ge 3$ we are in the *small-fused* regime $f_q < c_q$ unless $\chi$ scales like $\Omega(D^2)$.

**Required action:** State the dimension condition explicitly in the typeset paper, and have the engineer assert `f_q >= c_q or fall through to the wide-QR branch` (which is mathematically the same but semantically distinct). Add a `pytest` parametrising both regimes.

### F2. §3.4 cancellation argument is hand-wavy

The draft says the diagonal-phase rotation $Q\to Q\Phi, R\to\Phi^\dagger R$ "cancels in $\bar A$ identically when $|\phi|=1$, **provided** the upstream $\bar P$ is gauge-equivariant." This is true but needs to be *shown* in one or two lines, not asserted. The argument is:

$$
P=Q\Phi(\Phi^\dagger U_k) = Q U_k\quad\Rightarrow\quad \bar Q\Phi = (\bar P U_k^\dagger)\Phi,\;\; \bar U_k = (Q\Phi)^\dagger \bar P
$$

so the $\Phi$ factor in $\bar Q$ exactly compensates the $\Phi^\dagger$ from `copyltu(R\bar R^\dagger - \bar Q^\dagger Q)` in (3.4); the product is gauge-equivariant by construction. **Required action:** Promote this to a numbered Lemma in the LaTeX writeup; absorb it into the test suite as an explicit perturbation test (`gauge_invariance_test`).

### F3. §3.4 Tikhonov shift is a numerical band-aid, not a principled fix

The draft proposes $R \to R + \delta_R I$ with $\delta_R = 10^{-12} \|R\|_2$ for rank-deficient $R$. Two problems:

1. The Tikhonov shift on a triangular matrix breaks triangularity, so `solve_triangular` is no longer applicable; one must use `solve` instead. State that explicitly in the engineering section.
2. For the J1–J2 frustrated regime — the very regime where the stability argument is most needed — rank deficiency in the projector source is *exactly* what makes SVD-AD pathological. A $10^{-12}$ shift may be too small and a larger shift biases the projector. Better: **use the gauge fix $\phi=1$ for any $|R_{ii}| < 10^{-12}\|R\|$ AND zero the corresponding projector column** (i.e. let the truncation handle it). The $R^{-1}$ in (3.4) then need only be applied to the column-permuted *full-rank submatrix*.

**Required action:** Engineer must implement the rank-aware fallback (zero-column + reduced solve), not the naïve Tikhonov; add a `pytest` that places a deliberately rank-deficient $M$ and checks the AD gradient is finite and matches finite-difference within $10^{-6}$.

### F4. RQ3 framing — CPU honesty

The draft is careful to say "the 100× number is GPU-specific" (good) and predicts "1.2×–3× CPU forward speedup." But it **does not state the prediction is *speculative***. CPU LAPACK QR (`dgeqrf` + `dorgqr`) and CPU LAPACK SVD (`dgesdd`) have similar leading constants for square matrices; the QR advantage on CPU can vanish or even reverse for small $\chi$. **Required action:** The hypotheses table (§4, RQ3) should add an explicit *null-result* row: "if backward-only timing for QR is $\ge$ backward-only SVD on every cell, conclude that the Lorentzian-eigh backward of $RR^\dagger$ has the same leading cost as Lorentzian-SVD of $C^\dagger C$; this is a publishable methods note, not a failure." This protects against motivated reasoning when the engineer reports timings.

### F5. RQ5 — easy to fake by cherry-picking points

The 2×2 interaction-effect plot (RQ5) at a single $J_2/J_1=0.5$ is the headline finding; it is also the easiest to fake by selecting a seed where QR+metric happens to win. **Required action:** the engineer must report **all 3 seeds individually** in the 2×2 table (not only the mean) and the interaction term must be the seed-mean ± seed-std. Additionally, the hypothesis test should use a paired sign test across seeds, not eyeballing.

### F6. RQ4 / RQ5 - "metric-LBFGS+QR is fastest-and-stablest" prejudges the result

The hypothesis in §4 RQ4 *predicts* the conclusion. This is fine as a *hypothesis* but the draft must clearly distinguish a hypothesis from a finding. **Required action:** Rephrase the hypothesis row as "we test the prediction that ..."; require the engineer to log results before plotting so the plot rendering is not result-dependent.

### F7. Stretch goal — SR is silently dropped, OK if documented

The draft correctly says SR is a stretch goal and points to `arXiv:2512.05749`. However, the optimiser-axis table in instruction.md (Variables.D) lists SR as a variable. **Required action:** In the LaTeX paper, the optimiser axis must be reported with a clear footnote: "stochastic reconfiguration with warm-started SVD was not implemented this run; the three implemented optimisers cover the standard iPEPS optimisation methods (Adam, L-BFGS, metric-LBFGS) and answer RQ4–RQ5 as posed."

---

## What I am NOT critiquing

* The choice of dispatching `qr_canonical` via a registered hook in `./src/` rather than editing `./tenax/` directly is correct and matches the substrate's contribution policy (the `CLAUDE.md` of tenax requires upstream changes to go through PRs).
* The engineering harness (§5) lists the right tests and the right config knobs.
* The literature review covers the canonical references and the novelty claim is defensible.

---

## Verdict

The theoretical proposal is **physically and mathematically sound** modulo the seven items above. F1–F3 are the load-bearing technical fixes; F4–F7 are honesty / methodology guardrails. Once these are reflected in the LaTeX writeup and the engineer's test suite, the run can proceed.

**APPROVAL GRANTED: PROCEED TO TYPESETTING AND ENGINEERING.**
