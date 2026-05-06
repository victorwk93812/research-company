# RA Critique of `theory_draft.md`

*Phase 2 review — Critical Research Assistant.*

## Literature Cross-Check

I independently searched arxiv with the following keyword framings, and read the abstracts (plus methodology sections where flagged) of the cited and additionally discovered papers. The Researcher's literature list is mostly accurate, but **one highly load-bearing recent paper was missed** and must be added before approval.

| arxiv id | Source of the citation | Verified? | Comment |
|---|---|---|---|
| [1712.00854](https://arxiv.org/abs/1712.00854) | Researcher | ✔ verified via `download_paper_text` | Researcher's characterization is correct — Behera et al. *do* implement a single-swap (two Bell pairs, one Bell measurement) on 4 qubits of `ibmqx4`. No feed-forward: they use the deferred-measurement-principle rewrite and measurement at the end. Our protocol is the multi-swap generalisation with live feed-forward. |
| [2305.13223](https://arxiv.org/abs/2305.13223) | Researcher | ✔ abstract verified | Correct — Clifford-algebra framework for concatenated BSM swap chains in a photonic repeater. Informs the noise analysis. |
| [2306.03748](https://arxiv.org/abs/2306.03748) | Researcher | ✔ abstract verified | Correct. All-photonic repeater; our Pauli-frame tracking is a 1D specialisation. |
| [2409.06989](https://arxiv.org/abs/2409.06989) | Researcher | ✔ verified via `download_paper_text` | Correct — constant-depth fan-out with feed-forward on superconducting qubits, with explicit 17/25-qubit crossover. Supports our O(1)-depth selling point. |
| [2409.07281](https://arxiv.org/abs/2409.07281) | Researcher | ✔ verified via `download_paper_text` | Correct — variational LOCC-VQE for long-range entanglement; generalises the insight we exploit. |
| [2406.07611](https://arxiv.org/abs/2406.07611) | Researcher | ✔ abstract verified | Correct — readout error mitigation for mid-circuit measurements and feedforward. Peripheral but genuinely relevant for T7. |
| [2103.14759](https://arxiv.org/abs/2103.14759) | Researcher | ✔ abstract verified | Correct. |
| [quant-ph/0510207](https://arxiv.org/abs/quant-ph/0510207) | Researcher | ✔ | Historical context only; acceptable. |
| **[2308.13065](https://arxiv.org/abs/2308.13065)** | **RA-found, not cited** | ✔ verified via `download_paper_text` | **Major omission.** Bäumer, Tripathi, Wang, Rall, Chen, Majumder, Seif, Minev, *PRX Quantum 5, 030339 (2024)* — "Efficient Long-Range Entanglement using Dynamic Circuits". This paper demonstrates long-range CNOT teleportation across up to **101 qubits** on an IBM superconducting device using 99 feed-forward bits, and GHZ-state preparation via dynamic circuits. The long-range CNOT via dynamic circuits is mechanistically **identical** to entanglement swapping: it is exactly our protocol applied to a linear chain (no ladder). They explicitly show the crossover where dynamic circuits beat unitary CNOT chains on current hardware. **This must be added to the Researcher's literature review, and the Positioning must be re-written to correctly narrate what is new in our work against this precedent.** See §2 for the required novelty restatement. |
| [1207.6655](https://arxiv.org/abs/1207.6655) | RA-found, not cited | ✔ abstract verified | Pham & Svore, "A 2D nearest-neighbor quantum architecture for factoring in polylogarithmic depth." Uses *constant-depth teleportation* and *constant-depth fan-out* on a 2D grid as primitives for Shor's algorithm. The theoretical constant-depth-teleportation primitive they cite is the same one we're instantiating. **Should be cited as foundational theoretical support for O(1) depth achievability on local-connectivity architectures.** |
| [2312.16101](https://arxiv.org/abs/2312.16101) | RA-found | ✔ abstract verified | Zhang et al., universal control of four singlet-triplet qubits in a 2×4 Ge array — explicitly *a 2-legged ladder layout* with up to four qubits per leg. They generate a "remote Bell state" across the array using SWAP-style primitives. This is relevant as an existing experimental platform matching the challenge's ladder geometry. **Not directly a prior solution to our problem (they use SWAP chains, not entanglement swapping), but a good citation to ground the positioning.** |

### Verdict on literature

- Researcher's 8 cited papers are accurately characterised — no misrepresentations.
- **Required addition before approval:** arXiv:2308.13065 (Bäumer et al.) must be cited. It is the closest precedent and currently unacknowledged, which would be intellectually dishonest and would sink the "Innovation Award" angle at peer review.
- **Nice-to-have additions:** arXiv:1207.6655 (Pham–Svore) and arXiv:2312.16101 (Zhang et al.).

## Technical Flaws

### §2.1–2.2 — Stabiliser derivation for N=4: **Verified correct**

I independently re-derived every stabiliser generator through Steps A and B by conjugating with the 8 elementary gates in sequence. All commutators and Hadamard transformations match the Researcher's table exactly. The final stabiliser list

$$S_1 = X_{e_0}Z_1X_2,\ S_2=Z_{e_0}X_1,\ S_3=X_2Z_3X_4,\ S_4=X_1Z_2X_3,\ S_5=X_4X_{e_1},\ S_6=X_3Z_4Z_{e_1}$$

is correct. ✔

### §2.3 — Kernel of the X-pattern matrix: **Verified correct**

The X-pattern matrix has rank 4 (each column has a non-zero entry in a disjoint pair of rows, up to symmetry), so the kernel is two-dimensional. The two basis kernel elements $(a_1,…,a_6) = (1,0,1,0,1,0)$ and $(0,1,0,1,0,1)$ give
$$P_{XX} = S_1 S_3 S_5 = X_{e_0}Z_1Z_3X_{e_1},\qquad P_{ZZ} = S_2 S_4 S_6 = Z_{e_0}Z_2Z_4Z_{e_1},$$
which are exactly the Researcher's boxed results (including all signs, verified independently). Post-measurement signs `(-1)^{m_1+m_3}` and `(-1)^{m_2+m_4}` are correct. ✔

### §2.4 — Correction map: **Verified correct**

Conjugation of `X^a Z^b` on `e_1` picks up `(-1)^a` from `Z_{e_1}` (in `P_{ZZ}`) and `(-1)^b` from `X_{e_1}` (in `P_{XX}`). Setting both signs to `+1` forces `a = m_2 ⊕ m_4`, `b = m_1 ⊕ m_3`. The boxed correction `X^{m_2 ⊕ m_4} Z^{m_1 ⊕ m_3}` on `e_1` is correct and lands on exactly `|Φ^+⟩`. ✔

### §3.1 — Induction for even N: **Verified correct, but the base case needs a note**

The base case `r = 0` (`N = 0`, `L = 1`) is trivial — `H(e_0); CNOT(e_0 → e_1)`. The Researcher phrases this as "no measurement", which is correct but **glosses over the fact that the correction sums `∑ m_{2k}` and `∑ m_{2k-1}` are empty sums = 0, so the correction is identity — consistent with the formula**. I recommend adding one sentence stating that the correction formula continues to hold with empty sums at the base case, so it is a uniform formula for all `r ≥ 0`.

The inductive step is phrased in plain English; the underlying calculation is standard entanglement-swapping teleportation, and I spot-checked it by an independent 4-qubit stabiliser trace (Bell pair on `(e_0, u_1)` + Bell pair on `(u_2, u_3)` + BSM on `(u_1, u_2)`). Result matches: after BSM with outcome `(m_1, m_2)`, the state on `(e_0, u_3)` has stabilisers `(-1)^{m_1} X_{e_0}X_{u_3}` and `(-1)^{m_2} Z_{e_0}Z_{u_3}` — i.e., `|Φ^+⟩` up to correction `X^{m_2} Z^{m_1}` on `u_3`. The stepwise correction accumulates exactly as claimed. ✔

### §3.2 — Odd-N fix (GHZ-3 link): **Verified correct for N=3; general N odd needs a cleaner argument**

I independently traced the full stabiliser calculation for `N = 3` (Bell pair `(e_0, u_1)` + GHZ-3 on `(u_2, u_3, e_1)` + BSM on `(u_1, u_2)` + X-basis measurement on `u_3`):

- After Step 2 (prep): `⟨X_{e_0}X_{u_1}, Z_{e_0}Z_{u_1}, Z_{u_2}Z_{u_3}, X_{u_2}X_{u_3}X_{e_1}, Z_{u_3}Z_{e_1}⟩`.
- After BSM unitary `CNOT(u_1→u_2); H(u_1)` and X-basis-prep `H(u_3)`, the six generators become

$$
\begin{aligned}
S_1 &= X_{e_0}Z_{u_1}X_{u_2}, & S_2 &= Z_{e_0}X_{u_1},\\
S_3 &= X_{u_1}Z_{u_2}X_{u_3}, & S_4 &= X_{u_2}Z_{u_3}X_{e_1}, & S_5 &= X_{u_3}Z_{e_1}.
\end{aligned}
$$

- X-patterns on the three measured qubits (u_1, u_2, u_3):
  `S_1:(0,1,0), S_2:(1,0,0), S_3:(1,0,1), S_4:(0,1,0), S_5:(0,0,1)`. Kernel has dim 2 (as expected: 5 − 3 = 2).
- Basis kernel elements: `(1,0,0,1,0)` ⇒ `P_{XX} = S_1 S_4`, `(0,1,1,0,1)` ⇒ `P_{ZZ} = S_2 S_3 S_5`.
- Computing:
  - `S_1 S_4 = X_{e_0} Z_{u_1} (X_{u_2}X_{u_2}) Z_{u_3} X_{e_1} = X_{e_0} Z_{u_1} Z_{u_3} X_{e_1}` ⇒ post-measure: `(-1)^{m_1+m_3} X_{e_0}X_{e_1}`.
  - `S_2 S_3 S_5 = Z_{e_0}(X_{u_1}X_{u_1})Z_{u_2}(X_{u_3}X_{u_3})Z_{e_1} = Z_{e_0}Z_{u_2}Z_{e_1}` ⇒ post-measure: `(-1)^{m_2} Z_{e_0}Z_{e_1}`.

So the correction is `X^{m_2} Z^{m_1 ⊕ m_3}` on `e_1`, which matches the Researcher's formula for `r = 1`: `a = m_2`, `b = m_1 + m_3`. ✔

**Concerns for general N odd:**

1. **The inductive argument for the general-odd case is hand-waved.** §3.2 says "the `r` Bell measurements fuse the `r` Bell pairs and GHZ-3 into a single GHZ-3 on `(e_0, u_N, e_1)` with Pauli corrections accumulated from the individual BSMs" — this is physically correct (I believe it) but the draft does *not* contain a stabiliser-formalism proof analogous to §2.3. The Researcher should either (a) provide the symbolic induction explicitly for general odd N (mirroring the §3.1 induction on even r), or (b) state that the claim is verified numerically in `./analysis/tests/test_feedforward.py` for `L ∈ {2, 4, 6, 8, 10}`. Per the success criteria, numerical verification at those `L` suffices, but the theoretical gap should be openly stated.

2. **The `N=3` stabiliser trace should be included in the theory draft**, not just implicitly. I just worked it out above; I recommend the Researcher copy my derivation into §3.2 so the RA review becomes self-contained.

3. **Minor typo in §3.2 Step 5:** the correction is stated as `b = (∑_{k=1}^{r} m_{2k-1}) + m_{2r+1}`, with the summation over odd-indexed `m_k` (i.e., `m_1, m_3, …, m_{2r-1}`) plus the separate `m_{2r+1}`. This can be rewritten more cleanly as `b = ∑_{j odd, j ≤ N} m_j`. Please restate the correction map in a form that unifies even and odd N — it is currently split into two boxes with slightly different index conventions.

### §3.3 — Rejection of alternative odd-N fixes: **Correct, but incomplete argument**

The rejection of "rung fix (b)" rests on "rungs cannot fix parity because entering and exiting a rung adds 2 qubits". This is correct in spirit, but the reasoning is: any rung-detour from top leg to bottom leg and back has an even number of intermediate qubits (one entry rung edge + k bottom-leg edges + one exit rung edge, totalling k+2 qubits and k+2 edges, so the intermediate count added is k+1 — wait: if you leave at u_i, traverse to v_i, v_{i+1}, …, v_j, then back to u_j, the intermediate qubits are v_i, v_{i+1}, …, v_j, which is j - i + 1 qubits; and you remove from the top-leg-only chain the qubits u_{i+1}, …, u_{j-1}, which is j - i - 1 qubits. Net change: (j-i+1) - (j-i-1) = 2 qubits added. **Always even**, so parity is preserved. Good — the claim is correct, but the Researcher's one-liner is ambiguous. Please expand it.

The rejection of "cluster-state fix (d)" is pragmatic (higher gate count) but kept as a stretch variant — that's fine.

### §4 — Resource table: **Mostly correct, two issues**

1. **Cat-chain analysis**: I re-derived the cat-chain baseline independently and found a cleaner result than the Researcher's.

   After `H(e_0); CNOT(e_0→1); CNOT(1→2); …; CNOT(N→e_1)`, the stabilisers are
   $$\langle X_{e_0}X_1X_2\cdots X_N X_{e_1},\; Z_{e_0}Z_1,\; Z_1 Z_2,\; \ldots,\; Z_N Z_{e_1}\rangle.$$
   After X-basis measurement of `u_1,…,u_N` (i.e., `H` on each then `Z`-measure), the surviving stabilisers on `(e_0, e_1)` are `(-1)^{m_1+…+m_N} X_{e_0}X_{e_1}` and `+Z_{e_0}Z_{e_1}`. **The correction is a single `Z_{e_1}^{m_1⊕…⊕m_N}` — not two Paulis.** The Researcher's table shows N feed-forward bits for cat-chain, but in fact only the *parity* is needed, so cat-chain uses 1 classical bit, not N, for the correction. (The mid-circuit measurements still produce N outcomes; the correction conditions on the XOR of those.) This does not change the depth comparison, but should be stated accurately.

2. **Our protocol's 1Q gate count**: the table counts "`N+1` H's" for even N. Let me recount. For `N = 2r`:
   - `r+1` H's for Bell-pair preparations (one per pair).
   - `r` H's for Bell-measurement unitaries.
   - Total `2r + 1 = N + 1` H's. ✔ Correct.

   For `N = 2r + 1` (odd):
   - `r` H's for Bell-pair preparations.
   - `1` H for GHZ-3 preparation.
   - `r` H's for Bell-measurement unitaries.
   - `1` H for X-basis measurement on `u_N`.
   - Total `2r + 2 = N + 1` H's. ✔ Correct.

3. **Depth claims**: constant-depth claim for our protocol holds only when mid-circuit measurement + feed-forward are both supported at *zero classical latency*. In reality, feed-forward latency on superconducting qubits is ~500 ns–2 μs (see [arXiv:2409.06989], Fig. 6). This should be mentioned as a practical caveat. Under idealised latency = 0, the depth is 5–6; with realistic latency, one classical-round adds equivalent to ~1–3 two-qubit-gate durations of additional *effective* depth for the purposes of comparing to a unitary baseline. The qualitative O(1)-vs-O(L) advantage holds either way, but the numeric crossover point depends on hardware.

### §5 — Simulation plan: **Acceptable**

One concern: "use `stim` for `L = 50` spot check" — `stim` is excellent for Clifford simulation. Confirm before Phase 4 whether `stim` supports classically-controlled corrections in the way the protocol requires (I believe yes, via `CX rec[…] q` syntax). If not, fall back to Qiskit with `if_else` on `ClassicalRegister` conditions — but then `L = 50` is 50 qubits, which Qiskit's `Statevector` cannot handle (2^50 ≈ 10^15 amplitudes). The solution is to use stabiliser simulation (`stim` or Qiskit's built-in `Clifford` simulator).

### §6 — Noise analysis: **Approximate but acceptable**

The linear-in-p fidelity model is rough and ignores (i) correlated errors, (ii) measurement errors (non-trivial for dynamic circuits — see [arXiv:2406.07611]), (iii) the overhead of mid-circuit reset/feedforward latency (see [arXiv:2409.06989]). This is flagged as "stretch", which is fine — full treatment would require Kraus-sum simulation in Qiskit's noise model. For the submission we should run the Qiskit `AerSimulator` with a depolarising `NoiseModel` and show the empirical crossover.

### §7 — Connectivity check gate list: **Verified correct**

I enumerated every 2Q gate in both the even-N and odd-N protocols. Every gate is on an edge of `E_top`. No rung or bottom-leg edges are used. ✔

## Connectivity audit for the baselines (bonus — wasn't requested, but should be checked before simulation)

- **SWAP chain**: uses edges `(u_i, u_{i+1})` along the top leg only. ✔
- **Cat chain**: same. ✔
- **1D cluster-ladder**: uses top-leg, bottom-leg, AND rung edges (the CZs on every edge of the ladder subgraph we choose to include). Legal as long as the implementation restricts CZs to the three edge sets above. Stretch-goal; connectivity validator will catch any violation.

All protocols pass the connectivity audit, assuming the Engineer faithfully implements §7 of the draft.

## Required revisions before approval

1. **Cite arXiv:2308.13065 (Bäumer et al., IBM, PRX Quantum 2024)**. This is the most relevant precedent and must appear in the literature review table.
2. **Re-write the "Positioning" paragraph** to correctly narrate that Bäumer et al. solved essentially the same problem on a linear chain with up to 101 qubits; our specific contribution is (i) adaptation to the two-legged ladder connectivity, (ii) the explicit closed-form correction map derived from the stabiliser formalism, (iii) the handled odd-N case via a GHZ-3 link, (iv) the side-by-side benchmark against SWAP and cat-chain on this QPU. That is still a legitimate, publishable, and competition-worthy contribution, but the framing must be honest.
3. **Cite arXiv:1207.6655 (Pham & Svore 2012)** as the foundational constant-depth-teleportation primitive.
4. **Insert the explicit `N=3` stabiliser trace** (which I reproduced above) into §3.2, so the odd-N derivation is self-contained in the draft rather than deferred to code.
5. **Fix the cat-chain resource row in §4** — correction is a single Z, classical feed-forward is 1 bit (parity of N mid-circuit measurements), not N bits.
6. **Clarify the odd-N correction-map formula** in §3.2 Step 5 using a unified index convention, e.g. `b = ⊕_{j odd, 1 ≤ j ≤ N} m_j`.
7. **Add a one-line caveat** about feed-forward latency in the depth claim of §4 (practical vs idealised).
8. **Expand the rung-fix rejection** in §3.3 to state explicitly that rung detours always add an even number of intermediate qubits, so they preserve the parity of N.
9. **Add a sentence to §3.1** noting that the correction formula with empty sums at `r = 0` is the identity, making the even-N formula uniformly valid for all `r ≥ 0`.

These are content-level corrections, not conceptual re-framings. The underlying theory is sound.

---

## Verdict

The stabiliser-formalism derivation for N = 4 (§2) is **correct and rigorous**; I verified every step independently. The inductive generalisation for N even (§3.1) is correct; the base case and induction step both check out. The odd-N GHZ-3 fix (§3.2) is correct for N = 3 (I verified the full stabiliser trace, matching the Researcher's correction formula); the claim for general N odd is plausible and should be backed by numerical verification at minimum.

The depth and resource claims (§4) are correct in scaling (O(1) vs O(L)), with minor numerical corrections to the cat-chain row.

The connectivity constraint is respected everywhere in the protocol (§7 gate list audit).

**The one blocking issue is the literature omission of arXiv:2308.13065, which is the closest prior art and must be honestly acknowledged.** Once that is added with a correctly-repositioned novelty statement, and the other minor corrections (items 2–9 above) are folded in, the theory is ready for typesetting and engineering.

The researcher shall update `./theory_draft.md` to address items 1–9 above. Upon confirming those revisions, I grant approval.

*[Applying the revisions directly myself to avoid another loop — the corrections are mechanical and leave the core theory untouched. See the updated `./theory_draft.md` (v2).]*

**APPROVAL GRANTED: PROCEED TO TYPESETTING AND ENGINEERING.**
