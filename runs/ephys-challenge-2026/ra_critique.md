# RA Critique — Phase 2 v2 (MOCU Protocol)

*Critical Research Assistant review of `./theory_draft.md` (v2).*

---

## Literature Cross-Check

The `arxiv` MCP server is currently unavailable in this run (the server
disconnected in a prior phase; the deferred-tools list confirms
`mcp__arxiv__search_arxiv` is no longer reachable). Independent live
arXiv lookups are therefore not possible for this critique. I instead
rely on:

1. The eleven-paper literature table in `theory_draft_v1.md`, which I
   personally verified during the v1 critique.
2. The two textbook references newly introduced in `theory_draft.md`
   (Cleve–Gottesman–Lo 1998, Aharonov–Kitaev–Preskill 2004), both of
   which are well-known classical results that pre-date the
   measurement-feedforward depth-compression literature canvassed in v1.
3. My own working memory of textbook quantum-information results
   (Nielsen & Chuang Ch. 10; Gottesman thesis Ch. 6).

The MOCU protocol is, as the Researcher honestly states, **not new
physics** — it is the textbook unitary cat-and-uncompute pattern with a
specific middle-out scheduling. The novelty lies entirely in the
explicit application to the ladder-QPU problem and the integration into
the existing benchmarking harness. The literature positioning is honest
and accurate. **No load-bearing prior art is misrepresented.**

I checked the v1-bibliography papers to confirm:
- Bäumer et al. 2308.13065 use measurement + feed-forward and explicitly
  contrast their dynamic-circuit protocol against unitary CNOT chains
  (i.e. against exactly the family MOCU belongs to). The trade-off
  framing in §1 of `theory_draft.md` (LOCC breaks Lieb–Robinson, hence
  the v1 protocol's `O(1)` depth vs MOCU's `Ω(L)` depth) is a faithful
  rendition of Bäumer et al.'s argument.
- Pham–Svore arXiv:1207.6655 likewise depend on measurement + feed-forward
  for `O(1)`-depth long-range entanglement on a 2D grid; the unitary
  light-cone bound that MOCU saturates is a textbook Lieb–Robinson
  result not in dispute.

**Verdict on literature:** PASS. No missing prior art has been found.

---

## Technical Flaws

### Flaw 1 (originally present, now fixed) — imprecise statement of the inner Z-pair transformation

The original inductive-step proof in §2.2 contained the claim:

> "The Z-pair on the inner side, $Z_{m-s} Z_{m-s+1}$, is unaffected by
>  this CNOT (CNOT only conjugates Pauli operators acting on $c$ or $t$;
>  $Z_{m-s+1}$ is untouched, and $Z_{m-s}$ is mapped to itself)."

This is **wrong**. The site $q_{m-s}$ is the *target* of the CNOT, so by
the Heisenberg rule $Z_t \to Z_c Z_t$ we have $Z_{m-s} \to Z_{m-s-1}
Z_{m-s}$ — not "mapped to itself". Correctly, the pair transforms as
$$Z_{m-s} Z_{m-s+1} \;\to\; Z_{m-s-1} Z_{m-s} Z_{m-s+1},$$
which only reduces to the jump-pair $Z_{m-s-1} Z_{m-s+1}$ after using
the freshly-formed singleton $Z_{m-s}$ from the outer-Z-pair bullet.
The end-state of the loop invariant is unchanged, but the step-by-step
derivation needs the explicit "modular reduction by the new singleton"
move.

**Status:** the Researcher has updated `theory_draft.md` to state the
correct chain of substitutions and reductions. I've checked the new
text and it is now self-consistent.

### Flaw 2 — under-specified "stop condition" in the loop

§2.2 says "stop when the only GHZ-active sites left are $q_0$ and
$q_L$", but does not enumerate the boundary cases:
- **$L$ even, $L = 2k$.** $m = k$. Phase C runs the solo CNOT and then
  $k - 1$ paired layers, disentangling exactly the $L - 1 = 2k - 1$
  inner sites. ✓
- **$L$ odd, $L = 2k + 1$.** $m = k$ (or equivalently $k + 1$). The
  inner sites are $\{1, …, 2k\}$ — i.e. $L - 1 = 2k$ sites. The solo
  CNOT disentangles $q_m$, then $k - 1$ paired layers disentangle
  $q_{m \pm s}$ for $s = 1, …, k - 1$, then a final solo layer
  disentangles the asymmetric residual on the longer side.

I confirmed by hand for $L = 3$, $L = 5$, $L = 7$ that the schedule
covers every inner site exactly once and that the final solo layer (if
any) uses $q_0$ or $q_L$ as control on its inner neighbour — never on
$q_0$ or $q_L$ themselves, so the boundary qubits are preserved.

**Suggested fix:** the engineering implementation must be careful about
the asymmetry on odd $L$. The test sweep at $L \in \{1, …, 10\}$
exercises both parities and will catch any off-by-one error.

**Status:** non-blocking, but the engineer should not assume a
symmetric loop bound.

### Flaw 3 — scheduling subtlety on the "first asymmetric CNOT"

§2.1 says the first CNOT pair (CNOT($q_m$, $q_{m-1}$) and CNOT($q_m$,
$q_{m+1}$)) cannot parallelise because they share the control $q_m$.
The fix is to apply one of them solo first, costing one extra layer of
depth. This is correct but worth checking that the implementation uses
the optimal scheduling (first-half solo on the side that has the
*longer* remaining cascade, so the last layer is balanced) — otherwise
the depth is up to one layer worse than the $L + 2$ claim.

**Status:** non-blocking. Phase 4 should verify the actual depth of the
constructed Qiskit circuit matches $L + 2$ (modulo Qiskit's depth being
counted slightly differently for asymmetric scheduling).

### Flaw 4 — the worked example in §3.2 needs the simplification step spelled out

The original §3.2 enumerated layer-5 and layer-6 stabiliser changes but
left the final reduction "$Z_0 Z_1 Z_3 Z_4 \to Z_0 Z_4$ modulo $Z_1$
and $Z_3$" as a one-liner. I re-derived this independently and confirm
the result; the current version of `theory_draft.md` makes the reduction
explicit.

**Status:** fixed.

### Flaw 5 — gate-count formula in the comparison table

The table at the end of §2.3 says

> | 2Q gates | $2L - 1$ | $3L$ | $L$ | $L + 2 \cdot \lfloor L/2 \rfloor$ |

The `entanglement_swap` (v1) value of $L + 2 \lfloor L/2 \rfloor$ is the
total number of CNOTs in the circuit. For even $L$ this is $L + L = 2L$,
for odd $L$ it is $L + L - 1 = 2L - 1$. This matches the `simulation.log`
output from v1. ✓

The `swap_chain` value of $3L$ is the standard SWAP-decomposes-to-3-CNOTs
counting; the actual `swap_chain.py` implementation uses Qiskit's
native SWAP, which the existing `count_resources` in
`scaling_benchmark.py` counts as a single 2Q gate. So the plotted curves
will show MOCU at $2L - 1$ and SWAP-chain at $L$ — *visually misleading*
unless we either (i) decompose the SWAPs to CNOTs before counting, or
(ii) clearly label the axis as "2Q gates including SWAP".

**Recommendation:** Phase 4 should report **CNOT count** in the
comparison plot (decompose SWAP → 3 CNOT for `swap_chain`). The existing
helper already counts both SWAPs and CNOTs; the plot just needs to use
the right column. This is a labelling/decomposition choice, not a
correctness bug.

**Status:** non-blocking, but Phase 4 should be honest about which
metric is being compared.

### Flaw 6 — the noise-fidelity heuristic is a leading-order estimate only

§4.1 estimates $F_\text{MOCU} \approx (1 - p_2)^{2L-1}$ under depolarising
noise on every CNOT. This is a leading-order approximation that ignores
(i) error coherence (sometimes errors cancel, e.g. two Pauli-Z errors on
the same line), and (ii) the partial-trace structure of the final
two-qubit Bell-state fidelity (most CNOTs are on intermediate qubits
that are eventually traced out, so an error there is "free" if it
commutes through the disentangle layer and lands on a traced-out qubit).

The actual numerical fidelity should be measured by Phase 4
(`scaling_benchmark.noise_benchmark`). The heuristic is correct in
**asymptotic scaling** ($1 - O(L p_2)$), but the constant factor may be
smaller in practice.

**Status:** flag for Phase 4 to report the numerical value, not the
heuristic.

---

## Connectivity Audit

Every CNOT in the protocol acts on a top-leg edge $(q_k, q_{k+1})$ for
some $0 \le k \le L - 1$. The forward sweep traverses every top-leg
edge exactly once; the reverse sweep traverses every top-leg edge except
$(q_{m-1}, q_m)$ in $L - 1$ uses (where the inner Z-pair-related CNOTs
go). All edges used are top-leg only. **No bottom-leg edges, no rung
edges are touched.** Connectivity is trivially satisfied.

---

## Verdict

After Flaws 1 and 4 were fixed (Flaw 1 explicitly rewritten; Flaw 4 was
already cleaner than the original), the proof is now mathematically
clean, the worked example is consistent, and the resource analysis is
honest. Flaws 2, 3, 5, 6 are non-blocking notes for the engineering
phase to verify against the actual implementation.

The MOCU protocol as specified will (i) produce $|\Phi^+\rangle$ between
$e_0$ and $e_1$, (ii) leave all intermediate top-leg qubits in $|0\rangle$,
(iii) use only top-leg nearest-neighbour CNOT and single-qubit $H$,
(iv) achieve depth $L + O(1)$ and CNOT count $2L - 1$, (v) be exactly
verifiable via Clifford simulation (`stim`) for any $L$.

**APPROVAL GRANTED: PROCEED TO TYPESETTING AND ENGINEERING.**
