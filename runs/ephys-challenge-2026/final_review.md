# Final Review — Phase 5 v2 (MOGU Protocol Addition)

*Three-reviewer audit of the v2 measurement-free protocol added to the
2026 NCCU Institute of Applied Physics submission.*

The Review Board panel: **The Math Pedant**, **The Performance Hacker**,
and **The Domain Expert**. Each reports separately; the consolidated
verdict appears at the end.

The v1 protocol (constant-depth entanglement swapping with
mid-circuit measurement and feed-forward) and its review are preserved
in `final_review_v1.md`. Submission bundle `./submission/` is untouched
in v2 — competition-ready as of 2026-04-23.

---

## 1. The Math Pedant (`./theory_draft.md` + `./report/main.tex` Appendix)

### Scope of review

I re-derived the MOGU stabiliser propagation by hand for $L=4$ and
cross-checked against the stim trace from
`tests/test_mogu.py::test_stim_exact_fidelity`.

### Findings

1. **§2.1 (Forward sweep proof).** The orbit of $Z_m$ under $H + $ Phase B
   is the global $X$-string $X_0 X_1 \cdots X_L$. The other initial
   $Z_k$ generators each pick up a $Z_a$ factor from each CNOT they
   target, leaving the chain $\{Z_{k-1} Z_k\}$. Both claims are correct.
   ✔
2. **§2.2 (Reverse sweep proof, after RA Flaw 1 fix).** The four bullets
   in the inductive step now correctly track:
   - the big $X$-string losing its $X_{m-s}$ factor by $X_c \to X_c X_t$
     cancellation (with the existing $X_{m-s}$),
   - the outer Z-pair $Z_{m-s-1} Z_{m-s}$ collapsing to the single-site
     $Z_{m-s}$,
   - the inner Z-pair $Z_{m-s} Z_{m-s+1}$ becoming
     $Z_{m-s-1} Z_{m-s} Z_{m-s+1}$ and reducing to the jump-pair
     $Z_{m-s-1} Z_{m-s+1}$ via the freshly-formed singleton, and
   - all other generators untouched.

   The inductive invariant $A_s = A_{s-1} \setminus \{m-s, m+s\}$
   carries cleanly to the terminal $A = \{0, L\}$, at which point the
   stabilisers are exactly those of $\Phip_{(e_0,e_L)} \otimes
   \ket{0}^{\otimes(L-1)}$. ✔
3. **§3 (Worked example $L=4$).** Layer-by-layer Heisenberg trace
   reproduces the GHZ_5 stabilisers after layer 4 and the target
   stabiliser group after layer 6. I re-derived this myself and
   confirm the final reduction
   $\{X_0 X_4, Z_0 Z_4, Z_1, Z_2, Z_3\}$. ✔
4. **§3.3 (Boundary cases $L=1, 2, 3$).** All three small-$L$ recipes
   are correctly handled by `mogu.build_circuit`, as verified by the
   stim sweep for $L = 1, \dots, 10$ (Bell fidelity $= 1$ to machine
   precision). ✔
5. **§4 (Resource analysis).** Depth $= L + 2$ upper bound is correct;
   the actual Qiskit-reported depth is $L + 2$ for even $L$ and $L + 1$
   for odd $L$ (because the reverse-sweep solo CNOT and the first
   parallel pair can interleave with the forward sweep's last layer
   without violating the stabiliser invariants). 2Q-gate count $2L - 1$
   matches `tests/test_mogu.py::test_two_qubit_gate_count` exactly. ✔
6. **Appendix in `main.tex`.** Same content as `theory_draft.md`,
   appropriately compressed for the LaTeX surface. The appendix
   compiles cleanly with `xelatex` (12 pages total report, up from 9).
   The "back-references" to body sections use existing `\label{}`s
   (`sec:resources`, `sec:process`) — verified compile-time. ✔

### Verdict (Math Pedant)

All MOGU mathematical claims are correct; the proof is internally
consistent after RA Flaw 1 (mis-stated Z-pair conjugation rule) was
fixed; the worked $L = 4$ example reproduces the target stabiliser
group exactly; and the stim test suite cross-validates every claim
numerically. **PASS.**

---

## 2. The Performance Hacker (`./analysis/mogu.py` + tests)

### Scope of review

- Module structure and adherence to the v1 conventions.
- Correctness of the middle-out scheduling (left vs right solo).
- Connectivity, no-measurement, gate-count invariants.
- Test coverage and cross-validation.
- Integration into `scaling_benchmark.PROTOCOLS` and noise sweep.

### Findings

1. **Module structure.** `mogu.py` (76 lines) imports `_top_chain` from
   `entanglement_swap` and `n_qubits` from `ladder_graph` — reusing
   existing infrastructure, no duplication. The single public
   `build_circuit(L)` function returns a Qiskit `QuantumCircuit` with
   no `ClassicalRegister`, no measurements, only `H` and `CNOT` —
   the cleanest possible API for a unitary protocol. ✔
2. **Scheduling.** The "solo on the longer side" heuristic (lines
   42–48) is correctly implemented: when `right_n >= left_n` we apply
   `CNOT(q_m, q_{m+1})` solo first; otherwise the symmetric left
   choice. The subsequent parallel-expand loop (lines 50–60) advances
   one CNOT on each side per iteration, which is the depth-optimal
   schedule modulo the unavoidable solo. ✔
3. **No-measurement invariant.** Verified explicitly by the test
   `test_no_measurements`: zero `measure` instructions and zero
   classical bits at every $L$. ✔
4. **Connectivity.** All 10 $L$ values pass `validate_connectivity`. ✔
5. **Gate counts.** For $L = 1, \dots, 10$ the actual 2Q-gate count is
   $\{1, 3, 5, 7, 9, 11, 13, 15, 17, 19\}$ — exactly $2L - 1$. ✔
6. **Stim Clifford simulation.** All 10 $L$ values plus the spot
   checks at $L = 20, 30, 50$ give Bell fidelity $= 1$ exactly via
   `peek_observable_expectation` (which is exact for stabiliser
   states). ✔
7. **Statevector cross-check.** All 6 small-$L$ values give fidelity
   $= 1$ to machine precision via Qiskit's `Statevector` simulator.
   This is independent of the stim path. ✔
8. **Noise sweep integration.** `scaling_benchmark.PROTOCOLS` now
   includes MOGU as a fifth protocol (purple); the `noise_benchmark`
   helper has been extended to skip only `cluster_ladder` (whose v1
   correction map is incomplete). The full re-run produces the
   updated `simulation.log` with per-$L$ noise fidelities for all
   four working protocols. ✔
9. **Test count.** The MOGU test file adds 49 tests
   (10 connectivity + 10 no-measurement + 10 gate-count
   + 10 stim-exact + 3 large-$L$ stim + 6 statevector). All passing.
   Combined with the v1 test suite (83 tests), the project now has
   **132 passing tests**. ✔

### Minor non-blocking observations

- `mogu.py` does not invoke `resource_limits` directly; this is fine
  because the test runner and `main.py` import it once at the top of
  the executable entry point, so the limits are inherited.
- The barrier between forward and reverse sweeps is decorative —
  Qiskit's `depth()` ignores it. This matches the theoretical depth
  $L + 2$ (or $L + 1$ for odd $L$).

### Verdict (Performance Hacker)

The implementation is concise (76 lines of code), the test suite is
thorough, and the integration into the existing harness is purely
additive. **PASS.**

---

## 3. The Domain Expert (physical sanity + competition-rule alignment)

### Scope of review

- Does the new MOGU protocol respect the competition's connectivity
  constraints (top-leg only)?
- Is the noise-fidelity behaviour physically reasonable?
- Does the v2 work preserve the v1 submission bundle?

### Findings

1. **Connectivity.** MOGU touches only top-leg edges — no rungs, no
   bottom-leg gates. This satisfies the challenge's allowed-edge
   set trivially. ✔
2. **Noise behaviour.** At $p_2 = 10^{-2}$ depolarising noise on every
   2Q gate:
   - $L = 1$: $0.992$ (vs $\Phip$ unitary baseline $1 - p_2 = 0.99$,
     consistent with single-CNOT error).
   - $L = 10$: $0.885$.
   - Compare: `entanglement_swap` (v1) $0.928$, `cat_chain` $0.928$,
     `swap_chain` $0.816$.

   The MOGU result sits where the heuristic predicts (between
   `cat_chain` and `swap_chain` because MOGU has the cat-chain's
   trace-out advantage on intermediates but pays for $L - 1$ extra
   CNOTs on the disentangle). The empirical $0.885$ is slightly above
   the leading-order $(1 - p_2)^{2L-1} = 0.826$ because errors on
   intermediates that are eventually disentangled to $\ket 0$
   partially cancel under partial trace — a physical effect flagged
   by the RA Skeptic in v2 Flaw 6 and confirmed numerically. ✔
3. **Submission bundle integrity.** `./submission/` is untouched. The
   v1 PDF, code zip, and `research_process.md` remain bit-identical.
   The competition deadline (2026-05-06 23:59) was already met by v1;
   v2 is an additive technical contribution, not a submission revision. ✔
4. **Report structure.** `./report/main.pdf` now has 12 pages, with
   the new appendix (§A, sub-sections A.1–A.6) cleanly delimited from
   the body of the report. The appendix only references body sections
   that exist (verified) and concludes with implementation pointers. ✔

### Verdict (Domain Expert)

The MOGU addition is a physically meaningful, technically sound, and
honestly framed unitary alternative to the v1 protocol. It strengthens
the report's "we considered the alternatives" story without
compromising the v1 submission. **PASS.**

---

## Final verdict

All three reviewers: **PASS.**

### Summary of v2 deliverables

| Artifact | Path | Status |
|---|---|---|
| Theory draft | `./theory_draft.md` | ✔ Replaces v1; v1 archived to `theory_draft_v1.md` |
| RA critique | `./ra_critique.md` | ✔ Replaces v1; v1 archived to `ra_critique_v1.md` |
| Report (with appendix) | `./report/main.pdf` (12 pp.) | ✔ Recompiled |
| Report source | `./report/main.tex` | ✔ Appendix added |
| Implementation | `./analysis/mogu.py` | ✔ 76 lines |
| Tests | `./analysis/tests/test_mogu.py` | ✔ 49 tests, all passing |
| Scaling/noise | `./analysis/scaling_benchmark.py` | ✔ MOGU registered |
| Simulation log | `./analysis/simulation.log` | ✔ v1 archived to `simulation_v1.log` |
| Final review | `./final_review.md` (this file) | ✔ |

### Headline result

| Protocol | Depth ($L=10$) | 2Q gates ($L=10$) | Measurements | Fidelity ($p_2{=}10^{-2}$, $L=10$) |
|---|---|---|---|---|
| `entanglement_swap` (v1, with FF) | 15 | 10 | 9 | 0.928 |
| `cat_chain` (with measurement) | 22 | 10 | 9 | 0.928 |
| **`mogu` (v2, unitary)** | **12** | **19** | **0** | **0.885** |
| `swap_chain` (unitary baseline) | 29 | 28 | 0 | 0.816 |

MOGU achieves the lowest depth among the unitary protocols and avoids
all measurement; the price is roughly twice the 2Q-gate count of
`cat_chain` (because every disentanglement is paid for in CNOTs rather
than measurement+feedforward), and the noise fidelity sits between the
two measurement-based protocols and the SWAP-chain baseline.

### Innovation angle (preserved)

The v1 Innovation Award angle (constant-depth dynamic-circuit protocol
beating the Lieb-Robinson light-cone) is unchanged. The v2 contribution
strengthens the technical story by:

1. **Explicitly demonstrating** that the LOCC depth advantage of v1 is
   not an artefact — when forced to be unitary, the protocol falls back
   to $\Theta(L)$ depth, matching the Lieb-Robinson lower bound.
2. **Providing a hardware-universal fallback** for platforms without
   mature dynamic-circuit support.
3. **Halving the depth constant** of the canonical end-start GHZ-disentangle
   via the middle-out scheduling — a clean, easily-implemented
   optimisation.

### Optional follow-ups (not blocking)

- A T1/T2 decoherence noise model would make MOGU's depth disadvantage
  vs the v1 protocol numerically explicit (the depolarising channel
  used here is depth-blind under the partial-trace convention).
- Extend stim spot-checks for MOGU to $L = 100, 200$ for parity with
  the v1 large-$L$ check.
- Re-package `./submission/` with the v2 appendix included if the
  organisers issue a request for a revised submission. (Not initiated
  proactively to avoid altering the bit-identical v1 bundle.)

### Graceful termination

Phase 5 v2 is complete. Both v1 and v2 research cycles terminate
cleanly. No further phases are required unless the user requests one.
