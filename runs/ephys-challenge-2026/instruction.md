# Objective (v2 — Measurement-Free Variant)

**Add a fully unitary, measurement-free protocol that prepares `|Φ⁺⟩` on the
two far-end qubits `e0`, `e1` of the ladder QPU, complementing the existing
feed-forward entanglement-swap protocol (v1).**

The user has proposed the following idea:

> "First apply entanglement to the middle sites and distribute the
> entanglement further site-by-site until the boundaries, and then sweep
> back to cancel the entanglements."

This is a **Middle-Out Cat-Uncompute (MOCU)** protocol: a unitary
analogue of `cat_chain` (which uses X-basis measurements), with the
extra optimisation that the GHZ is grown **from the centre** so the
forward sweep depth is `≈⌈L/2⌉ + 2` instead of `L + 1`. The reverse
sweep then "shrinks" the GHZ from the inside outward, leaving `|Φ⁺⟩`
between `e0` and `e1` and `|0⟩` on every intermediate qubit.

This run is purely **additive**: do not alter the v1 submission bundle
or the existing test suite. v1 artifacts have been snapshotted to
`./instruction_v1.md`, `./theory_draft_v1.md`, `./ra_critique_v1.md`,
`./final_review_v1.md`. v1 source code in `./analysis/` is left in
place — new modules and tests are added next to it.

---

# Background: Why a Measurement-Free Variant?

The v1 entanglement-swap protocol achieves `O(1)` depth, but only
because Qiskit's dynamic-circuit primitives encode mid-circuit
measurement and classical feed-forward — a feature not yet supported on
all hardware. A **purely unitary** protocol:

- Runs on any hardware that supports nearest-neighbour CNOT (no MCM/FF).
- Has a clean closed-form correctness proof (Heisenberg-picture
  stabiliser propagation, no outcome branching).
- Pays for these wins with `O(L)` depth (Lieb-Robinson bound for unitary
  protocols on a 1D chain).

Compared to existing baselines:
- vs `swap_chain`: ~2× fewer 2Q gates (no SWAP triples).
- vs `cat_chain`: same gate count, but no measurements.
- vs `entanglement_swap` (v1): `O(L)` depth instead of `O(1)` — clear
  trade-off.

This is a **textbook unitary cat / disentangle** circuit. The main novelty
of the v2 work is the **middle-out scheduling** that halves the depth
constant, and the integration into the existing benchmarking
infrastructure for an honest side-by-side comparison.

---

# The MOCU Protocol — Definition

Top-leg qubits: `q_0 = e0, q_1, q_2, ..., q_{L-1}, q_L = e1` (so `L+1`
qubits). Choose middle `m = ⌊L/2⌋`.

## Phase A — "Plant" entanglement at the middle
- `H(q_m)` brings `q_m` into `|+⟩`.

## Phase B — "Spread" the GHZ outward (forward sweep)
- For each layer `r = 1, 2, ..., max(m, L-m)`:
  - In parallel, apply
    - `CNOT(q_{m+r-1}, q_{m+r})` if `m+r ≤ L`, and
    - `CNOT(q_{m-r+1}, q_{m-r})` if `m-r ≥ 0`.
- After this sweep, the top leg is in
  `GHZ_{L+1} = (|0...0⟩ + |1...1⟩)/√2`.

The first layer cannot do both `CNOT(q_m, q_{m-1})` and
`CNOT(q_m, q_{m+1})` in parallel because they share the control `q_m`.
So the forward depth is `1 + (max(m, L-m) + 1)` = `max(m, L-m) + 2`.

## Phase C — "Shrink" the GHZ inward (reverse sweep)
- For each layer `s = 1, 2, ..., max(m, L-m) - 1`:
  - In parallel, apply CNOTs that disentangle the innermost-still-GHZ
    qubits using their outermost-still-GHZ neighbour as control.
  - Concretely (for the symmetric case `m = ⌊L/2⌋`, `L` even):
    - layer 1: `CNOT(q_{m-1}, q_m)` — disentangles `q_m`. Single CNOT.
    - layer 2: `CNOT(q_{m-2}, q_{m-1})` and `CNOT(q_{m+1}, q_{m+2})` if
      we choose to disentangle right side now. **OR** the cleaner
      formulation below.

The cleanest reverse is **outward-from-middle, alternating**:
1. Disentangle `q_m`: `CNOT(q_{m-1}, q_m)`.
2. Disentangle `q_{m-1}` and `q_{m+1}` in parallel: `CNOT(q_{m-2}, q_{m-1})`
   and `CNOT(q_{m+2}, q_{m+1})`.
3. Disentangle `q_{m-2}` and `q_{m+2}` in parallel: `CNOT(q_{m-3}, q_{m-2})`
   and `CNOT(q_{m+3}, q_{m+2})`.
4. Continue until the only GHZ-active qubits left are `q_0` and `q_L`.

At each step, the control of the CNOT is one site further from the
boundary than the target, and that control is still in the active GHZ
(by induction). The target therefore receives the GHZ value of its
neighbour and XORs against itself, which gives `0` (because both are
equal in the GHZ branch).

Reverse sweep depth: `max(m, L-m)`.

**Total depth (asymptotic):** `2 · ⌈L/2⌉ + 2 = L + 2`.

**Total CNOT count:** `L` forward + `(L - 1)` reverse = `2L - 1`.

## Special cases
- `L = 1`: top leg `(q_0, q_1)`. `m = 0`. Just `H(q_0); CNOT(q_0, q_1)`.
  No reverse sweep needed.
- `L = 0` (degenerate, `e_0 = e_1`): not in scope.
- `L = 2`: `m = 1`. Forward: `H(q_1); CNOT(q_1, q_0)`, then
  `CNOT(q_1, q_2)`. GHZ_3. Reverse: `CNOT(q_0, q_1)`. q_1 = 0; (q_0, q_2) = `|Φ⁺⟩`.

## Stabiliser-formalism proof sketch

After Phase B, top leg is in GHZ_{L+1} with stabilisers
`{X_0 X_1 ... X_L, Z_i Z_{i+1} : i = 0..L-1}`.

We want the target stabilisers `{X_0 X_L, Z_0 Z_L, Z_1, Z_2, ..., Z_{L-1}}`
(i.e. `|Φ⁺⟩` on `(e_0, e_L)` ⊗ `|0⟩^{L-1}` on intermediates).

Each CNOT(`c`, `t`) in Phase C transforms the Pauli operators in the
Heisenberg picture: `X_c → X_c X_t`, `X_t → X_t`, `Z_c → Z_c`,
`Z_t → Z_c Z_t`. Track the stabiliser group through the reverse sweep
and verify the target group is reached.

For the reverse sweep "disentangle `q_m` first, then outward
alternating", the bookkeeping is straightforward and is presented as a
clean inductive proof in `theory_draft.md` (Phase 1 deliverable).

---

# Research Tasks

## V2-T1 — Stabiliser proof
- Derive the protocol stabiliser-by-stabiliser for `L = 4` explicitly.
- Generalise by induction on the number of disentanglement layers.
- Report in `theory_draft.md` (a new file replacing the v1 snapshot;
  the v1 snapshot is in `theory_draft_v1.md`).

## V2-T2 — Connectivity audit
- Confirm every gate uses only top-leg edges — bottom-leg and rung
  qubits are untouched in this protocol.
- The `validate_connectivity` helper in `./analysis/` already covers
  this; reuse it.

## V2-T3 — Engineering
- Implement `mocu.py` (or `unitary_chain.py`) under `./analysis/`
  with `build_circuit(L) -> QuantumCircuit`. **Do not modify**
  existing files; add a new module.
- Add unit tests under `./analysis/tests/test_mocu.py`:
  - Connectivity check for `L = 1..10`.
  - Exact Bell-fidelity = 1 via stim Clifford simulation for `L = 1..10`
    and spot checks at `L = 20, 30, 50`.
  - Statevector cross-check for `L = 1..6`.
- Register the new protocol in `scaling_benchmark.PROTOCOLS` so the
  noise + scaling sweeps include it automatically.

## V2-T4 — Comparison and write-up
- Re-run `simulation.log` with the four-protocol set extended to five.
- Add an appendix section to `./report/main.tex` documenting the new
  protocol, its stabiliser proof, and the updated benchmark figures.
- Update `./submission/research_process.md` to reflect the v2 addition
  if (and only if) the user later asks for a re-submission. For now,
  leave `submission/` untouched.

## V2-T5 — Final review
- Three-reviewer audit (Math Pedant, Performance Hacker, Domain Expert)
  of the new protocol, theorem, code, and tests. Output to
  `final_review.md` (replacing v1; v1 snapshot retained in
  `final_review_v1.md`).

---

# Workflow Cycle

Standard 5-phase company workflow. Follow `../../personas/*.md`.

1. **Phase 1 (Researcher)** → `./theory_draft.md`. Stabiliser proof of
   the MOCU protocol for arbitrary `L`. Compare gate count, depth,
   measurement count vs v1, swap_chain, cat_chain.
2. **Phase 2 (RA Skeptic)** → `./ra_critique.md`. Pedantic check of
   stabiliser propagation. Loop with Phase 1 until approval.
3. **Phase 3 (LaTeX Writer)** → append a new section to
   `./report/main.tex` ("§ A. Measurement-Free Variant — MOCU").
   Recompile `./report/main.pdf`.
4. **Phase 4 (Python Engineer)** → `./analysis/mocu.py` +
   `./analysis/tests/test_mocu.py` + register protocol in
   `scaling_benchmark.py`. Re-run `main.py`, log to
   `./analysis/simulation.log`.
5. **Phase 5 (Review Board)** → `./final_review.md`.

---

# Practical Constraints

- **Do not break v1.** The v1 submission bundle in `./submission/` is
  competition-ready and should remain bit-identical unless the user
  explicitly asks for a re-submission.
- **Additive code only.** Do not modify existing modules
  (`entanglement_swap.py`, `swap_chain.py`, `cat_chain.py`, etc.) or
  existing tests. Add new files alongside.
- **Reuse infrastructure.** `validate_connectivity`, `verification.py`,
  `stim_verification.py`, `resource_limits.py` should all work for the
  new protocol with no changes — register a new entry in the
  `PROTOCOLS` list and existing harnesses pick it up.
- **Clifford-only.** The MOCU protocol uses only `H` and `CNOT`, so
  stim simulation is exact and fast for any `L`.

---

# Success Criteria

The v2 run is complete if:

1. The MOCU protocol is proven stabiliser-correct for arbitrary `L`.
2. `test_mocu.py` passes for `L = 1..10` with Bell-fidelity = 1
   (machine precision via stim).
3. The new protocol is benchmarked side-by-side with the four existing
   baselines for depth, 2Q-gate count, and noise-fidelity at
   `p_2 = 10^{-2}` for `L = 1..10`.
4. `./report/main.pdf` includes the new appendix section.
5. `final_review.md` documents PASS from all three reviewers.

A valid partial outcome is if numerical noise robustness turns out to
be worse than measurement-based protocols — we report the honest
trade-off (Lieb-Robinson depth bound is the price of unitarity).
