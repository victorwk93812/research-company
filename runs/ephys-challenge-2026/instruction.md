# Objective

**Design, verify, and implement a scalable quantum circuit that places the two far-end qubits `e0` and `e1` of a two-legged ladder QPU into a Bell state, using only single-qubit gates and nearest-neighbour two-qubit gates.**

Our proposition (from `prompt.txt`) is to use **entanglement swapping along the top leg** rather than a plain SWAP chain or a GHZ-style unitary. The research run must prove this works, identify which Bell state it produces, generalise it to arbitrary ladder length `L`, benchmark it against baselines, and submit to the 2026 Institute of Applied Physics / NCCU challenge (see `2026_challenge_markdown/2026_challenge.md`).

**Key dates:**
- Preliminary submission deadline: **2026-05-06, 23:59**
- Finals (oral presentation): **2026-05-16** at NCCU IIR Campus
- Today: 2026-04-23 → ~2 weeks to preliminary, ~3 weeks to finals.

---

# Background: The QPU Layout

From Fig. 1 of the challenge (L=5 example, 12 qubits total):

```
  e0 ─── 1 ─── 2 ─── 3 ─── 4 ─── e1     (top leg)
         │     │     │     │
   6 ─── 7 ─── 8 ─── 9 ──── 10 ── 11    (bottom leg)
```

**Connectivity rules (what two-qubit gates are allowed):**
- Top leg: `e0–1, 1–2, 2–3, 3–4, 4–e1` (and analogously for longer `L`).
- Bottom leg: `6–7, 7–8, 8–9, 9–10, 10–11`.
- Rungs: `1–7, 2–8, 3–9, 4–10` — **only between the inner columns**, NOT between the end columns (`e0` is *not* connected to `6`, and `e1` is *not* connected to `11`).

**Scalable layout (Fig. 2):** for generic `L` the top leg has `L+1` qubits (`e0`, `u_1`, …, `u_{L-1}`, `e1`), the bottom leg has `L+1` qubits (`v_0`, `v_1`, …, `v_L`), and rungs exist only at `u_i–v_i` for `i = 1, …, L-1`. Let `N = L-1` be the number of inner top qubits.

---

# Our Starting Proposition (from `prompt.txt`)

Entanglement-swapping chain along the top leg:

1. **Prepare Bell pairs on alternating links:** apply `H` and `CNOT` to create $|\Phi^+\rangle$ on `(e0, 1)`, `(2, 3)`, `(4, e1)` — three independent Bell pairs, all formed with nearest-neighbour gates (legal).
2. **Couple the chain via Bell-type gates:** apply further gates to entangle `(1, 2)` and `(3, 4)` — these are nearest neighbours on the top leg (legal).
3. **Bell-measure the intermediate qubits:** measure qubits `1, 2, 3, 4` (or equivalently perform Bell measurements on pairs `(1,2)` and `(3,4)`); apply conditional single-qubit `X`/`Z` corrections on `e0` and `e1` to collapse the residual state into a definite Bell state.

This is textbook repeater-style entanglement swapping, specialised to four intermediate qubits. The research tasks below verify it is correct for `N=4`, extend it to arbitrary `N`, identify the output Bell state, and benchmark.

---

# Research Tasks

## T1 — Analytical verification for the N=4 case

- Using the stabiliser formalism (not brute-force statevector), explicitly track the stabiliser generators through every step of the protocol.
- Determine which of $\{|\Phi^\pm\rangle,|\Psi^\pm\rangle\}$ the pair `(e0, e1)` lands in as a function of the four measurement outcomes $m_1, m_2, m_3, m_4 \in \{0,1\}$.
- Derive the deterministic feed-forward correction map $(m_1,m_2,m_3,m_4) \mapsto (X^a Z^b)_\text{e0} \otimes (X^c Z^d)_\text{e1}$ that lands the pair in $|\Phi^+\rangle$ for every outcome.
- Pin down **exactly which gate sequence** implements step 2 ("entangle (1,2), (3,4)") — the prompt is ambiguous. Likely choice: `CNOT(1,2); CNOT(3,4); H(1); H(3)` followed by Z-basis measurement, which is the standard Bell-basis measurement on each pair. State this clearly.

## T2 — Generalisation to arbitrary L (the hard research problem)

- For **`N` even** (`L` odd): the proposition generalises directly — prepare Bell pairs on `(e0, u_1), (u_2, u_3), …, (u_{N-2}, u_{N-1}), (u_N, e1)` and Bell-measure the `N/2` inner pairs `(u_1,u_2), (u_3,u_4), …, (u_{N-1},u_N)`. Prove correctness by induction on the number of swap links.
- For **`N` odd** (`L` even): the simple top-leg chain breaks the pairing. Find a fix. Candidate strategies:
  - (a) Use one bottom-leg qubit via a rung to "absorb" the parity mismatch.
  - (b) Prepare a 3-qubit GHZ link instead of a Bell pair at one spot in the chain and teleport through it.
  - (c) Use a **different topology** entirely, e.g. route through the bottom leg for a segment.
- Compare these fixes on depth, gate count, and classical-communication overhead.
- For **`N = 0`** (`L = 1`, `e0` directly adjacent to `e1`): trivial — one `H + CNOT`. Include as base case.

## T3 — Circuit depth and resource analysis

- **Depth (with mid-circuit measurement + classical feed-forward):** show the protocol is `O(1)` in depth — all Bell pairs can be prepared in parallel, all Bell measurements can be performed in parallel, feed-forward corrections are single-qubit. This is the **key selling point** vs a SWAP chain.
- **Depth (unitary-only, post-selected):** if we defer all measurements to the end, depth is still `O(1)` but success probability drops as $4^{-N/2}$.
- **Depth (unitary-only, non-post-selected):** convert Bell measurements into coherent unitary corrections controlled on the intermediate qubits — gives a direct `O(L)` depth circuit; compare with SWAP-chain `O(L)`.
- **Gate count:** for each variant, tabulate total 1Q and 2Q gates as a function of `L`.
- **Measurement count:** `N = L-1` mid-circuit measurements + `2 × number_of_measurements` bits of classical feed-forward.

## T4 — Baseline protocols to compete against (so we know our solution is non-trivial)

Implement and benchmark all of the following in simulation:

1. **SWAP-chain:** prepare Bell pair on `(e0, 1)`, then SWAP `1↔2, 2↔3, …, (N-1)↔N, N↔e1`. Depth `O(L)`, no measurements.
2. **Direct unitary creation:** `H(e0)` then a chain of CNOTs `e0→1, 1→2, …, N→e1` (a cat / GHZ-like state on the whole chain), then disentangle the middle via further CNOTs. This is the "brute-force" answer.
3. **Our entanglement-swapping protocol** (the main contribution): depth `O(1)` with feed-forward.
4. **Measurement-based ladder variant** (optional, stretch goal): use the bottom-leg ancillas to prepare a cluster state on the ladder and teleport `e0` to `e1` through it. Might give a cleaner circuit for `N` odd.

## T5 — Python simulation (under `./analysis/`, managed with `uv`)

- `uv init` under `./analysis/`. Add deps: `qiskit` (primary), `numpy`, `scipy`, `matplotlib`. Optional: `stim` for fast stabiliser simulation of our protocol.
- Resource block at the top of every entry script (16 GB mem cap, 4 BLAS threads) — see Engineer persona.
- Implement each protocol (T4 items 1–4) as a function `build_circuit(L) -> QuantumCircuit`.
- **Verification routine:** for every `L` from 1 to 10 (and at least one spot-check at `L=50` via stabiliser simulation), confirm that after applying the circuit + feed-forward corrections, the reduced density matrix on `(e0, e1)` has fidelity $>1-10^{-9}$ with the declared target Bell state. Average over all $2^N$ measurement outcome branches.
- Verify on both Qiskit's `AerSimulator` (shot-based) and `Statevector` (exact, for small `L`).
- Export circuit diagrams (`qc.draw("mpl")`) under `./analysis/figures/`.
- Log a full sweep to `./analysis/simulation.log`.

## T6 — Final Bell state identification

- State clearly in the report which Bell state the protocol targets (expected: $|\Phi^+\rangle$ after Pauli corrections, but verify).
- Show the explicit derivation of the outcome-to-correction map from T1.

## T7 — Robustness sanity-checks (stretch)

- How does the protocol perform under a depolarising noise channel on each gate (`p ∈ {10^{-3}, 10^{-2}}`)? Does the `O(1)` depth advantage translate into higher fidelity than the SWAP-chain baseline at realistic noise levels? This is probably the strongest "Innovation Award" angle.

---

# Directory Layout (overrides CLAUDE.md default `./src/`)

Because the user specified `analysis/` for Python work, the per-run layout is:

```
runs/ephys-challenge-2026/
  prompt.txt                       (source proposition)
  instruction.md                   (this file)
  2026_challenge_markdown/         (challenge statement, Chinese + English)
  theory_draft.md                  (Phase 1, Researcher)
  ra_critique.md                   (Phase 2, RA Skeptic)
  report/
    main.tex                       (Phase 3, LaTeX Writer)
    main.pdf
    figures/                       (circuit diagrams, depth/fidelity plots)
  analysis/                        (Phase 4, Python Engineer — replaces ./src/)
    pyproject.toml                 (uv-managed)
    uv.lock
    swap_chain.py
    cat_chain.py
    entanglement_swap.py           (our protocol)
    cluster_ladder.py              (stretch)
    verification.py                (fidelity checks, outcome-averaging)
    scaling_benchmark.py           (depth / gate count vs L)
    tests/
      test_small_L.py
      test_stabiliser_consistency.py
      test_feedforward.py
    figures/
    simulation.log
  final_review.md                  (Phase 5, Review Board)
  submission/                      (the competition-ready bundle)
    submission.pdf                 (copy of report/main.pdf + cover)
    circuit_diagram.pdf
    code_bundle.zip
    research_process.md            (required by competition rules §3)
```

---

# Deliverables (per CLAUDE.md workflow, adapted for competition)

## Phase 1 — Researcher → `./theory_draft.md`
- Brief literature note on entanglement swapping (Żukowski–Zeilinger 1993, quantum repeater literature — Briegel–Dür–Cirac–Zoller 1998 — and measurement-based quantum computing on cluster states). arXiv MCP search required.
- Formal definition of the ladder graph, connectivity, and notation.
- Full stabiliser-formalism derivation for T1.
- Inductive generalisation for T2 (even case) and at least one candidate fix for the odd case.
- Depth / gate-count analysis from T3.
- Positioning: why is this more interesting than SWAP-chain or GHZ disentangle?

## Phase 2 — RA Skeptic → `./ra_critique.md`
- Independent literature check (NCCU is a Taiwan institution; this is a student-level open challenge — check if entanglement swapping solutions have been proposed for ladder QPUs previously in e.g. IBM / Quantinuum architecture papers).
- Pedantic review of the stabiliser derivation — are all signs correct, all Pauli corrections consistent?
- Check the odd-`N` fix works; call out if T2 has gaps.
- Check that **every** two-qubit gate in every protocol uses only listed edges — connectivity violations would disqualify us.
- Loop with Phase 1 until approval.

## Phase 3 — LaTeX Writer → `./report/main.tex`
- XeLaTeX, `revtex4-2` or `article` with physics preamble.
- Fig. 1 reproduction of the ladder, Fig. 2 circuit diagram, Fig. 3 depth/fidelity plots.
- Must satisfy competition rule §3 required items: circuit diagram + explanation, declared Bell state, verification method, code description, research-process description.

## Phase 4 — Python Engineer → `./analysis/`
- `uv init` the project. Pinned Python 3.11+.
- Implement all four protocols from T4, run the verification sweep from T5.
- Unit tests in `./analysis/tests/`. Golden tests: Bell fidelity > 1 − 10⁻⁹ averaged over outcomes for `L ∈ {1, …, 10}`.
- Pipe full run to `./analysis/simulation.log`.

## Phase 5 — Review Board → `./final_review.md`
- Math pedant: stabiliser derivation correctness.
- Performance hacker: code cleanliness, vectorisation, correctness of feed-forward logic.
- Domain expert: does the submitted package satisfy every line-item in competition rule §3?
- Produce the final `submission/` bundle ready for upload to the registration URL.

---

# Practical Constraints

- **Language / tooling:** Python 3.11+, Qiskit 1.x or 2.x (state at run time), managed by `uv`. No conda, no pip-install-requirements.
- **Memory:** 16 GB cap; 4 BLAS threads. Enforced by the standard resource block.
- **Reproducibility:** log git commit, `qiskit.__version__`, seed all RNGs.
- **Connectivity hygiene:** every 2Q gate must be checked against the allowed edge list before any simulation — write a `validate_connectivity(qc, edges)` helper and call it on every constructed circuit.
- **Competition hygiene:** §3 of the rules says "if AI tools are used, they should be explicitly listed, and the scope of use explained." The `research_process.md` in `submission/` must declare Claude Code's role honestly.
- **Scope discipline:** do NOT attempt to solve this on a real IBM / Quantinuum device — simulation only, unless a free-tier backend is available and noise benchmarking (T7) specifically calls for it.

---

# Success Criteria

The run is a competition-ready success if by end of Phase 5:

1. The entanglement-swapping protocol is proven (both stabiliser-analytically and numerically) to produce a named Bell state on `(e0, e1)` for `L ∈ {1, …, 10}` with fidelity > 1 − 10⁻⁹.
2. A correctness argument covers **all `L ≥ 1`**, including odd `N`.
3. Depth, gate count, and measurement count are tabulated vs `L` and compared against the SWAP-chain and cat-chain baselines.
4. The `submission/` directory is complete and satisfies every bullet under competition rule §3.
5. An "Innovation Award" angle is explicitly highlighted in the report — either the `O(1)` depth advantage under feed-forward, the noise-fidelity comparison (T7), or the cluster-state variant (T4 stretch).

A valid partial outcome (still worth submitting) is if the odd-`N` fix turns out to cost extra depth — we report the honest tradeoff.
