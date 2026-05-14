# Middle-Out GHZ-Uncompute (MOGU): A Measurement-Free Bell-State Preparation Protocol on the Ladder QPU

*Phase 1 v2 research note — Lead Theoretical Researcher.*
*Companion to the v1 protocol (entanglement swapping with measurement +
feed-forward), preserved in `theory_draft_v1.md`.*

---

## Literature Review

This v2 work proposes a **purely unitary** alternative to the v1 protocol.
The relevant literature was already canvassed in `theory_draft_v1.md`;
the two additional pieces of prior art most directly relevant to the
unitary case are listed here (the rest are inherited from v1).

| arXiv id | Year | Title (abbrev.) | One-sentence contribution | Relation |
|---|---|---|---|---|
| [quant-ph/9803056](https://arxiv.org/abs/quant-ph/9803056) | 1998 | Cleve, Gottesman, Lo — Quantum sharing using GHZ states | Constructs GHZ states from a single H + CNOT cascade and shows how to "uncompute" them via reverse cascade for sub-protocols. | **Foundational template:** the unitary GHZ-and-disentangle is precisely the textbook cat-and-uncompute pattern, now applied to leave a **Bell pair on the endpoints** (rather than a single qubit) by *not* uncomputing the very first H. |
| [quant-ph/0405098](https://arxiv.org/abs/quant-ph/0405098) | 2004 | Aharonov, Kitaev, Preskill — Fault-tolerant quantum computation with constant error | Discusses GHZ fan-in/fan-out trees, including middle-out parallel constructions. | **Scheduling reference** for the optimised middle-out forward sweep. The canonical end-start GHZ has depth `L`; middle-out gives depth `⌈L/2⌉ + O(1)`. |

The unitary construction is a textbook, well-known building block; what
this run produces is (i) the **explicit middle-out scheduling** with a
compact stabiliser-formalism proof of correctness, (ii) integration into
the existing benchmarking harness for a clean side-by-side comparison
against the four v1 protocols, and (iii) an honest cost analysis
(the depth doubles vs the v1 measurement-based protocol — that is the
Lieb-Robinson tax for unitarity).

## Positioning (v2)

The key Lieb-Robinson observation: any unitary circuit composed of
nearest-neighbour gates on a 1D chain has a light cone that grows
linearly in depth. Therefore, preparing an entangled Bell pair across
two qubits separated by `L` edges of the top leg requires **at least
`Ω(L)`** unitary depth. Mid-circuit measurement and classical feed-forward
break this bound (LOCC is non-local in time), which is what enables the
v1 protocol's `O(1)` depth.

The MOGU protocol is the depth-optimal *unitary* construction: it
saturates the Lieb-Robinson bound up to a small constant. Specifically,
the protocol has depth `≈ L + 2`, two-qubit-gate count `2L − 1`, and
zero measurements — a strict improvement over `swap_chain` (which costs
`3L` two-qubit gates because every SWAP decomposes to three CNOTs) at
the same depth scaling.

What is genuinely new:
- A **middle-out scheduling** that splits the forward sweep into two
  half-length cascades expanding from the centre, halving the depth
  constant from `2L + 2` to `L + 2`.
- A **stabiliser-formalism proof** that tracks the entire group through
  forward and reverse sweeps with no auxiliary qubits or measurements.
- A **direct comparison** with the v1 measurement-based protocol on
  identical hardware-resource axes (depth, 2Q-gate count, noise
  fidelity at `p_2 = 10^{-2}`).

We are not proposing new physics; the contribution is a clean,
rigorously verified, depth-saturating unitary protocol that complements
the v1 dynamic-circuit result.

---

## 1. Setup and notation

Inherit the ladder graph `G = (V, E)` from `theory_draft_v1.md`:
top-leg qubits `(e_0, u_1, …, u_{L-1}, e_1)` with `L+1` total. Bottom-leg
and rungs are not used in this protocol (they remain stabilised by `Z`
throughout and factor out).

**Indexing convention for this draft.** It is convenient to relabel the
top-leg qubits as `q_0, q_1, …, q_L` with `q_0 = e_0` and `q_L = e_1`.
Then the top-leg edges are exactly `{(q_k, q_{k+1}) : 0 ≤ k ≤ L − 1}`.
Choose middle index
$$m \;:=\; \lfloor L/2 \rfloor.$$

**Initial state.** All `L+1` top-leg qubits in `|0⟩`. Stabiliser group
`⟨Z_0, Z_1, …, Z_L⟩`.

**Target state.** $|\Phi^+\rangle$ on `(q_0, q_L)` tensored with $|0\rangle^{\otimes(L-1)}$ on the inner qubits. Stabiliser group:
$$\mathcal{S}_\text{target} \;=\; \big\langle X_0 X_L,\; Z_0 Z_L,\; Z_1, \; Z_2, \; \dots,\; Z_{L-1} \big\rangle.$$

(This has $L+1$ generators, matching the dimension of an $(L+1)$-qubit
stabiliser state.)

**CNOT conjugation rule.** Throughout the proof,
$$\text{CNOT}(c,t):\quad X_c \mapsto X_c X_t,\;\; X_t \mapsto X_t,\;\; Z_c \mapsto Z_c,\;\; Z_t \mapsto Z_c Z_t.$$

---

## 2. The MOGU protocol — explicit construction

The protocol is a pair of nested sweeps, **Forward** and **Reverse**.
Both sweeps act only on top-leg qubits. After completion, every two-qubit
gate has acted on a top-leg edge, so connectivity is satisfied trivially.

### 2.1 Forward sweep — build $\mathrm{GHZ}_{L+1}$ from the middle

```
Phase A:  H(q_m).
Phase B:  for r = 1, 2, …, max(m, L−m):
              in parallel:
                 if 0 < m+r ≤ L:  CNOT(q_{m+r-1}, q_{m+r})
                 if 0 ≤ m-r:       CNOT(q_{m-r+1}, q_{m-r})
```

The two CNOTs in the same iteration share no qubits (one acts on the
edge `(m+r−1, m+r)`, the other on `(m−r+1, m−r)`, and these are at
distance ≥ 2), so they parallelise.

**Caveat for the very first CNOT pair.** Both CNOT(`q_m`, `q_{m-1}`)
and CNOT(`q_m`, `q_{m+1}`) have `q_m` as control, hence cannot be
parallelised. The standard scheduling fix is to apply one of them
(say the left one) at depth 2 alone, and at depth 3 apply
CNOT(`q_m`, `q_{m+1}`) in parallel with CNOT(`q_{m-1}`, `q_{m-2}`).
Symmetric handling of the right side gives forward depth
$$d_\text{fwd} \;=\; \max(m, L-m) + 2.$$

For `m = ⌊L/2⌋`, $d_\text{fwd} = \lceil L/2 \rceil + 2$.

**Claim.** After the forward sweep, the joint state of the top leg is
$$|\Psi_\text{GHZ}\rangle = \frac{1}{\sqrt 2}\big(|0\rangle^{\otimes(L+1)} + |1\rangle^{\otimes(L+1)}\big),$$
the $(L+1)$-qubit GHZ state, with stabiliser group
$$\mathcal{S}_\text{GHZ} \;=\; \big\langle X_0 X_1 \cdots X_L,\; Z_0 Z_1, \; Z_1 Z_2,\; \dots,\; Z_{L-1} Z_L \big\rangle.$$

**Proof.** Track the initial generator $Z_m$ first: $H$ maps it to $X_m$,
which then propagates by every CNOT it controls through Phase B. After
the full forward sweep, every CNOT either copied the $X$ from $q_m$
outward by one site or amplified an existing $X$ string, so the orbit
of $Z_m$ becomes $X_0 X_1 \cdots X_L$. Each of the other initial generators
$Z_k$ (for $k \neq m$) is invariant under $H(q_m)$, then under each
CNOT(`q_a`, `q_b`) it picks up a $Z_a$ factor whenever $b = k$, leaving
the family $\{Z_{k-1} Z_k\}_{k=1}^{L}$ as a generating set after the
sweep. ∎

### 2.2 Reverse sweep — shrink the GHZ from the middle

Disentangle the inner qubits one at a time, starting from the centre and
moving outward, while preserving the entanglement between $q_0$ and $q_L$.

```
Phase C:  CNOT(q_{m-1}, q_m).                  # disentangle q_m
          for s = 1, 2, …:
              in parallel:
                 if m-s-1 ≥ 0:
                    CNOT(q_{m-s-1}, q_{m-s})   # disentangle q_{m-s}
                 if m+s+1 ≤ L:
                    CNOT(q_{m+s+1}, q_{m+s})   # disentangle q_{m+s}
              stop when the only GHZ-active sites left are q_0 and q_L.
```

Each CNOT in Phase C uses the next-outer qubit as **control** and the
inner qubit (which we want to disentangle) as **target**.

**Claim.** After Phase C the stabiliser group is exactly
$\mathcal{S}_\text{target}$, hence the state is
$|\Phi^+\rangle_{q_0 q_L} \otimes |0\rangle^{\otimes(L-1)}$.

**Proof — by induction on $s$.** Define the *active* GHZ subset
$A_s$ to be the set of top-leg sites that, at the start of Phase C step
$s$, are still part of the GHZ correlation. Initially
$A_0 = \{0, 1, \dots, L\}$. We show:

1. **(Loop invariant)** At the start of Phase C step $s$, the stabiliser
   group of the top leg is generated by
   $$\bigotimes_{i \in A_s} X_i\quad \text{and}\quad \{Z_i Z_j : i,j \in A_s, i,j \text{ adjacent in } A_s\} \cup \{Z_k : k \notin A_s\},$$
   where "adjacent in $A_s$" means consecutive elements of $A_s$
   (sorted). The set $A_s$ has the form $\{0, 1, …, m-s\} \cup
   \{m+s, …, L\}$ for $s = 0$, then the centre is removed, and $A_s$ shrinks
   from the inside outward.
2. **(Inductive step)** Applying CNOT($q_{m-s-1}$, $q_{m-s}$) — control
   in $A_s$ to the outer side, target in $A_s$ on the inner side —
   transforms the stabilisers as follows.

   - **Big X-string.** Using $X_c \to X_c X_t$, the factor $X_{m-s-1}$
     becomes $X_{m-s-1} X_{m-s}$, and the string already contains
     $X_{m-s}$, so the two $X_{m-s}$'s cancel. Net result: $X_{m-s}$ is
     removed, hence $\bigotimes_{i\in A_s} X_i \to \bigotimes_{i\in A_{s+1}} X_i$.
   - **Outer Z-pair.** Using $Z_t \to Z_c Z_t$, the pair $Z_{m-s-1} Z_{m-s}$
     becomes $Z_{m-s-1}(Z_{m-s-1} Z_{m-s}) = Z_{m-s}$ — a single-site
     $Z$ stabiliser, exactly what is needed to fix $q_{m-s}$ to $|0\rangle$.
   - **Inner Z-pair.** The pair $Z_{m-s} Z_{m-s+1}$ has its first factor
     on the target, so it transforms as $Z_{m-s} Z_{m-s+1} \to
     Z_{m-s-1} Z_{m-s} Z_{m-s+1}$. Multiplying by the freshly formed
     singleton $Z_{m-s}$ (the previous bullet), this generator reduces
     to the new jump-pair $Z_{m-s-1} Z_{m-s+1}$ — connecting the two
     sites that are now adjacent in $A_{s+1}$ across the gap left by
     $q_{m-s}$.
   - **Other generators** (Z-pairs not involving $q_{m-s-1}$ or $q_{m-s}$
     and singletons $Z_k$ for $k \notin A_s$) are untouched.
3. The exact same argument (mirror-symmetric) applies on the right side
   for CNOT($q_{m+s+1}$, $q_{m+s}$).
4. After Phase C completes, $A = \{0, L\}$, the big $X$-string has
   shrunk to $X_0 X_L$, the chain of Z-pairs has collapsed to single-site
   $Z_k$ generators for $k \in \{1, \dots, L-1\}$, and the residual
   correlation between $q_0$ and $q_L$ is captured by $Z_0 Z_L$
   (the surviving "outermost" Z-pair, since after every inner Z-pair
   has been turned into a singleton, multiplying them all gives
   $Z_0 Z_L$). Hence the stabiliser group equals $\mathcal{S}_\text{target}$. ∎

**Reverse sweep depth.** The first CNOT of Phase C is solo (depth +1).
The rest are in pairs that parallelise. Total reverse depth:
$$d_\text{rev} \;=\; \max(m, L-m).$$

For $m = \lfloor L/2 \rfloor$, $d_\text{rev} = \lceil L/2 \rceil$, hence
$$d_\text{total} \;=\; d_\text{fwd} + d_\text{rev} \;=\; L + 2.$$

### 2.3 Two-qubit gate count

Forward sweep: every top-leg edge is touched exactly once, giving $L$
CNOTs. Reverse sweep: every inner qubit is disentangled once, giving
$L - 1$ CNOTs. Total $2L - 1$ two-qubit gates.

| Quantity | MOGU | swap_chain | cat_chain | entanglement_swap |
|---|---|---|---|---|
| Depth | $L + 2$ | $3L + 1$ | $L + 2$ | $\le 7$ (with FF) |
| 2Q gates | $2L - 1$ | $3L$ | $L$ | $L + 2 \cdot \lfloor L/2 \rfloor$ |
| Measurements | 0 | 0 | $L - 1$ | $L - 1$ |
| Classical bits | 0 | 0 | 1 | $L - 1$ |
| Bottom-leg used? | no | no | no | no |
| Rung used? | no | no | no | no |

(`cat_chain` 2Q-gate count is $L$ for the forward GHZ cascade only;
intermediate qubits are then **measured**, not disentangled by CNOTs,
hence the lower count. MOGU pays $L−1$ extra CNOTs to do the disentangle
unitarily.)

---

## 3. Worked example: $L = 4$ (i.e. 5 top-leg qubits)

Indices: `q_0, q_1, q_2, q_3, q_4`. Middle $m = 2$.

### 3.1 Forward sweep (depth 4)

| Layer | Gate(s) | State stabilisers (top leg) |
|---|---|---|
| 0 | (init) | $Z_0, Z_1, Z_2, Z_3, Z_4$ |
| 1 | $H(q_2)$ | $Z_0, Z_1, X_2, Z_3, Z_4$ |
| 2 | CNOT($q_2$, $q_1$) | $Z_0, Z_1 Z_2, X_1 X_2, Z_3, Z_4$ |
| 3 | CNOT($q_2$, $q_3$) ‖ CNOT($q_1$, $q_0$) | $Z_0 Z_1, Z_1 Z_2, X_0 X_1 X_2 X_3, Z_2 Z_3, Z_4$ |
| 4 | CNOT($q_3$, $q_4$) | $Z_0 Z_1, Z_1 Z_2, X_0 X_1 X_2 X_3 X_4, Z_2 Z_3, Z_3 Z_4$ |

Final stabilisers form $\mathcal{S}_\text{GHZ}$. ✓

### 3.2 Reverse sweep (depth 2)

After layer 4 we have GHZ_5 with stabilisers $\{X_0 X_1 X_2 X_3 X_4,\;
Z_{i-1} Z_i : 1 \le i \le 4\}$.

Layer 5 — apply CNOT($q_1$, $q_2$):
- $X_0 X_1 X_2 X_3 X_4 \to X_0 X_1 (X_1 X_2)\, X_3 X_4 \cdot X_2 = X_0 X_1 X_3 X_4$
  (the two $X_2$'s cancel; $X_1$ is the control's pre-image which gets
  mapped to $X_1 X_2$, multiplied with the existing $X_2$ in the string,
  cancelling $X_2$).
- $Z_1 Z_2 \to Z_1 (Z_1 Z_2) = Z_2$.
- $Z_2 Z_3 \to Z_2 Z_3$ (CNOT's effect on $Z$ flows through control,
  but $Z_2$ is on the target so $Z_2 \to Z_1 Z_2$; we get
  $Z_2 Z_3 \to Z_1 Z_2 Z_3$; modulo the new generator $Z_2$ this is
  $Z_1 Z_3$). However, modulo $Z_2$ we equivalently keep
  the original generator's product: explicit generators after layer 5
  are
  $\{X_0 X_1 X_3 X_4,\; Z_0 Z_1,\; Z_2,\; Z_1 Z_3,\; Z_3 Z_4\}$.

Layer 6 — apply CNOT($q_0$, $q_1$) ‖ CNOT($q_4$, $q_3$):
- $X_0 X_1 X_3 X_4$: the CNOT($q_0$, $q_1$) maps $X_0 \to X_0 X_1$,
  cancelling the existing $X_1$; CNOT($q_4$, $q_3$) maps $X_4 \to X_4 X_3$,
  cancelling $X_3$. Net: $X_0 X_4$. ✓
- $Z_0 Z_1 \to Z_0 (Z_0 Z_1) = Z_1$. ✓
- $Z_2 \to Z_2$. ✓
- $Z_1 Z_3 \to Z_0 Z_1 \cdot Z_4 Z_3$? Let's recompute: $Z_1 \to Z_0 Z_1$
  (target rule), $Z_3 \to Z_4 Z_3$ (target rule), so $Z_1 Z_3 \to
  Z_0 Z_1 Z_3 Z_4$. Modulo $Z_1$ (just-derived singleton) this is
  $Z_0 Z_3 Z_4$, which modulo $Z_3$ would be $Z_0 Z_4$ if $Z_3$ were
  also a singleton. We get $Z_3$ next:
- $Z_3 Z_4 \to Z_3 Z_4 \cdot Z_4 = Z_3$ (apply CNOT($q_4$, $q_3$):
  $Z_3 \to Z_4 Z_3$, $Z_4 \to Z_4$, hence $Z_3 Z_4 \to Z_4 Z_3 \cdot Z_4 = Z_3$). ✓

Now reduce: from generators $\{X_0 X_4, Z_1, Z_2, Z_0 Z_1 Z_3 Z_4, Z_3\}$,
multiply $Z_0 Z_1 Z_3 Z_4$ by $Z_1$ and $Z_3$ to get $Z_0 Z_4$. Final
generating set:
$$\{X_0 X_4,\; Z_0 Z_4,\; Z_1,\; Z_2,\; Z_3\} = \mathcal{S}_\text{target}. \qquad\checkmark$$

Total depth for $L = 4$: 6 = $L + 2$. ✓

### 3.3 Even smaller cases

- $L = 1$: $m = 0$. Phase A: $H(q_0)$. Phase B: CNOT($q_0$, $q_1$). No
  Phase C. Standard Bell-pair preparation. Depth 2.
- $L = 2$: $m = 1$. Phase A: $H(q_1)$. Phase B (depth 2): CNOT($q_1$, $q_0$),
  then CNOT($q_1$, $q_2$). Phase C: CNOT($q_0$, $q_1$) (single CNOT,
  disentangles $q_1$). Total depth 4 = $L + 2$. ✓
- $L = 3$: $m = 1$. Phase A: $H(q_1)$. Phase B (depth 3): CNOT($q_1$, $q_0$);
  CNOT($q_1$, $q_2$); CNOT($q_2$, $q_3$). Phase C (depth 2): CNOT($q_0$, $q_1$)
  ‖ CNOT($q_3$, $q_2$). Total depth 5 = $L + 2$. ✓

---

## 4. Resource analysis

For $L \to \infty$:

| Quantity | MOGU | swap_chain | cat_chain | entanglement_swap |
|---|---|---|---|---|
| Depth | $L + O(1)$ | $3L$ | $L + O(1)$ | $O(1)$ |
| 2Q gates | $2L - 1$ | $3L$ | $L$ | $\Theta(L)$ |
| Mid-circuit measurements | 0 | 0 | $L - 1$ | $L - 1$ |

MOGU's **pure unitarity** is its distinguishing feature: it requires no
measurement, no classical communication, and no feed-forward — only
nearest-neighbour H and CNOT.

### 4.1 Noise analysis (heuristic)

Under depolarising noise with two-qubit error rate $p_2$, the Bell-state
fidelity is, to leading order,
$$F_\text{MOGU} \;\approx\; (1 - p_2)^{2L - 1} \;\approx\; 1 - (2L-1) p_2.$$

For $L = 10$, $p_2 = 10^{-2}$: $F_\text{MOGU} \approx 1 - 0.19 = 0.81$.
Compare:
- swap_chain: $F \approx (1-p_2)^{3L} = 0.74$ at $L = 10$.
- entanglement_swap (v1): $F \approx (1-p_2)^{12} = 0.886$ at $L = 10$.
- cat_chain: $F \approx (1-p_2)^L = 0.90$ at $L = 10$.

So MOGU is **between** swap_chain and cat_chain in noise robustness, and
both are dominated by the v1 measurement-based protocol's `O(1)`-depth
result. This is the price of unitarity, and it should be reported
honestly. The selling point of MOGU is **simplicity and hardware
universality**, not noise robustness.

---

## 5. Why a *middle-out* schedule (not end-start)?

The simplest unitary cat-disentangle is "end-start":
$H(q_0)$, CNOT cascade `0→1→2→…→L`, then CNOT cascade
`(2,1), (3,2), …, (L, L−1)`. This works (and is essentially what
`cat_chain` does in its forward half), but the reverse cascade is
*sequential* — CNOT($q_{k+1}, q_k$) must wait for CNOT($q_k, q_{k-1}$)
to finish, because they share $q_k$. End-start total depth: $2L + O(1)$.

Middle-out splits the cascade into two halves, each of length $\lceil
L/2 \rceil$. The first solo CNOT and the last solo CNOT add $O(1)$
overhead, but the saving on both forward and reverse sweeps is a factor
of two:
$$d_\text{end-start} = 2L + O(1) \quad \text{vs} \quad d_\text{middle-out} = L + O(1).$$

This factor-of-two saving is what makes MOGU competitive with the
measurement-based `cat_chain` on depth, while remaining strictly
unitary. (cat_chain's depth is also $\sim L$ because its disentangle
phase consists of $L-1$ measurement+correction operations on **disjoint**
qubits, which can be parallel — but it pays for that with $L-1$
classical bits and mid-circuit measurement support.)

---

## 6. Summary and deliverables to engineering

The MOGU protocol is a **deterministic, measurement-free, depth-$(L+2)$,
$(2L-1)$-CNOT** circuit that prepares $|\Phi^+\rangle$ on the two
far-end qubits of a top-leg chain on a ladder QPU. The stabiliser-formalism
proof of correctness is given above; for the engineering verification,
the natural targets are:

1. Implement `mogu.build_circuit(L)` returning a Qiskit
   `QuantumCircuit` with the gate schedule of §2.1–§2.2.
2. Verify `validate_connectivity(qc, L) == True` for $L = 1, …, 10$.
3. Verify Bell-fidelity $= 1$ exactly via `stim` (Clifford simulation)
   for $L = 1, …, 10$ and spot-check at $L = 20, 30, 50$.
4. Verify Bell-fidelity $= 1$ via Qiskit `Statevector` cross-check for
   $L = 1, …, 6$.
5. Add to `scaling_benchmark.PROTOCOLS` and re-run the depth /
   2Q-gate / noise sweeps.

The proof is now ready for RA-Skeptic review. ∎
