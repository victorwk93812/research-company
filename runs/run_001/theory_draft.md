# Theory Draft: SU(2)-Symmetric Block-Sparse Tensor Contractions for DMRG on the Heisenberg Chain

## Literature Review

The following arxiv records were consulted via the `arxiv` MCP server (sort=relevance; abstracts read for all; methodology sections pulled via `download_paper_text` for the two load-bearing references).

| arXiv id | Title (shortened) | One-sentence contribution | Relation to this proposal |
|---|---|---|---|
| cond-mat/0012319 (McCulloch & Gulácsi, 2001) | The Non-Abelian DMRG Algorithm | First DMRG formulation in which block states are full SU(2) multiplets; single-site reduced matrix elements with 9j couplings. | **Building block.** Our block-sparse structure is exactly the reduced-multiplet representation derived there. |
| cond-mat/0701428 (McCulloch, 2007) | From DMRG to Matrix Product States | Re-derives DMRG in MPS language and shows how abelian and non-abelian symmetries impose block-sparsity on MPS tensors. | **Building block.** Provides the MPS-level formulation we adopt for the two-site update. |
| 1202.5664 (Weichselbaum, 2012) | Non-abelian symmetries in tensor networks: QSpace | Unified "QSpace" data structure: reduced multiplet tensor + Clebsch-Gordan tensor factorized as a tensor product. | **Building block.** Our memory model for storing symmetric tensors follows the multiplet/CGC split. |
| 1910.13736 (Weichselbaum, 2019) | X-Symbols for Non-Abelian Symmetries | Tabulates contraction coefficients of CGT pairs so CGCs never need to be stored explicitly. | **Extends** the QSpace approach — we note it as the route to avoid CGC storage overhead in production codes; our prototype still stores CGCs but we flag the extension. |
| 1203.2222 (Singh, 2012 thesis) | Tensor Networks with Abelian and Non-Abelian Symmetries | Pedagogical derivation of symmetric MPS/PEPS; U(1) and SU(2) worked out in full. | **Independent baseline** for our derivation; we cross-checked our block-sparse selection rules against this reference. |
| 1501.05504 (Hubig, McCulloch, Schollwöck, Wolf, 2015) | Single-Site DMRG with Subspace Expansion | Strictly single-site DMRG + subspace expansion, compatible with non-abelian symmetries. | **Independent baseline.** Shows that the non-abelian framework is not restricted to two-site DMRG. |
| 1708.09213 (Ran et al., 2017) | Lecture Notes of Tensor Network Contractions | Broad review of TN contraction algorithms and their complexity. | **Independent baseline** for dense U(1) complexity estimates used in §6. |

## Positioning

Everything in §3–§5 below is a re-derivation from the references above — there is no new physics. What is **new in this proposal** is (i) a self-contained memory-scaling argument that makes the reduction-per-symmetry-sector explicit for the spin-1/2 antiferromagnetic Heisenberg chain at fixed bond dimension, parameterised so the prototype in Phase 4 can directly verify it, and (ii) a concrete, minimal block-sparse data structure (a dict keyed by conserved charges, with reduced multiplet blocks only) that gives the SU(2) reduction without needing Clebsch-Gordan tensors to be stored explicitly in the memory accounting. We build on (not replace) McCulloch–Gulácsi (arXiv:cond-mat/0012319) and Weichselbaum's QSpace (arXiv:1202.5664); the X-symbol optimisation (arXiv:1910.13736) is acknowledged but out of scope for the prototype.

---

## 1. Setup: Hilbert space, Hamiltonian, and symmetries

Consider the isotropic spin-1/2 antiferromagnetic Heisenberg chain on $L$ sites with open boundary conditions:

$$
\hat H \;=\; J \sum_{i=1}^{L-1} \hat{\mathbf S}_i \cdot \hat{\mathbf S}_{i+1}, \qquad J>0.
$$

The local Hilbert space is $\mathcal H_i \cong \mathbb{C}^2$, spanned by $|\tfrac12, \pm\tfrac12\rangle$. The full Hilbert space is $\mathcal H = \bigotimes_{i=1}^L \mathcal H_i$, with $\dim\mathcal H = 2^L$.

**Symmetries.** $\hat H$ commutes with every component of $\hat{\mathbf S}_{\rm tot} = \sum_i \hat{\mathbf S}_i$. Hence the global symmetry is the full non-abelian $SU(2)$, with two labels per basis state:
- the multiplet (total spin) $S \in \{0, \tfrac12, 1, \tfrac32, \ldots\}$, and
- the $z$-projection $M \in \{-S, -S+1, \ldots, +S\}$.

A dense U(1) simulation retains only $\hat S_{\rm tot}^z$ conservation (labels $M$). An SU(2) simulation retains both, and block-diagonalises by $S$.

**Boundary conditions.** Open; this is what DMRG uses.

## 2. MPS and bond-dimension conventions

An MPS on $L$ sites is

$$
|\Psi\rangle \;=\; \sum_{\{s_i\}} A^{s_1} A^{s_2} \cdots A^{s_L} \, |s_1 s_2 \cdots s_L\rangle,
$$

with $A^{s_i} \in \mathbb C^{\chi_{i-1} \times \chi_i}$ and $s_i \in \{\uparrow, \downarrow\}$. We fix a uniform maximum bond dimension $\chi$. In the middle of the chain, both virtual legs have dimension $\chi$.

## 3. U(1) block structure (baseline)

Under $\hat S_{\rm tot}^z$ conservation, the left virtual leg carries a charge $M_L \in \tfrac12 \mathbb Z$, the physical leg carries $m \in \{\pm\tfrac12\}$, and the right leg carries $M_R$. Non-zero entries of $A^{s}$ are constrained by the selection rule

$$
M_R \;=\; M_L + m.
$$

Group the bond Hilbert space by charge: $\chi = \sum_{M} d_M^{\mathrm{U(1)}}$, where $d_M^{\mathrm{U(1)}}$ is the number of basis states with $\hat S_z = M$ on the bond. The MPS tensor block-diagonalises into blocks $A^{s}_{M_L, M_R}$ of shape $d_{M_L}^{\mathrm{U(1)}} \times d_{M_R}^{\mathrm{U(1)}}$, non-zero only for $M_R = M_L + s_z$.

**Memory (U(1) block-sparse).** Summing over blocks,

$$
\mathcal{M}_{\mathrm{U(1)}} \;=\; \sum_{s \in \{\pm\tfrac12\}} \sum_{M_L} d_{M_L}^{\mathrm{U(1)}} \, d_{M_L + s}^{\mathrm{U(1)}}.
$$

If the multiplicities are roughly equal across the abelian sectors, $d_M^{\mathrm{U(1)}} \sim \chi/N_M^{\mathrm{U(1)}}$ where $N_M^{\mathrm{U(1)}}$ is the number of occupied $M$-sectors, giving

$$
\mathcal{M}_{\mathrm{U(1)}} \;\sim\; 2\,\chi^2 / N_M^{\mathrm{U(1)}}.
$$

Dense (no symmetry) memory is $2\chi^2$.

## 4. SU(2) block structure (proposed representation)

Under full $SU(2)$ we label each bond not by projections $M$ but by multiplets $S$, each with internal dimension $2S+1$. Decompose the bond space into a direct sum of multiplets:

$$
\mathcal V_{\rm bond} \;=\; \bigoplus_S \, \mathbb C^{d_S^{\mathrm{SU(2)}}} \otimes \mathcal D^{(S)},
$$

where $d_S^{\mathrm{SU(2)}}$ is the **multiplicity** of the multiplet (the "reduced" dimension) and $\mathcal D^{(S)}$ is the $(2S+1)$-dimensional irreducible representation (carrier space). The **total bond dimension** in the underlying dense basis is

$$
\chi \;=\; \sum_S (2S+1)\, d_S^{\mathrm{SU(2)}}.
$$

**Wigner–Eckart / Clebsch–Gordan factorisation.** For any irreducible tensor operator $\hat T^{(k)}$ (here $k=\tfrac12$ for the physical spin-1/2 site, $k=1$ for the interaction $\hat{\mathbf S}\cdot\hat{\mathbf S}$), the Wigner–Eckart theorem states:

$$
\langle S'M'\,\alpha' \,\|\, \hat T^{(k)}_q \,\|\, S M\, \alpha\rangle
\;=\;
\underbrace{C^{S'M'}_{S M;\, k q}}_{\text{Clebsch–Gordan}}
\cdot
\underbrace{\langle S'\alpha' \,\|\, \hat T^{(k)} \,\|\, S\alpha\rangle}_{\text{reduced matrix element}}.
$$

The left-hand side is an object of shape $(2S'{+}1) d_{S'} \times (2S{+}1) d_S$. The **reduced matrix element** on the right has shape $d_{S'} \times d_S$ and is independent of $M, M', q$. The Clebsch–Gordan factor $C$ is fixed by the group and need not be stored per-tensor (it is a universal object tabulated once; see arXiv:1910.13736 for the X-symbol further compression).

The MPS tensor of the spin-1/2 chain, viewed as an irreducible tensor operator on the virtual bond (the physical leg transforms as $k=\tfrac12$), therefore splits as

$$
(A^{s})_{(S_L \alpha_L M_L),\,(S_R \alpha_R M_R)}
\;=\;
C^{S_R M_R}_{S_L M_L;\, \tfrac12\, s}
\cdot
\tilde A^{S_L \to S_R}_{\alpha_L,\alpha_R},
$$

with $S_R \in \{|S_L - \tfrac12|, S_L + \tfrac12\}$ (the two "fusion channels") and the **reduced tensor** $\tilde A$ having shape $d_{S_L} \times d_{S_R}$.

## 5. Block-sparse memory accounting

The object that must actually be stored is $\tilde A^{S_L \to S_R}$, the set of reduced matrix elements. The Clebsch–Gordan factor is global. So:

$$
\boxed{\;
\mathcal{M}_{\mathrm{SU(2)}}
\;=\;
\sum_{S_L} \sum_{S_R \in \{S_L \pm \tfrac12\}} d_{S_L}^{\mathrm{SU(2)}} \, d_{S_R}^{\mathrm{SU(2)}}.
\;}
$$

Contrast this with the U(1) memory at the same total bond dimension $\chi = \sum_S (2S+1)\,d_S^{\mathrm{SU(2)}}$. Each multiplet of spin $S$ contributes $(2S{+}1)$ dense basis states to the U(1) sectors, distributed over $M \in \{-S,\ldots,S\}$. Therefore

$$
d_M^{\mathrm{U(1)}} \;=\; \sum_{S \ge |M|} d_S^{\mathrm{SU(2)}}.
$$

The U(1) storage implicitly pays for the full $(2S+1)$ internal-dimension factor *per multiplet and per leg*; the SU(2) storage pays for it *once, globally, via the CGC tensor*. The saving scales as the mean multiplet dimension on the bond,

$$
\mathcal R \;\equiv\; \frac{\mathcal M_{\mathrm{U(1)}}}{\mathcal M_{\mathrm{SU(2)}}} \;\approx\; \langle (2S+1) \rangle_{\rm bond}^{\,2} / (\text{a mild selection-rule factor}),
$$

so that at $\chi \sim 10^2$, where spins up to $S \sim 3$–$4$ appear on the bond with significant weight, one obtains an order-of-magnitude memory reduction consistent with the benchmarks of McCulloch & Gulácsi (arXiv:cond-mat/0012319) and Weichselbaum (arXiv:1202.5664).

## 6. Two-site DMRG update in the block-sparse formulation

Standard two-site DMRG sweeps build the effective bond tensor $\Theta$, applies the effective Hamiltonian $\hat H_{\rm eff}$, and re-truncates via SVD. Under SU(2):

1. **Build $\Theta$.** Contract neighbouring reduced MPS tensors:
   $$
   \tilde\Theta^{S_L \to S_R}_{\alpha_L, \alpha_R}
   \;=\;
   \sum_{S_M}\sum_{\alpha_M}
   F(S_L, \tfrac12, S_M, \tfrac12, S_R)\;
   \tilde A^{S_L \to S_M}_{\alpha_L,\alpha_M}\,
   \tilde A^{S_M \to S_R}_{\alpha_M,\alpha_R},
   $$
   where $F(\cdot)$ is a **9j symbol** (equivalent to a sum over the intermediate $M$-projections of a product of two Clebsch–Gordan factors, Racah's identity) that collapses the two physical-leg couplings into one compounded coupling on the two-site fused leg. This is the "single contraction per symmetry sector" structure of arXiv:cond-mat/0012319, Eq. (8).
2. **Apply $\hat H_{\rm eff}$.** The Heisenberg interaction $\hat{\mathbf S}\cdot\hat{\mathbf S}$ is a scalar operator ($k=0$ under SU(2)), so its reduced matrix elements are block-diagonal in $S$. Its action on $\tilde\Theta$ is a per-sector matrix product; no $M$-index loops appear.
3. **SVD / truncation.** Performed per $(S_L, S_R)$ block. Entanglement eigenvalues are $(2S{+}1)$-fold degenerate, so truncation naturally respects the symmetry.

**Per-sector cost.** A contraction on a dense bond of dimension $\chi$ costs $O(\chi^3)$. A per-multiplet contraction on the reduced block of size $d_S \sim \chi / \langle 2S{+}1\rangle$ costs $O(d_S^3)$. Summed over multiplets and selection-rule-allowed pairs, the SU(2) contraction cost drops by roughly $\langle 2S{+}1\rangle^2$ relative to dense, matching Singh (arXiv:1203.2222, §6).

## 7. What the Phase 4 prototype will measure

For fixed $\chi$, random symmetric MPS tensor with a prescribed multiplet spectrum on each bond, we compare:
- $\mathcal M_{\mathrm{dense}}$ — $2\chi^2$ float64 entries per MPS tensor.
- $\mathcal M_{\mathrm{U(1)}}$ — sum over abelian selection-rule-allowed blocks.
- $\mathcal M_{\mathrm{SU(2)}}$ — sum over reduced multiplet blocks ($\tilde A$ only; CGC treated as global).

We sweep $\chi \in \{16, 32, 64, 128, 256\}$ and report both (i) absolute memory in bytes and (ii) the ratio $\mathcal R_{\mathrm{U(1)} \to \mathrm{SU(2)}}$ as a function of the maximum multiplet $S_{\max}$ appearing on the bond.

**Accounting convention.** Following the QSpace convention of arXiv:1202.5664, the Clebsch–Gordan coefficient tensor is treated as a shared, group-universal table and is *not* counted in the SU(2) memory figure. We report "storage per MPS tensor, excluding shared CGC tables." A naive implementation that redundantly stores CGCs per tensor would erase the apparent gain — the engineer must avoid this pitfall. (The X-symbol approach of arXiv:1910.13736 eliminates CGC storage entirely at contraction time; out of scope here but worth noting.) We predict $\mathcal R$ grows roughly linearly with $\langle 2S+1\rangle$ and saturates around $\mathcal R \approx 3$–$6$ for realistic Heisenberg-chain bond spectra at $\chi \le 256$, consistent with Table I of arXiv:cond-mat/0012319 (where an SO(4) calculation at $m{=}300$ matched U(1)$\times$U(1) at $m{=}1700$-plus).

## 8. Deliverables summary

1. **Mathematical derivation (this document):** block-sparse representation under SU(2), explicit Wigner–Eckart factorisation of the MPS tensor, two-site DMRG update in multiplet form.
2. **Python prototype (Phase 4, in `./src/`):** synthetic symmetric MPS tensors with a spin-1/2 Heisenberg-like bond spectrum; measure $\mathcal M$ under dense / U(1) / SU(2) parameterisations.
3. **Formal report (Phase 3, in `./report/`):** the above cast as a short LaTeX note with the benchmark plots.

End of theory draft.
