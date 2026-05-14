# Code Structure: iPEPS Symmetry Optimization

Architecture overview based on the actual source files under `./src/`.

## Top-level layout (`./src/`)

```
src/
├── _resource_block.py        # CPU/thread/memory caps — imported FIRST
├── models.py                 # Two-site gates: TFIM, Heisenberg, J1-J2
├── ctmrg_qr.py               # QR-canonical projector (the actual algorithm)
├── ctmrg_qr_register.py      # Monkey-patch tenax to use it
├── ctmrg_qr_strong_eps.py    # Bigger Lorentzian eps override
├── lbfgs_small_history.py    # Monkey-patch tenax L-BFGS memory_size
├── optimizer_harness.py      # run_cell(): assemble iPEPSConfig, call tenax
├── run_one_cell.py           # One-cell argv wrapper (subprocess entry)
├── benchmark.py              # Sweep driver: forks one subprocess per cell
├── analysis.py               # JSONL → CSV + matplotlib figures
├── make_tables.py            # CSV → LaTeX tables
├── finalize_phase8*.py       # Poll for completion, run analysis, write status
├── configs/*.yaml            # Cell grids
└── data/*.jsonl, logs/, figures/
```

## Modularization principle

**Four layers**, each importing only the layer below it:

| Layer | Files | Job |
|---|---|---|
| **L1 Algorithm** | `ctmrg_qr.py`, `models.py` | Pure math: QR-canonical projector forward + custom_vjp Lorentzian-eigh backward; spin-1/2 bond gates |
| **L2 Tenax integration** | `ctmrg_qr_register.py`, `ctmrg_qr_strong_eps.py`, `lbfgs_small_history.py` | Monkey-patches that let our algorithm appear inside tenax's CTM machinery without forking tenax |
| **L3 Per-cell driver** | `optimizer_harness.py`, `run_one_cell.py` | One (model, χ, mode, optimizer, seed) cell → `CellResult` |
| **L4 Sweep + report** | `benchmark.py`, `analysis.py`, `make_tables.py`, `finalize_*.py` | Subprocess fan-out, JSONL aggregation, figures, LaTeX |

Each subprocess re-imports L1→L3 from scratch — JIT cache isolation per cell, which is why the Phase-6 OOM didn't propagate.

## L1: algorithm core

### `ctmrg_qr.py` — the actual contribution

Forward pipeline: `M → QR → canonical_gauge(Q,R) → ρ = R R† → eigh(ρ) → P = Q · U[:,top-k]`.

- `_canonical_qr(M)` — thin QR with diag(R) phase rotated real-positive so the factorisation is smooth across small M perturbations.
- `_lorentzian_eigh(M)` — `@jax.custom_vjp` symmetric eigh. Forward = `jnp.linalg.eigh`; backward replaces `F_ij = 1/(w_j − w_i)` with the Lorentzian-regularised `F^ε_ij = (w_j − w_i)/((w_j − w_i)² + ε²)` (Francuz PRR 7, 013237). `ε = max(ε_abs, ε_rel · max|w|)` caps `‖F‖ ≤ 1/(2ε)` near degeneracy.
- `qr_canonical_projector(M, χ)` — composes the two. JAX's built-in QR backward handles the upstream half; our custom_vjp handles only the truncation-coupling step.
- `_qr_canonical_with_eigh_only` — non-truncated variant used by the FD gradient-check harness.

### `models.py`

`heisenberg_two_site_gate`, `tfim_two_site_gate(h)`, `j1j2_two_site_gate(J1, J2)` → real `(d,d,d,d)` tensors with axes `(s1,s2,s1',s2')` matching tenax's convention.

## L2: tenax integration (no upstream fork)

### `ctmrg_qr_register.py` — the wrapping point

Replaces **`tenax.algorithms._ctm_projector._compute_projector_tensor`** with `_patched(C1g, C4g, chi, projector_method, ...)`. When `projector_method == "qr_canonical"`:

1. Unwraps `DenseTensor`/`SymmetricTensor` → raw `jax.Array`.
2. Concatenates `M = [C1g | C4g]` along col-axis.
3. Calls `qr_canonical_projector(M, chi)`.
4. Re-wraps via `_ctmp._wrap_dense_projector` with the original fused-leg index, returning identical `(P, P)`.

After PR #341/#342/#343, multiple tenax modules grabbed `_compute_projector_tensor` by name at import time, so the patcher **rebinds in all seven importers**: `_ctm_tensor`, `_ctm_tensor_moves`, `_ctm_tensor_paired_moves`, `_ctm_tensor_c4v`, `_ctm_compiled_moves`, `_split_ctm_tensor`, `_split_ctm_tensor_moves`. Also extends `tenax.algorithms.ad_utils._PM_STR_TO_INT` so the string `"qr_canonical"` survives the JIT config round-trip.

### `ctmrg_qr_strong_eps.py`

Single function `use_strong_eps(rel=1e-3)` overwrites `ctmrg_qr._LORENTZ_EPS_REL/ABS` at module level — must be called *before* `register_qr_canonical()` so the patched projector sees the new eps.

### `lbfgs_small_history.py`

Replaces `tenax.algorithms.ipeps_optimize._build_optimizer`; rebuilds the optax chain with `memory_size=4` instead of 10 for the L-BFGS branch (Phase-6 OOM fix). Adam/CG fall through.

## L3: per-cell driver

### `optimizer_harness.py` — `run_cell(...)`

Single public entrypoint. Orthogonal axes: `model, D, chi, ctmrg_mode, optimizer, seed, h_or_J2, num_steps, metric_precond, gs_c4v, chi_extrap, su_init, gs_implicit_ad`.

`_build_config(...)` spells out **every** AD-relevant knob in a single `iPEPSConfig` so the run is reproducible against tenax default drifts:

- **`CTMConfig`** (forward+adjoint): `chi, max_iter=80, min_iter=10, conv_tol=5e-7, projector_method, projector_backward, ad_regularize_svd=True, ad_backward_method="vjp", gmres_tol=1e-6, gmres_restart=20, gmres_maxiter=200, adjoint_solver="bicgstab", adjoint_maxiter=50, adjoint_tol=1e-8, adjoint_tikhonov=1e-6, adjoint_arnoldi_precheck=True, adjoint_arnoldi_threshold=5.0, forward_gauge="phase"`.
- **`iPEPSConfig`** (AD path + optimiser): `max_bond_dim, unit_cell, su_init, gs_implicit_ad` (SVD→True, QR→False per PR #346 policy validator), `gs_explicit_ad_warmup=3, gs_explicit_ad_steps=10` (lowered from 20 to dodge new-tenax χ=16 NaN), `gs_optimizer, gs_num_steps, gs_learning_rate, gs_conv_tol=1e-15` (disable early-stop on SU-stationary plateau), `gs_line_search_method="hager_zhang"`, `gs_metric_precond`, `gs_c4v`, `gs_stall_recovery="reset"` under C4v.

`gs_callback` hook captures `energy/grad_norm/delta_energy` per step → `CellResult.energies/grad_norms/...`.

**Post-optimization guardrails** (no AD):
- `_ctm_post_check(...)` — re-runs `python_loop_ctm_converge` at the optimized tensor and one extra `_make_jit_ctm_step(...)` to measure `‖F(C★) − C★‖`. Handles new tenax's `(envs, eps_T)` tuple return.
- `_chi_extrap_check(...)` — re-evaluates `E_CTM(χ)` over a tuple of χ values via `compute_energy_ctm_tensor[_2site]` to check monotonic descent.
- `_VARIATIONAL_FLOORS` — flags Heisenberg cells with `E < −0.66944` as non-variational (tenax issue #328 trace).

### `run_one_cell.py`

argparse wrapper: spec JSON → `run_cell` → `CellResult.to_dict()` → JSON. Runs in a fresh subprocess.

## L4: sweep + report

- **`benchmark.py`** — YAML cell grid; for each `(cell × seed)` forks `uv run python run_one_cell.py <spec_json> <out_path>` with per-cell timeout and global budget. Appends to `data/<label>.jsonl` after every cell.
- **`analysis.py`** — `_load → _aggregate → plot_*`. Per-(mode, seed) Wong palette via `_LINE_PALETTE`, metric-on lines dashed. Outputs energy-trajectory, gradient-stability, χ-extrap, 2×2 interaction, wallclock figures.
- **`make_tables.py`** — CSV → LaTeX tables (`summary_table.tex`, `opt_x_mode.tex`).
- **`finalize_phase8*.py`** — polls jsonl, runs analysis + tables, writes `phase8_status.md`.

## Tenax APIs we wrapped or hooked

| Wrapping point | Reason |
|---|---|
| `tenax.algorithms._ctm_projector._compute_projector_tensor` | inject QR-canonical projector under JAX tracing |
| `tenax.algorithms.{_ctm_tensor, _ctm_tensor_moves, _ctm_tensor_paired_moves, _ctm_tensor_c4v, _ctm_compiled_moves, _split_ctm_tensor, _split_ctm_tensor_moves}._compute_projector_tensor` | rebind name-imported references post PR #341/#342 |
| `tenax.algorithms.ad_utils._PM_STR_TO_INT / _PM_INT_TO_STR` | survive JIT config round-trip |
| `tenax.algorithms.ipeps_optimize._build_optimizer` | shrink L-BFGS memory_size 10→4 |
| `tenax.optimize_gs_ad` | top-level call from `run_cell` |
| `tenax.{CTMConfig, iPEPSConfig}` | explicit AD-knob spec |
| `tenax.algorithms._ctm_python_loop.{python_loop_ctm_converge, _make_jit_ctm_step}` | gradient-free post-checks (CTM residual, χ-extrap) |
| `tenax.algorithms._ctm_tensor.{compute_energy_ctm_tensor, compute_energy_ctm_tensor_2site}` | χ-extrap energy re-eval |
| `tenax.algorithms._ctm_tensor_convergence.{CHECKERBOARD_NEIGHBORS, _max_env_leaf_diff}` | 2-site neighbour map + fixed-point residual |
| `tenax.core.tensor.{DenseTensor, SymmetricTensor}` | unwrap/rewrap inside the projector patch |
