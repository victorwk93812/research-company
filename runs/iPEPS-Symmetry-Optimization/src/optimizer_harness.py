"""Per-cell run wrapper around tenax.optimize_gs_ad with full per-step logging.

A "cell" is the tuple (model, D, chi, ctmrg_mode, optimizer, seed).  We
instantiate an iPEPSConfig with the appropriate toggles, call
``optimize_gs_ad``, and intercept per-step metrics via the ``gs_callback``
hook tenax provides.  The result is a ``CellResult`` dataclass with
trajectories stored as numpy arrays.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import _resource_block  # noqa: F401  -- must run BEFORE jax import

import jax
import jax.numpy as jnp
import numpy as np

# Force the QR-canonical projector to be registered before tenax is exercised.
# Phase-6: install the strong-eps Lorentzian override BEFORE registration so
# the J1-J2(0.5) std::bad_alloc is bounded by the larger eps.
import ctmrg_qr_strong_eps

ctmrg_qr_strong_eps.use_strong_eps(rel=1e-3)

import ctmrg_qr_register

ctmrg_qr_register.register_qr_canonical()

# Phase-7: shrink L-BFGS history (memory_size 10 -> 4) before tenax exercises
# its optimizer factory.  This dodges the OOM that killed Phase-6 cells
# beyond seed-0.  The patch wraps tenax in-place; do NOT edit upstream.
import lbfgs_small_history  # noqa: F401

lbfgs_small_history.patch_lbfgs_memory_size(memory_size=4)

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad

from models import gate_for, ModelSpec


@dataclass
class CellResult:
    """All per-step trajectories and final values for one (model, D, chi, mode, opt, seed) cell."""

    spec: dict
    energies: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    delta_energies: list[float] = field(default_factory=list)
    step_times: list[float] = field(default_factory=list)  # wall time per AD step
    final_energy: float = float("nan")
    final_grad_norm: float = float("nan")
    total_wall_time: float = float("nan")
    converged: bool = False
    n_steps: int = 0
    error: str | None = None  # exception summary if failed
    # Post-optimization CTM convergence diagnostic (no gradient tracking).
    # Set by ``_ctm_post_check``: did CTMRG converge at the final A, B?
    # If False, the reported energy is non-variational.
    ctm_converged_post: bool | None = None
    ctm_iterations_post: int | None = None
    ctm_sv_diff_post: float | None = None
    ctm_residual_post: float | None = None  # element-wise max of |F(C★)−C★|
    # Variational sanity: is the final energy below a known physical floor?
    # E.g. QMC for Heisenberg ≈ -0.66944. If True, the cell is non-variational
    # (CTM is converged but the optimizer is exploiting truncation error to
    # push below the physical ground state — tenax issue #328 documented for
    # 2-site implicit-AD at chi < 16 with gs_c4v=False).
    below_variational_floor: bool | None = None
    variational_floor_reference: float | None = None

    def to_dict(self) -> dict:
        return {
            **self.spec,
            "final_energy": self.final_energy,
            "final_grad_norm": self.final_grad_norm,
            "total_wall_time": self.total_wall_time,
            "n_steps": self.n_steps,
            "converged": self.converged,
            "error": self.error,
            "energies": list(self.energies),
            "grad_norms": list(self.grad_norms),
            "step_times": list(self.step_times),
            "ctm_converged_post": self.ctm_converged_post,
            "ctm_iterations_post": self.ctm_iterations_post,
            "ctm_sv_diff_post": self.ctm_sv_diff_post,
            "ctm_residual_post": self.ctm_residual_post,
            "below_variational_floor": self.below_variational_floor,
            "variational_floor_reference": self.variational_floor_reference,
        }


def _build_config(
    D: int,
    chi: int,
    ctmrg_mode: str,
    optimizer: str,
    unit_cell: str,
    num_steps: int,
    metric_precond: bool,
    gs_c4v: bool = False,
) -> iPEPSConfig:
    """Map our orthogonal axes to a fully-spec'd iPEPSConfig.

    Phase-7: ``gs_c4v=True`` enables the C4v sublattice-rotation parametrisation
    (single coefficient vector $c$ for $A$ in a C4v basis, $B$ derived as
    $U_{\\sigma^y/2}A$). This is required for variational correctness on the
    2-site path at $\\chi<16$; tenax issue #328.
    """
    # ``svd`` is the Francuz baseline; ``qr_canonical`` is our new mode;
    # ``eigh`` kept available for ablations.
    projector_method = ctmrg_mode

    ctm = CTMConfig(
        chi=chi,
        max_iter=80,
        conv_tol=5e-7,
        projector_method=projector_method,
        projector_backward="lorentzian" if ctmrg_mode == "svd" else "auto",
        forward_gauge="phase",
        ad_backward_method="vjp",
        ctm_conv_method="elementwise",
        verbose=False,
    )

    cfg = iPEPSConfig(
        max_bond_dim=D,
        ctm=ctm,
        unit_cell=unit_cell,
        gs_optimizer=optimizer,
        gs_num_steps=num_steps,
        gs_learning_rate=0.05 if optimizer == "adam" else 1e-3,
        gs_conv_tol=1e-7,
        gs_verbose=False,
        gs_log_interval=1,
        gs_metric_precond=metric_precond and (optimizer == "lbfgs"),
        gs_max_grad_norm=2.0,
        su_init=False,  # use random init for reproducibility across modes
        gs_c4v=gs_c4v,
        # gs_c4v=True requires "reset" stall recovery (the "noise" recovery
        # path is incompatible with the C4v coefficient-vector parametrisation).
        gs_stall_recovery="reset" if gs_c4v else None,
    )
    return cfg


def run_cell(
    model: str,
    D: int,
    chi: int,
    ctmrg_mode: str,
    optimizer: str,
    seed: int,
    h_or_J2: float = 0.0,
    num_steps: int = 25,
    metric_precond: bool = False,
    gs_c4v: bool = False,
    chi_extrap: tuple[int, ...] = (),
) -> CellResult:
    """One (model, D, chi, ctmrg_mode, optimizer, seed) benchmark cell.

    optimizer in {"adam", "lbfgs"}; metric_precond toggles Rader metric
    preconditioning (only honored when optimizer == "lbfgs"). ``gs_c4v``
    enables the C4v sublattice-rotation parametrisation. ``chi_extrap`` is a
    tuple of $\\chi$ values to re-evaluate $E_{\\rm CTM}$ at on the
    converged tensor (pure forward, no AD); the resulting trajectory is
    written to ``CellResult.spec['chi_extrap']``.
    """
    spec = dict(
        model=model,
        D=D,
        chi=chi,
        ctmrg_mode=ctmrg_mode,
        optimizer=optimizer,
        metric_precond=metric_precond,
        seed=seed,
        h_or_J2=h_or_J2,
        num_steps=num_steps,
        gs_c4v=gs_c4v,
        chi_extrap_targets=list(chi_extrap),
    )
    res = CellResult(spec=spec)

    if model == "tfim":
        unit_cell = "1x1"
        # 1x1 ansatz for TFIM (paramagnetic phase favours uniform state).
    else:
        unit_cell = "2site"

    gate = gate_for(model, h_or_J2)
    cfg = _build_config(
        D, chi, ctmrg_mode, optimizer, unit_cell, num_steps, metric_precond, gs_c4v=gs_c4v
    )

    # Random init — same seed for both ctmrg_modes so they see the same start.
    key = jax.random.PRNGKey(seed)
    if unit_cell == "1x1":
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (D, D, D, D, 2)) + 1j * jax.random.normal(k2, (D, D, D, D, 2))
        # Real init is enough; for TFIM the gate is real.
        A = jnp.real(A).astype(jnp.float64)
        A_init: Any = A
    else:
        keys = jax.random.split(key, 4)
        A = jax.random.normal(keys[0], (D, D, D, D, 2)).astype(jnp.float64)
        B = jax.random.normal(keys[1], (D, D, D, D, 2)).astype(jnp.float64)
        A_init = (A, B)

    last_energy = [None]
    step_t0 = [time.perf_counter()]

    def cb(metrics: dict[str, Any]) -> None:
        now = time.perf_counter()
        step_t = now - step_t0[0]
        step_t0[0] = now
        e = float(metrics.get("energy", float("nan")))
        gn = float(metrics.get("grad_norm", float("nan")))
        de = float(metrics.get("delta_energy", float("nan")))
        res.energies.append(e)
        res.grad_norms.append(gn)
        res.delta_energies.append(de)
        res.step_times.append(step_t)
        last_energy[0] = e

    cfg.gs_callback = cb

    t0 = time.perf_counter()
    final_tensors = None
    try:
        out = optimize_gs_ad(gate, A_init, cfg)
        if unit_cell == "1x1":
            A_opt, _env, E_gs = out
            final_tensors = (A_opt,)
        else:
            (A_opt, B_opt), _envs, E_gs = out
            final_tensors = (A_opt, B_opt)
        res.final_energy = float(E_gs)
        res.converged = True
    except Exception as exc:  # noqa: BLE001
        res.error = repr(exc)[:200]
        res.final_energy = float(last_energy[0]) if last_energy[0] is not None else float("nan")
    res.total_wall_time = time.perf_counter() - t0
    res.n_steps = len(res.energies)
    if res.grad_norms:
        res.final_grad_norm = float(res.grad_norms[-1])

    # Post-optimization CTM convergence diagnostic (no gradient tracking).
    # This is the user-requested guardrail: if the CTM is not actually a fixed
    # point at the converged variational tensor, the reported E_var is not
    # variational and must not be trusted as a physical energy.
    if final_tensors is not None:
        try:
            _ctm_post_check(final_tensors, ctmrg_mode, chi, unit_cell, res)
        except Exception as exc:  # noqa: BLE001
            res.ctm_converged_post = None
            # Don't overwrite a real exception above; just record the post-check failure.
            if res.error is None:
                res.error = f"post-CTM check failed: {repr(exc)[:150]}"

    # Phase-7 chi-extrapolation cross-check: re-evaluate E_CTM at
    # multiple chi values on the converged variational tensor (pure
    # forward, no AD).  If E_CTM(chi) is monotonically increasing
    # toward chi -> infinity, the cell is variationally convergent.
    if final_tensors is not None and chi_extrap:
        try:
            extrap_results = _chi_extrap_check(
                final_tensors,
                ctmrg_mode,
                unit_cell,
                gate,
                tuple(chi_extrap),
            )
            res.spec["chi_extrap_results"] = extrap_results
        except Exception as exc:  # noqa: BLE001
            res.spec["chi_extrap_results"] = []
            res.spec["chi_extrap_error"] = repr(exc)[:200]

    # Variational-floor sanity: even with a converged CTM env, the 2-site
    # implicit-AD path at chi < 16 is documented (tenax issue #328) to push
    # E below the physical ground state by exploiting truncation error.
    # Flag any cell whose final energy violates a known physical floor.
    floor = _VARIATIONAL_FLOORS.get(model)
    if floor is not None:
        res.variational_floor_reference = floor
        if not (res.final_energy != res.final_energy):  # not nan
            res.below_variational_floor = res.final_energy < floor
    return res


# Known physical-ground-state lower-bound references used for the
# below-variational-floor sanity check.  Conservatively use the BEST
# documented variational/QMC value as the floor; anything below is
# non-physical and almost certainly a CTM-AD artifact.
_VARIATIONAL_FLOORS = {
    "heisenberg": -0.66944,    # QMC, infinite-D
    # J1-J2 reference values are J2/J1-dependent; we omit the floor for
    # the J1-J2 model (it would need a sweep table).  The Heisenberg floor
    # is the most informative for the headline RQ1 / RQ4 cells.
    "tfim": None,              # No conservative floor; depends on h/J.
    "j1j2": None,
}


def _ctm_post_check(tensors, ctmrg_mode: str, chi: int, unit_cell: str, res: CellResult) -> None:
    """Run CTMRG to convergence at the final tensor with NO gradient tracking;
    record (converged, iterations, sv_diff, residual) on the CellResult.

    The post-check uses the standard ``python_loop_ctm_converge`` path, which
    reports SV-difference convergence info. We additionally compute the
    element-wise max ``|F(C★)−C★|`` across one further CTMRG sweep as the
    fixed-point residual.
    """
    from tenax.algorithms._ctm_python_loop import (
        _make_jit_ctm_step,
        python_loop_ctm_converge,
    )
    from tenax.algorithms._ctm_tensor_convergence import (
        CHECKERBOARD_NEIGHBORS,
        _max_env_leaf_diff,
    )

    if unit_cell == "1x1":
        site_tensors = {(0, 0): tensors[0]}
        neighbors = {(0, 0): {"left": (0, 0), "right": (0, 0), "up": (0, 0), "down": (0, 0)}}
    else:
        site_tensors = {(0, 0): tensors[0], (1, 0): tensors[1]}
        neighbors = CHECKERBOARD_NEIGHBORS

    envs, info = python_loop_ctm_converge(
        site_tensors,
        neighbors,
        chi=chi,
        max_iter=80,
        min_iter=10,
        conv_tol=5e-7,
        conv_method="elementwise",
        renormalize=True,
        projector_method=ctmrg_mode,
        projector_backward="auto",
    )
    res.ctm_converged_post = bool(info.converged)
    res.ctm_iterations_post = int(info.iterations)
    res.ctm_sv_diff_post = float(info.sv_diff)

    # One further sweep to measure the fixed-point residual ‖F(C★) − C★‖.
    jit_step = _make_jit_ctm_step(neighbors)
    envs_next = jit_step(
        site_tensors,
        envs,
        chi=chi,
        projector_method=ctmrg_mode,
        renormalize=True,
        projector_backward="auto",
    )
    residual = 0.0
    for c in envs:
        residual = max(residual, _max_env_leaf_diff(envs[c], envs_next[c]))
    res.ctm_residual_post = float(residual)


def _chi_extrap_check(
    tensors,
    ctmrg_mode: str,
    unit_cell: str,
    gate,
    chi_values: tuple[int, ...],
) -> list[tuple]:
    """For each chi in ``chi_values``, run CTMRG to convergence on the
    converged variational tensor and report ``(chi, E_ctm, ctm_iters,
    ctm_residual)``.  Pure forward — no gradient tracking.

    Used as a Phase-7 sanity check: if E_CTM(chi) monotonically increases
    with chi (i.e. the variational floor relaxes downward as chi grows),
    the cell is in the convergent regime.  If E_CTM(chi) is non-monotone
    or decreases, the cell is non-variational at finite chi.
    """
    from tenax.algorithms._ctm_python_loop import (
        _make_jit_ctm_step,
        python_loop_ctm_converge,
    )
    from tenax.algorithms._ctm_tensor import (
        compute_energy_ctm_tensor,
        compute_energy_ctm_tensor_2site,
    )
    from tenax.algorithms._ctm_tensor_convergence import (
        CHECKERBOARD_NEIGHBORS,
        _max_env_leaf_diff,
    )

    if unit_cell == "1x1":
        site_tensors = {(0, 0): tensors[0]}
        neighbors = {(0, 0): {"left": (0, 0), "right": (0, 0), "up": (0, 0), "down": (0, 0)}}
    else:
        site_tensors = {(0, 0): tensors[0], (1, 0): tensors[1]}
        neighbors = CHECKERBOARD_NEIGHBORS

    out: list[tuple] = []
    for chi_val in chi_values:
        chi_int = int(chi_val)
        try:
            envs, info = python_loop_ctm_converge(
                site_tensors,
                neighbors,
                chi=chi_int,
                max_iter=80,
                min_iter=10,
                conv_tol=5e-7,
                conv_method="elementwise",
                renormalize=True,
                projector_method=ctmrg_mode,
                projector_backward="auto",
            )
            # Fixed-point residual
            jit_step = _make_jit_ctm_step(neighbors)
            envs_next = jit_step(
                site_tensors,
                envs,
                chi=chi_int,
                projector_method=ctmrg_mode,
                renormalize=True,
                projector_backward="auto",
            )
            residual = 0.0
            for c in envs:
                residual = max(residual, _max_env_leaf_diff(envs[c], envs_next[c]))

            # Energy: 1-site or 2-site dispatch
            if unit_cell == "1x1":
                E_chi = float(
                    compute_energy_ctm_tensor(tensors[0], envs[(0, 0)], gate)
                )
            else:
                E_chi = float(
                    compute_energy_ctm_tensor_2site(
                        tensors[0],
                        tensors[1],
                        envs[(0, 0)],
                        envs[(1, 0)],
                        gate,
                    )
                )
            out.append(
                (chi_int, E_chi, int(info.iterations), float(residual))
            )
        except Exception as exc:  # noqa: BLE001
            out.append((chi_int, float("nan"), -1, float("nan")))
            # Continue to next chi rather than aborting the whole sweep.
    return out
