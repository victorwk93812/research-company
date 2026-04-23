"""Main driver: runs the tests, benchmarks, and saves figures.

Invoked via `uv run python main.py | tee simulation.log`.
"""
from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import resource_cap  # noqa: F401  (apply BLAS cap first)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ctmrg import run_ctmrg, fixed_point_residual, double_layer
from ad_pipeline import unrolled_observable, implicit_observable
from benchmark import time_forward_backward, finite_diff_check


HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
RESULTS = HERE / "results.jsonl"


def log(msg):
    print(msg, flush=True)


def make_site_tensor(d: int, D: int, key, bias_ones: float = 3.0) -> jax.Array:
    """Random site tensor biased to a gapped regime."""
    A = jax.random.normal(key, (d, D, D, D, D))
    A = A + bias_ones * jnp.ones_like(A) * (jnp.arange(A.size).reshape(A.shape) == 0)
    return A


def run_unit_tests():
    log("\n================ UNIT TESTS ================")
    from tests import test_qr_backward, test_fixed_point, test_qr_vs_svd, test_toy_minimum

    for name, mod in [
        ("qr_backward", test_qr_backward),
        ("fixed_point", test_fixed_point),
        ("qr_vs_svd", test_qr_vs_svd),
        ("toy_minimum", test_toy_minimum),
    ]:
        log(f"--- {name} ---")
        fns = [getattr(mod, x) for x in dir(mod) if x.startswith("test_")]
        for f in fns:
            try:
                f()
                log(f"  [PASS] {f.__name__}")
            except Exception as e:
                log(f"  [FAIL] {f.__name__}: {e!r}")


def run_ad_benchmarks():
    log("\n============ AD BENCHMARKS ============")
    results = []
    key = jax.random.PRNGKey(123)
    for D in [2, 3]:
        chi = 2 * D * D
        n_steps = 8
        A = make_site_tensor(d=2, D=D, key=key)
        for projector in ("qr", "svd"):
            for mode in ("unrolled", "implicit"):
                loss_name = f"{projector}-{mode}"
                if mode == "unrolled":
                    def L(A, p=projector, c=chi, n=n_steps):
                        return unrolled_observable(A, c, n, projector=p)
                else:
                    def L(A, p=projector, c=chi, n=n_steps):
                        return implicit_observable(A, c, n, projector=p)
                try:
                    r = time_forward_backward(L, A, name=f"D={D}-chi={chi}-{loss_name}")
                    ad_dir, fd_dir = finite_diff_check(L, A, h=1e-4)
                    result = {
                        "D": D, "chi": chi, "n_steps": n_steps,
                        "projector": projector, "mode": mode,
                        "forward_ms": r.forward_ms, "backward_ms": r.backward_ms,
                        "total_ms": r.total_ms,
                        "ad_directional": ad_dir, "fd_directional": fd_dir,
                        "fd_ad_rel_err": abs(ad_dir - fd_dir) / (abs(fd_dir) + 1e-30),
                    }
                    log(f"  {r.name}: fwd={r.forward_ms:.1f}ms bwd={r.backward_ms:.1f}ms "
                        f"tot={r.total_ms:.1f}ms  AD={ad_dir:+.4e} FD={fd_dir:+.4e} "
                        f"rel_err={result['fd_ad_rel_err']:.2e}")
                    results.append(result)
                except Exception as e:
                    log(f"  [ERR] D={D} chi={chi} {loss_name}: {e!r}")
    with open(RESULTS, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    return results


def run_projector_step_microbench():
    log("\n============ PROJECTOR MICRO-BENCH ============")
    import time as _time
    from ctmrg import qr_projector, svd_projector
    key = jax.random.PRNGKey(0)
    out = []
    for D in [2, 3, 4]:
        D2 = D * D
        for chi in [D2, 2 * D2, 3 * D2]:
            M = jax.random.normal(key, (chi * D2, chi * D2))
            M = 0.5 * (M + M.T)
            _ = qr_projector(M, chi).block_until_ready()
            _ = svd_projector(M, chi).block_until_ready()
            n_rep = 5
            t0 = _time.time()
            for _ in range(n_rep):
                _ = qr_projector(M, chi).block_until_ready()
            t1 = _time.time()
            for _ in range(n_rep):
                _ = svd_projector(M, chi).block_until_ready()
            t2 = _time.time()
            qr_ms = (t1 - t0) / n_rep * 1000.0
            svd_ms = (t2 - t1) / n_rep * 1000.0
            log(f"  D={D} chi={chi:3d}  QR={qr_ms:6.2f}ms  SVD={svd_ms:6.2f}ms  "
                f"speedup={svd_ms/qr_ms:.2f}x")
            out.append({"D": D, "chi": chi, "qr_ms": qr_ms, "svd_ms": svd_ms})
    return out


def run_optimization_demo():
    log("\n============ OPTIMIZATION DEMO ============")
    import optax
    key = jax.random.PRNGKey(11)
    D, chi, n_steps = 2, 4, 8
    A = make_site_tensor(d=2, D=D, key=key, bias_ones=1.0)
    traces = {}
    for projector in ("qr", "svd"):
        A_cur = A
        opt = optax.adam(1e-2)
        opt_state = opt.init(A_cur)

        def loss(A, p=projector):
            return unrolled_observable(A, chi, n_steps, projector=p)
        vag = jax.value_and_grad(loss)
        t0 = time.time()
        losses, gnorms = [], []
        for i in range(30):
            v, g = vag(A_cur)
            losses.append(float(v))
            gnorms.append(float(jnp.linalg.norm(g)))
            updates, opt_state = opt.update(g, opt_state)
            A_cur = optax.apply_updates(A_cur, updates)
        t1 = time.time()
        log(f"  {projector}: start={losses[0]:.4f} end={losses[-1]:.4f} "
            f"wall={t1-t0:.1f}s")
        traces[projector] = {"losses": losses, "gnorms": gnorms}
    return traces


def main():
    log(f"=== QR-CTMRG end-to-end AD pipeline, run started {time.ctime()} ===")
    log(f"jax={jax.__version__}")
    run_unit_tests()
    proj_bench = run_projector_step_microbench()
    ad_bench = run_ad_benchmarks()
    opt_traces = run_optimization_demo()

    summary = {
        "projector_microbench": proj_bench,
        "ad_benchmarks": ad_bench,
        "optimization_traces": opt_traces,
    }
    with open(HERE / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log("\n=== Done. ===")


if __name__ == "__main__":
    main()
