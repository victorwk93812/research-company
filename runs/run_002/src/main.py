"""Entrypoint: configure JAX for x64 CPU under a 16 GB/4-thread cap, then
run the reduced benchmark grid."""

from __future__ import annotations

import resource_cap  # noqa: F401  (must be first to install limits)

import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np

from benchmark import run_grid


def main() -> None:
    outdir = Path(__file__).resolve().parent
    (outdir / "figures").mkdir(exist_ok=True)
    print("JAX:", jax.__version__, "| NumPy:", np.__version__)
    print("CPU devices:", jax.devices())
    print("PID:", os.getpid(), "| x64:", jax.config.jax_enable_x64)
    run_grid(str(outdir), n_steps=int(os.environ.get("N_STEPS", "40")))


if __name__ == "__main__":
    main()
