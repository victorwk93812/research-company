"""Unit test: gradient-descent on a 2-parameter toy surface recovers the known minimum."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import resource_cap  # noqa: F401
import jax
import jax.numpy as jnp
import optax


def test_toy_minimum():
    """A 2-d quadratic with a known minimum at (3, -2). Adam should find it."""
    def loss(theta):
        x, y = theta
        return (x - 3.0) ** 2 + 2.0 * (y + 2.0) ** 2

    theta = jnp.array([0.0, 0.0])
    opt = optax.adam(0.1)
    opt_state = opt.init(theta)
    value_and_grad = jax.value_and_grad(loss)
    for _ in range(500):
        v, g = value_and_grad(theta)
        updates, opt_state = opt.update(g, opt_state)
        theta = optax.apply_updates(theta, updates)
    v = float(loss(theta))
    print(f"[toy_minimum] final theta = {theta}, loss = {v:.6e}")
    assert v < 1e-4, f"toy minimum not reached: loss={v}"
    assert abs(float(theta[0]) - 3.0) < 1e-2
    assert abs(float(theta[1]) + 2.0) < 1e-2


if __name__ == "__main__":
    test_toy_minimum()
    print("Toy-minimum test PASSED.")
