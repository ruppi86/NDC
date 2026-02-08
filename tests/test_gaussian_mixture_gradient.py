import numpy as np

from NDC.dynamics.landscape import GaussianMixture


def _numeric_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        grad[i] = (f(x1) - f(x2)) / (2 * eps)
    return grad


def test_gaussian_mixture_gradient_matches_numeric():
    gm = GaussianMixture.from_params(
        dim=2,
        params={
            "centers": [[0.0, 0.0], [1.0, 0.0]],
            "scales": [1.0, 1.0],
            "weights": [1.0, 0.5],
        },
    )
    x = np.array([0.25, -0.5])
    analytic = gm.gradient(x, 0.0)
    numeric = _numeric_grad(lambda z: gm.potential(z, 0.0), x)
    assert np.allclose(analytic, numeric, rtol=1e-3, atol=1e-4)
