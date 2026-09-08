"""Synthetic dataset generators.

Every generator takes an explicit `rng` so that results are reproducible and no
global NumPy random state is touched.
"""

import numpy as np


def make_linear_data(
    n: int,
    d: int,
    rng: np.random.Generator,
    noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw data from a linear model with Gaussian noise.

    The design matrix carries the intercept coordinate in its first column, so
    the model is h(x) = theta^T x with no separate bias term.

    Args:
        n: number of examples.
        d: number of features, including the intercept coordinate.
        rng: source of randomness.
        noise: standard deviation of the additive Gaussian noise on y.

    Returns:
        X: (n, d) design matrix; column 0 is all ones.
        theta: (d,) the parameters used to generate y.
        y: (n,) targets, y = X @ theta + eps.
    """
    if d < 1:
        raise ValueError(f"d must be at least 1 (the intercept), got {d}")
    X = np.ones((n, d), dtype=np.float64)
    X[:, 1:] = rng.normal(size=(n, d - 1))
    theta = rng.normal(size=d)
    y = X @ theta + noise * rng.normal(size=n)
    return X, theta, y
