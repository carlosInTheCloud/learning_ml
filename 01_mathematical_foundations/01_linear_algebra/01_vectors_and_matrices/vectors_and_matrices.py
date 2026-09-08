"""Vectors and matrices — your implementation.

Fill in each body. The signatures, shapes, and error conditions are already
fixed; nothing but the algorithm is left to write.

`matmul` and `matmul_outer` must not use `@`, `np.dot`, `np.matmul`, or
`np.einsum`. Build the product from elementwise multiplication and summation —
that is the whole exercise. The later functions may use `@` freely.

Run the tests with:

    pytest 01_mathematical_foundations/01_linear_algebra/01_vectors_and_matrices
"""

import numpy as np


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices via the inner-product reading of the definition.

    Entry (i, j) of the product is the inner product of row i of A with column j
    of B. Compute all of them at once with broadcasting: view A as (m, k, 1) and
    B as (1, k, p), multiply elementwise to get (m, k, p), and sum over the
    shared axis k.

    Args:
        A: (m, k)
        B: (k, p)

    Returns:
        (m, p) matrix product.

    Raises:
        ValueError: if the inner dimensions do not agree.
    """
    raise NotImplementedError


def matmul_outer(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices as a sum of outer products.

    Column t of A and row t of B form a rank-one matrix. Sum those k rank-one
    matrices. Looping over t (the shared dimension) is expected here; looping
    over rows or columns of the result is not.

    Args:
        A: (m, k)
        B: (k, p)

    Returns:
        (m, p) matrix product.

    Raises:
        ValueError: if the inner dimensions do not agree.
    """
    raise NotImplementedError


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the intercept is carried inside x.

    Args:
        X: (n, d - 1) raw features, one example per row.

    Returns:
        (n, d) design matrix whose first column is all ones.
    """
    raise NotImplementedError


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the linear model at every example at once.

    Entry i of the result is theta^T x^(i).

    Args:
        X: (n, d) design matrix, one example per row.
        theta: (d,) parameters.

    Returns:
        (n,) vector of predictions.

    Raises:
        ValueError: if the shapes do not conform.
    """
    raise NotImplementedError


def gram(X: np.ndarray) -> np.ndarray:
    """Compute the Gram matrix X^T X.

    Args:
        X: (n, d) design matrix.

    Returns:
        (d, d) Gram matrix.
    """
    raise NotImplementedError
