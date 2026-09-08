"""Vectors and matrices — reference implementation.

Matrix multiplication is built twice, from two different readings of the same
definition, using only elementwise multiplication and summation. Neither
`matmul` nor `matmul_outer` may use `@`, `np.dot`, `np.matmul`, or `np.einsum`:
the point is to construct the operation, not to call it.

The remaining functions apply those ideas to the shapes this program actually
uses — a design matrix whose rows are examples.
"""

import numpy as np


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices via the inner-product reading of the definition.

    Entry (i, j) of the product is the inner product of row i of A with column j
    of B. Broadcasting computes all of them at once: A is viewed as (m, k, 1)
    and B as (1, k, p), their elementwise product is (m, k, p), and summing over
    the shared axis k collapses it to (m, p).

    Args:
        A: (m, k)
        B: (k, p)

    Returns:
        (m, p) matrix product.

    Raises:
        ValueError: if the inner dimensions do not agree.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"expected two 2-D arrays, got {A.ndim}-D and {B.ndim}-D")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"inner dimensions disagree: {A.shape} and {B.shape}")
    return (A[:, :, None] * B[None, :, :]).sum(axis=1)


def matmul_outer(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices as a sum of outer products.

    Column t of A and row t of B form a rank-one matrix, and the product is the
    sum of those k rank-one pieces. This reading is what makes the Gram matrix a
    sum over examples, and is the shape of every low-rank factorization later in
    the program.

    Args:
        A: (m, k)
        B: (k, p)

    Returns:
        (m, p) matrix product.

    Raises:
        ValueError: if the inner dimensions do not agree.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"expected two 2-D arrays, got {A.ndim}-D and {B.ndim}-D")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"inner dimensions disagree: {A.shape} and {B.shape}")
    m, k = A.shape
    p = B.shape[1]
    total = np.zeros((m, p), dtype=np.float64)
    for t in range(k):
        # A[:, t] is (m,), B[t, :] is (p,); the outer product is (m, p).
        total += A[:, t, None] * B[None, t, :]
    return total


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the intercept is carried inside x.

    With x_0 = 1 the affine model b + w^T x becomes theta^T x, and no separate
    bias term is needed anywhere in the program before part 9.

    Args:
        X: (n, d - 1) raw features, one example per row.

    Returns:
        (n, d) design matrix whose first column is all ones.
    """
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {X.ndim}-D")
    n = X.shape[0]
    return np.hstack([np.ones((n, 1), dtype=np.float64), X])


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the linear model at every example at once.

    Entry i of the result is theta^T x^(i). Read column-wise, the whole vector
    is a linear combination of the columns of X with weights theta, which is
    exactly the statement that predictions live in the column space of X.

    Args:
        X: (n, d) design matrix, one example per row.
        theta: (d,) parameters.

    Returns:
        (n,) vector of predictions.

    Raises:
        ValueError: if the shapes do not conform.
    """
    if X.ndim != 2 or theta.ndim != 1:
        raise ValueError(f"expected (n, d) and (d,), got {X.shape} and {theta.shape}")
    if X.shape[1] != theta.shape[0]:
        raise ValueError(f"shapes disagree: {X.shape} and {theta.shape}")
    return X @ theta


def gram(X: np.ndarray) -> np.ndarray:
    """Compute the Gram matrix X^T X.

    Symmetric by construction, (d, d) rather than (n, n), and the sum over
    examples of the outer products x^(i) (x^(i))^T. It is the matrix that
    appears on the left of the normal equations.

    Args:
        X: (n, d) design matrix.

    Returns:
        (d, d) Gram matrix.
    """
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {X.ndim}-D")
    return X.T @ X
