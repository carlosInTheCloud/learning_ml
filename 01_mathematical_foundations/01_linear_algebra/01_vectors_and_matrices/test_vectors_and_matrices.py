"""Tests for 1.1.1 Vectors and matrices.

These run against the stub, so they fail until you implement it. Passing the
whole suite is what "completed the subtopic" means.

    pytest 01_mathematical_foundations/01_linear_algebra/01_vectors_and_matrices
"""

import ast
import pathlib

import numpy as np
import pytest

from mlfs import make_linear_data
from vectors_and_matrices import add_intercept, gram, matmul, matmul_outer, predict

RNG_SEED = 0
TOL = 1e-12  # float64 matmul of well-scaled data; far above rounding, far below error


# --------------------------------------------------------------------------
# matmul: agrees with the operator it is reconstructing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(3, 4, 2), (1, 5, 3), (4, 1, 4), (2, 2, 2), (7, 3, 1)])
@pytest.mark.parametrize("fn", [matmul, matmul_outer])
def test_matmul_reproduces_the_builtin_product(fn, shape):
    """Both constructions equal NumPy's product on rectangular and degenerate shapes."""
    m, k, p = shape
    rng = np.random.default_rng(RNG_SEED)
    A = rng.normal(size=(m, k))
    B = rng.normal(size=(k, p))
    assert np.allclose(fn(A, B), A @ B, atol=TOL)


@pytest.mark.parametrize("fn", [matmul, matmul_outer])
def test_matmul_returns_the_shape_the_definition_predicts(fn):
    """An (m, k) by (k, p) product is (m, p) — the shared dimension disappears."""
    rng = np.random.default_rng(RNG_SEED)
    assert fn(rng.normal(size=(3, 5)), rng.normal(size=(5, 2))).shape == (3, 2)


def test_the_two_constructions_agree_with_each_other():
    """Inner-product and outer-product readings are the same operation."""
    rng = np.random.default_rng(RNG_SEED)
    A = rng.normal(size=(6, 4))
    B = rng.normal(size=(4, 5))
    assert np.allclose(matmul(A, B), matmul_outer(A, B), atol=TOL)


@pytest.mark.parametrize("fn", [matmul, matmul_outer])
def test_matmul_is_associative_but_not_commutative(fn):
    """(AB)C = A(BC) on conformable matrices; AB != BA for a generic square pair."""
    rng = np.random.default_rng(RNG_SEED)
    A, B, C = rng.normal(size=(3, 4)), rng.normal(size=(4, 5)), rng.normal(size=(5, 2))
    assert np.allclose(fn(fn(A, B), C), fn(A, fn(B, C)), atol=TOL)

    P, Q = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    assert not np.allclose(fn(P, Q), fn(Q, P), atol=1e-8)


@pytest.mark.parametrize("fn", [matmul, matmul_outer])
def test_identity_leaves_a_matrix_unchanged_on_either_side(fn):
    """I is the multiplicative identity, and its size differs on each side."""
    rng = np.random.default_rng(RNG_SEED)
    A = rng.normal(size=(4, 3))
    assert np.allclose(fn(np.eye(4), A), A, atol=TOL)
    assert np.allclose(fn(A, np.eye(3)), A, atol=TOL)


@pytest.mark.parametrize("fn", [matmul, matmul_outer])
def test_mismatched_inner_dimensions_are_rejected(fn):
    """A (3, 4) by (5, 2) product is undefined and must not silently broadcast."""
    rng = np.random.default_rng(RNG_SEED)
    with pytest.raises(ValueError):
        fn(rng.normal(size=(3, 4)), rng.normal(size=(5, 2)))


@pytest.mark.parametrize("name", ["matmul", "matmul_outer"])
def test_matmul_is_constructed_rather_than_called(name):
    """Neither construction may delegate to @, np.dot, np.matmul or np.einsum.

    The subtopic is about building the operation, so this reads the source
    rather than the output: a correct answer that calls the operator is not an
    answer to this exercise.
    """
    source = (pathlib.Path(__file__).parent / "vectors_and_matrices.py").read_text()
    fn_node = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    banned = {"dot", "matmul", "einsum", "inner", "tensordot"}
    for node in ast.walk(fn_node):
        assert not isinstance(node, ast.MatMult), f"{name} uses the @ operator"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in banned, f"{name} calls np.{node.func.attr}"


# --------------------------------------------------------------------------
# add_intercept: the x_0 = 1 convention
# --------------------------------------------------------------------------

def test_add_intercept_prepends_ones_and_preserves_the_features():
    """The result gains one column, all ones, and the original data is untouched."""
    rng = np.random.default_rng(RNG_SEED)
    raw = rng.normal(size=(6, 3))
    X = add_intercept(raw)
    assert X.shape == (6, 4)
    assert np.all(X[:, 0] == 1.0)
    assert np.allclose(X[:, 1:], raw, atol=TOL)


def test_add_intercept_handles_a_single_example_and_no_features():
    """One row still gets its one; zero raw features yields the all-ones column."""
    assert add_intercept(np.array([[2.0, 3.0]])).shape == (1, 3)
    empty = add_intercept(np.zeros((4, 0)))
    assert empty.shape == (4, 1)
    assert np.all(empty == 1.0)


def test_intercept_makes_the_affine_model_linear():
    """b + w^T x equals theta^T x once the ones column carries b as theta_0."""
    rng = np.random.default_rng(RNG_SEED)
    raw = rng.normal(size=(5, 2))
    w, b = rng.normal(size=2), 1.7
    theta = np.concatenate([[b], w])
    assert np.allclose(predict(add_intercept(raw), theta), raw @ w + b, atol=TOL)


# --------------------------------------------------------------------------
# predict: one equation for all n examples
# --------------------------------------------------------------------------

def test_predict_matches_the_per_example_inner_products():
    """Entry i of X theta is theta^T x^(i), computed one example at a time."""
    rng = np.random.default_rng(RNG_SEED)
    X, theta, _ = make_linear_data(n=20, d=4, rng=rng)
    expected = np.array([theta @ X[i] for i in range(X.shape[0])])
    assert np.allclose(predict(X, theta), expected, atol=TOL)


def test_predict_is_a_linear_combination_of_the_columns_of_X():
    """X theta = sum_j theta_j (column j) — the column reading of the product."""
    rng = np.random.default_rng(RNG_SEED)
    X, theta, _ = make_linear_data(n=15, d=3, rng=rng)
    combination = sum(theta[j] * X[:, j] for j in range(X.shape[1]))
    assert np.allclose(predict(X, theta), combination, atol=TOL)


def test_predict_handles_a_single_example_and_a_single_feature():
    """Degenerate shapes still return an (n,) vector, never a scalar."""
    assert predict(np.array([[1.0, 2.0]]), np.array([3.0, 4.0])).shape == (1,)
    assert predict(np.ones((5, 1)), np.array([2.0])).shape == (5,)
    assert np.allclose(predict(np.ones((5, 1)), np.array([2.0])), 2.0, atol=TOL)


def test_predict_rejects_a_parameter_vector_of_the_wrong_length():
    """A (n, 3) design matrix cannot be paired with a 2-vector of parameters."""
    with pytest.raises(ValueError):
        predict(np.ones((4, 3)), np.ones(2))


# --------------------------------------------------------------------------
# gram: the matrix behind the normal equations
# --------------------------------------------------------------------------

def test_gram_equals_X_transpose_times_X_and_is_symmetric():
    """G = X^T X, and G^T = G for every X."""
    rng = np.random.default_rng(RNG_SEED)
    X, _, _ = make_linear_data(n=30, d=5, rng=rng)
    G = gram(X)
    assert G.shape == (5, 5)
    assert np.allclose(G, X.T @ X, atol=TOL)
    assert np.allclose(G, G.T, atol=TOL)


def test_gram_is_the_sum_of_outer_products_over_examples():
    """X^T X = sum_i x^(i) (x^(i))^T — the identity that makes it a sum over data."""
    rng = np.random.default_rng(RNG_SEED)
    X, _, _ = make_linear_data(n=25, d=4, rng=rng)
    accumulated = sum(np.outer(X[i], X[i]) for i in range(X.shape[0]))
    assert np.allclose(gram(X), accumulated, atol=1e-10)


def test_gram_quadratic_form_is_never_negative():
    """theta^T X^T X theta = (X theta)^T (X theta) >= 0, for any theta.

    The Gram matrix is positive semi-definite, which is why the normal equations
    have a minimum rather than a saddle. Proved properly in 1.1.4.
    """
    rng = np.random.default_rng(RNG_SEED)
    X, _, _ = make_linear_data(n=40, d=6, rng=rng)
    G = gram(X)
    for _ in range(25):
        theta = rng.normal(size=6)
        assert theta @ G @ theta >= -TOL


def test_gram_of_collinear_data_is_singular():
    """A duplicated feature makes X rank-deficient, so X^T X cannot be inverted.

    This is the failure that motivates regularization in part 4.
    """
    rng = np.random.default_rng(RNG_SEED)
    raw = rng.normal(size=(20, 2))
    X = add_intercept(np.hstack([raw, raw[:, [0]]]))  # column 3 duplicates column 1
    assert abs(np.linalg.det(gram(X))) < 1e-8
