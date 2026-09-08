# 1.1.1 Vectors and Matrices — Exercises

Attempt every problem before opening [the solutions](01_vectors_and_matrices_solutions.md). A worked solution you have read is not a problem you have solved.

Notation follows [conventions.md](../../../OKF/conventions.md): vectors are columns, $X \in \mathbb{R}^{n \times d}$ has examples as rows, and $x^{(i)} \in \mathbb{R}^{d}$ is the $i$-th example.

Problems marked $\star$ are harder than the rest. They are not optional, but they are worth more time.

---

## A. Derivations and proofs

**1. Associativity.**
Let $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times p}$, $C \in \mathbb{R}^{p \times q}$. Prove that $(AB)C = A(BC)$ by expanding entry $(i, j)$ of each side into a double sum and justifying the exchange of summation order. State the shape of the result.

**2. Transposes of products and of symmetric sandwiches.**

&nbsp;&nbsp;(a) Prove $(A^\top)^\top = A$ directly from the definition.
&nbsp;&nbsp;(b) Using $(AB)^\top = B^\top A^\top$ from the theory, prove $(ABC)^\top = C^\top B^\top A^\top$.
&nbsp;&nbsp;(c) Prove that $A^\top A$ is symmetric for **every** matrix $A$, square or not.
&nbsp;&nbsp;(d) Suppose $S$ is symmetric and $B$ is any matrix for which $B^\top S B$ is defined. Prove $B^\top S B$ is symmetric.

**3. The column reading.**
Starting from the entry definition $(Ax)_i = \sum_{j} A_{ij} x_j$, prove that

$$
Ax = \sum_{j=1}^{n} x_j A_{:,j}
$$

where $A_{:,j}$ is column $j$ of $A$. Then state, in one sentence, what this says about the set of vectors expressible as $X\theta$ as $\theta$ ranges over $\mathbb{R}^{d}$.

**4. The Gram matrix as a sum over examples.**
Prove that

$$
X^\top X = \sum_{i=1}^{n} x^{(i)} \left( x^{(i)} \right)^\top
$$

Identify which of the four readings of the matrix product in section 5 you are using, and say what the shared index is summing over.

**5. Commutativity.**

&nbsp;&nbsp;(a) Give explicit $2 \times 2$ matrices $A$ and $B$ with $AB \neq BA$. Show both products.
&nbsp;&nbsp;(b) Prove that any two diagonal matrices of the same size commute.
&nbsp;&nbsp;(c) $\star$ Suppose $A \in \mathbb{R}^{n \times n}$ satisfies $AM = MA$ for **every** $M \in \mathbb{R}^{n \times n}$. Prove that $A = cI$ for some scalar $c$. *Hint: test $A$ against the matrices $E^{(rs)}$ that have a single $1$ in position $(r, s)$ and zeros elsewhere.*

**6. Rank-one structure.**
Let $u \in \mathbb{R}^{m}$ and $v \in \mathbb{R}^{p}$ be nonzero, and let $M = u v^\top$.

&nbsp;&nbsp;(a) What is the shape of $M$? Give the entry $M_{ij}$.
&nbsp;&nbsp;(b) Prove that every column of $M$ is a scalar multiple of $u$, and identify the scalar.
&nbsp;&nbsp;(c) Prove that $M x$ is a scalar multiple of $u$ for every $x \in \mathbb{R}^{p}$, and identify the scalar.
&nbsp;&nbsp;(d) In one sentence, explain why reading 4 of section 5 is therefore called "a sum of rank-one pieces."

**7. What the intercept column does.**
Let $\tilde{X} \in \mathbb{R}^{n \times (d-1)}$ hold raw features and let $X = [\mathbf{1} \;\; \tilde{X}] \in \mathbb{R}^{n \times d}$ prepend a column of ones.

&nbsp;&nbsp;(a) Show that for $\theta = (b, w_1, \ldots, w_{d-1})^\top$, the $i$-th entry of $X\theta$ equals $b + w^\top \tilde{x}^{(i)}$.
&nbsp;&nbsp;(b) Compute the entry $(X^\top X)_{11}$ (the top-left entry, in 1-based indexing) in terms of $n$.
&nbsp;&nbsp;(c) Compute $(X^\top X)_{1j}$ for $j > 1$, and express it using the column means of $\tilde{X}$.
&nbsp;&nbsp;(d) What does it mean about the data if $(X^\top X)_{1j} = 0$ for all $j > 1$?

**8. Non-negativity of the quadratic form.**

&nbsp;&nbsp;(a) Prove $\theta^\top X^\top X \theta = \sum_{i=1}^{n} \left( \theta^\top x^{(i)} \right)^2$ for every $\theta \in \mathbb{R}^{d}$, naming each identity you use.
&nbsp;&nbsp;(b) Deduce that $\theta^\top X^\top X \theta \geq 0$ always.
&nbsp;&nbsp;(c) Prove that $\theta^\top X^\top X \theta = 0$ if and only if $X\theta = 0$.
&nbsp;&nbsp;(d) $\star$ Give a nonzero $\theta$ and a matrix $X$ with $n > d$ for which $X\theta = 0$. What is true of the columns of your $X$?

---

## B. Implementation

Work in `vectors_and_matrices.py`. Run the suite with:

```bash
pytest 01_mathematical_foundations/01_linear_algebra/01_vectors_and_matrices
```

**9. `matmul` — the inner-product reading.**
Implement matrix multiplication using only elementwise multiplication and summation. No `@`, no `np.dot`, no `np.matmul`, no `np.einsum` — a test reads your source and enforces this. Use broadcasting rather than looping over entries: view $A$ as $(m, k, 1)$ and $B$ as $(1, k, p)$, and collapse the shared axis. Raise `ValueError` when the inner dimensions disagree.

**10. `matmul_outer` — the outer-product reading.**
Implement the same product as a sum of $k$ rank-one matrices. A loop over the shared dimension $t$ is expected; a loop over rows or columns of the result is not. The same prohibitions apply.

**11. `add_intercept`, `predict`, `gram`.**
Implement the three ML-facing functions. These may use `@` freely. Match the shapes and the error conditions in the docstrings exactly.

**12. Cost of the grouping.**
Write a short script (not part of the shipped module) that, for $X \in \mathbb{R}^{n \times d}$ with $n = 200{,}000$ and $d = 200$, times both groupings of $X^\top X \theta$:

&nbsp;&nbsp;(a) $(X^\top X)\theta$ — form the Gram matrix first.
&nbsp;&nbsp;(b) $X^\top (X\theta)$ — apply to the vector first.

Confirm the two results agree to within floating-point tolerance, report both timings, and compare the ratio you measure against the ratio the operation counts in section 9 predict. If they disagree, propose an explanation.

---

## C. Conceptual

**13. The orientation of the design matrix.**
This program fixes $X \in \mathbb{R}^{n \times d}$ with examples as rows. In two or three sentences each:

&nbsp;&nbsp;(a) Give one concrete reason to prefer this orientation over $d \times n$.
&nbsp;&nbsp;(b) Describe a specific bug that arises from mixing the two conventions in one codebase, and explain why testing on square data would fail to catch it.

**14. When forming $X^\top X$ is the wrong move.**
Section 9 shows that computing $X^\top(X\theta)$ beats $(X^\top X)\theta$ for a single evaluation. Give a situation in which forming $X^\top X$ explicitly is nevertheless the right choice, and one further situation — beyond raw operation count — in which it is a bad idea even though you will reuse it. Name the resource that runs out in each case.

**15. The memory cost of the broadcast product.**
Your `matmul` materializes an intermediate array of shape $(m, k, p)$.

&nbsp;&nbsp;(a) For $m = k = p = 1000$ and `float64`, how many bytes is that intermediate?
&nbsp;&nbsp;(b) A production BLAS routine computes the same product in fixed extra memory. Explain what it does differently.
&nbsp;&nbsp;(c) `matmul_outer` uses $O(mp)$ extra memory rather than $O(mkp)$. What did it give up to get that?
