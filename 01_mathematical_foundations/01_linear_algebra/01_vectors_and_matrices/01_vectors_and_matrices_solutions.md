# 1.1.1 Vectors and Matrices — Solutions

Solutions to [the exercises](01_vectors_and_matrices_exercises.md), in the same order and numbering.

---

## A. Derivations and proofs

### 1. Associativity

Let $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times p}$, $C \in \mathbb{R}^{p \times q}$. Both $(AB)C$ and $A(BC)$ are $m \times q$: the first is $(m \times p)(p \times q)$, the second is $(m \times k)(k \times q)$.

Fix $i \in \lbrace 1, \ldots, m \rbrace$ and $j \in \lbrace 1, \ldots, q \rbrace$. Expanding the left side,

$$
\begin{aligned}
\left( (AB)C \right)_{ij}
  &= \sum_{s=1}^{p} (AB)_{is}\, C_{sj} \\
  &= \sum_{s=1}^{p} \left( \sum_{t=1}^{k} A_{it} B_{ts} \right) C_{sj} \\
  &= \sum_{s=1}^{p} \sum_{t=1}^{k} A_{it} B_{ts} C_{sj}
\end{aligned}
$$

and the right side,

$$
\begin{aligned}
\left( A(BC) \right)_{ij}
  &= \sum_{t=1}^{k} A_{it}\, (BC)_{tj} \\
  &= \sum_{t=1}^{k} A_{it} \left( \sum_{s=1}^{p} B_{ts} C_{sj} \right) \\
  &= \sum_{t=1}^{k} \sum_{s=1}^{p} A_{it} B_{ts} C_{sj}
\end{aligned}
$$

The two final expressions are sums of the same finite collection of terms $A_{it} B_{ts} C_{sj}$, indexed over the same rectangle $\lbrace 1, \ldots, k \rbrace \times \lbrace 1, \ldots, p \rbrace$, in different orders. Finite sums of real numbers may be reordered freely — associativity and commutativity of addition in $\mathbb{R}$ — so the two agree. Since $i$ and $j$ were arbitrary, $(AB)C = A(BC)$. $\blacksquare$

The finiteness matters. For infinite sums, exchanging the order requires absolute convergence, and the exchange can genuinely fail without it.

### 2. Transposes

**(a)** $\left( (A^\top)^\top \right)_{ij} = (A^\top)_{ji} = A_{ij}$, applying the definition twice. $\blacksquare$

**(b)** Group and apply the product rule twice:

$$
(ABC)^\top = \left( (AB)C \right)^\top = C^\top (AB)^\top = C^\top B^\top A^\top
$$

The first equality is associativity (exercise 1), which lets us bracket the triple product however we like. $\blacksquare$

**(c)** For $A \in \mathbb{R}^{m \times n}$, the product $A^\top A$ is $(n \times m)(m \times n) = n \times n$, so it is square and transposing it is meaningful. Then

$$
(A^\top A)^\top = A^\top (A^\top)^\top = A^\top A
$$

using the product rule and then part (a). Note that no assumption on $A$ was used: this holds for every matrix, rectangular included. $\blacksquare$

**(d)** With $S^\top = S$,

$$
(B^\top S B)^\top = B^\top S^\top (B^\top)^\top = B^\top S B
$$

$\blacksquare$ This is the fact that makes symmetry survive a change of basis, which is why covariance matrices and Hessians stay symmetric under every transformation applied to them later in the program.

### 3. The column reading

Let $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^{n}$. Consider the right-hand side and evaluate its $i$-th coordinate:

$$
\begin{aligned}
\left( \sum_{j=1}^{n} x_j A_{:,j} \right)_i
  &= \sum_{j=1}^{n} \left( x_j A_{:,j} \right)_i && \text{coordinates of a sum} \\
  &= \sum_{j=1}^{n} x_j \left( A_{:,j} \right)_i && \text{coordinates of a scalar multiple} \\
  &= \sum_{j=1}^{n} x_j A_{ij}                   && \text{entry } i \text{ of column } j \text{ is } A_{ij} \\
  &= (Ax)_i                                      && \text{definition of the product}
\end{aligned}
$$

True for every $i$, so the vectors are equal. $\blacksquare$

**Consequence.** As $\theta$ ranges over all of $\mathbb{R}^{d}$, the vector $X\theta$ ranges over exactly the set of linear combinations of the columns of $X$ — their span, a subspace of $\mathbb{R}^{n}$ of dimension at most $d$. When $d < n$, most vectors in $\mathbb{R}^{n}$ are unreachable by any choice of parameters, which is why fitting is approximation rather than solution.

### 4. The Gram matrix as a sum over examples

Compute entry $(j, l)$ of $X^\top X$ directly:

$$
\begin{aligned}
(X^\top X)_{jl}
  &= \sum_{i=1}^{n} (X^\top)_{ji} X_{il} && \text{definition of the product} \\
  &= \sum_{i=1}^{n} X_{ij} X_{il}        && \text{definition of transpose} \\
  &= \sum_{i=1}^{n} x^{(i)}_j x^{(i)}_l  && \text{row } i \text{ of } X \text{ is } (x^{(i)})^\top \\
  &= \sum_{i=1}^{n} \left( x^{(i)} \left( x^{(i)} \right)^\top \right)_{jl}
      && \text{entry } (j,l) \text{ of an outer product}
\end{aligned}
$$

True for every $(j, l)$, hence the matrix identity. $\blacksquare$

This is **reading 4**, the sum of outer products. The shared index of the product $X^\top X$ is the one being summed away, and here that index runs over the $n$ **examples**. Each example contributes one $d \times d$ rank-one matrix, and the Gram matrix is their total — which is exactly why it can be accumulated in a single pass over data too large to hold in memory.

### 5. Commutativity

**(a)** Take

$$
A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}
\qquad
B = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}
$$

Then

$$
AB = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}
\qquad
BA = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}
$$

These differ in every diagonal entry. (They are not even close: the products have disjoint supports.)

**(b)** Let $D$ and $E$ be diagonal, so $D_{it} = 0$ unless $t = i$. Then

$$
(DE)_{ij} = \sum_{t} D_{it} E_{tj} = D_{ii} E_{ij}
$$

since only $t = i$ contributes. As $E$ is diagonal, $E_{ij} = 0$ unless $i = j$, so $DE$ is diagonal with $(DE)_{ii} = D_{ii} E_{ii}$. Running the identical argument for $ED$ gives $(ED)_{ii} = E_{ii} D_{ii}$. Real multiplication commutes, so the two agree entry by entry. $\blacksquare$

**(c)** $\star$ Let $E^{(rs)}$ denote the matrix with a $1$ in position $(r, s)$ and zeros elsewhere, so $E^{(rs)}_{tu} = \mathbf{1}[t = r]\,\mathbf{1}[u = s]$. Compute both products:

$$
\left( A E^{(rs)} \right)_{ij} = \sum_{t} A_{it} E^{(rs)}_{tj} = A_{ir}\, \mathbf{1}[j = s]
$$

$$
\left( E^{(rs)} A \right)_{ij} = \sum_{t} E^{(rs)}_{it} A_{tj} = \mathbf{1}[i = r]\, A_{sj}
$$

By hypothesis these are equal for all $i, j, r, s$. Two choices finish it.

*Off-diagonal entries vanish.* Fix $r$ and $s$, and take $j = s$ and any $i \neq r$. The left side is $A_{ir}$; the right side is $\mathbf{1}[i = r] A_{ss} = 0$. Hence $A_{ir} = 0$ whenever $i \neq r$.

*Diagonal entries are all equal.* Take $i = r$ and $j = s$. The left side is $A_{rr}$ and the right side is $A_{ss}$, so $A_{rr} = A_{ss}$ for every $r$ and $s$.

So $A$ is diagonal with a single repeated value $c$, that is, $A = cI$. The converse is immediate, since $cI$ commutes with everything. $\blacksquare$

### 6. Rank-one structure

**(a)** $M = uv^\top$ is $(m \times 1)(1 \times p) = m \times p$, with entries $M_{ij} = u_i v_j$.

**(b)** Column $j$ of $M$ has $i$-th entry $u_i v_j$, that is, $M_{:,j} = v_j\, u$. Every column is the same vector $u$, rescaled by the corresponding entry of $v$. $\blacksquare$

**(c)** For $x \in \mathbb{R}^{p}$, associativity (exercise 1) lets us regroup:

$$
Mx = (u v^\top) x = u (v^\top x) = (v^\top x)\, u
$$

The middle step is legitimate because $v^\top x$ is $1 \times 1$, a scalar, and scalars commute with everything. So $Mx$ is always a multiple of $u$, with coefficient $v^\top x$. $\blacksquare$

**(d)** By (c) the image of $M$ is contained in $\mathrm{span}\lbrace u \rbrace$, which is one-dimensional when $u \neq 0$ — the defining property of a rank-one matrix. Reading 4 therefore writes any product as a sum of $k$ such pieces, each of which collapses its input onto a single direction.

### 7. What the intercept column does

Write $X = [\mathbf{1} \;\; \tilde{X}]$, so $X_{i1} = 1$ and $X_{ij} = \tilde{X}_{i,j-1}$ for $j > 1$.

**(a)** By the row reading, entry $i$ of $X\theta$ is the inner product of row $i$ of $X$ with $\theta$:

$$
(X\theta)_i = 1 \cdot b + \sum_{j=2}^{d} \tilde{X}_{i,j-1} w_{j-1} = b + w^\top \tilde{x}^{(i)}
$$

The affine model in $\tilde{x}$ is a linear model in $x$. $\blacksquare$

**(b)** $(X^\top X)_{11} = \sum_{i=1}^{n} X_{i1} X_{i1} = \sum_{i=1}^{n} 1 = n$.

**(c)** For $j > 1$,

$$
(X^\top X)_{1j} = \sum_{i=1}^{n} X_{i1} X_{ij} = \sum_{i=1}^{n} \tilde{X}_{i,j-1} = n\, \bar{\tilde{x}}_{j-1}
$$

where $\bar{\tilde{x}}_{j-1}$ is the mean of raw feature $j-1$ across the data. The first row of the Gram matrix is $n$ times the vector of column means, with $n$ itself in the leading position.

**(d)** $(X^\top X)_{1j} = 0$ for all $j > 1$ exactly when every raw feature has mean zero — the data is **centered**. Centering therefore decouples the intercept from the other parameters in the Gram matrix, and this is the reason centering is a standard preprocessing step rather than a cosmetic one. Part 3 returns to it.

### 8. Non-negativity of the quadratic form

**(a)** Two identities, named as used:

$$
\begin{aligned}
\theta^\top X^\top X \theta
  &= \left( X\theta \right)^\top \left( X\theta \right)
      && \text{since } (X\theta)^\top = \theta^\top X^\top \text{, the product rule for transposes} \\
  &= \sum_{i=1}^{n} \left( X\theta \right)_i^2
      && \text{definition of the inner product of a vector with itself} \\
  &= \sum_{i=1}^{n} \left( \theta^\top x^{(i)} \right)^2
      && \text{row reading: } (X\theta)_i = \theta^\top x^{(i)}
\end{aligned}
$$

$\blacksquare$

**(b)** Each term is the square of a real number, hence non-negative; a finite sum of non-negative reals is non-negative. $\blacksquare$

**(c)** ($\Leftarrow$) If $X\theta = 0$ then every term of the sum is $0^2 = 0$, so the total is $0$.

($\Rightarrow$) Suppose the sum is $0$. Every term is non-negative, so a single strictly positive term would make the total strictly positive. Hence every term is zero, so $(X\theta)_i = 0$ for all $i$, so $X\theta = 0$. $\blacksquare$

The quadratic form is therefore positive semi-definite but **not** positive definite in general: it vanishes on any $\theta$ that the columns of $X$ annihilate.

**(d)** $\star$ Take

$$
X = \begin{bmatrix} 1 & 1 \\ 2 & 2 \\ 3 & 3 \end{bmatrix} \in \mathbb{R}^{3 \times 2},
\qquad
\theta = \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

Then $X\theta = 0$ with $\theta \neq 0$ and $n = 3 > 2 = d$. The two columns of $X$ are identical, hence linearly dependent; by the column reading, $X\theta$ is the combination $1 \cdot A_{:,1} + (-1) \cdot A_{:,2}$, which cancels exactly.

More is true, and it matters in part 4: whenever the columns are linearly dependent there is a nonzero $\theta$ with $X\theta = 0$, so $X^\top X$ is singular and the least squares solution is not unique — adding any multiple of such a $\theta$ changes the parameters without changing a single prediction. The test `test_gram_of_collinear_data_is_singular` exhibits exactly this.

---

## B. Implementation

### 9. `matmul`

```python
if A.ndim != 2 or B.ndim != 2:
    raise ValueError(f"expected two 2-D arrays, got {A.ndim}-D and {B.ndim}-D")
if A.shape[1] != B.shape[0]:
    raise ValueError(f"inner dimensions disagree: {A.shape} and {B.shape}")
return (A[:, :, None] * B[None, :, :]).sum(axis=1)
```

`A[:, :, None]` has shape $(m, k, 1)$ and `B[None, :, :]` has shape $(1, k, p)$. Broadcasting expands both to $(m, k, p)$, so the elementwise product at position $(i, t, j)$ is $A_{it} B_{tj}$ — every term of every entry's sum, held at once. Summing over `axis=1` collapses the shared index $t$, leaving $(m, p)$ with entry $(i, j)$ equal to $\sum_t A_{it} B_{tj}$, which is the definition.

The shape checks are not optional. Without them, mismatched inputs can still broadcast — `(3, 4)` against `(4, 2)` fails loudly, but some mismatched shapes produce a silently wrong array instead of an error.

### 10. `matmul_outer`

```python
m, k = A.shape
p = B.shape[1]
total = np.zeros((m, p), dtype=np.float64)
for t in range(k):
    total += A[:, t, None] * B[None, t, :]
return total
```

`A[:, t, None]` is $(m, 1)$ and `B[None, t, :]` is $(1, p)$; broadcasting their product gives the $(m, p)$ outer product $A_{:,t} B_{t,:}$. Accumulating over $t$ gives $\sum_t A_{:,t} B_{t,:}$, which is reading 4.

The loop is over the shared dimension only. Each iteration does a fully vectorized outer product over all $mp$ entries, so the Python-level overhead is $k$ iterations rather than $mkp$.

### 11. `add_intercept`, `predict`, `gram`

```python
def add_intercept(X):
    n = X.shape[0]
    return np.hstack([np.ones((n, 1), dtype=np.float64), X])

def predict(X, theta):
    return X @ theta

def gram(X):
    return X.T @ X
```

Three points worth noticing. `predict` returns shape $(n,)$, not $(n, 1)$ — a one-dimensional array, matching the convention that a vector of predictions is a vector. `gram` uses `X.T`, which in NumPy is a **view** rather than a copy, so no data is moved. And `add_intercept` on an $(n, 0)$ array correctly yields the $(n, 1)$ all-ones matrix, the model with no features but an intercept, which is the constant predictor — the baseline every model in part 2 is measured against.

### 12. Cost of the grouping

```python
import time
import numpy as np

rng = np.random.default_rng(0)
n, d = 200_000, 200
X = rng.normal(size=(n, d))
theta = rng.normal(size=d)

t0 = time.perf_counter()
left = (X.T @ X) @ theta          # form the Gram matrix first
t1 = time.perf_counter()
right = X.T @ (X @ theta)         # apply to the vector first
t2 = time.perf_counter()

print(f"(X^T X) theta : {t1 - t0:.4f} s")
print(f"X^T (X theta) : {t2 - t1:.4f} s")
print(f"ratio         : {(t1 - t0) / (t2 - t1):.1f}x")
print(f"agree         : {np.allclose(left, right)}")
```

The two results agree to floating-point tolerance — they are the same quantity.

**Predicted ratio.** Grouping (a) costs $nd^2 + d^2 \approx nd^2 = 8 \times 10^{9}$ multiply-adds. Grouping (b) costs $2nd = 8 \times 10^{7}$. The operation counts predict roughly **100×**.

**Measured ratio.** You will typically observe something smaller — often 20× to 60×. The measurement is not wrong and neither is the analysis; they count different things.

The reason is *arithmetic intensity*. $X^\top X$ is a matrix–matrix product, which performs $O(nd^2)$ operations on $O(nd)$ data, so each element loaded from memory is reused $O(d)$ times. A tuned BLAS keeps those reused blocks in cache and runs near the processor's peak floating-point rate. $X\theta$ is a matrix–vector product: $O(nd)$ operations on $O(nd)$ data, each element used once. It is limited by memory bandwidth, not arithmetic, and runs at a small fraction of peak.

So grouping (b) does a hundred times less arithmetic but does it far less efficiently per operation. It still wins decisively — just by less than a naive count suggests. The general lesson holds for the rest of the program: **operation counts predict scaling, not wall-clock time.** When $n$ grows by 10, both the prediction and the measurement grow by 10; the constant between them is a property of the machine.

---

## C. Conceptual

### 13. The orientation of the design matrix

**(a)** NumPy stores arrays in row-major order, so with examples as rows a single example occupies a contiguous block of memory. Every operation that touches examples one at a time or in batches — shuffling, splitting into train and test, drawing a mini-batch in part 9 — then reads contiguous memory rather than striding across it. The convention also matches pandas and scikit-learn, so data can move between the mathematics and the libraries without a transpose that someone will eventually forget.

**(b)** A function written for $d \times n$ input, handed an $n \times d$ array, will compute $X\theta$ with $\theta$ of the wrong length and raise a shape error — the good case. The bad case is a function that internally computes something like $X X^\top$ instead of $X^\top X$: it returns an $n \times n$ matrix rather than $d \times d$, which for $n = 10^{6}$ is an attempt to allocate eight terabytes, or, for moderate $n$, a perfectly well-formed matrix of the wrong quantity that flows onward silently.

Square test data hides all of it. If $n = d$ then $X$ and $X^\top$ have the same shape, every product remains defined, and every result has a plausible shape. The bug appears only on the first rectangular input, which in practice is real data rather than the test fixture. This is why the suite for this subtopic checks shapes such as $(3, 4, 2)$ and $(7, 3, 1)$ and never relies on square cases.

### 14. When forming $X^\top X$ is the wrong move

**When it is right:** whenever you will solve with the same $X$ many times. Ridge regression over a grid of twenty values of $\lambda$ needs $X^\top X + \lambda I$ for each; forming the $O(nd^2)$ Gram matrix once and reusing it costs far less than twenty passes over the data. The same applies to any direct solver used repeatedly with different right-hand sides. The resource being conserved is **time**, and the saving comes from amortizing one expensive pass over many cheap solves.

**When it is wrong despite reuse:** forming $X^\top X$ squares the condition number of $X$. Roughly, if $X$ carries $\kappa$ as a measure of how nearly dependent its columns are, then $X^\top X$ carries $\kappa^2$, and you lose about half of your available significant digits before the solve even begins. On borderline data this converts a merely difficult problem into a numerically hopeless one, and methods that factor $X$ directly — QR, or the SVD of 1.1.6 — avoid the squaring entirely. The resource that runs out here is **precision**: the answer is not slow, it is wrong.

A third case is worth naming: when $d$ is very large, $X^\top X$ has $d^2$ entries and simply may not fit. There the resource is **memory**, and iterative methods that only ever compute $X^\top(Xv)$ are the only option.

### 15. The memory cost of the broadcast product

**(a)** The intermediate has $m \times k \times p = 10^{9}$ entries, at 8 bytes each for `float64`:

$$
10^{9} \times 8 = 8 \times 10^{9} \text{ bytes} = 8 \text{ GB}
$$

The inputs and the output together occupy about 24 MB. The temporary is more than three hundred times the size of the problem.

**(b)** A production BLAS never materializes the full set of products. It tiles the computation: it takes a block of rows of $A$ and a block of columns of $B$ small enough to sit in cache, accumulates that block's contribution directly into the corresponding block of the output, and moves on. Each output entry is updated in place as the shared index is traversed, so extra memory is a fixed number of cache-sized tiles regardless of the input size. This is also why it is fast — the blocking is chosen so operands are reused many times while resident in fast memory.

**(c)** `matmul_outer` makes the same trade in miniature: it accumulates into the $(m, p)$ output one rank-one contribution at a time, so only one $(m, p)$ temporary exists at any moment. What it gives up is doing everything in a single fused operation. It pays $k$ separate passes over the output array, each incurring Python-level loop overhead and re-traversing $mp$ elements, where the broadcast version traverses everything once. For small $k$ that cost is negligible; the broadcast version is preferable only while the $(m, k, p)$ temporary still fits comfortably in memory, which for realistic sizes it does not.

This is the general shape of the trade-off, and it recurs: **batch everything for speed, stream for memory.** Mini-batch gradient descent in part 1.4 is the same decision made about data instead of about intermediates.
