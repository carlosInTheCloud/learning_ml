# 1.1.1 Vectors and Matrices

**Part 1 — Mathematical Foundations · Topic 1.1 — Linear Algebra · Core**

**Prerequisites:** none. This is the first subtopic of the program.

---

## Symbols in this subtopic

Every symbol used below, how to say it, and what to call it out loud. Skim it now; the text glosses each one again where it first appears.

| Symbol | Say it | What it is |
|---|---|---|
| $n$ | "en" | number of examples — the rows of your data |
| $d$ | "dee" | number of features — the columns |
| $\mathbb{R}^{d}$ | "are dee" | the space of lists of $d$ real numbers |
| $x \in \mathbb{R}^{d}$ | "x is in are dee" | $x$ is a list of $d$ real numbers |
| $x_j$ | "x jay" | feature $j$ of a generic example |
| $x^{(i)}$ | "x i" | the $i$-th example — parentheses mark an index, not a power |
| $X$ | "capital X" | the design matrix — the whole table of data |
| $\theta$ | "theta" | the parameter vector — one weight per feature |
| $X\theta$ | "X theta" | the vector of predictions, one per example |
| $y$ | "y" | the vector of true labels |
| $A_{ij}$ | "A i jay" | the entry of $A$ in row $i$, column $j$ |
| $A_{:,j}$ | "column jay of A" | one whole column |
| $A^\top$ | "A transpose" | $A$ reflected across its diagonal |
| $x^\top z$ | "x transpose z" | the inner product of $x$ and $z$ — a single number |
| $X^\top X$ | "X transpose X" | the Gram matrix |
| $\sum_{i=1}^{n}$ | "sum from i equals one to en" | add up over all $n$ examples |
| $I$ | "the identity" | the matrix that changes nothing |

---

## 1. Why this comes first

A learning algorithm is handed a table. Rows are things that happened — a house that sold, an email that arrived, a patient who was scanned. Columns are the **features**: the individual numbers recorded about each one. Somewhere there is also a column of outcomes, and the job is to find a rule that turns a row into its outcome.

You could write that rule as a loop. For each of the $n$ rows, multiply each of the $d$ features by its weight, add them up, and record the answer. That description is correct, and it is how almost everyone first understands regression.

It is also the wrong level of abstraction, for three reasons that will not go away:

1. **It is slow.** A Python loop over a million rows is thousands of times slower than the same arithmetic dispatched to a linear algebra library, which uses cache blocking and vector instructions you are not going to write yourself.
2. **It hides the structure.** Written as a loop, "the vector of all predictions" is an accident of the code. Written as $X\theta$ — say "*X theta*": the table of data $X$ ("capital X") multiplied by the list of weights $\theta$ ("theta", a Greek letter, the standard name for a model's parameters) — it is a single object with properties — and one of those properties, that every achievable prediction vector lies in the column space of $X$, is the entire geometric content of least squares.
3. **It does not survive differentiation.** In part 1.2 you will differentiate a cost with respect to $\theta$. Differentiating $X\theta$ is a one-line matrix calculus rule. Differentiating a loop is not a thing you can do.

So the first job is to build the language in which the rest of the program is written. This subtopic covers exactly that: what vectors and matrices are, what multiplying them means, and why the design matrix is laid out the way it is. Norms and distances are 1.1.2; subspaces and projections are 1.1.3. Here we build the objects and the product.

---

## 2. Vectors

A **vector** in $\mathbb{R}^{d}$ — say "*are dee*", the space of all lists of $d$ real numbers — is an ordered list of $d$ real numbers. When you see $x \in \mathbb{R}^{d}$, say "*x is in are dee*": it announces the type of $x$ the way a signature announces the type of a function argument. Throughout this program a vector is a **column**:

$$
x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \in \mathbb{R}^{d}
$$

Two readings of the same object, and you need both:

- **As a point.** $x$ is a location in $d$-dimensional space. This reading makes distance, angle, and projection meaningful — the content of 1.1.2 and 1.1.3.
- **As data.** $x$ is one example — one house, one email, one patient — and $x_j$, say "*x jay*", is one **feature** of it: a single number recorded about that example. How many features there are is $d$.

A house might be recorded by its floor area in square feet, its number of bedrooms, the year it was built, and its distance in miles to the nearest school. Then $d = 4$, and that house *is* the vector

$$
x = \begin{bmatrix} 1850 \\ 3 \\ 1974 \\ 0.8 \end{bmatrix} \in \mathbb{R}^{4}
$$

with $x_1 = 1850$, $x_2 = 3$, and so on.

A feature need not be something physically measured with an instrument. It can be a count, a category turned into a number — has a garage: $1$ or $0$ — or a quantity derived from other features, such as price per square foot. Three things are required of it, and only these three: it is **one number**, it is present for **every** example, and it always sits in the **same position**. That last requirement is the one people underestimate. If $x_2$ means "bedrooms" for one house and "bathrooms" for another, the vector is meaningless and every result in this program silently breaks. Turning messy records into features that satisfy all three is the subject of part 3.

Two operations are defined, and they are the only two:

$$
(x + z)_j = x_j + z_j
\qquad\qquad
(c\,x)_j = c\,x_j
$$

for $c \in \mathbb{R}$. Everything else in linear algebra is built from these. Combining them gives the **linear combination**, the single most important expression in the subject:

$$
c_1 v_1 + c_2 v_2 + \cdots + c_k v_k
$$

The set of all linear combinations of $v_1, \ldots, v_k$ is their **span**, written $\mathrm{span}\lbrace v_1, \ldots, v_k \rbrace$. Hold on to the word: when we ask in part 4 which prediction vectors a linear model can possibly produce, the answer will be "the span of the columns of $X$," and nothing more needs to be said.

### 2.1 The inner product

For $x, z \in \mathbb{R}^{d}$,

$$
x^\top z = \sum_{j=1}^{d} x_j z_j \in \mathbb{R}
$$

Read the left side as "*x transpose z*", and the whole line as "*x transpose z is the sum, from j equals one to dee, of x jay times z jay*". The raised $\top$ is the **transpose**, defined properly in section 6; for now read it as the mark that turns a column on its side. The result is a single number, which is what $\in \mathbb{R}$ is asserting.

A column times a column is not defined; $x^\top z$ works because transposing $x$ makes it a $1 \times d$ row, and a $1 \times d$ times a $d \times 1$ is $1 \times 1$. The bookkeeping is not pedantry — it is what makes every later shape check mechanical.

The inner product is **symmetric** ($x^\top z = z^\top x$) and **linear in each argument**:

$$
x^\top (a z + b w) = a\, x^\top z + b\, x^\top w
$$

Both follow immediately from the definition by splitting the sum. The geometric meaning — that $x^\top z$ measures how much of $x$ points along $z$ — is developed in 1.1.2, where it is needed. Here it is a sum of products, and that is enough.

---

## 3. Matrices

A **matrix** $A \in \mathbb{R}^{m \times n}$ — say "*A is in are, em by en*" — is a rectangular array with $m$ rows and $n$ columns, with $A_{ij}$ — "*A i jay*" — the entry in row $i$, column $j$. Rows first, always.

Like vectors, matrices carry two readings, and confusing them is the single most common source of transposed-shape bugs:

- **As data.** A table. Row $i$ is the $i$-th example; column $j$ holds feature $j$ across all examples. This is what the design matrix is.
- **As a linear map.** $A$ is a function that eats a vector in $\mathbb{R}^{n}$ and returns a vector in $\mathbb{R}^{m}$, and it is precisely a function satisfying

$$
A(a x + b z) = a\,(Ax) + b\,(Az)
$$

The second reading is the deeper one. Every linear map from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$ is a matrix, and every matrix is such a map — the two notions are the same notion. This is why "linear model" and "matrix" turn out to be the same word.

---

## 4. The matrix–vector product

For $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^{n}$, the product $Ax \in \mathbb{R}^{m}$ is defined entry-wise by

$$
(Ax)_i = \sum_{j=1}^{n} A_{ij} x_j
$$

One definition, two readings. Learn both; different arguments need different ones.

### 4.1 The row reading

Entry $i$ of $Ax$ is the inner product of row $i$ of $A$ with $x$:

$$
Ax = \begin{bmatrix}
  a_1^\top x \\ a_2^\top x \\ \vdots \\ a_m^\top x
\end{bmatrix}
\qquad \text{where } a_i^\top \text{ is row } i \text{ of } A
$$

This is the reading that makes $X\theta$ mean *"score every example."* Row $i$ of $X$ is the example $x^{(i)}$, so entry $i$ of $X\theta$ is $\theta^\top x^{(i)}$ — the model's prediction for example $i$. One product, all $n$ predictions.

### 4.2 The column reading

Group the same double sum the other way. Writing $A_{:,j}$ — say "*column jay of A*"; the colon means "all rows", borrowed from array-slicing notation — for column $j$ of $A$,

$$
Ax = x_1 A_{:,1} + x_2 A_{:,2} + \cdots + x_n A_{:,n}
$$

**$Ax$ is a linear combination of the columns of $A$, with the entries of $x$ as coefficients.**

Verify that these agree by expanding the right-hand side at coordinate $i$. The $j$-th term contributes $x_j (A_{:,j})_i = x_j A_{ij}$, so the sum at coordinate $i$ is $\sum_j A_{ij} x_j$, which is the definition.

The column reading is the one that answers *"what can this model produce?"* As $\theta$ ranges over all of $\mathbb{R}^{d}$, $X\theta$ ranges over the span of the columns of $X$ — a subspace of $\mathbb{R}^{n}$ of dimension at most $d$. When $d < n$, which is the usual case, most vectors $y \in \mathbb{R}^{n}$ are **not** reachable. Least squares is the problem of finding the reachable vector closest to $y$, and 1.1.3 makes that precise.

---

## 5. The matrix–matrix product

For $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times p}$, the product $AB \in \mathbb{R}^{m \times p}$ has entries

$$
(AB)_{ij} = \sum_{t=1}^{k} A_{it} B_{tj}
$$

The shared dimension $k$ is summed away; it must match, and it disappears. $(m \times k)(k \times p) \to (m \times p)$.

The definition admits four groupings of the same triple sum. Each is a genuine tool.

**Reading 1 — entries as inner products.** $(AB)_{ij}$ is row $i$ of $A$ dotted with column $j$ of $B$. This is the definition read literally, and the one used to prove identities entry by entry.

**Reading 2 — columns.** Column $j$ of $AB$ is $A$ applied to column $j$ of $B$:

$$
(AB)_{:,j} = A\,B_{:,j}
$$

So a matrix product is just the matrix–vector product done to every column at once. This makes $AB$ the *composition* of the two maps: apply $B$, then apply $A$.

**Reading 3 — rows.** Symmetrically, row $i$ of $AB$ is row $i$ of $A$ applied to $B$ from the left.

**Reading 4 — sum of outer products.** Group the sum over the shared index $t$ last instead of first:

$$
AB = \sum_{t=1}^{k} A_{:,t}\, B_{t,:}
$$

Each term is a column times a row — an $m \times p$ matrix of **rank one**. The product is a sum of $k$ rank-one pieces.

This fourth reading is the one people meet last and use most. It is why the Gram matrix is a sum over examples (section 8), it is the shape of every low-rank factorization in 1.1.7 and part 7, and it is how a gradient accumulates contributions example by example in part 4. The implementation builds the product twice, once by reading 1 and once by reading 4, precisely so that both are in your hands.

### 5.1 Associative, distributive, not commutative

$$
(AB)C = A(BC)
\qquad
A(B + C) = AB + AC
\qquad
AB \neq BA \text{ in general}
$$

Associativity is proved by expanding both sides into the same double sum over the two shared indices and exchanging the order of finite summation — an exercise. Non-commutativity is not a defect; it is the content. $AB$ means "do $B$, then $A$," and composing operations in the other order is a different operation. Even when both products are defined and square, they generally differ.

Associativity has teeth. It is a mathematical identity, so both orders give the same answer — but they need not cost the same, and section 9 shows a case where the difference is a factor of thousands.

---

## 6. Transpose

The **transpose** $A^\top$ reflects a matrix across its diagonal:

$$
(A^\top)_{ij} = A_{ji}
\qquad
A \in \mathbb{R}^{m \times n} \implies A^\top \in \mathbb{R}^{n \times m}
$$

The rule that matters:

$$
(AB)^\top = B^\top A^\top
$$

**Proof.** Both sides are $p \times m$, so it is enough to compare entries. Fix $i$ and $j$:

$$
\begin{aligned}
\left( (AB)^\top \right)_{ij}
  &= (AB)_{ji}                       && \text{definition of transpose} \\
  &= \sum_{t=1}^{k} A_{jt} B_{ti}    && \text{definition of the product} \\
  &= \sum_{t=1}^{k} (B^\top)_{it} (A^\top)_{tj} && \text{definition of transpose, twice} \\
  &= (B^\top A^\top)_{ij}            && \text{definition of the product}
\end{aligned}
$$

Since $i$ and $j$ were arbitrary, the matrices are equal. $\blacksquare$

The order reverses, and it must: $A$ is $m \times k$ and $B$ is $k \times p$, so $A^\top B^\top$ would be $k \times m$ times $p \times k$, which is not even defined unless $m = p$. The shapes tell you the rule before the algebra does.

---

## 7. Matrices worth naming

| Name | Definition | Where it shows up |
|---|---|---|
| Identity $I_n$ | $I_{ij} = \mathbf{1}[i = j]$ | $AI = A$, $IA = A$; the map that does nothing |
| Diagonal | $A_{ij} = 0$ for $i \neq j$ | Per-feature scaling; eigenvalue matrices in 1.1.5 |
| Symmetric | $A^\top = A$ | Gram matrices, covariance, Hessians — everything with a spectral theorem |
| Orthogonal | $Q^\top Q = I$ | Rotations; the factors of the SVD in 1.1.6 |

Symmetric matrices deserve the emphasis. Nearly every matrix this program actually optimizes over is symmetric, and symmetry is what buys real eigenvalues, orthogonal eigenvectors, and the whole spectral apparatus of 1.1.5.

---

## 8. The design matrix

Here is the convention this program uses everywhere, stated once:

$$
X \in \mathbb{R}^{n \times d}, \qquad
\text{row } i \text{ of } X \text{ is } \left( x^{(i)} \right)^\top
$$

Say it as "*capital X is in are, en by dee; row i of X is x i transpose*". The symbol $x^{(i)}$ is read "*x i*" — the parentheses around the index are there precisely so it is not mistaken for a power, since $x^2$ and $x^{(2)}$ mean entirely different things: the square of $x$, and the second example.

$n$ examples down, $d$ features across. Examples are **rows**. This matches NumPy, pandas, and scikit-learn, so the mathematics and the code agree without a transpose sitting between them. A good deal of the literature uses the opposite convention; mixing the two is the most common way to produce a program that runs, returns the wrong shape, and gives no error.

Note the small clash of readings: $x^{(i)}$ is a **column** vector in $\mathbb{R}^{d}$, but it appears in $X$ as a **row**. Hence the transpose in the line above. This is not an inconsistency to be fixed — it is the reason $X^\top$ appears as often as it does.

### 8.1 The intercept

A linear model should be able to represent $h(x) = b + w^\top x$ with an offset $b$. Rather than carry $b$ separately, prepend a constant coordinate $x_0 = 1$ to every example. Then with $\theta = (b, w_1, \ldots)^\top$,

$$
\theta^\top x = b \cdot 1 + w_1 x_1 + \cdots = b + w^\top x
$$

The affine model becomes linear, at the cost of one column of ones. Every $d$ in this program counts that column. `add_intercept` in the implementation is this one idea.

Part 9 drops the convention: neural network layers keep weights $W$ and biases $b$ apart, because a layer is applied to many different inputs and appending ones to each is wasteful and awkward. The convention is a convenience, not a law.

### 8.2 Three objects you will meet constantly

With $X \in \mathbb{R}^{n \times d}$, $\theta \in \mathbb{R}^{d}$, $y \in \mathbb{R}^{n}$:

$$
X\theta \in \mathbb{R}^{n}
\qquad
X\theta - y \in \mathbb{R}^{n}
\qquad
X^\top X \in \mathbb{R}^{d \times d}
$$

The first is every prediction. The second is every residual. The third, $X^\top X$ — say "*X transpose X*" — is the **Gram matrix** (rhymes with "programme"), and it repays a close look.

Apply the outer-product reading of section 5 to $X^\top X$. The shared index runs over the $n$ examples, so

$$
X^\top X = \sum_{i=1}^{n} x^{(i)} \left( x^{(i)} \right)^\top
$$

Three things follow, all of which matter later:

- **It is symmetric.** $(X^\top X)^\top = X^\top (X^\top)^\top = X^\top X$, using section 6 twice.
- **It is $d \times d$, not $n \times n$.** With a million examples and twenty features, it is a $20 \times 20$ matrix. The data size has been summed away, which is what makes the normal equations tractable in part 4.
- **Its quadratic form is never negative.** For any $\theta$,

$$
\theta^\top X^\top X \theta = (X\theta)^\top (X\theta) = \sum_{i=1}^{n} \left( \theta^\top x^{(i)} \right)^2 \geq 0
$$

using $(X\theta)^\top = \theta^\top X^\top$ from section 6. That inequality is why the least squares cost has a minimum rather than a saddle. It is the definition of positive semi-definiteness, developed properly in 1.1.4.

---

## 9. What it costs

Computing $AB$ for $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times p}$ requires $mkp$ multiply-add operations: one per entry of the $m \times p$ result, times $k$ terms in each sum. Multiplying two $n \times n$ matrices by this definition is $O(n^3)$.

Associativity says $(AB)C$ and $A(BC)$ are equal. It says nothing about their cost. Take the extremely common case

$$
X^\top X \theta
\qquad
X \in \mathbb{R}^{n \times d},\ \theta \in \mathbb{R}^{d}
$$

- **Left to right:** form $X^\top X$ at $O(nd^2)$, then apply it to $\theta$ at $O(d^2)$. Total $O(nd^2)$.
- **Right to left:** form $X\theta$ at $O(nd)$, then $X^\top(X\theta)$ at $O(nd)$. Total $O(nd)$.

With $n = 10^6$ and $d = 100$, that is $10^{10}$ operations against $10^{8}$ — a hundredfold difference from moving parentheses, with identical output. Iterative methods in part 1.4 and part 4 depend on the second grouping; forming the Gram matrix once and reusing it is right when you will solve repeatedly with the same $X$. Neither is universally correct, which is exactly why you have to know it is a choice.

---

## 10. Implementation

Five functions, in `vectors_and_matrices.py`. The two products must be built out of elementwise multiplication and summation — no `@`, no `np.dot`, no `np.einsum`. A test reads the source and enforces it, because the exercise is to construct the operation rather than to call it.

`matmul` is reading 1, made vectorized by broadcasting rather than by looping over entries:

```python
return (A[:, :, None] * B[None, :, :]).sum(axis=1)
```

$A$ is viewed as $(m, k, 1)$ and $B$ as $(1, k, p)$; broadcasting stretches both to $(m, k, p)$, the elementwise product holds every term $A_{it}B_{tj}$, and summing over axis 1 collapses the shared index. Note the cost: that intermediate array holds $mkp$ floats. For $1000 \times 1000$ matrices it is eight gigabytes, whereas a real BLAS routine does the same arithmetic in fixed extra memory. Building the operation and using the library version are different activities, and this is why the second one exists.

`matmul_outer` is reading 4, accumulating rank-one pieces:

```python
for t in range(k):
    total += A[:, t, None] * B[None, t, :]
```

The loop runs over the shared dimension, not over examples or entries, so each iteration is a full vectorized outer product. The memory is $O(mp)$ rather than $O(mkp)$ — the same trade a streaming algorithm makes when it accumulates over data it cannot hold at once.

The remaining three are the ML shapes: `add_intercept` prepends the ones column of section 8.1, `predict` computes $X\theta$, and `gram` computes $X^\top X$.

---

## 11. Where this breaks

- **Silent broadcasting.** NumPy will happily compute `A * B` elementwise when you meant `A @ B`, and if the shapes broadcast you get a wrong answer with no error. Shape assertions are not decoration.
- **`(n,)` is not `(n, 1)`.** A one-dimensional array is neither a row nor a column, and NumPy's rules for it differ from both. `X @ theta` with `theta` of shape `(d,)` returns `(n,)`; with shape `(d, 1)` it returns `(n, 1)`. Mixing them produces `(n, n)` outer products where you expected a vector — often the real cause of a mysterious memory blow-up.
- **Transposed conventions.** Code written against a $d \times n$ design matrix will run against an $n \times d$ one whenever $n = d$, and only then. Test on non-square data.
- **Collinear columns.** If one feature is a copy or an exact linear combination of others, $X^\top X$ is singular and cannot be inverted. Nothing above breaks, but the model in part 4 will. The test suite includes this case so you see it now rather than later.
- **Cost is not symmetry.** $(AB)C$ and $A(BC)$ agree in value and can differ by orders of magnitude in time. The compiler will not fix it for you.

---

## 12. Check yourself

Answer without looking back. If any of these needs the text, the section it comes from has not landed yet.

1. State the two readings of $Ax$, and say which one answers "what predictions can this model produce?"
2. Why does $(AB)^\top = B^\top A^\top$ reverse the order? Give the shape argument, not the entry-wise proof.
3. $X$ is $n \times d$. What are the shapes of $X^\top X$ and $XX^\top$, and which one do you want when $n$ is a million?
4. Write $X^\top X$ as a sum over examples. Which reading of the matrix product gives it?
5. Why is $\theta^\top X^\top X \theta \geq 0$ for every $\theta$? One line.
6. You need $X^\top X\theta$ once, with $n = 10^6$ and $d = 50$. Which grouping, and how many operations does each take?
7. What exactly goes wrong if you pass a `(d, 1)` array where the code expects `(d,)`?

---

## 13. What comes next

**1.1.2 Matrix operations and norms** puts a size on these objects: vector and matrix norms, the geometry of the inner product, and the Cauchy–Schwarz inequality. Once vectors have lengths, "the closest reachable prediction" of section 4.2 becomes a well-posed question — and 1.1.3 answers it with projection.
