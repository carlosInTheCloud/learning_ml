---
type: Guideline
title: Conventions
description: Notation contract, math rendering rules, derivation policy, and code standards for all material in this repository.
tags: [conventions, notation, rendering, code-standards]
status: stable
generated: { by: human:carlos.espinosa, at: 2026-09-07T13:32:57Z }
trigger: always_on
---

# Conventions

Mechanical standards for authored material. [goal.md](goal.md) says what the work is for; this file says how it is written so that a hundred files produced across a hundred sessions read as one document.

## 1. Notation contract

Fixed for the whole program. A subtopic that needs a symbol not listed here defines it on first use and does not redefine one that is.

### Symbols

| Symbol | Meaning |
|---|---|
| $n$ | number of training examples |
| $d$ | number of features, including the intercept coordinate |
| $k$ | number of classes, clusters, or mixture components |
| $x^{(i)} \in \mathbb{R}^{d}$ | the $i$-th training example |
| $y^{(i)}$ | the label of the $i$-th example |
| $x_j$ | the $j$-th feature of a generic example |
| $X \in \mathbb{R}^{n \times d}$ | design matrix |
| $y \in \mathbb{R}^{n}$ | label vector |
| $\theta \in \mathbb{R}^{d}$ | model parameters |
| $h_\theta(x)$ | hypothesis / model prediction |
| $J(\theta)$ | the objective being minimized |
| $L(\hat{y}, y)$ | per-example loss |
| $\mathcal{L}(\theta)$ | likelihood |
| $\ell(\theta)$ | log-likelihood |
| $\alpha$ | learning rate |
| $\lambda$ | regularization strength |
| $\nabla_\theta J$ | gradient of $J$ with respect to $\theta$ |
| $H$ | Hessian |
| $\sigma(\cdot)$ | logistic sigmoid |
| $\mathbf{1}[\cdot]$ | indicator function |
| $\mathbb{E}[\cdot]$, $\mathrm{Var}(\cdot)$ | expectation, variance |
| $\|\cdot\|_2$ | Euclidean norm |

### Rules

1. **Vectors are columns.** $x^{(i)} \in \mathbb{R}^{d}$ is a column vector. Vectors are not bolded; the symbol table carries the type.
2. **Examples are rows of $X$.** Row $i$ of the design matrix is $\left(x^{(i)}\right)^{\top}$. This is the orientation NumPy and scikit-learn use, so code and math agree without a transpose in between. Half the literature uses the transpose — never mix the two inside the program.
3. **The intercept is absorbed.** Every example carries $x_0 = 1$, so $h_\theta(x) = \theta^{\top} x$ needs no separate bias term. Deep learning is the exception: from part 9 onward, weights $W$ and biases $b$ are separate, because that is how layers are actually built.
4. **Denominator layout for derivatives.** $\nabla_\theta J$ has the same shape as $\theta$; $\partial y / \partial x$ for $y \in \mathbb{R}^{m}$, $x \in \mathbb{R}^{n}$ is $n \times m$. Every derivation states the shape of its result.
5. **Indices.** Example index is a parenthesized superscript, $x^{(i)}$. Feature and component indices are subscripts, $x_j$, $\theta_j$. Iteration/step index is a bracketed superscript, $\theta^{[t]}$.
6. **Transpose is `^\top`**, rendering $\theta^{\top}$ — not `^T`.
7. **Estimates and predictions take hats:** $\hat{\theta}$, $\hat{y}$.

## 2. Math rendering

Files must render in **GitHub** (MathJax) and **MarkText** (KaTeX). The two engines overlap but are not identical, so material is written to their intersection. Neither is the binding constraint on its own — each rejects things the other accepts — so the rules below come from testing both, not from either engine's documentation.

- **Inline math:** `$ ... $`
- **Display math:** `$$ ... $$`, each delimiter on its own line, with a blank line before the opening one.

### Prohibited

| Do not use | Instead | Why |
|---|---|---|
| ` ```math ` fenced blocks | `$$ ... $$` | Renders on GitHub; MarkText shows a code block |
| `\[ ... \]`, `\( ... \)` | `$$ ... $$`, `$ ... $` | Not recognized by GitHub's markdown math |
| `\operatorname`, `\operatorname*` | `\underset{...}{\arg\min}` | Blocked by GitHub's macro allowlist |
| `\argmin`, `\argmax` | `\underset{\theta}{\arg\min}` | KaTeX-only; undefined on GitHub |
| `\mathbb{1}` | `\mathbf{1}` | Blackboard bold has no digit glyphs on GitHub |
| `\underbrace`, `\overbrace` | a following line of prose, or `\text` inside `aligned` | The brace glyph draws malformed on GitHub |
| `\{ ... \}` | `\lbrace ... \rbrace`, or `[ ... ]` for indicators | Fails on GitHub; markdown escaping is the likely cause |
| `\bm` | `\boldsymbol` | Not in KaTeX |
| bare `\begin{align}` | `aligned` inside `$$` | `align` is unsupported |
| `\newcommand`, `\DeclareMathOperator` | write the expression out | GitHub renders each block independently — macros do not carry across blocks or files |
| `\label`, `\ref` | number equations in prose | Unsupported in both |

### Verified working in both

`aligned` · `array` · `cases` · `matrix` / `pmatrix` / `bmatrix` / `vmatrix` · `\\` line breaks inside `aligned` · `\arg\min` with a subscript · `\underset` · `\mathop{...}\limits` · `\|`, `\lVert`/`\rVert`, `\Vert` · `\,` `\;` `\quad` · `\mathbb` (letters) · `\mathbf` · `\mathcal` · `\mathrm` · `\boldsymbol` · `\text` · `\top` · `\frac` · `\sum` · `\int` · `\partial` · `\nabla` · `\hat` · underscores inside `\text` · inline math containing underscores.

### Standard spellings

Fixed so that identical expressions are written identically everywhere:

| Expression | Write | Renders |
|---|---|---|
| argmin | `\underset{\theta}{\arg\min}` | $\underset{\theta}{\arg\min}$ |
| argmax | `\underset{\theta}{\arg\max}` | $\underset{\theta}{\arg\max}$ |
| indicator | `\mathbf{1}[y = k]` | $\mathbf{1}[y = k]$ |
| norm | `\|\theta\|_2^2` | $\|\theta\|_2^2$ |
| variance | `\mathrm{Var}(\cdot)` | $\mathrm{Var}(\cdot)$ |
| transpose | `\theta^\top` | $\theta^\top$ |

`\arg\min_{\theta}` also renders in both and is acceptable inline, where `\underset` sets awkwardly. In display math use `\underset`.

> Verified in GitHub and MarkText, September 2026. Re-test if either renderer changes.

## 3. Derivation policy

"No black boxes" from [goal.md](goal.md), made operational:

- **Derive in full** anything specific to machine learning: every model, loss, optimization routine, update rule, and bound.
- **Derive once, then cite** the mathematical results that part 1 covers. After part 1, a derivation may invoke them by name with a link back to the subtopic that established them.
- **State without proving** standard results that part 1 does not cover — the spectral theorem, measure-theoretic foundations of probability, convergence theorems from real analysis. State the result precisely, state its hypotheses, cite a source, and move on. Do not silently assume it.
- **Never use a result whose hypotheses have not been stated.** A cited theorem the reader cannot check the conditions of is a black box wearing a name.

## 4. Originality

All material is written from scratch. The program names university courses to calibrate difficulty, not to track their contents.

- **Never reproduce** another course's lecture notes, problem sets, solutions, figures, slides, or distinctive phrasing — from any source, including memory. This applies to exercises above all: an exercise that reads like it was lifted from a known problem set must be rewritten.
- **Freely use** standard mathematical results, standard notation, and the topics themselves. Facts, theorems, and curriculum outlines are not anyone's property; the normal equations belong to no one.
- **Cite** any external source a subtopic draws on — a textbook, a paper, a set of published notes — by name in the theory file.
- The test: could this derivation, exercise, or explanation have been written by someone who had learned the subject and then closed every book? If not, rewrite it.

## 5. Code standards

- **NumPy, vectorized.** No Python loop over training examples. Loops over iterations, layers, or folds are fine. The implementation should be readable as the equation it came from.
- **Shapes are documented and asserted.** Every public function's docstring gives the shape of each argument and its return using the symbols above — `(n, d)`, `(d,)`. Shape assertions belong in the tests.
- **`float64` throughout**, until part 9 makes precision itself the subject.
- **Randomness is seeded.** `rng = np.random.default_rng(0)`, passed explicitly. No calls to the global `np.random` state.
- **Type hints on public functions.**
- **Numerical stability is handled, not mentioned.** Log-sum-exp where sums of exponentials appear; solve rather than invert; explicit conditioning or regularization where a matrix may be near-singular.

### Stubs, solutions, and the shared package

- The stub (`{subtopic_name}.py`) carries the full signature, a docstring giving every shape, and `raise NotImplementedError`. It is not a sketch: everything except the body is finished, so the reader implements the algorithm and nothing else.
- The solution (`{subtopic_name}_solution.py`) is byte-for-byte compatible in signature. A test written against one runs unchanged against the other.
- Verify a suite by putting the solution in the stub's place and running it. Ship the stub unimplemented.
- Shared infrastructure — gradient checking, synthetic data, plotting — lives in `mlfs/` at the repository root and is imported directly (`from mlfs import numerical_gradient`). The root `conftest.py` makes this work from any depth.
- An implementation is promoted to `mlfs/models/` only after its own subtopic is complete, and never before.

### Tests

Every implementation ships with `test_{subtopic_name}.py`:

- **Gradient checks** against central finite differences, with relative error below `1e-6` for `float64`.
- **Equivalence** with a closed-form solution, or with `scikit-learn` / `scipy` on synthetic data, to a stated tolerance.
- **Shape and edge cases:** single example, single feature, perfectly separable or perfectly collinear data.
- Tests state what they verify. A test named `test_gradient` that asserts a number is not a verification.
- Tests are written against the stub's signature and must fail cleanly — `NotImplementedError`, not an import error — before the reader has implemented anything.

## 6. Prose

- The theory file opens with **why the problem exists** before any formalism.
- One idea per section; heading levels do not skip.
- Derivations show intermediate steps. A step justified by a non-obvious identity names the identity.
- British/American spelling is not policed; be consistent within a file.
