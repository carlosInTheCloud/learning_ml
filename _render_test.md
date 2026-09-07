# Math rendering smoke test

Disposable. Open in **MarkText** and view on **GitHub**. Note any block that fails in either.

---

**1. Inline math.** The design matrix is $X \in \mathbb{R}^{n \times d}$ and the parameter vector is $\theta \in \mathbb{R}^{d}$.

**2. Display math.**

$$
J(\theta) = \frac{1}{2n} \sum_{i=1}^{n} \left( \theta^\top x^{(i)} - y^{(i)} \right)^2
$$

**3. Multi-line derivation (`aligned`).**

$$
\begin{aligned}
\nabla_\theta J(\theta)
  &= \frac{1}{n} \sum_{i=1}^{n} \left( \theta^\top x^{(i)} - y^{(i)} \right) x^{(i)} \\
  &= \frac{1}{n} X^\top (X\theta - y)
\end{aligned}
$$

**4. Matrices.**

$$
X = \begin{bmatrix}
  x_1^{(1)} & \cdots & x_d^{(1)} \\
  \vdots    & \ddots & \vdots \\
  x_1^{(n)} & \cdots & x_d^{(n)}
\end{bmatrix}
\qquad
\Sigma = \frac{1}{n} X^\top X
$$

**5. Operators and norms.**

$$
\hat{\theta} = \operatorname*{arg\,min}_{\theta \in \mathbb{R}^d} \; \|X\theta - y\|_2^2 + \lambda \|\theta\|_2^2
$$

**6. Cases.**

$$
\ell(z) =
\begin{cases}
  0        & \text{if } z \geq 1 \\
  1 - z    & \text{otherwise}
\end{cases}
$$

**7. Annotation and bold symbols.**

$$
\underbrace{\mathbb{E}\left[(\hat{f}(x) - f(x))^2\right]}_{\text{MSE}}
= \underbrace{\text{Bias}^2}_{\boldsymbol{b}} + \underbrace{\text{Var}}_{\boldsymbol{v}}
$$

**8. Underscores next to text** (a known GitHub parser trap): the loss $L_{\text{train}}$ versus $L_{\text{test}}$.

**9. Fenced `math` block** (GitHub-only syntax; expected to render as a *code block* in MarkText — confirming this is why we avoid it):

```math
p(y \mid x; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y - \theta^\top x)^2}{2\sigma^2}\right)
```

**10. Code block with math nearby.**

```python
import numpy as np

def grad(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gradient of MSE: (1/n) X^T (X theta - y)."""
    n = X.shape[0]
    return X.T @ (X @ theta - y) / n
```
