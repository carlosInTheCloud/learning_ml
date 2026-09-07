# Math rendering — round 2

Round 1 result: GitHub rejects `\operatorname` (macro allowlist), and appears to strip the backslash from `\,` and `\|` inside `$$` before the math engine runs. MarkText accepted all of it.

Below: every candidate replacement. **Report which letters render correctly in each viewer.** A candidate passes only if it works in both.

---

## A. arg min

**A1** — `\arg\min` with subscript

$$
\hat{\theta} = \arg\min_{\theta \in \mathbb{R}^d} J(\theta)
$$

**A2** — `\underset`

$$
\hat{\theta} = \underset{\theta \in \mathbb{R}^d}{\arg\min} \, J(\theta)
$$

**A3** — `\mathop` with `\limits`

$$
\hat{\theta} = \mathop{\arg\min}\limits_{\theta \in \mathbb{R}^d} J(\theta)
$$

**A4** — bare `\argmin` (KaTeX may define it; MathJax likely not)

$$
\hat{\theta} = \argmin_{\theta \in \mathbb{R}^d} J(\theta)
$$

**A5** — unstarred `\operatorname`, to confirm the whole macro is blocked

$$
\hat{\theta} = \operatorname{arg\,min}_{\theta} J(\theta)
$$

---

## B. Norms

**B1** — `\|` (round 1 suggests the backslashes are stripped)

$$
J(\theta) = \|X\theta - y\|_2^2 + \lambda \|\theta\|_2^2
$$

**B2** — `\lVert` / `\rVert`

$$
J(\theta) = \lVert X\theta - y \rVert_2^2 + \lambda \lVert \theta \rVert_2^2
$$

**B3** — `\Vert`

$$
J(\theta) = \Vert X\theta - y \Vert_2^2 + \lambda \Vert \theta \Vert_2^2
$$

---

## C. Thin spaces

**C1** — `\,`

$$
\mathbb{E}[X] = \int x \, p(x) \, dx
$$

**C2** — `\;`

$$
\mathbb{E}[X] = \int x \; p(x) \; dx
$$

**C3** — `\quad`

$$
\mathbb{E}[X] = \int x \quad p(x) \quad dx
$$

---

## D. Other escape-prone constructs

**D1** — `\\` line breaks inside `aligned` (used in every multi-line derivation)

$$
\begin{aligned}
\nabla_\theta J &= X^\top (X\theta - y) \\
\theta^{[t+1]} &= \theta^{[t]} - \alpha \nabla_\theta J
\end{aligned}
$$

**D2** — `\{` `\}` braces

$$
\mathbb{1}\{ y^{(i)} = k \}
$$

**D3** — `\_` and underscores in `\text`

$$
L_{\text{train}} < L_{\text{test}}
$$

**D4** — inline math with an underscore mid-sentence: the value $\theta_j$ at step $t$.

---

## E. Control

**E1** — fenced `math` block. Expected: renders on GitHub, shows as a code block in MarkText.

```math
p(y \mid x; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y - \theta^\top x)^2}{2\sigma^2}\right)
```
