# Course 2 — Lesson 4 — Lab: Train a 2-layer network (NumPy)

**Prerequisites:** [Lesson 4.d — Python training loop](lesson_4d.md), [Lesson 4.b — Backpropagation](lesson_4b.md).

In this lab you implement a full **training loop** for a two-layer network (one hidden layer with **ReLU**, output with **sigmoid**), including **L2 regularization**, on a **tiny fixed dataset** so your outputs can be checked against a reference solution.

---

## The dataset

To keep outputs **deterministic** and easy to verify, use **exactly** the data below and **`np.random.seed(42)`** before initializing weights.

We simulate **$m = 4$** rides. Features are **Power** (row 1) and **Cadence** (row 2), scaled to $[0,1]$-ish values. Labels: **$1$ = Bonk**, **$0$ = No Bonk**.

| Ride | Power | Cadence | Label |
|------|--------|---------|--------|
| 1 | High | High | Bonk ($1$) |
| 2 | High | Low | No Bonk ($0$) |
| 3 | Low | High | No Bonk ($0$) |
| 4 | Low | Low | No Bonk ($0$) |

```python
import numpy as np

# Set seed for reproducible weight initialization
np.random.seed(42)

# Features: Row 1 is Power, Row 2 is Cadence. Shape: (2, 4)
X = np.array([[0.9, 0.8, 0.2, 0.1],
              [0.9, 0.2, 0.8, 0.1]])

# Labels: 1 = Bonk, 0 = No Bonk. Shape: (1, 4)
Y = np.array([[1, 0, 0, 0]])
```

---

## Your tasks

1. **Initialize** $\mathbf{W}^{[1]}$, $\mathbf{b}^{[1]}$, $\mathbf{W}^{[2]}$, $\mathbf{b}^{[2]}$. Use **`np.random.randn(...) * 0.01`** for weights and **`np.zeros(...)`** for biases. Choose shapes consistent with **$n_x = 2$** inputs, a chosen hidden size **$n_h$**, and **one** output logit (e.g. **$\mathbf{W}^{[1]}$**: $(n_h, 2)$, **$\mathbf{b}^{[1]}$**: $(n_h, 1)$, **$\mathbf{W}^{[2]}$**: $(1, n_h)$, **$\mathbf{b}^{[2]}$**: $(1, 1)$).

2. **Write a complete `train()` loop** (or script) that includes, each iteration:
   - **Forward propagation:** $\mathbf{Z}^{[1]}$, $\mathbf{A}^{[1]}$ (ReLU), $\mathbf{Z}^{[2]}$, $\mathbf{A}^{[2]}$ (sigmoid).
   - **Cost:** mean **binary cross-entropy** plus the **L2** penalty $\dfrac{\lambda}{2m}\bigl(\|\mathbf{W}^{[1]}\|_F^2 + \|\mathbf{W}^{[2]}\|_F^2\bigr)$ (pick a small $\lambda \ge 0$; $\lambda = 0$ is allowed if you document it).
   - **Backpropagation:** $\mathrm{d}\mathbf{Z}^{[2]}$, gradients for layer 2, $\mathrm{d}\mathbf{Z}^{[1]}$ with ReLU mask, gradients for layer 1, including **L2** on $\mathrm{d}\mathbf{W}^{[1]}$ and $\mathrm{d}\mathbf{W}^{[2]}$ when $\lambda > 0$.
   - **Gradient descent** update for all parameters with learning rate **$\alpha$**.

3. **Print** the cost at **iteration 0** (after the first forward + cost, before or after the first update—state which convention you use) and at **iteration 999** (e.g. run **1000** iterations with indices $0 \ldots 999$).

Use the same layout as in the lessons: **examples as columns** in $\mathbf{X}$ and $\mathbf{Y}$.

---

## Reference

Compare your numbers to the provided solution script when available: `lesson_4_lab_answers.py`.
