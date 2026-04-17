# Course 2 — Lesson 4.d — Neural Networks: Python Functions

After this lesson you should be able to:

- Map each line of a small **NumPy** training loop to [Lesson 4.b](lesson_4b.md) (forward, cost, backprop, L2).
- Use layout **examples as columns** (`X.shape[1] == m`) consistently with **$\mathbf{W}\mathbf{x}$**-style formulas.
- Read **$\texttt{lambd}$** (L2 strength) in the cost and in **$\mathrm{d}\mathbf{W}$** updates.

**Prerequisites:** [Lesson 4.a](lesson_4a.md), [Lesson 4.b](lesson_4b.md), and optionally [Lesson 4.c](lesson_4c.md) (manual numbers).

This lesson shows a **minimal** `train_neural_network` routine: one **hidden** layer with **ReLU**, one **output** with **sigmoid**, **binary cross-entropy** loss, **L2** on weights, and **batch** gradient descent over all $m$ examples each iteration. It is the same math as Lessons 4.a–4.b, expressed in code.

**Layout:** `X` has shape `(n_x, m)` and `Y` has shape `(1, m)`—each **column** is one example. That matches `np.dot(W1, X)` so that `W1` is `(n_h, n_x)` and `Z1` is `(n_h, m)`.

**Naming:** The regularization strength is called `lambd` because `lambda` is a **reserved keyword** in Python.

---

## Training loop (full function)

```python
import numpy as np


def train_neural_network(X, Y, num_iterations, alpha, lambd):
    """
    Trains a 2-layer neural network using gradient descent.
    """
    m = X.shape[1]

    # ---------------------------------------------------------
    # PRE-LOOP: INITIALIZATION
    # (Assume initialize_parameters creates our starting W and b)
    # ---------------------------------------------------------
    W1, b1, W2, b2 = initialize_parameters()

    # ---------------------------------------------------------
    # THE TRAINING LOOP
    # ---------------------------------------------------------
    for i in range(num_iterations):

        # --- STEP 1: FORWARD PROPAGATION ---
        Z1 = np.dot(W1, X) + b1
        A1 = np.maximum(0, Z1)  # ReLU

        Z2 = np.dot(W2, A1) + b2
        A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid

        # --- STEP 2: CALCULATE COST (To monitor learning) ---
        # Standard cross-entropy plus L2 weight decay penalty
        cross_entropy = -(1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
        l2_penalty = (lambd / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        cost = cross_entropy + l2_penalty

        # --- STEP 3: BACKPROPAGATION ---
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T) + (lambd / m) * W2
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        relu_derivative = (Z1 > 0).astype(float)
        dZ1 = np.dot(W2.T, dZ2) * relu_derivative
        dW1 = (1 / m) * np.dot(dZ1, X.T) + (lambd / m) * W1
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # --- STEP 4: UPDATE PARAMETERS ---
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        # --- STEP 5: PRINT PROGRESS ---
        if i % 100 == 0:
            print(f"Iteration {i} | Cost: {cost:.4f}")

    # Return the trained weights so they can be used to predict new data
    return W1, b1, W2, b2
```

---

## How each step maps to the lessons

| Code | Lesson link |
|------|-------------|
| `Z1`, `A1` | [4.a](lesson_4a.md): linear + ReLU hidden layer |
| `Z2`, `A2` | [4.a](lesson_4a.md): linear + sigmoid output |
| `cross_entropy` | Mean binary cross-entropy over $m$ examples |
| `l2_penalty` | [4.b](lesson_4b.md): $\frac{\lambda}{2m}(\|W_1\|_F^2 + \|W_2\|_F^2)$ |
| `dZ2`, `dW2`, `db2` | [4.b](lesson_4b.md) Phase 1 (+ L2 on `dW2`) |
| `relu_derivative`, `dZ1`, `dW1`, `db1` | [4.b](lesson_4b.md) Phase 2 (+ L2 on `dW1`) |
| parameter updates | Gradient descent with learning rate `alpha` |

---

## Details worth a second look

- **`initialize_parameters`:** Not defined above—you supply it (e.g. small random `W`, zeros for `b`, or He/Xavier-style scaling). Shapes must match `X`: `W1.shape == (n_h, n_x)`, `b1.shape == (n_h, 1)`, `W2.shape == (1, n_h)`, `b2.shape == (1, 1)`.
- **`np.log(A2)`:** If `A2` can hit exactly `0` or `1`, the log can be `-inf`. Production code often **clips** probabilities (e.g. `1e-15`, `1 - 1e-15`) or uses a numerically stable BCE implementation.
- **Biases:** L2 penalty includes **only** `W1` and `W2`, not `b1`/`b2`, matching Lesson 4.b.
- **ReLU derivative at 0:** `(Z1 > 0)` uses **False → 0**, same convention as in Lesson 4.c.

---

## Key takeaways

- One loop iteration = **forward** → **cost** → **backward** → **update**; that is one step of batch gradient descent.
- **`m = X.shape[1]`** appears in averages for both cost and gradients; vectorization over examples is exactly the “superpower” from [Lesson 4.a](lesson_4a.md).
- **`dZ2 = A2 - Y`** is the same shortcut as in Lesson 4.b for sigmoid + mean BCE.
- **`(lambd / m) * W`** in `dW1` / `dW2` pairs with the L2 term in the cost; set `lambd = 0` to drop regularization.
