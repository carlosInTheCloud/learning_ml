# Course 2 — Lesson 4.c — Neural Networks: Manual Example

After this lesson you should be able to:

- Trace **forward** and **backward** one step for a two-layer net and align each step with [Lesson 4.b](lesson_4b.md) (Phases 1 and 2).
- Interpret the **sign** of $\mathrm{d}Z^{[2]}$ and explain **zero** gradients caused by **zero** hidden activations and by the **ReLU** mask.
- Read a small **gradient-descent** update and say which parameters move and why.

**Prerequisites:** [Lesson 4.a — Forward propagation](lesson_4a.md), [Lesson 4.b — Backpropagation](lesson_4b.md).

This lesson walks through **one training example** by hand: forward pass, backward pass, and one gradient-descent update. We use **binary cross-entropy** + **sigmoid** at the output and **ReLU** in the hidden layer, matching Lessons 4.a–4.b.

**Roadmap:** setup → forward prediction → backward (“blame”) → parameter update.

**Convention:** **$m = 1$** example, so **$1/m = 1$** and averages in the gradient formulas reduce to “just the value.” **Examples as column vectors**; **$\mathbf{W}^{[1]}$** is **$3 \times 2$** (three neurons, two inputs); **$\mathbf{W}^{[2]}$** is **$1 \times 3$** (one output logit).

**Rounding:** Displayed decimals are rounded; chained steps use the same rounded values so the story stays consistent (not full floating-point precision).

---

## The setup

### 1. Input and true label

Take a single input (two features) and label $y = 1$:

$$
\mathbf{x} = \mathbf{A}^{[0]} = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix}, \qquad y = 1.
$$

### 2. Layer 1 (3 neurons, ReLU)

$$
\mathbf{W}^{[1]} = \begin{bmatrix}
0.3 & 0.6 \\
-0.1 & 0.6 \\
0.2 & 0.8
\end{bmatrix}, \qquad
\mathbf{b}^{[1]} = \begin{bmatrix} -0.1 \\ -0.2 \\ 0 \end{bmatrix}.
$$

### 3. Layer 2 (1 neuron, sigmoid)

$$
\mathbf{W}^{[2]} = \begin{bmatrix} 0.9 & 0.3 & 0.5 \end{bmatrix}, \qquad b^{[2]} = 0.
$$

(The middle entry $0.3$ multiplies the second hidden unit; it will **not** affect the forward value when that unit is zero after ReLU.)

---

## The prediction

```text
x = A^[0]  -->  W1, b1  -->  Z1  --ReLU-->  A1  -->  W2, b2  -->  Z2  --sigmoid-->  y_hat = A^[2]
       ^                                              |
       └---------------- backprop ---------------------┘
```

### Step 1: Forward propagation

Push $\mathbf{x}$ through the network to get $\hat{y}$.

#### Layer 1: linear step ($\mathbf{Z}^{[1]}$)

$$
\mathbf{Z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} = \begin{bmatrix} 0.5 \\ 0.0 \\ 0.6 \end{bmatrix}.
$$

So the **pre-activation** vector is:

$$
\mathbf{Z}^{[1]} = \begin{bmatrix} z^{[1]}_1 \\ z^{[1]}_2 \\ z^{[1]}_3 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.0 \\ 0.6 \end{bmatrix}.
$$

#### Layer 1: activation ($\mathbf{A}^{[1]}$, ReLU)

$\mathrm{ReLU}(z) = \max(0, z)$:

$$
\mathbf{A}^{[1]} = \begin{bmatrix} 0.5 \\ 0.0 \\ 0.6 \end{bmatrix}.
$$

None of the $z^{[1]}_j$ were negative, so **$\mathbf{A}^{[1]} = \mathbf{Z}^{[1]}$** (here the second coordinate is exactly $0$, so it stays $0$).

#### Layer 2: linear step ($Z^{[2]}$)

Use $\mathbf{A}^{[1]}$ as the input to the output neuron:

$$
Z^{[2]} = \mathbf{W}^{[2]} \mathbf{A}^{[1]} + b^{[2]} = 0.9(0.5) + 0.3(0.0) + 0.5(0.6) + 0 = 0.75.
$$

#### Layer 2: activation (prediction $\hat{y}$)

Sigmoid $\sigma(z) = 1 / (1 + e^{-z})$:

$$
\hat{y} = A^{[2]} = \sigma(0.75) \approx 0.679.
$$

The network predicts probability **$\hat{y} \approx 0.679$**; the true label is **$y = 1$**, so the prediction is **too low**—we will see a **negative** $\mathrm{d}Z^{[2]}$.

**Sanity check:** Because $\hat{y} < y$, gradient descent should nudge weights and biases so that **$Z^{[2]}$** (and thus $\hat{y}$) tends to **increase** on the next forward pass—consistent with the sign of $\mathrm{d}Z^{[2]}$ and the updates below.

---

## The blame ($m = 1$)

In [Lesson 4.b](lesson_4b.md), gradients include a **$\frac{1}{m}$** when averaging over $m$ examples. Here **$m = 1$**, so $\frac{1}{m}\sum_{i=1}^{m}(\cdot) = (\cdot)$: the **same** formulas apply; you simply **omit** the explicit $1/m$. General backprop does **not** drop $1/m$—this is specific to a single-example minibatch.

With one example, **$\mathrm{d}\mathbf{W}^{[\ell]} = \mathrm{d}\mathbf{Z}^{[\ell]} (\text{inputs to layer } \ell)^{\mathsf T}$** matches the averaged rule with $m=1$.

### Phase 1: output layer (layer 2)

Maps to [Lesson 4.b — Phase 1: Output layer](lesson_4b.md#phase-1-output-layer-layer-2).

#### 1. Final error $\mathrm{d}Z^{[2]}$

Same shortcut as in Lesson 4.b (mean BCE + sigmoid). Here $A^{[2]} = \hat{y}$:

$$
\mathrm{d}Z^{[2]} = A^{[2]} - y = \hat{y} - y \approx 0.679 - 1 = -0.321.
$$

The **negative** sign means the model’s probability was **below** the target.

#### 2. Weight gradient $\mathrm{d}\mathbf{W}^{[2]}$

$$
\mathrm{d}\mathbf{W}^{[2]} = \mathrm{d}Z^{[2]} \bigl(\mathbf{A}^{[1]}\bigr)^{\mathsf T} \approx -0.321 \begin{bmatrix} 0.5 & 0.0 & 0.6 \end{bmatrix} = \begin{bmatrix} -0.160 & 0.0 & -0.193 \end{bmatrix}.
$$

The **middle** entry is **0.0**: neuron 2 sent **$0$** forward, so its weight receives **no gradient** for this example.

#### 3. Bias gradient $\mathrm{d}b^{[2]}$

For $m = 1$:

$$
\mathrm{d}b^{[2]} = \mathrm{d}Z^{[2]} \approx -0.321.
$$

---

### Phase 2: hidden layer (layer 1)

Maps to [Lesson 4.b — Phase 2: Hidden layer](lesson_4b.md#phase-2-hidden-layer-layer-1).

#### 4. Hidden error $\mathrm{d}\mathbf{Z}^{[1]}$

**Step A — push error back through $\mathbf{W}^{[2]}$:**

$$
\bigl(\mathbf{W}^{[2]}\bigr)^{\mathsf T} \mathrm{d}Z^{[2]} \approx -0.321 \begin{bmatrix} 0.9 \\ 0.3 \\ 0.5 \end{bmatrix} = \begin{bmatrix} -0.289 \\ -0.096 \\ -0.160 \end{bmatrix}.
$$

**Step B — ReLU gate** using $\mathbf{Z}^{[1]} = [0.5,\, 0.0,\, 0.6]^{\mathsf T}$:

| Neuron | $z^{[1]}_j$ | Gate $\mathbb{1}\{z^{[1]}_j > 0\}$ |
|--------|-------------|-------------------------------------|
| 1 | $0.5$ | $1$ |
| 2 | $0.0$ | $0$ (treated as “off”; derivative at $0$ is a convention) |
| 3 | $0.6$ | $1$ |

$$
\mathrm{d}\mathbf{Z}^{[1]} = \begin{bmatrix} -0.289 \\ 0 \\ -0.160 \end{bmatrix}.
$$

Neuron 2’s error is **zeroed**: it was **inactive** in the forward pass ($z^{[1]}_2 = 0$). That is different from a **chronically dead** ReLU, which is **mostly** off for **many** inputs during training; here neuron 2 simply carries **no signal for this one example**.

#### 5. Weight gradient $\mathrm{d}\mathbf{W}^{[1]}$

$$
\mathrm{d}\mathbf{W}^{[1]} = \mathrm{d}\mathbf{Z}^{[1]} \mathbf{x}^{\mathsf T} \approx \begin{bmatrix} -0.289 \\ 0 \\ -0.160 \end{bmatrix} \begin{bmatrix} 1 & 0.5 \end{bmatrix} = \begin{bmatrix} -0.289 & -0.145 \\ 0 & 0 \\ -0.160 & -0.080 \end{bmatrix}.
$$

#### 6. Bias gradient $\mathrm{d}\mathbf{b}^{[1]}$

For $m = 1$:

$$
\mathrm{d}\mathbf{b}^{[1]} = \mathrm{d}\mathbf{Z}^{[1]} \approx \begin{bmatrix} -0.289 \\ 0 \\ -0.160 \end{bmatrix}.
$$

---

## The update (gradient descent)

Use learning rate **$\alpha = 0.1$**. Rule: $\theta \leftarrow \theta - \alpha \, \mathrm{d}\theta$.

### Layer 2

**Weights:**

$$
\mathbf{W}^{[2]} \leftarrow \begin{bmatrix} 0.9 & 0.3 & 0.5 \end{bmatrix} - 0.1 \begin{bmatrix} -0.160 & 0.0 & -0.193 \end{bmatrix} \approx \begin{bmatrix} 0.916 & 0.300 & 0.519 \end{bmatrix}.
$$

**Bias:**

$$
b^{[2]} \leftarrow 0 - 0.1(-0.321) = 0.032.
$$

The **first and third** weights and the **bias** move to **increase** the pre-activation toward the positive direction that raises $\hat{y}$ toward $1$.

### Layer 1

**Weights:**

$$
\mathbf{W}^{[1]} \leftarrow \mathbf{W}^{[1]} - 0.1 \, \mathrm{d}\mathbf{W}^{[1]} \approx \begin{bmatrix} 0.329 & 0.614 \\ -0.1 & 0.6 \\ 0.216 & 0.808 \end{bmatrix}.
$$

**Biases:**

$$
\mathbf{b}^{[1]} \leftarrow \begin{bmatrix} -0.1 \\ -0.2 \\ 0 \end{bmatrix} - 0.1 \begin{bmatrix} -0.289 \\ 0 \\ -0.160 \end{bmatrix} \approx \begin{bmatrix} -0.071 \\ -0.2 \\ 0.016 \end{bmatrix}.
$$

The **middle row** (neuron 2) of $\mathbf{W}^{[1]}$ and $b^{[1]}_2$ is **unchanged**: $\mathrm{d}\mathbf{Z}^{[1]}_2 = 0$, so that neuron **learns nothing** from this example. **ReLU sparsity:** when the unit is inactive, the backward signal through that unit is **zero** for that forward pass.

---

## Try it

- If **$y = 0$** instead of $1$, what sign do you expect for $\mathrm{d}Z^{[2]}$ if $\hat{y}$ is still $0.679$?
- If **$z^{[1]}_2$** were a small **positive** number instead of $0$, how would the gate row and $\mathrm{d}\mathbf{W}^{[2]}_2$ change (qualitatively)?

---

## Key takeaways

- Forward: $\mathbf{Z}^{[1]} \to \mathrm{ReLU} \to \mathbf{A}^{[1]}$; then $Z^{[2]}$, then **sigmoid** $\to \hat{y}$ (see [Lesson 4.a](lesson_4a.md)).
- **$\mathrm{d}Z^{[2]} = A^{[2]} - y = \hat{y} - y$** (this loss/sigmoid pairing); sign tells you if probability was too high or too low.
- **$\mathrm{d}\mathbf{W}^{[2]}$** is proportional to $\mathbf{A}^{[1]}$; if a hidden unit is $0$, its outgoing weight gets **$0$** gradient.
- **$\mathrm{d}\mathbf{Z}^{[1]}$** multiplies backprop through $\mathbf{W}^{[2]}$ and then **masks** by ReLU; inactive ReLU units get **$0$** error for that pass.
- With **$m = 1$**, the **same** backprop formulas as Lesson 4.b apply; $\frac{1}{m}\sum(\cdot)$ is just $(\cdot)$—not a different rule.
