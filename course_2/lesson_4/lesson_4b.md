# Course 2 — Lesson 4.b — Neural Networks: Backward Propagation

**Prerequisite:** [Lesson 4.a — Forward propagation](lesson_4a.md) (how $\mathbf{Z}^{[\ell]}$, $\mathbf{A}^{[\ell]}$, and matrix shapes fit together).

After this lesson you should be able to:

- State why backpropagation uses the **chain rule** and what “flowing error backward” means.
- Write the key gradients for a **two-layer** classifier with **binary cross-entropy** and **sigmoid** output.
- Apply **L2 regularization** to weight gradients and interpret **weight decay**.

**Notation:** superscripts $^{[1]}$, $^{[2]}$ denote layers. We use $\mathbf{A}^{[0]} = \mathbf{X}$ for inputs. Shorthand $\mathrm{d}\mathbf{W}^{[\ell]}$ means $\partial J / \partial \mathbf{W}^{[\ell]}$ (same for $\mathbf{b}$, $\mathbf{Z}$), matching common course/code conventions. The symbol $\mathrm{d}$ is **not** a total derivative—it is a **label** for “gradient of $J$ with respect to …,” same idea as `dW` in NumPy code.

---

## Backward propagation

For a neural network doing classification (e.g. “Bonk” vs. “No Bonk”), we use the same **binary cross-entropy** cost as in logistic regression. The only change is that $\hat{y}$ (here $\mathbf{A}^{[2]}$) comes from several layers:

$$
J = -\frac{1}{m} \sum_{i=1}^{m} \Bigl( y^{(i)} \log a^{[2](i)} + \bigl(1 - y^{(i)}\bigr) \log \bigl(1 - a^{[2](i)}\bigr) \Bigr)
$$

where $m$ is the number of examples, $y^{(i)} \in \{0,1\}$, and $a^{[2](i)}$ is the predicted probability for example $i$.

If **forward propagation** is “make a prediction,” **backpropagation** is “learn from the error.”

In Lesson 2 the gradient was simple because there was only one layer. In a neural network, if the output $\hat{y}$ is wrong, we must decide:

- How much was **layer 2** to blame?
- How much was **layer 1** to blame?

We use the **chain rule** from calculus to propagate the error **backward** through the network.

**Roadmap:** computation graph → chain rule and shorthand → **phase 1** (output layer) → **phase 2** (hidden layer) → gradient descent → **L2** and weight decay.

---

## The computation graph

To keep backpropagation organized, we picture the network as a **computation graph**.

- Compute the derivative of the loss with respect to the output activation (e.g. $\mathrm{d}\mathbf{A}^{[2]}$), then fold in the **sigmoid** to get $\mathrm{d}\mathbf{Z}^{[2]}$.
- Multiply by derivatives of each activation (**ReLU** for the hidden layer) where they appear.
- Multiply by derivatives of affine transforms (weights, biases) to get $\mathrm{d}\mathbf{W}^{[\ell]}$, $\mathrm{d}\mathbf{b}^{[\ell]}$, and to pass error to earlier $\mathrm{d}\mathbf{Z}^{[\ell]}$.

It is like an **assembly line running in reverse**: start at the loss and walk back through each operation.

---

## The calculus of “blame”

If the network predicts a “Bonk” probability of $0.90$ but the true label was $0$ (you felt great), the cost is large—the model made a serious error.

Backpropagation is the algorithm that attributes that error **upstream**:

- Was **neuron 3 in layer 2** (illustrative indexing—not a specific lab neuron) a main contributor?
- Or was **neuron 1 in layer 1** passing bad information to layer 2?

To improve the network we need the gradient of the cost $J$ with respect to **every weight** (and typically every bias): $\partial J / \partial w$ for all layers.

---

## The chain rule

A neural network is a **nested** composition of functions—for example, schematically (read $\sigma$ as sigmoid, $g$ as ReLU):

$$
J\Bigl(\sigma\bigl(\mathbf{Z}^{[2]}( \cdots )\bigr)\Bigr), \quad \mathbf{A}^{[1]} = g\bigl(\mathbf{Z}^{[1]}(\mathbf{X})\bigr), \quad \mathbf{Z}^{[2]} = \mathbf{W}^{[2]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]}
$$

The earlier one-line nest $J(\mathbf{A}^{[2]}(\mathbf{Z}^{[2]}(\cdots)))$ is only a **cartoon**—it hides $\sigma$ and $g$. So we **start at the final error** and multiply backward through the derivative of **each** step (loss → output activation → sigmoid → $\mathbf{Z}^{[2]}$ → … → ReLU → $\mathbf{Z}^{[1]}$). That is the chain rule.

---

## Shorthand notation

In papers and code, writing $\partial J / \partial (\cdot)$ everywhere is heavy. A common shorthand (used in courses and in `dW`, `db`, `dZ` style code) is:

| Shorthand | Meaning | Question it answers |
|-----------|---------|---------------------|
| $\mathrm{d}\mathbf{W}^{[2]}$ | $\partial J / \partial \mathbf{W}^{[2]}$ | How should layer 2’s weights change? |
| $\mathrm{d}\mathbf{b}^{[2]}$ | $\partial J / \partial \mathbf{b}^{[2]}$ | How should layer 2’s biases change? |
| $\mathrm{d}\mathbf{Z}^{[1]}$ | $\partial J / \partial \mathbf{Z}^{[1]}$ | How much error reached the **linear** part of layer 1? |

(Same pattern for $\mathrm{d}\mathbf{W}^{[1]}$, $\mathrm{d}\mathbf{b}^{[1]}$, $\mathrm{d}\mathbf{Z}^{[2]}$.)

---

## The “reverse assembly line”

Forward pass: **left → right**. Backward pass: **right → left**. For a two-layer network (one hidden, one output), the order is:

1. **Output:** Compute the error signal at the final linear step, $\mathrm{d}\mathbf{Z}^{[2]}$.
2. **Layer 2:** From $\mathrm{d}\mathbf{Z}^{[2]}$, compute $\mathrm{d}\mathbf{W}^{[2]}$ and $\mathrm{d}\mathbf{b}^{[2]}$.
3. **Pass blame back:** Propagate through layer 2’s weights to get $\mathrm{d}\mathbf{Z}^{[1]}$ (error at layer 1’s linear output), including the **ReLU** derivative.
4. **Layer 1:** Compute $\mathrm{d}\mathbf{W}^{[1]}$ and $\mathrm{d}\mathbf{b}^{[1]}$ from $\mathrm{d}\mathbf{Z}^{[1]}$.

After those gradients are computed, update parameters with **gradient descent** and learning rate $\alpha$—see [Gradient descent update](#gradient-descent-update) below for the update equations.

Assume **vectorized** data: input $\mathbf{X} = \mathbf{A}^{[0]}$, labels $\mathbf{Y}$, and $m$ examples stacked as **columns** (same layout as [Lesson 4.a](lesson_4a.md)).

---

## Phase 1: Output layer (layer 2)

### Shapes (reference)

Fix **$n_x$** inputs, **$n_h$** hidden units, **$m$** examples, **one** output logit (binary classification). With **examples as columns**:

| Object | Shape | Notes |
|--------|--------|--------|
| $\mathbf{X} = \mathbf{A}^{[0]}$ | $n_x \times m$ | |
| $\mathbf{W}^{[1]}$, $\mathrm{d}\mathbf{W}^{[1]}$ | $n_h \times n_x$ | |
| $\mathbf{b}^{[1]}$, $\mathrm{d}\mathbf{b}^{[1]}$ | $n_h \times 1$ | broadcasts over columns |
| $\mathbf{Z}^{[1]}$, $\mathbf{A}^{[1]}$, $\mathrm{d}\mathbf{Z}^{[1]}$ | $n_h \times m$ | |
| $\mathbf{W}^{[2]}$, $\mathrm{d}\mathbf{W}^{[2]}$ | $1 \times n_h$ | one output neuron |
| $\mathbf{b}^{[2]}$, $\mathrm{d}\mathbf{b}^{[2]}$ | $1 \times 1$ | |
| $\mathbf{Z}^{[2]}$, $\mathbf{A}^{[2]}$, $\mathrm{d}\mathbf{Z}^{[2]}$, $\mathbf{Y}$ | $1 \times m$ | one probability per example |

If your code stores examples as **rows** instead, transposes swap—keep dimensions consistent with your forward pass.

### 1. Final error $\mathrm{d}\mathbf{Z}^{[2]}$

With **binary cross-entropy** and **sigmoid** on the output, the derivative of $J$ with respect to $\mathbf{Z}^{[2]}$ simplifies to a **prediction minus label** form:

$$
\mathrm{d}\mathbf{Z}^{[2]} = \mathbf{A}^{[2]} - \mathbf{Y}
$$

Same shape as $\mathbf{A}^{[2]}$ and $\mathbf{Y}$ (e.g. $1 \times m$ for one output logit across $m$ examples).

This **clean** form depends on the setup: **mean** binary cross-entropy over $m$ examples, **sigmoid** on the output, and the loss written in the usual way. It does **not** apply unchanged to other losses (e.g. softmax + multi-class cross-entropy) or to different reductions (sum vs mean).

### 2. Weight gradient $\mathrm{d}\mathbf{W}^{[2]}$

Layer 2’s linear map used activations $\mathbf{A}^{[1]}$ as inputs. With the usual layout (examples as columns):

$$
\mathrm{d}\mathbf{W}^{[2]} = \frac{1}{m} \, \mathrm{d}\mathbf{Z}^{[2]} \bigl(\mathbf{A}^{[1]}\bigr)^{\mathsf T}
$$

The transpose $\bigl(\mathbf{A}^{[1]}\bigr)^{\mathsf T}$ aligns dimensions so $\mathrm{d}\mathbf{W}^{[2]}$ matches the shape of $\mathbf{W}^{[2]}$.

### 3. Bias gradient $\mathrm{d}\mathbf{b}^{[2]}$

Average the error over all $m$ examples (sum over the example dimension, then divide by $m$):

$$
\mathrm{d}\mathbf{b}^{[2]} = \frac{1}{m} \sum_{i=1}^{m} \mathrm{d}\mathbf{z}^{[2](i)}
$$

where each $\mathrm{d}\mathbf{z}^{[2](i)}$ is column $i$ of $\mathrm{d}\mathbf{Z}^{[2]}$ (one scalar per example when $\mathrm{d}\mathbf{Z}^{[2]}$ is $1 \times m$). Then $\mathrm{d}\mathbf{b}^{[2]}$ is a single $1 \times 1$ bias gradient. In NumPy terms: `np.sum(dZ2, axis=1, keepdims=True) / m` when the example axis matches your layout. Conceptually: **$\mathrm{d}\mathbf{b}^{[2]}$ is the mean of the output-layer errors across examples.**

---

## Phase 2: Hidden layer (layer 1)

### 4. Hidden error $\mathrm{d}\mathbf{Z}^{[1]}$

Push $\mathrm{d}\mathbf{Z}^{[2]}$ backward through layer 2’s weights, then apply the **ReLU** derivative **element-wise** (the “activation gate”):

$$
\mathrm{d}\mathbf{Z}^{[1]} = \bigl(\mathbf{W}^{[2]}\bigr)^{\mathsf T} \mathrm{d}\mathbf{Z}^{[2]} \odot \mathbb{1}\bigl\{ \mathbf{Z}^{[1]} > 0 \bigr\}
$$

Here $\odot$ is element-wise product, and $\mathbb{1}\{\mathbf{Z}^{[1]} > 0\}$ is 1 where ReLU is active and 0 where ReLU is “off.” At $z = 0$ the ReLU derivative is undefined in theory; implementations almost always pick **0** (or sometimes 1) at that point—it rarely matters in practice.

### 5. Weight gradient $\mathrm{d}\mathbf{W}^{[1]}$

Layer 1’s inputs are $\mathbf{X} = \mathbf{A}^{[0]}$:

$$
\mathrm{d}\mathbf{W}^{[1]} = \frac{1}{m} \, \mathrm{d}\mathbf{Z}^{[1]} \mathbf{X}^{\mathsf T}
$$

### 6. Bias gradient $\mathrm{d}\mathbf{b}^{[1]}$

Again, average the error over examples:

$$
\mathrm{d}\mathbf{b}^{[1]} = \frac{1}{m} \sum_{i=1}^{m} \mathrm{d}\mathbf{z}^{[1](i)}
$$

(same averaging pattern as $\mathrm{d}\mathbf{b}^{[2]}$, applied to $\mathrm{d}\mathbf{Z}^{[1]}$).

---

## Gradient descent update

After computing the six gradient objects above, update parameters:

$$
\mathbf{W}^{[\ell]} \leftarrow \mathbf{W}^{[\ell]} - \alpha \, \mathrm{d}\mathbf{W}^{[\ell]}, \qquad
\mathbf{b}^{[\ell]} \leftarrow \mathbf{b}^{[\ell]} - \alpha \, \mathrm{d}\mathbf{b}^{[\ell]} \quad \text{for } \ell = 1, 2.
$$

---

## L2 regularization in backpropagation

Add to the cost a penalty that discourages large weights. Sum over **weight matrices** only (here $\ell \in \{1, 2\}$; biases are not included):

$$
J_{\text{total}} = J + \frac{\lambda}{2m} \Bigl( \bigl\| \mathbf{W}^{[1]} \bigr\|_F^2 + \bigl\| \mathbf{W}^{[2]} \bigr\|_F^2 \Bigr)
$$

More generally, $\frac{\lambda}{2m} \sum_{\ell} \| \mathbf{W}^{[\ell]} \|_F^2$ over all layers $\ell$ that have weights. Here $\|\cdot\|_F$ is the Frobenius norm (sum of squares of all entries) and $\lambda \ge 0$ controls strength.

For each weight matrix,

$$
\frac{\partial}{\partial \mathbf{W}^{[\ell]}} \left( \frac{\lambda}{2m} \bigl\| \mathbf{W}^{[\ell]} \bigr\|_F^2 \right) = \frac{\lambda}{m} \mathbf{W}^{[\ell]}
$$

So you **add** $\frac{\lambda}{m}\mathbf{W}^{[\ell]}$ to the **unregularized** $\mathrm{d}\mathbf{W}^{[\ell]}$ from backprop. The **bias** gradients are unchanged—L2 usually penalizes only **weights**.

### Regularized weight gradients

Keep $\mathrm{d}\mathbf{Z}^{[2]}$ and the rest of the error propagation **the same**; only the weight gradients pick up the extra term:

$$
\mathrm{d}\mathbf{W}^{[2]} = \frac{1}{m} \, \mathrm{d}\mathbf{Z}^{[2]} \bigl(\mathbf{A}^{[1]}\bigr)^{\mathsf T} + \frac{\lambda}{m} \mathbf{W}^{[2]}
$$

$$
\mathrm{d}\mathbf{W}^{[1]} = \frac{1}{m} \, \mathrm{d}\mathbf{Z}^{[1]} \mathbf{X}^{\mathsf T} + \frac{\lambda}{m} \mathbf{W}^{[1]}
$$

$\mathrm{d}\mathbf{b}^{[1]}$ and $\mathrm{d}\mathbf{b}^{[2]}$ are unchanged.

---

## Why this is called “weight decay”

Plug the regularized $\mathrm{d}\mathbf{W}$ into gradient descent:

$$
\mathbf{W}^{[\ell]} \leftarrow \mathbf{W}^{[\ell]} - \alpha \left( \mathrm{d}\mathbf{W}^{[\ell]}_{\text{data}} + \frac{\lambda}{m} \mathbf{W}^{[\ell]} \right)
$$

Rearrange:

$$
\mathbf{W}^{[\ell]} \leftarrow \left(1 - \frac{\alpha\lambda}{m}\right) \mathbf{W}^{[\ell]} - \alpha \, \mathrm{d}\mathbf{W}^{[\ell]}_{\text{data}}
$$

So each step **shrinks** $\mathbf{W}^{[\ell]}$ toward zero by a factor $\bigl(1 - \frac{\alpha\lambda}{m}\bigr)$ before applying the usual update from the data gradient. That is why frameworks often expose L2 penalty under the name **`weight_decay`**.

In libraries (e.g. PyTorch), the exact mapping between **`weight_decay`** and “classical L2 on the loss” can differ by **optimizer**—plain SGD often matches the story above; **AdamW** uses **decoupled** weight decay that is not identical to adding $\frac{\lambda}{m}\mathbf{W}$ to the gradient of $J$ in every case. The intuition—shrink weights toward zero—still holds.

---

## Why we do not regularize biases

In typical deep learning practice, **biases are not L2-penalized**. Weight matrices control **how strongly** the network mixes inputs and features—where overfitting and large “wiggly” solutions often show up. Biases mainly **shift** decision thresholds; they add less to model complexity in the usual sense, so the penalty is applied to $\mathbf{W}$, not $\mathbf{b}$.

---

## Key takeaways

- Backprop is the chain rule applied **systematically** on the computation graph (see [Lesson 4.a](lesson_4a.md) for the forward pass you differentiate through).
- For **mean** BCE + **sigmoid** on a **binary** output, $\mathrm{d}\mathbf{Z}^{[2]} = \mathbf{A}^{[2]} - \mathbf{Y}$ in the standard setup; other losses need different expressions.
- Errors flow backward: through $\mathbf{W}^{[2]}$, then through **ReLU** via element-wise masking (derivative 0 at $z=0$ is a convention).
- **$\mathrm{d}\mathbf{W}^{[\ell]}$** pairs the layer’s error with **inputs to that layer**; **$\mathrm{d}\mathbf{b}^{[\ell]}$** averages error over examples.
- **L2** adds $\frac{\lambda}{m}\mathbf{W}^{[\ell]}$ to $\mathrm{d}\mathbf{W}^{[\ell]}$, which leads to **weight decay** in the update; framework **`weight_decay`** may differ slightly by optimizer.
