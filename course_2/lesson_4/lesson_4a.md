# Course 2 — Lesson 4.a — Neural Network Foundations & Forward Propagation

## The first leap: from scalar to matrix

In the lab, we had one output $\hat{y}$ and two features $(x_1, x_2)$. In a neural network, a single layer might have four neurons, all looking at those same two features.

### Step 1: Group the inputs into a vector

Stack the features into a column vector:

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

### Step 2: Build the weight matrix

This is a **4 × 2** matrix:

- **4 rows** → four neurons ($z_1$ to $z_4$)
- **2 columns** → two inputs ($x_1$, $x_2$)

$$
\mathbf{W} = \begin{bmatrix}
w_{1,1} & w_{1,2} \\
w_{2,1} & w_{2,2} \\
w_{3,1} & w_{3,2} \\
w_{4,1} & w_{4,2}
\end{bmatrix}
$$

### Step 3: Bias vector

$$
\mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \end{bmatrix}
$$

### Step 4: Combine everything

1. $z_1 = w_{1,1} x_1 + w_{1,2} x_2 + b_1$
2. $z_2 = w_{2,1} x_1 + w_{2,2} x_2 + b_2$
3. $z_3 = w_{3,1} x_1 + w_{3,2} x_2 + b_3$
4. $z_4 = w_{4,1} x_1 + w_{4,2} x_2 + b_4$

Writing this out for every neuron is inefficient. Instead, we bundle all the weights into a matrix $\mathbf{W}$ and the inputs into a vector $\mathbf{x}$:

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

where:

- $\mathbf{W}$ is a **4 × 2** matrix (four neurons, two inputs each).
- $\mathbf{x}$ is a **2 × 1** column vector.
- $\mathbf{b}$ is a **4 × 1** vector.
- $\mathbf{z}$ is a **4 × 1** vector of **pre-activations**.

---

## The activation function: adding “soul” to the math

If we only kept multiplying by matrices, the whole network would reduce to one giant linear equation (and a straight line cannot learn much). We need to **squish** the output of every neuron using an activation function $g(\mathbf{z})$.

While we used sigmoid for classification, the modern standard for hidden layers is **ReLU**:

$$
g(z) = \max(0, z)
$$

ReLU is simple: if the input is negative, it outputs zero; if positive, it passes the value through. This creates the **sparsity** we discussed earlier and makes the math much faster for deep networks.

---

## The anatomy of a layer

### First “deep” mental model

In standard deep learning terminology, a **layer** refers to the entire process of transforming an input into an activation.

Conceptually:

```text
input (x) → [layer 1] → [layer 2] → output (ŷ)
```

Each layer is:

1. **Linear step:** $\mathbf{z} = \mathbf{W}\mathbf{a}^{[\ell-1]} + \mathbf{b}$  
   This is the raw score: a linear combination of the previous layer’s signals.

2. **Activation step:** $\mathbf{a} = g(\mathbf{z})$  
   This is the squished or rectified signal. The activation $\mathbf{a}$ is what is passed forward as the input to the next layer.

### Why $\mathbf{a}$ is the product, not $\mathbf{z}$?

If we only returned $\mathbf{z}$ (the linear part), stacking many layers would be mathematically similar to having **one** linear layer. Why? Because a linear function of a linear function is still linear (e.g. $2 \times 3 \times x$ is just $6x$).

By passing the activation $\mathbf{a}$, we **break** linearity. That lets the network learn complex patterns (like the shape of a bike frame or the waveform of a voice) that a single linear map cannot represent.

In one layer, every neuron sees the same raw data, but each neuron is tuned to something different. To visualize, suppose $x_1$ is **power** and $x_2$ is **duration**.

- **Neuron 1** has its own weights $(w_{1,1}, w_{1,2})$ and bias $b_1$. It might be looking for **sprints**—large weight on $x_1$ (power) and small weight on $x_2$ (duration).
- **Neuron 2** has its own weights $(w_{2,1}, w_{2,2})$ and bias $b_2$. It might be looking for **endurance**—large weight on $x_2$ (duration) and moderate weight on $x_1$.

To understand why this works, it helps to think in terms of **coordinate transformations**.

### 1. The “manifold” intuition

Imagine your “bonk” data as a messy pile of blue and red tangled yarn on a table. A single logistic regression (a straight line) is like a stiff ruler: no matter how you move it, you cannot perfectly separate the colors if they are tangled. A hidden layer acts like a **space warper**. When layer 1 applies $g(\mathbf{W}\mathbf{x} + \mathbf{b})$, it stretches, twists, and folds the coordinate system. It takes tangled data and **untangles** it into a new space where blue and red are easier to separate.

By the time the data reaches the final layer, the yarn has been untangled enough that a simple straight line (logistic regression) can separate the classes.

### 2. The hierarchy of abstraction

Each layer performs **feature synthesis**: it turns granular data into more conceptual data.

Example (Trek Émonda–style scenario):

- **Input layer:** Raw sensor data (watts, seconds, heartbeats)—meaningless in isolation.
- **Layer 1 (interpreter):** Combines power and duration into something like **energy flux**—a concept built from raw numbers.
- **Layer 2 (contextualizer):** Combines energy flux with heart rate into **aerobic strain**—a higher-level concept.

Why does this help training? The second layer does not have to reason about messy raw power directly; it trains on **refined concepts** from the first layer. Predicting a bonk from “extreme fatigue” is easier than from “242 W” alone.

### 3. The composition of functions

In calculus this is **function composition**: $f(g(h(x)))$. A neural network is a large **universal function approximator**. The **universal approximation theorem** says that with at least one hidden layer, enough neurons, and a non-linear activation (sigmoid or ReLU), you can approximate a wide class of functions.

Stacking layers builds a kind of **logic circuit**:

- Layer 1: “Is this true?”
- Layer 2: “If [layer 1, neuron A] is true and [layer 1, neuron B] is false, then …”

The activation from layer 1 “works” for layer 2 because layer 1 has already extracted signal from noise—it passes a **summary** forward.

### Why does it work? Linear separability and basis change

#### The problem: the XOR proof

A classic illustration is **XOR**: two inputs $(x_1, x_2)$; output 1 when exactly one input is active, 0 when both are on or both are off. Plotted in 2D, **no single straight line** (logistic regression) can separate the classes. A single layer implements a **linear decision boundary**; if the data are not linearly separable, one layer cannot solve XOR.

#### The solution: changing the basis

Passing $\mathbf{x}$ through the first layer to get $\mathbf{a}^{[1]}$ is a **basis transformation**. In input space, your axes are $x_1$ and $x_2$; there the points may be tangled.

The first layer computes:

$$
\mathbf{a}^{[1]} = \operatorname{ReLU}\bigl(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}\bigr)
$$

This **projects** the data (often into a higher-dimensional space) and then **clips** it with the activation. By the time data reaches the second layer, the **coordinates have changed**: the first layer has built new dimensions that are non-linear combinations of the originals.

**Cover’s theorem** (informally): complex patterns mapped into a high-dimensional space are **more likely** to become linearly separable than in a low-dimensional space.

#### The “universal approximation” angle

Why does layer 1’s output “work” for layer 2? Because of **composition**. If $f$ is linear, then $f(f(x))$ is still linear. But with a non-linearity $\sigma$, the map $f(\sigma(x))$ becomes **piecewise** linear (or more expressive). Each neuron in layer 1 adds a **hinge** (ReLU) or **curve** (sigmoid) to the landscape; layer 2 **sums** those pieces to build richer shapes.

- **Layer 1:** Simple geometric primitives (half-spaces, lines, planes).
- **Layer 2:** Combines those primitives into bumps and valleys.
- **Layer 3+:** Combines those into arbitrarily complex decision boundaries.

**Summary:** One layer only solves problems that are linearly separable; much real data is not. Non-linear activations map data into new coordinates where features are **higher-level combinations** of inputs. Stacking such maps lets us approximate complex functions from simpler, piecewise parts.

---

## Forward propagation

**Forward propagation** is the process of taking an input $\mathbf{x}$ and pushing it through the network to obtain a prediction. It is a sequence of matrix operations. Consider a network with **two layers** (one hidden, one output).

### Layer 1 (hidden layer)

Pre-activation:

$$
\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}
$$

Then apply ReLU:

$$
\mathbf{A}^{[1]} = \max(0, \mathbf{Z}^{[1]})
$$

(element-wise).

### Layer 2 (output layer)

Use $\mathbf{A}^{[1]}$ as the input to the next linear map:

$$
\mathbf{Z}^{[2]} = \mathbf{W}^{[2]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]}
$$

For binary classification, apply **sigmoid** at the output:

$$
\hat{y} = \mathbf{A}^{[2]} = \sigma(\mathbf{Z}^{[2]})
$$

---

## Why vectorization is a “superpower”

In earlier labs we used `np.dot(X, w)`. Neural networks do the same, but for **every example at once**.

For a hidden layer with 2 inputs and 5 neurons:

1. $\mathbf{W}^{[1]}$ has shape **5 × 2** (five neurons, each with two weights).
2. The activation $\mathbf{a}^{[1]}$ has **five** values passed to the next layer.

If you have $m = 1000$ rides and two features (e.g. power $x_1$ and cadence $x_2$):

- $\mathbf{X}$ can be **2 × 1000** (features × examples).
- $\mathbf{W}^{[1]}$ remains **5 × 2**.
- $\mathbf{Z}^{[1]}$ becomes **5 × 1000**.

One matrix multiply updates all 1000 examples at once. That is why large datasets and deep models are practical—**GPUs** are built for these operations.

---

## The non-linearity requirement

What if we removed activations $g(\mathbf{z})$ and passed $\mathbf{Z}$ through unchanged?

The network **collapses**: a linear function of a linear function is still linear. Without ReLU or sigmoid, a 100-layer “deep” network would have no more expressive power than a single linear regression for representation purposes. Activations are what let the network learn **bends** and **curves** in the data.
