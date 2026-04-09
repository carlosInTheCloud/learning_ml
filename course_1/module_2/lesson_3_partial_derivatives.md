# Course 1 - Module 2 - Lesson 3: Calculus for Machine Learning

## Partial Derivatives (The Compass of ML)

In the real world, models don't just have one input (x). They have thousands of inputs (x₁, x₂, x₃, …) and thousands of weights (w₁, w₂, w₃, …).

A **Partial Derivative** allows us to look at a complex formula and ask: *"If I change only ONE variable (like the weight of a single pixel), how much does the error change?"*

---

## The Golden Rule: The Cold Shoulder

When you take a partial derivative with respect to x (written as ∂ƒ/∂x), you treat every other variable (like y, z, or w) as if it were a boring, flat constant (like the number 5).

- If it's **attached to x:** It stays there as a multiplier.
- If it's **standing alone:** Its derivative is 0.

---

## A Numerical Example

Let's look at the function:

> ƒ(x, y) = 3x² + 5xy + y³

### Find ∂ƒ/∂x (treat y as a constant)

1. **Term 1 (3x²):** Use the Power Rule → **6x**
2. **Term 2 (5xy):** Treat 5 and y as constants. They are just "hitching a ride" on the x. The derivative of x is 1. → **5y**
3. **Term 3 (y³):** This term has no x in it at all. It's treated as a constant. → **0**

> ∂ƒ/∂x = **6x + 5y**

### Find ∂ƒ/∂y (treat x as a constant)

1. **Term 1 (3x²):** No y here. → **0**
2. **Term 2 (5xy):** Treat 5 and x as constants. The derivative of y is 1. → **5x**
3. **Term 3 (y³):** Power Rule → **3y²**

> ∂ƒ/∂y = **5x + 3y²**

---

## The Gradient: Combining Partial Derivatives

In **Gradient Descent**, we collect all the partial derivatives into one vector called the **Gradient** (∇ƒ):

> ∇ƒ = [∂ƒ/∂x, ∂ƒ/∂y]

For our example:

> ∇ƒ = [6x + 5y, 5x + 3y²]

The Gradient is like a **compass**: it points exactly in the direction of the steepest "uphill" slope. Since we want to **minimize** error, the computer simply looks at the gradient and moves in the **opposite direction**.

### Evaluating the Gradient at a Point

The gradient is a formula until you plug in numbers. Let's evaluate at the point (1, 2):

> ∇ƒ(1, 2) = [6(1) + 5(2), 5(1) + 3(2²)] = [6 + 10, 5 + 12] = **[16, 17]**

This is a concrete vector. It tells us: "From the point (1, 2), the steepest uphill direction is [16, 17]." Gradient Descent would step in the **opposite** direction: [-16, -17].

### Connection to Linear Algebra

Notice that the gradient is a **vector** — everything from Module 1 applies here. The **magnitude** of the gradient (how long the arrow is) tells you how steep the slope is. Its **direction** tells you which way. The dot product between the gradient and a step direction tells you how much the function would change along that step. This is where Linear Algebra and Calculus meet.

---

## The Gradient Descent Update Rule

Now we can write out the actual formula that every ML model uses to learn:

> w_new = w_old - α * ∇L(w_old)

Where:

- **w_old** is the current weight vector
- **α** (alpha) is the **learning rate** — how big a step you take
- **∇L(w_old)** is the gradient of the loss at the current weights
- The **minus sign** makes you go *downhill* (opposite the gradient)

This single equation is the engine of Machine Learning. Everything else — derivatives, chain rule, partial derivatives — exists to compute ∇L.

---

## Partial Derivatives + The Chain Rule

In a real ML model, the loss function is often nested. We need both partial derivatives *and* the chain rule from Lesson 2.

### Example: A Single-Weight Loss Function

Suppose our loss for one data point is:

> L(w) = (200w - 3.7)²

We want ∂L/∂w — how does the loss change when we nudge the weight?

**Identify the layers:**

- **Inner:** u = 200w - 3.7
- **Outer:** L = u²

**Apply the Chain Rule:**

1. **Outer derivative:** 2(200w - 3.7)
2. **Inner derivative (with respect to w):** 200

**Combine:**

> ∂L/∂w = 2(200w - 3.7) * 200 = **400(200w - 3.7)**

**Evaluate at w = 0.02:**

> ∂L/∂w = 400(200 * 0.02 - 3.7) = 400(4 - 3.7) = 400 * 0.3 = **120**

This tells the model: "At w = 0.02, increasing w slightly will increase the loss by 120 per unit. The slope is positive, so move w in the *negative* direction to reduce loss."

**Run one Gradient Descent step** with learning rate α = 0.001:

> w_new = 0.02 - 0.001 * 120 = 0.02 - 0.12 = **-0.10**

The model just adjusted its weight from 0.02 to -0.10. It would repeat this process — compute gradient, step, compute gradient, step — until the loss stops decreasing.

---

## Practice Problems

### Drill 3.1: The "Cold Shoulder" Drill

Find the partial derivatives for this function. Remember: when you focus on one letter, the other letter becomes "invisible" (a constant).

> ƒ(x, y) = 4x³ + 2x²y² + 7y + 10

**Find ∂ƒ/∂x (treat y as a constant):**

1. 4x³ → 12x²
2. 2x²y² → 4xy² (y² is a constant multiplier)
3. 7y → 0 (no x)
4. 10 → 0 (constant)

> ∂ƒ/∂x = **12x² + 4xy²**

**Find ∂ƒ/∂y (treat x as a constant):**

1. 4x³ → 0 (no y)
2. 2x²y² → 4x²y (x² is a constant multiplier)
3. 7y → 7
4. 10 → 0 (constant)

> ∂ƒ/∂y = **4x²y + 7**

---

### Drill 3.2: The "Mixed Term" Challenge

> ƒ(x, y) = 5x²y³ + 10x + 2y² + 100

**Find ∂ƒ/∂x (treat y as a constant):**

1. 5x²y³ → 10xy³ (y³ is a constant multiplier)
2. 10x → 10
3. 2y² → 0 (no x)
4. 100 → 0 (constant)

> ∂ƒ/∂x = **10xy³ + 10**

**Find ∂ƒ/∂y (treat x as a constant):**

1. 5x²y³ → 15x²y² (x² is a constant multiplier)
2. 10x → 0 (no y)
3. 2y² → 4y
4. 100 → 0 (constant)

> ∂ƒ/∂y = **15x²y² + 4y**

---

### Drill 3.3: Three Variables

> ƒ(x, y, z) = x²z + 3yz² + 2x

**Find ∂ƒ/∂x (treat y and z as constants):**

1. x²z → 2xz (z is a constant multiplier)
2. 3yz² → 0 (no x)
3. 2x → 2

> ∂ƒ/∂x = **2xz + 2**

**Find ∂ƒ/∂y (treat x and z as constants):**

1. x²z → 0 (no y)
2. 3yz² → 3z² (z² is a constant multiplier)
3. 2x → 0 (no y)

> ∂ƒ/∂y = **3z²**

**Find ∂ƒ/∂z (treat x and y as constants):**

1. x²z → x² (x² is a constant multiplier)
2. 3yz² → 6yz (y is a constant multiplier)
3. 2x → 0 (no z)

> ∂ƒ/∂z = **x² + 6yz**

**The Gradient:**

> ∇ƒ = [2xz + 2, 3z², x² + 6yz]

This is a 3D vector — one component per variable. In a real ML model with 1000 weights, the gradient would be a 1000-dimensional vector.

---

### Drill 3.4: Chain Rule + Partial Derivative

Find ∂L/∂w for the following loss function and evaluate at w = 0.1:

> L(w) = (50w + 2)³

**Solution:**

**Identify the layers:**

- **Inner:** u = 50w + 2
- **Outer:** L = u³

**Apply the Chain Rule:**

1. **Outer derivative:** 3(50w + 2)²
2. **Inner derivative (with respect to w):** 50

**Combine:**

> ∂L/∂w = 3(50w + 2)² * 50 = **150(50w + 2)²**

**Evaluate at w = 0.1:**

> ∂L/∂w = 150(50 * 0.1 + 2)² = 150(5 + 2)² = 150 * 49 = **7350**

---

## Common Mistakes and Gotchas

- **Treating mixed terms as if both variables disappear.** For ∂/∂x of 5xy, the answer is 5y (not 0). The y is a constant — it stays. Only the x "reacts" to the derivative.
- **Confusing ∂ƒ/∂x with dƒ/dx.** The curly ∂ means **partial** — other variables are held constant. The straight d means **total** — it accounts for how other variables might also depend on x. In most ML contexts you'll see ∂, because we're always asking about one weight at a time.
- **Forgetting that the gradient is a vector, not a number.** Each component is the partial derivative for one variable. A function of 3 variables has a 3D gradient; a function of 1000 variables has a 1000D gradient.
- **Dropping the chain rule on nested loss functions.** When computing ∂L/∂w for something like (200w - 3.7)², don't forget to multiply by the inner derivative (200). The outer derivative alone gives you 2(200w - 3.7), but without the * 200 you're off by a factor of 200.

---

## Key Takeaways

- A **partial derivative** (∂ƒ/∂x) measures how the output changes when you nudge *one* variable, holding all others constant.
- The rule is simple: give the "Cold Shoulder" to every variable except the one you're differentiating with respect to — treat them as constants.
- The **Gradient** (∇ƒ) collects all partial derivatives into a single vector that points in the direction of steepest ascent. Gradient Descent moves in the **opposite** direction to minimize loss.
- The **update rule** — w_new = w_old - α * ∇L(w_old) — is the single equation that powers all of Machine Learning.
- Partial derivatives combine naturally with the **Chain Rule** — this is exactly how backpropagation computes ∂loss/∂w for each weight in a neural network.
- The gradient is a **vector** from Module 1 — its magnitude is the steepness, its direction is "which way is up."

## What's Next

With derivatives (Lesson 1), the chain rule (Lesson 2), and partial derivatives (Lesson 3), you now have the complete calculus toolkit for understanding **Gradient Descent** — the algorithm that powers nearly all of modern Machine Learning. In **Module 3**, we'll put it all together and see how a model actually learns step by step.
