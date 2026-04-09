# Course 1 - Module 2 - Lesson 2: Calculus for Machine Learning

## The Chain Rule (The "Onion" Rule)

In Machine Learning, we often have functions "nested" inside each other. A common example is the Sigmoid or ReLU activation function in a neural network, which wraps around the weighted sum of inputs. To find the derivative of a nested function, we use the **Chain Rule**.

---

## The Conceptual Idea

Think of a nested function as an **onion**.

- ƒ(g(x)) means g(x) is the **inner layer** and ƒ is the **outer layer**.
- To find the derivative, you "peel" it:
  1. Take the derivative of the **outside** (leaving the inside exactly as it is).
  2. Multiply it by the derivative of the **inside**.

**The Formula:**

> If y = ƒ(g(x)), then:
>
> dy/dx = ƒ'(g(x)) * g'(x)

In words: *derivative of the outer, evaluated at the inner* **×** *derivative of the inner*.

---

## A Numerical Walkthrough

Let's find the derivative of y = (4x² + 1)³.

**First, identify the layers:**

- Let **u** = 4x² + 1 (the **inner** function)
- Then y = u³ (the **outer** function)

**Step 1: The Outside.** Differentiate y = u³ with respect to u. The 3 comes down, the new power is 2 — but leave the inner function untouched:

> 3(4x² + 1)²

**Step 2: The Inside.** Now differentiate u = 4x² + 1 with respect to x:

> 8x

**Step 3: Combine.** Multiply Step 1 by Step 2.

> dy/dx = 3(4x² + 1)² * 8x = **24x(4x² + 1)²**

---

## Why This Matters for Your ML Study

When you reach **Backpropagation**, you'll see that the computer calculates how to change a weight (w) by "chaining" together the derivatives of every layer between the weight and the final error.

Here's what that looks like for a simple 2-layer network:

```
input --[w₁]--> layer₁ --[w₂]--> layer₂ ---> loss
```

To figure out how to adjust w₁, the model needs to know: "How does changing w₁ affect the loss?" But w₁ doesn't touch the loss directly — it goes through layer₁ and layer₂ first. So we chain:

> ∂loss/∂w₁ = ∂loss/∂layer₂ * ∂layer₂/∂layer₁ * ∂layer₁/∂w₁

Each `*` is the Chain Rule in action. If you have 5 layers, you have 5 derivatives multiplied together.

### The Vanishing Gradient Problem

There's a catch: when you chain many small derivatives together, the product can become **extremely small**. The sigmoid function's maximum derivative is only 0.25 — chain five of those together and you get 0.25⁵ ≈ 0.001. The gradients "vanish," and the early layers of the network essentially stop learning.

This is *the* reason **ReLU** replaced sigmoid in deep networks. ReLU's derivative is either 0 or 1, so chaining many of them together doesn't shrink the gradient. It's a direct consequence of how the Chain Rule works at scale.

---

## Practice Problems (Two Layers)

### Problem 1

ƒ(x) = (3x + 5)²

**Solution:**

- **u** = 3x + 5 (inner), y = u² (outer)

1. **Outside:** 2(3x + 5)
2. **Inside:** 3
3. **Combine:** 2(3x + 5) * 3 = **6(3x + 5) = 18x + 30**

---

### Problem 2

ƒ(x) = (x² + 10)⁴

**Solution:**

- **u** = x² + 10 (inner), y = u⁴ (outer)

1. **Outside:** 4(x² + 10)³
2. **Inside:** 2x
3. **Combine:** 4(x² + 10)³ * 2x = **8x(x² + 10)³**

---

### Problem 3

ƒ(x) = (x³ - 7)⁵

**Solution:**

- **u** = x³ - 7 (inner), y = u⁵ (outer)

1. **Outside:** 5(x³ - 7)⁴
2. **Inside:** 3x²
3. **Combine:** 5(x³ - 7)⁴ * 3x² = **15x²(x³ - 7)⁴**

---

### Problem 4

ƒ(x) = (2x² + 3x)²

**Solution:**

- **u** = 2x² + 3x (inner), y = u² (outer)

1. **Outside:** 2(2x² + 3x) = 4x² + 6x
2. **Inside:** 4x + 3
3. **Combine:** (4x² + 6x)(4x + 3) = 16x³ + 12x² + 24x² + 18x = **16x³ + 36x² + 18x**

---

### Problem 5: Spot the Layers

ƒ(x) = √(x² + 1)

*Hint:* Rewrite the square root as a power: √(x² + 1) = (x² + 1)^(1/2).

**Solution:**

- **u** = x² + 1 (inner), y = u^(1/2) (outer)

1. **Outside:** (1/2)(x² + 1)^(-1/2)
2. **Inside:** 2x
3. **Combine:** (1/2)(x² + 1)^(-1/2) * 2x = **x / √(x² + 1)**

---

## Three Layers: Extending the Chain

When a function has three nested layers, we just keep peeling. The chain rule extends naturally:

> If y = ƒ(g(h(x))), then:
>
> dy/dx = ƒ'(g(h(x))) * g'(h(x)) * h'(x)

Peel the outermost shell, then the middle, then the core — and multiply them all together.

### Numerical Example

Let's look at y = [(2x² + 1)³ + 5]².

**Identify the three layers:**

- **h(x)** = 2x² + 1 (the core)
- **g(u)** = u³ + 5 (the middle)
- **ƒ(v)** = v² (the outer shell)

**Layer 1 (The Outer Shell):** Something squared.

> Derivative: 2[(2x² + 1)³ + 5]¹

**Layer 2 (The Middle):** (2x² + 1)³ + 5.

> Derivative: 3(2x² + 1)² (the +5 becomes 0)

**Layer 3 (The Core):** 2x² + 1.

> Derivative: 4x

**The Final Chain (Multiply them all):**

> dy/dx = 2[(2x² + 1)³ + 5] * 3(2x² + 1)² * 4x = **24x[(2x² + 1)³ + 5](2x² + 1)²**

---

## Practice Problems (Three Layers)

### Problem 1

Find the derivative ƒ'(x) for:

ƒ(x) = [(4x + 2)² + 10]³

**Solution:**

**Layer 1 (Outer):** Something cubed.

> 3[(4x + 2)² + 10]²

**Layer 2 (Middle):** (4x + 2)² + 10.

> 2(4x + 2) (the +10 becomes 0)

**Layer 3 (Inner):** 4x + 2.

> 4

**Combine:**

> ƒ'(x) = 3[(4x + 2)² + 10]² * 2(4x + 2) * 4 = **24(4x + 2)[(4x + 2)² + 10]²**

---

## Common Mistakes and Gotchas

- **Forgetting to multiply by the inner derivative.** This is the #1 chain rule mistake. You correctly differentiate the outer function but stop there. Always ask yourself: "Did I multiply by the derivative of what's inside?"
- **Replacing the inner function instead of leaving it intact.** In Step 1, you differentiate the outer function *while leaving the inner expression untouched*. For y = (4x² + 1)³, Step 1 gives you 3(4x² + 1)², NOT 3(8x)².
- **Leading constants aren't the "outer" function.** In ƒ(x) = 5(2x³ + 4)², the outer function is ( )², not 5 * ( ). The 5 is a constant multiple that comes along for the ride. The layers are: outer = u², inner = 2x³ + 4, then multiply the whole thing by 5.
- **Not recognizing hidden nesting.** Functions like √(x² + 1), sin(3x), or e^(2x) are all nested — the square root, sine, and exponential are "outer" wrappers around an "inner" expression. If in doubt, ask: "Can I rewrite this as ƒ(something)?"

---

## Key Takeaways

- The **Chain Rule** handles nested functions: peel from the outside in, multiply all the derivatives together.
- **Always name your layers first** — identify the inner and outer functions before you start differentiating.
- For two layers: dy/dx = ƒ'(g(x)) * g'(x).
- For three layers: dy/dx = ƒ'(g(h(x))) * g'(h(x)) * h'(x). The pattern extends to any depth.
- In **Backpropagation**, the Chain Rule is applied across every layer of a neural network — each layer's derivative is multiplied together to figure out how to adjust the weights.
- Chaining many small derivatives causes the **Vanishing Gradient Problem** — the reason deep networks use ReLU instead of sigmoid.

## What's Next

In Lesson 2, we learned how to differentiate nested functions. In **Lesson 3**, we'll tackle **Partial Derivatives** — what happens when a function has multiple inputs (like multiple weights in a model), and we need to find the slope with respect to just one of them while holding the others constant.
