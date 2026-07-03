# Course 1 - Module 2 - Lesson 1: Calculus for Machine Learning

If Linear Algebra is the "language" we use to describe our data, Calculus is the **"engine"** that allows a Machine Learning model to learn. When a model makes a mistake, Calculus tells it exactly how much to tweak its weights to be more accurate next time.

## The Derivative (The "Slope" of Learning)

In Machine Learning, we want to minimize **Error**. To do that, we need to know: *If I change my input slightly, how much does my output change?*

That rate of change is the **Derivative**.

---

## Visualizing the Derivative

Imagine a hilly landscape representing your model's "Loss" (error).

- If you are at the top of a hill, the slope is **steep**.
- If you are at the bottom of a valley (the goal!), the slope is **zero**.

The derivative tells you the **slope of the tangent line** at any specific point on a curve. But the slope isn't just a number — its **sign** and **size** both matter:

- **Positive slope:** The function is going uphill (increasing). To reduce loss, move **left** (decrease the weight).
- **Negative slope:** The function is going downhill (decreasing). To reduce loss, move **right** (increase the weight).
- **Zero slope:** You've reached a flat spot — either a **minimum** (the goal), a maximum, or a saddle point.
- **Large slope:** You're far from the minimum — take a bigger step.
- **Small slope:** You're getting close — take a smaller step.

This is exactly the logic behind **Gradient Descent**: read the sign and size of the derivative, then step in the opposite direction.

### A Note on Notation

You'll see derivatives written two ways:

- **ƒ'(x)** — Lagrange notation. Compact, common in ML papers and textbooks.
- **dy/dx** — Leibniz notation. Makes the "rate of change" meaning explicit: "the change in y per change in x."

They mean the same thing for single-variable functions. In Lesson 2, you'll see **∂y/∂x** — the curly ∂ signals a **partial derivative**, used when a function has multiple variables.

---

## The Power Rule (The First Tool in Your Kit)

You don't need to calculate limits by hand like in high school. In ML, we use rules. The most fundamental is the **Power Rule**.

If you have a function where x is raised to a power n:

> ƒ(x) = xⁿ

The derivative is found by **bringing the power down** to the front and **subtracting 1** from the exponent:

> ƒ'(x) = n * xⁿ⁻¹

**Examples:**

| ƒ(x) | ƒ'(x) | Why |
|---|---|---|
| x¹ | 1 * x⁰ = **1** | The slope of a straight line is its coefficient |
| x² | 2x | |
| x⁵ | 5x⁴ | |
| 7x³ | 7 * 3x² = **21x²** | Constants come along for the ride (see below) |
| x⁰ = 1 | 0 * x⁻¹ = **0** | The Power Rule proves the Constant Rule |

---

## Derivative Rules

### The Constant Rule

The derivative of a plain number (like 5 or 100) is always **0**. A flat line has no slope.

### The Constant Multiple Rule

If ƒ(x) = c * g(x), then ƒ'(x) = c * g'(x). Constants just "come along for the ride" — you differentiate what's attached to x and leave the constant alone.

> ƒ(x) = 7x³ → ƒ'(x) = 7 * 3x² = 21x²

### The Sum Rule

If you have a sum of terms, take the derivative of each part separately:

> ƒ(x) = x² + x³ → ƒ'(x) = 2x + 3x²

---

## Why This Matters

In the **Linear Regression** algorithm, we use a "Cost Function" (usually **Mean Squared Error**) which often looks like a parabola (x²).

1. We take the **derivative** of that error.
2. The derivative tells us which way is **"downhill."**
3. The model takes a step in that direction. This is called **Gradient Descent**.

### A Concrete Example: Gradient Descent in One Step

Let's say our model has one weight *w*, and the loss function is:

> L(w) = (w - 3)²

This is a parabola centered at w = 3 — the minimum is at w = 3 (where the loss is 0).

**Take the derivative:**

> L'(w) = 2(w - 3)

**Now evaluate at different points:**

| Current w | L'(w) | Meaning | Action |
|---|---|---|---|
| w = 5 | 2(5 - 3) = **+4** | Positive slope — going uphill to the right | Move left (decrease w) |
| w = 1 | 2(1 - 3) = **-4** | Negative slope — going uphill to the left | Move right (increase w) |
| w = 3 | 2(3 - 3) = **0** | Zero slope — at the bottom of the valley | Stop — you've found the minimum |

The derivative always points you toward w = 3, the minimum. This is Gradient Descent in its simplest form: **read the slope, step the other way, repeat.**

---

## Rules Summary

| Rule | Formula | Example |
|---|---|---|
| **Power Rule** | ƒ(x) = xⁿ → ƒ'(x) = nxⁿ⁻¹ | x⁴ → 4x³ |
| **Constant Rule** | ƒ(x) = c → ƒ'(x) = 0 | 10 → 0 |
| **Constant Multiple** | ƒ(x) = c * g(x) → ƒ'(x) = c * g'(x) | 7x³ → 21x² |
| **Sum Rule** | ƒ(x) = g(x) + h(x) → ƒ'(x) = g'(x) + h'(x) | x² + x³ → 2x + 3x² |

---

## Practice Problems

### Quick Check: The First Calculus Drill

Find the derivatives (ƒ'(x)) for these three functions:

**1.** ƒ(x) = x⁴

> ƒ'(x) = 4x³

**2.** ƒ(x) = 3x² + 10

> ƒ'(x) = 6x + 0 = **6x**

**3.** ƒ(x) = 2x³ + 5x

> ƒ'(x) = 6x² + 5x⁰ = 6x² + 5(1) = **6x² + 5**

---

### Finding the Minimum

**4.** Given ƒ(x) = x² - 6x + 10, find ƒ'(x) and determine the value of x where the slope is zero.

**Solution:**

> ƒ'(x) = 2x - 6

Set the derivative equal to zero and solve:

> 2x - 6 = 0
>
> 2x = 6
>
> x = **3**

At x = 3, the slope is zero — this is the **minimum** of the function. This is exactly what Gradient Descent automates: it keeps stepping until it finds the x where ƒ'(x) = 0.

---

## Key Takeaways

- The **derivative** measures the rate of change — how much the output shifts when you nudge the input.
- The **sign** tells you direction (uphill or downhill); the **magnitude** tells you steepness.
- The **Power Rule** (ƒ(x) = xⁿ → ƒ'(x) = nxⁿ⁻¹) handles the vast majority of derivatives you'll see early on.
- **Gradient Descent** is just "read the derivative, step the opposite way, repeat" until the slope hits zero.
- dy/dx and ƒ'(x) mean the same thing. You'll see ∂y/∂x in Lesson 2 when functions have multiple variables.
