# Course 2 - Lesson 2: Appendix

## Appendix A: Deriving the Logit from the Sigmoid

This appendix shows how we "unbend" the Sigmoid curve back into a straight line. We take the Logistic Function and solve for the linear equation wx + b — essentially putting the machine in reverse.

---

### The Derivation: Unbending the Curve

Let p be the probability output of our model: p = P(y=1 | x; w,b).

For clean notation, let z = wx + b. Our starting point is the standard Logistic Function:

```
         1
p = -----------
    1 + e⁻ᶻ
```

**Step 1: Invert both sides.** Get z out of the denominator by taking the reciprocal:

> 1/p = 1 + e⁻ᶻ

**Step 2: Isolate the exponential term.** Subtract 1 from both sides:

> 1/p - 1 = e⁻ᶻ

**Step 3: Find a common denominator.** Rewrite 1 as p/p so we can combine the left side into a single fraction:

> 1/p - p/p = (1 - p) / p = e⁻ᶻ

So:

> (1 - p) / p = e⁻ᶻ

**Step 4: Cancel the base e.** Take the natural logarithm (ln) of both sides to bring -z down from the exponent:

> ln((1 - p) / p) = -z

**Step 5: Remove the negative sign.** Multiply both sides by -1:

> -ln((1 - p) / p) = z

**Step 6: Flip the fraction.** Use the logarithm property -ln(a/b) = ln(b/a) to flip the fraction inside the logarithm:

> ln(p / (1 - p)) = z

**Step 7: Replace z** with our original linear combination:

> **ln(p / (1 - p)) = wx + b**

This is the **Logit Function**. The left side is the log-odds of bonking. The right side is a straight line. Logistic Regression is Linear Regression on the log-odds scale.

### Numerical Verification

Using the worked example from the lesson (w = 0.05, b = -10, x = 220W, ŷ = 0.731):

> ln(0.731 / (1 - 0.731)) = ln(0.731 / 0.269) = ln(2.717) = **1.0**
>
> wx + b = 0.05 * 220 + (-10) = 11 - 10 = **1.0** ✓

The algebra checks out — the logit of the probability recovers the original linear combination.

---

## Appendix B: The Origin of the Logistic Function

### Pierre François Verhulst (1838)

In the early 19th century, a mathematician named **Verhulst** was studying population growth. The prevailing theory (Malthusian growth) stated that populations grow exponentially: more people make more babies, who make more babies.

Unrestricted exponential growth naturally involves Euler's number (e), because eˣ is the only mathematical function where the rate of growth is exactly equal to the current size (its derivative is itself).

But Verhulst realized this was physically impossible. A population can't grow to infinity; it eventually runs out of food or space (a "carrying capacity"). He needed an equation that said:

- "Start by growing exponentially..."
- "...but as you get closer to the maximum capacity, slow down and flatten out."

He wrote this as a differential equation:

```
dP/dt = rP(1 - P/K)
```

Where P is population, r is growth rate, and K is maximum capacity. When you solve this differential equation with calculus, the resulting formula **is** the Logistic Function.

### Why Use e? The Information Theory Connection

In Appendix A, we went from sigmoid → logit (unbending the curve into a line). Here we go the other way — starting from log-odds and deriving the sigmoid. Both paths arrive at the same place, confirming the two functions are true inverses.

The e isn't a coincidence — it is deeply tied to how we mathematically measure **Odds**.

If we set our linear model equal to the natural log-odds:

> ln(p / (1-p)) = z

Let's solve for p (the probability):

1. Raise e to both sides to remove the ln:

> p / (1-p) = eᶻ

2. Multiply both sides by (1-p):

> p = eᶻ(1 - p) = eᶻ - peᶻ

3. Move all p terms to the left:

> p + peᶻ = eᶻ

4. Factor out p:

> p(1 + eᶻ) = eᶻ

5. Isolate p:

> p = eᶻ / (1 + eᶻ)

6. Divide top and bottom by eᶻ to clean up:

> p = 1 / (1 + e⁻ᶻ)

**Boom.** The Logistic Function. The e is there because taking the inverse of the natural logarithm inherently requires e.

---

[Back to Lesson 2: Logistic Regression](lesson_2_logistic_regression.md)
