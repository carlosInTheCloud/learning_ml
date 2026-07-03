# Course 1 - Module 1 - Lesson 3: Mathematical Foundations

## Determinants and Matrix Inversion

Now that we know how to "move" data using matrices (Transformations), we need to know two things:

1. **The Scale:** How much did the matrix grow or shrink the "area" of our data? (The **Determinant**)
2. **The Rewind:** Can we undo the move and get back to the original data? (The **Inverse**)

---

## The Determinant (det(A))

The determinant is a single number that tells you how the "volume" of your space changes after the transformation.

- If det(**A**) = 2: The matrix doubles the area/volume of your data.
- If det(**A**) = 0.5: The matrix shrinks the area by half.
- If det(**A**) = 0: This is the **"Red Flag."** The matrix has squashed your data so flat (into a line or a point) that information has been lost forever.

### The 2×2 Formula

```
     | a  b |
A =  | c  d |

det(A) = ad - bc
```

**Can a determinant be negative?** Yes — a negative determinant means the transformation "flips" the orientation of the space (like a mirror reflection), in addition to scaling it.

#### Worked Example

Let's find the determinant of matrix **A**:

```
     | 3  1 |
A =  | 2  2 |
```

1. Multiply the main diagonal: 3 * 2 = 6
2. Multiply the opposite diagonal: 1 * 2 = 2
3. Subtract: 6 - 2 = 4

Since the determinant is **4**, this matrix is "Healthy." It expands the space by 4x and is fully invertible.

#### Geometric Intuition: The Determinant as Area

The determinant tells you the area of the parallelogram that the matrix creates from a unit square. Here's a visual:

```
Before (Unit Square):          After (Transformed):

  (0,1) --- (1,1)                     * (a+c, b+d)
    |         |                       / \
    |   area  |                      /   \
    |   = 1   |                     /     \
  (0,0) --- (1,0)           (c,d) *  area  * (a,b)
                                    \  = |det(A)|
                                     \   /
                                      \ /
                                       * (0,0)
```

The two column vectors of the matrix become the sides of a parallelogram. The absolute value of the determinant is the area of that parallelogram. If det = 0, the parallelogram has collapsed into a line — no area, no way to recover the original shape.

### The 3×3 Determinant

In 3D, the determinant tells you how much the "unit cube" is stretched or squashed by the transformation. The calculation uses a method called **Laplace Expansion** (or "Expansion by Minors").

#### The "Checkerboard" Method

To find the determinant of a 3×3 matrix, we break it down into three smaller 2×2 determinants.

If we have matrix **A**:

```
     | a  b  c |
A =  | d  e  f |
     | g  h  i |
```

We pick the top row (a, b, c) and multiply each by the 2×2 matrix that remains when you "cross out" that number's row and column.

**The Formula:**

```
det(A) = a * det| e  f | - b * det| d  f | + c * det| d  e |
                | h  i |          | g  i |          | g  h |
```

Note the **minus sign** on b. We always follow a **plus-minus-plus** pattern (+ - +) across the row.

#### Worked Example

Let's find the determinant of:

```
     | 1  2  3 |
A =  | 0  4  5 |
     | 1  0  6 |
```

**Step 1: Expand along the top row**

Take **1**: Cross out its row and column. We are left with:

```
| 4  5 |
| 0  6 |

det = (4 * 6) - (5 * 0) = 24
```

Result: 1 * 24 = **24**

Take **2** (remember the minus!): Cross out its row and column. We are left with:

```
| 0  5 |
| 1  6 |

det = (0 * 6) - (5 * 1) = -5
```

Result: -2 * (-5) = **+10**

Take **3**: Cross out its row and column. We are left with:

```
| 0  4 |
| 1  0 |

det = (0 * 0) - (4 * 1) = -4
```

Result: 3 * (-4) = **-12**

**Step 2: Add them up**

> 24 + 10 + (-12) = **22**

### Why the Determinant is a "Red Flag" Detector

If you calculate this and the result is 0, it means your 3D space was squashed into a 2D plane or a 1D line. In the language of Lesson 1, a zero determinant means the **columns of the matrix are linearly dependent** — they don't span the full space. The matrix has redundant "directions" and has lost the ability to distinguish certain inputs from each other.

> **In ML terms:** If your data matrix has a 0 determinant, your model cannot distinguish between certain inputs because they've been "squashed" onto the same spot. This is why we check for **"Singular Matrices"** (determinant = 0) before trying to train a model.

### Properties of Determinants

A few key properties that are useful for sanity-checking your work:

- **det(AB) = det(A) * det(B)** — Scaling factors multiply. If **A** doubles area and **B** triples it, **AB** scales area by 6.
- **det(A⁻¹) = 1 / det(A)** — The inverse reverses the scaling. If **A** doubles area, **A⁻¹** halves it.
- **det(I) = 1** — The identity matrix doesn't scale anything.
- **det(Aᵀ) = det(A)** — Transposing doesn't change the determinant.

### Determinants Only Exist for Square Matrices (n × n)

**The Geometric Reason: Area vs. Volume**

Remember that the determinant measures how much a "unit" of space (an area or a volume) is scaled.

- In 2D (2×2), we measure the change in **Area**.
- In 3D (3×3), we measure the change in **Volume**.

If you have a 2×3 matrix, you are taking a 3D input and squashing it onto a 2D plane (Dimensionality Reduction).

- You can't ask "How much did the volume change?" because the volume has effectively become zero — the 3D object is now a 2D shadow.
- Because the "input" space and "output" space have different dimensions, there is no single "scaling factor" that describes the transformation.

For rectangular data (more features than samples, or vice versa), we use the **Pseudo-inverse** (Moore-Penrose Inverse) — we'll revisit this when we cover Least Squares.

---

## The Matrix Inverse (A⁻¹)

The inverse is the **"Undo" button**. If **Av** = **w**, then **A⁻¹w** = **v**.

**The Catch:** You can only invert a matrix if its determinant is **NOT zero**.

If the determinant is zero, the data was squashed into a lower dimension (like a 2D plane becoming a 1D line). You can't mathematically "un-squash" a line back into a plane because you don't know where the points originally were.

### Why This Matters for ML

In a classic ML algorithm called **Linear Regression**, the computer has to "solve" for the best weights. The closed-form solution (the **Normal Equation**) is:

> **w** = (**X**ᵀ**X**)⁻¹ **X**ᵀ**y**

To compute this, we must invert the matrix **X**ᵀ**X**. If your data has redundant features (like we discussed with Span in Lesson 1), the determinant of **X**ᵀ**X** becomes zero, the matrix becomes **"Singular"** (non-invertible), and the math breaks.

### Near-Singular Matrices: The "Almost Zero" Problem

In practice, a determinant of exactly 0 is rare. What's far more common — and more dangerous — is a determinant that's *very close* to zero. This makes the inverse **numerically unstable**: tiny rounding errors in your data explode into huge errors in the result.

This instability is measured by the **condition number** of a matrix. A high condition number means "this matrix is almost singular — don't trust the inverse."

The fix? **Regularization**. Instead of inverting **X**ᵀ**X** directly, we add a small value λ to the diagonal (borrowing from the Identity Matrix in Lesson 2):

> **w** = (**X**ᵀ**X** + λ**I**)⁻¹ **X**ᵀ**y**

This is exactly what **Ridge Regression** does. The λ**I** term nudges the determinant away from zero, stabilizing the inverse. It's a trade-off: you lose a tiny bit of accuracy to gain a huge amount of stability.

### The 2×2 Inverse Formula

To find the inverse of **A**:

```
     | a  b |
A =  | c  d |
```

1. **Find the Determinant:** det(**A**) = ad - bc
2. **Swap and Flip:**
   - Swap the positions of a and d.
   - Change the signs of b and c (make them negative).

```
          1      |  d  -b |
A⁻¹ = ------- *  | -c   a |
       det(A)
```

### Worked Example

```
     | 4  7 |
A =  | 2  6 |
```

**Step 1: Check if the inverse exists**

Before we do anything else, we must check if an inverse even exists.

> det(**A**) = ad - bc = (4 * 6) - (7 * 2) = 24 - 14 = **10**

Since the determinant is 10 (not zero), the matrix is **"Invertible."**

**Step 2: The "Swap and Flip"**

Now we create a new matrix based on our original **A**:

1. Swap the positions of the numbers on the main diagonal (4 and 6).
2. Flip the signs of the other two numbers (7 and 2).

```
Temporary matrix:

|  6  -7 |
| -2   4 |
```

**Step 3: Divide by the Determinant**

Finally, we multiply every number in our temporary matrix by 1/det(**A**), which is 1/10 (or 0.1):

```
         1    |  6  -7 |   |  0.6  -0.7 |
A⁻¹ = ----- * | -2   4 | = | -0.2   0.4 |
        10
```

**The "Proof": Does it actually work?**

If we did this correctly, multiplying the original matrix **A** by its inverse **A⁻¹** should result in the Identity Matrix (**I**).

```
     | 4  7 |   |  0.6  -0.7 |
A *  | 2  6 | * | -0.2   0.4 |
```

- Top-Left: (4 * 0.6) + (7 * -0.2) = 2.4 - 1.4 = **1**
- Top-Right: (4 * -0.7) + (7 * 0.4) = -2.8 + 2.8 = **0**
- Bottom-Left: (2 * 0.6) + (6 * -0.2) = 1.2 - 1.2 = **0**
- Bottom-Right: (2 * -0.7) + (6 * 0.4) = -1.4 + 2.4 = **1**

```
        | 1  0 |
A*A⁻¹ = | 0  1 | = I  ✓
```

---

## Common Mistakes and Gotchas

- **Forgetting the +/-/+ sign pattern in 3×3 expansion.** The middle term always gets a minus sign. It's tempting to make everything positive — double-check by writing out the signs first: (+a, -b, +c).
- **Computing the inverse without checking the determinant first.** Always calculate det(**A**) before attempting the inverse. If it's zero, stop — the inverse doesn't exist. If it's very close to zero, the inverse exists but shouldn't be trusted.
- **Assuming det(A + B) = det(A) + det(B).** This is **false**. Determinants distribute over multiplication (det(**AB**) = det(**A**) * det(**B**)), but NOT over addition.
- **Confusing the "swap and flip" positions.** In the 2×2 inverse, you swap the **diagonal** (a ↔ d) and negate the **off-diagonal** (b, c). A common mistake is to swap b and c instead — that gives a wrong answer.

---

## Practice Problems

### Problem 1: The Determinant "Red Flag"

Calculate the determinant for matrix **B**. Does this matrix have an inverse? Why or why not?

```
     | 2  4 |
B =  | 3  6 |
```

**Solution:**

> det(**B**) = (2 * 6) - (4 * 3) = 12 - 12 = **0**

The determinant is zero. The data has been squashed into a line and **cannot be inverted**. This is a singular matrix — its columns are linearly dependent (column 2 is just 2 * column 1).

---

### Problem 2: The 2×2 Inverse

Find the inverse of matrix **C**:

```
     | 1  2 |
C =  | 3  4 |
```

**Solution:**

**Find the determinant:**

> det(**C**) = (1 * 4) - (2 * 3) = 4 - 6 = **-2**

**Swap and Flip:**

```
|  4  -2 |
| -3   1 |
```

**Multiply by 1/det = -1/2:**

```
             |  4  -2 |   | -2     1   |
C⁻¹ = -1/2 * | -3   1 | = |  1.5  -0.5 |
```

**Validate (C * C⁻¹ = I):**

```
| 1  2 |   | -2     1   |
| 3  4 | * |  1.5  -0.5 |
```

- Top-Left: (1 * -2) + (2 * 1.5) = -2 + 3 = **1**
- Top-Right: (1 * 1) + (2 * -0.5) = 1 - 1 = **0**
- Bottom-Left: (3 * -2) + (4 * 1.5) = -6 + 6 = **0**
- Bottom-Right: (3 * 1) + (4 * -0.5) = 3 - 2 = **1**

```
         | 1  0 |
C*C⁻¹ =  | 0  1 | = I  ✓
```

---

### Problem 3: The 3D Volume

Find the determinant for the 3×3 matrix **D**:

```
     | 1  0  2 |
D =  | 0  1  0 |
     | 3  0  1 |
```

**Solution:**

Expand along the top row:

Take **1**: Cross out its row and column.

```
| 1  0 |
| 0  1 |

det = (1 * 1) - (0 * 0) = 1
```

Result: 1 * 1 = **1**

Take **0** (with minus sign): Cross out its row and column.

```
| 0  0 |
| 3  1 |

det = (0 * 1) - (0 * 3) = 0
```

Result: -0 * 0 = **0**

Take **2**: Cross out its row and column.

```
| 0  1 |
| 3  0 |

det = (0 * 0) - (1 * 3) = -3
```

Result: 2 * (-3) = **-6**

**Add them up:**

> 1 + 0 + (-6) = **-5**

The determinant is -5. The transformation scales volume by a factor of 5 and flips the orientation.

---

### Problem 4: Solving a System with the Inverse

Given matrix **A** and vector **b**, solve **Ax** = **b** by finding **A⁻¹**.

```
     | 1  0 |       | 3 |
A =  | 2  1 |  b =  | 8 |
```

**Solution:**

**Step 1: Find the determinant.**

> det(**A**) = (1 * 1) - (0 * 2) = 1

**Step 2: Find the inverse (Swap and Flip, divide by det = 1).**

```
      | 1   0 |
A⁻¹ = | -2  1 |
```

**Step 3: Multiply A⁻¹ by b to get x.**

> x₁ = (1 * 3) + (0 * 8) = **3**
>
> x₂ = (-2 * 3) + (1 * 8) = -6 + 8 = **2**

> **x** = [3, 2]

**Verify:** Does **Ax** = **b**?

> (1 * 3) + (0 * 2) = 3 ✓
>
> (2 * 3) + (1 * 2) = 8 ✓

**Follow-up:** Check that det(**A**) * det(**A⁻¹**) = 1. We have det(**A**) = 1 and det(**A⁻¹**) = (1 * 1) - (0 * -2) = 1. Indeed, 1 * 1 = 1 ✓. This confirms the property det(**A⁻¹**) = 1/det(**A**).

---

## Key Takeaways

- The **determinant** measures how much a matrix scales area (2D) or volume (3D). It only exists for square matrices.
- A **zero determinant** is a red flag — it means the columns are linearly dependent, the space has been collapsed, and the transformation cannot be undone.
- A **negative determinant** means the transformation flips orientation (mirror reflection).
- The **inverse** (**A⁻¹**) is the "undo" button: **A** * **A⁻¹** = **I**. It only exists when det(**A**) ≠ 0.
- In practice, watch out for **near-zero determinants** (ill-conditioning). **Regularization** (adding λ**I**) is how ML stabilizes the inverse.
- The **Normal Equation** for Linear Regression depends on matrix inversion — this is where all these concepts meet in a real algorithm.
