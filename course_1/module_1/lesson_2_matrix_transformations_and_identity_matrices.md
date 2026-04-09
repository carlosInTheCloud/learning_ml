# Course 1 - Module 1 - Lesson 2: Mathematical Foundations

## Matrix Transformations and Identity Matrices

In Lesson 1, we treated vectors as individual arrows. In Lesson 2, we group those arrows together into a **Matrix**.

If a vector is a single data point, a matrix is a **Transformation**. In Machine Learning, when we multiply a vector by a matrix, we are "pushing" that vector through a function that rotates, scales, or flips it.

---

## What is a Matrix?

A matrix is a rectangular array of numbers. We can view a matrix **A** as a collection of column vectors:

> **A** = [**v₁** **v₂**]

For example:

```
A = | 2  0 |
    | 0  2 |
```

If we multiply this matrix by a vector **x**, the matrix acts as a "scaling machine."

---

## Matrix-Vector Multiplication (The Transformation)

When you multiply a matrix by a vector, you are calculating a **Linear Combination** of the matrix's columns, using the components of the vector as the weights.

If **A** = [**v₁** **v₂**] and **x** = [x₁, x₂], then:

> **Ax** = x₁**v₁** + x₂**v₂**

### Scaling Example

In Machine Learning, this is exactly what happens when you "normalize" your data — for instance, if you want to shrink large house prices so they are easier for an algorithm to process.

Suppose we have a matrix **S** that is designed to double the x-coordinate and triple the y-coordinate of any vector it touches:

```
S = | 2  0 |
    | 0  3 |
```

Let's take a simple vector **v** = [4, 1]. Geometrically, this is an arrow pointing 4 units right and 1 unit up.

To find the new vector (the "transformed" version of **v**), we multiply the matrix by the vector. We do this by taking a linear combination of the columns of **S**, using the components of **v** as the weights:

```
Sv:
   4 * | 2 | + 1 * | 0 |
       | 0 |       | 3 |
```

Step-by-step calculation:

```
1. Scale col 1:
   4 * | 2 | = | 8 |
       | 0 |   | 0 |

2. Scale col 2:
   1 * | 0 | = | 0 |
       | 3 |   | 3 |

3. Add them up:
   | 8 | + | 0 | = | 8 |
   | 0 |   | 3 |   | 3 |
```

The resulting vector is **[8, 3]**.

The matrix **S** acted like a "function." It took our input arrow and stretched it out. If we had a thousand different vectors (a whole dataset), and we multiplied the matrix by all of them, the entire "cloud" of data would stretch in exactly the same way.

### Rotation Example (Conceptual)

What if the matrix wasn't just diagonal numbers?

```
R = |  0  -1 |
    |  1   0 |
```

If you multiply this matrix **R** by our vector [4, 1]:

```
Rv:
   4 * |  0 | + 1 * | -1 | = |  0 | + | -1 | = | -1 |
       |  1 |       |  0 |   |  4 |   |  0 |   |  4 |
```

The vector [4, 1] was rotated 90 degrees counter-clockwise to become **[-1, 4]**.

> **In ML terms:** We use these transformations to change how the computer "sees" the data — rotating it or scaling it until the patterns (like the difference between a "spam" email and a "real" email) become statistically obvious.

---

## Dimension Compatibility

### Can we multiply a 3D vector by a 2D matrix?

**Answer: No (usually).**

If you have a 2D matrix (2 rows, 2 columns) and a 3D vector (x, y, z):

- The matrix only has "instructions" for two dimensions (two columns).
- The 3D vector has three pieces of information.
- The matrix doesn't know what to do with the third component (z).

### The Exception: Dimensionality Reduction (2×3 matrix)

You *can* multiply a 3D vector by a matrix that is 2 rows by 3 columns.

- **Matrix (2×3):** 2 rows, 3 columns.
- **Vector (3×1):** 3 rows.
- **Result:** A 2D vector.

In Machine Learning, this is called **Dimensionality Reduction**. You are taking high-dimensional data (like 3D coordinates) and "squashing" it onto a 2D plane (like a shadow). This is a core technique (often called **PCA** or **Linear Projection**) used to simplify complex data while keeping the most important information.

#### Worked Example: 3D → 2D

Imagine we have 3D data (like a point in a room) and we want to "squash" it down into a 2D flat map (like a shadow on the floor).

**The Matrix (A):** A 2×3 matrix. 2 rows (the new dimensions) and 3 columns (to match our 3D input).

**The Input Vector (v):** A 3D vector representing a data point.

```
A = | 1  0  0 |      v = |  5 |
    | 0  1  0 |          |  8 |
                         | 12 |
```

In this specific matrix, the first row says "Keep the x value" and the second row says "Keep the y value." The lack of a third row means we are going to "discard" the z value.

**The Transformation (Av):**

We multiply the 2×3 matrix by the 3×1 vector. Following our rule, we take a linear combination of the three columns of **A** using the components of **v** as weights:

```
1. Scale col 1:
    5 * | 1 | = | 5 |
        | 0 |   | 0 |

2. Scale col 2:
    8 * | 0 | = | 0 |
        | 1 |   | 8 |

3. Scale col 3:
   12 * | 0 | = | 0 |
        | 0 |   | 0 |

4. Add them up:
   | 5 | + | 0 | + | 0 | = | 5 |
   | 0 |   | 8 |   | 0 |   | 8 |
```

Our 3D point [5, 8, 12] has been reduced to a 2D point **[5, 8]**.

**Why is this "Machine Learning"?**

In the real world, the matrix wouldn't just be zeros and ones. It would be a "Learned Matrix." Imagine **v** is a vector representing a customer:

- x = Age
- y = Income
- z = Credit Score

If a bank wants to simplify their analysis, they might use a matrix to combine these three numbers into just two "Meta-Features" (like Financial Stability and Spending Power). The matrix would "weight" the age, income, and credit score to produce those two new numbers.

**Dimensions summary:**

| | Shape |
|---|---|
| **Input** | 3×1 vector |
| **Transformation** | 2×3 matrix |
| **Output** | 2×1 vector |
| **Constraint** | The "3" in the matrix columns must match the "3" in the vector rows |

### Can we multiply a 2D vector by a 3D matrix?

**Answer: Yes, if the matrix is 3×2.**

If you have a matrix with 3 rows and 2 columns, and a 2D vector (x, y):

- The matrix has 2 columns (the "socket").
- The vector has 2 components (the "pins").
- Result: A 3D vector.

In ML, this is called **Feature Expansion**. You are taking a simple set of data and "lifting" it into a higher-dimensional space to find more complex patterns.

#### Worked Example: 2D → 3D

**The Input Vector (v):** A 2D point [2, 3].

**The Matrix (E):** A 3×2 matrix. 3 rows (the new dimensions) and 2 columns (to match our 2D input).

```
E = | 1  0 |      v = | 2 |
    | 0  1 |          | 3 |
    | 1  1 |
```

In this matrix:

- The first row keeps the x value (2).
- The second row keeps the y value (3).
- The third row creates a new feature by adding x and y together (2 + 3 = 5).

Step-by-step:

```
1. Scale col 1:
   2 * | 1 | = | 2 |
       | 0 |   | 0 |
       | 1 |   | 2 |

2. Scale col 2:
   3 * | 0 | = | 0 |
       | 1 |   | 3 |
       | 1 |   | 3 |

3. Add them up:
   | 2 | + | 0 | = | 2 |
   | 0 |   | 3 |   | 3 |
   | 2 |   | 3 |   | 5 |
```

**Why is this "Machine Learning"?**

Imagine you have two groups of points on a flat piece of paper that are all tangled together. You can't draw a single straight line to separate them. However, if you "lift" one group higher into 3D space (by creating a new feature like x² + y²), you can then slide a flat sheet of paper (a **hyperplane**) between the two groups to separate them perfectly.

**Dimensions summary:**

| | Shape |
|---|---|
| **Input** | 2×1 vector |
| **Transformation** | 3×2 matrix |
| **Output** | 3×1 vector |
| **Constraint** | The "2" in the matrix columns must match the "2" in the vector rows |

---

## How to Think About a Matrix as Data

This is known as the **Column Picture** of matrix multiplication.

### The "Tabular Data" Mental Model

Think of your matrix as a spreadsheet where each column represents a specific "concept" or "feature" (like Price History or Inventory Levels).

- **The Matrix (The Knowledge Base):** A collection of columns.
- **The Vector (The Decision Maker):** A list of weights (scalars) that tells the computer how much to "trust" or "value" each column.
- **The Result:** A new vector that is a linear combination of that data.

### A Real-World Example: Portfolio Management

Imagine you have a matrix **M** representing two stocks and their performance over 3 days (3 rows, 2 columns):

```
M = |  2  1 |
    | -1  3 |
    |  0  2 |
```

- Column 1: Stock A's performance (2, -1, 0).
- Column 2: Stock B's performance (1, 3, 2).

If your Weight Vector **w** = [10, 5], it means you own 10 shares of Stock A and 5 shares of Stock B.

When you calculate **Mw**:

```
1. Scale col 1:
   10 * |  2 | = |  20 |
        | -1 |   | -10 |
        |  0 |   |   0 |

2. Scale col 2:
    5 * | 1 | = |  5 |
        | 3 |   | 15 |
        | 2 |   | 10 |

3. Add them up:
   |  20 | + |  5 | = | 25 |
   | -10 |   | 15 |   |  5 |
   |   0 |   | 10 |   | 10 |
```

### The "Golden Rule" Check

Since your matrix has 2 columns (2 stocks), your weight vector must have 2 rows (2 weights). If you tried to apply 3 weights to 2 stocks, the "RC" rule would break — you'd have a weight with no stock to attach it to!

---

## Composition of Transformations

We've seen a matrix act on a single vector, but we haven't seen what happens when two matrices meet. This is often called **Composition of Transformations**.

### 1. The Conceptual Idea: A Chain Reaction

Think of matrix multiplication as a **pipeline**.

- Matrix **A** might be a "Rotate 90 degrees" machine.
- Matrix **B** might be a "Scale by 2" machine.

If you multiply them together (**AB**), you are creating a new, single machine that does both jobs at once. When you plug a vector into this new machine, it rotates *and* scales in one step.

### 2. The "Golden Rule" (RC Again!)

For **A** × **B** to work:

- The number of **Columns** in **A** must match the number of **Rows** in **B**.

> **A** (m × n) × **B** (n × p) = **Result** (m × p)

### 3. How to Calculate It (The Dot Product Method)

To find the number that goes into a specific slot of the resulting matrix, you take a **row** from the first matrix and **dot product** it with a **column** of the second matrix.

Numerical example:

```
A = | 1  2 |      B = | 5  6 |
    | 3  4 |          | 7  8 |
```

**Top-Left corner:** Row 1 of **A** [1, 2] · Column 1 of **B** [5, 7]

> (1 × 5) + (2 × 7) = 5 + 14 = 19

**Top-Right corner:** Row 1 of **A** [1, 2] · Column 2 of **B** [6, 8]

> (1 × 6) + (2 × 8) = 6 + 16 = 22

**Bottom-Left corner:** Row 2 of **A** [3, 4] · Column 1 of **B** [5, 7]

> (3 × 5) + (4 × 7) = 15 + 28 = 43

**Bottom-Right corner:** Row 2 of **A** [3, 4] · Column 2 of **B** [6, 8]

> (3 × 6) + (4 × 8) = 18 + 32 = 50

```
AB = | 19  22 |
     | 43  50 |
```

### A Very Important Warning: Order Matters!

In regular math, 5 × 2 is the same as 2 × 5. **In matrices, AB is NOT the same as BA.**

If you rotate a house 90 degrees and then move it 10 feet right, it ends up in a different spot than if you moved it 10 feet right first and then rotated it. In ML, the order of your "layers" changes everything.

### Why This Matters for ML

When you hear about "Deep Learning," the "Deep" just refers to many matrices multiplied together:

> Output = **W₃** × (**W₂** × (**W₁** × input))

By multiplying these matrices, the computer combines many simple transformations into one incredibly complex pattern-recognition machine.

---

## The Identity Matrix (The "1" of Matrices)

The **Identity Matrix** (denoted as **I**) is a special square matrix with 1s on the diagonal and 0s everywhere else.

```
I = | 1  0 |
    | 0  1 |
```

**The Magic Property:** Multiplying any vector **v** by the Identity Matrix **I** leaves the vector exactly as it is.

> **Iv** = **v**

> **In ML terms:** We use the Identity Matrix as a starting point for many algorithms. It represents a transformation that "does nothing" to the data.

### Why the Identity Matrix Matters for ML

Every layer in a Neural Network is essentially a matrix. When data (a vector) enters a layer, the matrix transforms it into a new space where the patterns are easier for the computer to see. If the matrix is "tall," it might be compressing the data; if it's "wide," it might be expanding it.

The Identity Matrix (**I**) is the "Number 1" of the matrix world. While it doesn't change the vector it touches right now, it is essential for the "algebra" of Machine Learning for three main reasons:

#### 1. The "Base Case" for Learning

When we start training a Neural Network, we often initialize the weights. If we start with a matrix of all zeros, the model is "dead" — no information can pass through. If we start with an Identity Matrix, the information passes through perfectly (**Iv** = **v**), giving the model a clean slate to start "tweaking" the weights from.

#### 2. Matrix Inversion (The "Backwards" Step)

In algebra, if you have 5x = 10, you multiply by the reciprocal (1/5) to get x.

In matrices, there is no "division." Instead, we use the **Inverse Matrix** (**A⁻¹**).

The definition of an inverse is:

> **A** × **A⁻¹** = **I**

We need the Identity Matrix to define what it means to "undo" a transformation. If a matrix rotates a vector 90 degrees, its inverse rotates it back, resulting in the Identity (the original state).

#### 3. Changes and Residuals

In modern ML (like **ResNets** or **Transformers**), we often want a model to learn only the *change* to the data, rather than the whole transformation. We use a formula like:

> Output = **I** × **x** + small_change(**x**)

This means: "Take the original data (**I**) and add a little bit of new insight to it." Without the Identity Matrix, the model would have to relearn the entire input from scratch at every single step.

### Summary of Identity Matrix

| | |
|---|---|
| **Definition** | A square matrix with 1 on the diagonal and 0 elsewhere |
| **Purpose** | Acts as the "Neutral Element" |
| **Analogy** | It's like the "Reset" button or a clear pane of glass. It doesn't distort the image (the vector), but it provides the frame we need to start building more complex lenses. |

---

## Practice Problems

### Problem 1: Scaling

Calculate **Sv** for:

```
S = | 5    0   |      v = | 10 |
    | 0    0.1 |          | 50 |
```

**Solution:**

```
Sv:
   10 * | 5 | + 50 * | 0   |
        | 0 |        | 0.1 |

   = | 50 | + | 0 | = | 50 |
     |  0 |   | 5 |   |  5 |
```

The x-component was stretched (10 → 50) and the y-component was shrunk (50 → 5). In data science, this is how we "squash" features that are too large (like house prices) and "boost" features that are too small.

---

### Problem 2: Identity

What is **Iv** for **v** = [3, 9, 21]? Why keep it around?

**Solution:**

```
     | 1  0  0 |   |  3 |   |  3 |
Iv = | 0  1  0 | * |  9 | = |  9 |
     | 0  0  1 |   | 21 |   | 21 |
```

The vector is unchanged. The Identity Matrix acts as a clean slate or "neutral" starting point — it's the "do nothing" transformation. We keep it around because it's essential for defining inverses, initializing neural network weights, and building residual connections.

---

### Problem 3: Dimensions

Is (2×3) × (3×1) possible? How many rows in the result?

**Solution:**

Yes, this is possible.

- **The "RC" Rule Check:** The matrix is 2×3 (2 rows, 3 columns). The vector is 3×1 (3 rows, 1 column).
- Because the "inner numbers" (3 and 3) match, the multiplication is allowed.
- **Why it works:** Even though the matrix only has 2 rows (making it "shorter" than the vector), it has 3 columns. Each of those 3 columns acts as a "bucket" for one of the 3 numbers in your vector.
- **The Result:** A 2×1 vector. You have successfully performed **Dimensionality Reduction** — taking 3 pieces of info and squashing them into 2.

---

