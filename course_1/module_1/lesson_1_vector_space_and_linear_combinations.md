# Course 1 - Module 1 - Lesson 1: Mathematics Fundamentals

## Vector Space and Linear Combinations

In Machine Learning, data is almost always represented as vectors. If you are predicting house prices, a single house is a vector where each element is a feature (square footage, number of rooms, etc.).

### The Concept

A **vector** is an object that has both magnitude and direction, usually written as:

> **v** = [v1, v2, v3, ..., vn]

1. **Vector Addition:** You add vectors element-wise. Geometrically, this is the "tip-to-tail" method.
2. **Scalar Multiplication:** Multiplying a vector by a number (a scalar) stretches or shrinks it.
3. **Linear Combination:** This is the most critical part of ML. It is the sum of scaled vectors:

> **W** = c1 **v1** + c2 **v2**

---

## Vector Addition: Combining Information

When we add two vectors, **u** and **v**, we are essentially combining their features.

- **Algebraically:** We add the corresponding components.

> **u** = [3, 1] and **v** = [1, 2]
>
> **u** + **v** = [3+1, 1+2] = [4, 3]

- **Geometrically:** This is the **Parallelogram Law**. If you place the tail of **v** at the head of **u**, the resulting vector from the origin to the new tip is the sum.

> **In ML terms:** Imagine **u** represents the "size" of a house and **v** represents its "location quality." Adding them (often with weights) creates a combined score for that house.

### Adding Multiple Vectors

When we move beyond adding just two vectors, we enter the world of **Linear Combinations**. This is the secret sauce of Machine Learning. Almost every model you will build — from a simple line of best fit to a massive neural network — is essentially just a very long list of vectors being added together.

Adding multiple vectors follows the **Associative** and **Commutative** properties. This means the order doesn't matter, and how you group them doesn't matter.

If we have three vectors **u**, **v**, and **w**:

> **u** + **v** + **w** = [u1 + v1 + w1, u2 + v2 + w2, ..., un + vn + wn]

You simply sum all the first components, then all the second components, and so on.

Geometrically, adding multiple vectors is like following a treasure map. You start at the origin (0, 0), follow the first arrow, then start the second arrow where the first one ended, and the third where the second ended. The resulting vector (the **Resultant**) is the straight line from the very beginning to the very end.

---

## Vector Multiplication

### Scalar Multiplication: Scaling Influence

A **scalar** is just a regular number (like 2, -0.5, or π). When you multiply a vector **v** by a scalar *c*, you change its magnitude (length) and potentially its direction (if *c* is negative).

- **Algebraically:** Multiply every component by *c*.

> c = 2, **v** = [1, 3] → 2**v** = [2, 6]

- **Geometrically:** You are stretching or shrinking the arrow along the same line.

> **In ML terms:** This is how we apply **Weights**. If a model decides that "Square Footage" is twice as important as "Year Built," it will multiply the square footage vector by a scalar of 2.

### The Dot Product: Measuring Similarity

While we can multiply a vector by a scalar, we also often "multiply" two vectors together using the **Dot Product** (**u** · **v**). This results in a single number (a scalar), not a new vector.

> **u** · **v** = (u1 × v1) + (u2 × v2) + ... + (un × vn)

**Example:** Given the vectors **u** = [2, 3] and **v** = [4, 5]:

> **u** · **v** = (2 × 4) + (3 × 5) = 8 + 15 = 23

**Why it matters:** The dot product tells us how much two vectors point in the same direction.

- If the dot product is **high**, the vectors are similar.
- If it is **zero**, the vectors are **orthogonal** (perpendicular / unrelated).
- If it is **negative**, they point in opposite directions.

> **In ML terms:** This is the core of **"Pattern Matching."** A model takes your input data, does a dot product with its learned weights, and the resulting number tells it how closely your data matches a specific pattern (like "this image is a cat").

---

## Linear Combination

A **Linear Combination** takes several vectors, scales them, and adds them together to create a new vector.

| | |
|---|---|
| **Input** | Scalars and Vectors |
| **Output** | A Vector |
| **Formula** | **y** = c1 **v1** + c2 **v2** |
| **Intuition** | "I am mixing 2 parts of Vitamin A and 3 parts of Vitamin B to create a new Supplement Vector." |

### Linear Combination Example

Let's ground this in a concrete numerical example. Imagine we are building a simple model to predict a "Fitness Score" based on two factors: **Hours of Sleep** and **Kilometers Cycled**.

#### The Setup

We have two vectors representing these features:

- **v1** = [1, 0] — The "Sleep" dimension
- **v2** = [0, 1] — The "Cycling" dimension

Now, let's say a specific person slept 8 hours and cycled 30 km. In ML, the "weights" (scalars) are what the model learns. Let's assume the model decided:

- c1 = 0.5 — Each hour of sleep adds 0.5 to the score
- c2 = 0.2 — Each km cycled adds 0.2 to the score

#### The Calculation

We apply the Linear Combination formula:

> **y** = c1(8 **v1**) + c2(30 **v2**)

**Step 1: Scale the vectors (Scalar Multiplication)**

- Scaled Sleep: 0.5 × [8, 0] = [4, 0]
- Scaled Cycling: 0.2 × [0, 30] = [0, 6]

**Step 2: Add the results (Vector Addition)**

> **y** = [4, 0] + [0, 6] = [4, 6]

**The Result:** The final vector [4, 6] represents the contribution of each activity to the fitness goal. If we wanted a single number (the "Score"), we would just sum those components: 4 + 6 = 10.

#### What if the vectors aren't "clean" [1, 0]?

Most data is messy. Imagine two different workouts that both affect "Endurance" and "Strength":

- **Workout A** (**v1**): [2, 1] — High endurance, low strength
- **Workout B** (**v2**): [1, 2] — Low endurance, high strength

If you do 2 of Workout A and 3 of Workout B:

> **y** = 2[2, 1] + 3[1, 2] = [4, 2] + [3, 6] = [7, 8]

By combining these vectors, you ended up with a total "benefit" of 7 units of endurance and 8 units of strength. This is exactly how a neural network layer combines multiple inputs to find a new "hidden" state.

---

## Dot Product vs. Linear Combination

Think of a **Linear Combination** as a process of *building something new* (a vector), while the **Dot Product** is a process of *measuring something* (a scalar).

### 1. Linear Combination (The "Recipe")

A Linear Combination takes several vectors, scales them, and adds them together to create a new vector.

| | |
|---|---|
| **Input** | Scalars and Vectors |
| **Output** | A Vector |
| **Formula** | **y** = c1 **v1** + c2 **v2** |
| **Intuition** | "I am mixing 2 parts of Vitamin A and 3 parts of Vitamin B to create a new Supplement Vector." |

### 2. Dot Product (The "Comparison")

A Dot Product takes two vectors and multiplies their corresponding parts to see how much they "overlap" or align.

| | |
|---|---|
| **Input** | Two Vectors (of the same size) |
| **Output** | A Scalar (a single number) |
| **Formula** | **u** · **v** = u1 v1 + u2 v2 |
| **Intuition** | "How much does this person's health profile (Vector A) match the ideal athlete profile (Vector B)?" |

### 3. Where They Meet (The "Matrix-Vector" Secret)

Here is where people often get them confused (and where the magic of ML happens):

When we multiply a **Matrix** by a **Vector**, we are actually doing **both** at once.

- **From one perspective:** You are taking a Linear Combination of the columns of the matrix.
- **From the other perspective:** You are calculating the Dot Product of the vector with each row of the matrix.

In Machine Learning, we usually have a "Weight Vector" **w** and a "Data Vector" **x**. When we want to make a prediction, we calculate:

> Prediction = **w** · **x**

Because this results in a single number (like a price or a probability), we use the **Dot Product**. If we wanted to transform the data into a new multi-dimensional space, we would use **Linear Combinations**.

---

## Span

The concept of **Span** is one of the most important ideas in Linear Algebra because it defines the "reach" of your data. If your Machine Learning model's features don't have enough "Span," the model can never learn certain patterns, no matter how much data you throw at it.

### The Formal Definition

The **Span** of a set of vectors {**v1**, **v2**, ..., **vn**} is the collection of **all possible linear combinations** of those vectors.

Mathematically, it's every possible vector **y** you can create by changing the scalars:

> **y** = a **v1** + b **v2** + c **v3** + ...

### Visualizing Span in 2D

Think of Span as the "territory" you can cover if those vectors were your only directions of travel.

1. **A Single Vector:** If you only have **v** = [1, 1], you can only scale it (2**v**, -5**v**, 0.1**v**). You are stuck on a single infinite **line** passing through the origin.
2. **Two Linearly Dependent Vectors:** **u** = [1, 1] and **v** = [-1, -1]. **v** adds no new information. It's just **u** pointing the other way. You are still stuck on that same line.
3. **Two Linearly Independent Vectors:** If **u** = [1, 0] and **v** = [0, 1], you can reach **any point** in the 2D plane. By mixing these two, you can "walk" to any (x, y) coordinate.

### The "Redundancy" Problem in ML

In Machine Learning, we call linearly dependent vectors **Redundant Features**.

Imagine you are predicting health and you have two features:

- Height in inches
- Height in centimeters

Since centimeters are just inches multiplied by a scalar (2.54), these two vectors are **Linearly Dependent**.

- They don't increase the Span of your model.
- They point in the exact same "direction" in your data space.
- Adding the second one doesn't help the model learn anything new; it just makes the math more crowded.

### The "Basis"

If a set of vectors is linearly independent and their span covers the entire space (like the 2D plane), we call that set a **Basis**. In ML, we want our features to be as close to a basis as possible — meaning they each provide unique, non-overlapping information.

---

## Practice Problems

### Problem 1: The "Fitness Tracking" Scenario

You have two "Activity Vectors" representing the impact on Endurance and Strength:

- **Running** **r** = [3, 1] — 3 units of Endurance, 1 unit of Strength
- **Weightlifting** **w** = [1, 4] — 1 unit of Endurance, 4 units of Strength

If you perform 3 units of Running and 2 units of Weightlifting, what is the Resultant Vector **y** representing your total gains?

**Solution:**

> **y** = 3**r** + 2**w**
>
> **y** = [3×3, 3×1] + [2×1, 2×4]
>
> **y** = [9, 3] + [2, 8]
>
> **y** = [11, 11]

---

### Problem 2: Visualizing the "Span"

Consider two vectors:

- **u** = [1, 1]
- **v** = [-1, -1]

**Part A:** Calculate the linear combination 2**u** + 2**v**.

**Solution:**

> **y** = 2**u** + 2**v** = 2[1, 1] + 2[-1, -1] = [2, 2] + [-2, -2] = [0, 0]

The resulting vector has no length.

**Part B:** The **Span** of these two vectors is restricted to a **single line** through the origin (the line y = x). It does **not** cover the entire 2D plane because **v** is just a scalar multiple of **u** (**v** = -1 · **u**), making them linearly dependent.

---

### Problem 3: The Dot Product "Signal"

In a simple recommendation system, a User's Interest Vector is **a** = [5, 2] (where 5 is interest in "Action" and 2 is interest in "Comedy").

There are two movies available:

- **Movie X:** [4, 1]
- **Movie Y:** [1, 5]

**Task:** Calculate the dot product for both and determine which movie the system should recommend.

**Solution:**

Movie X:

> **a** · Movie X = (5 × 4) + (2 × 1) = 20 + 2 = 22

Movie Y:

> **a** · Movie Y = (5 × 1) + (2 × 5) = 5 + 10 = 15

**Movie X** has a higher dot product (22 > 15), meaning it aligns more closely with the user's interests. The system should recommend **Movie X**.

---

