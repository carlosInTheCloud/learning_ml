# Course 2 - Lesson 1: Simple Linear Regression

The name **Simple Linear Regression** comes from combining two different ideas: one about the shape of the model and one about a statistical phenomenon discovered in the 1800s.

---

## The "Linear" Part (The Shape)

This is called **Linear** because the relationship between your input (Power) and your output (Speed) is modeled as a straight line.

In algebra, a linear equation is any equation where the variables (x) are only to the first power. No x², no √x.

- **Linear:** y = wx + b (a straight line)
- **Non-Linear:** y = wx² + b (a curve)

## The "Regression" Part (The History)

In plain English, "regress" usually means to go backward or get worse. The name comes from a scientist named **Francis Galton** who was studying the heights of parents and their children.

He noticed a pattern:

- If parents were extremely tall, their children tended to be tall, but **shorter** than the parents.
- If parents were extremely short, their children tended to be short, but **taller** than the parents.

He called this **"Regression toward the Mean."** The data points "regressed" (moved back) toward the average height of the population.

Today, we use the term "Regression" for any model that predicts a **continuous numerical value** (like 22.5 km/h, $500,000, or 72 degrees).

- **Classification:** Predicting a "Category" (Is it a dog or a cat?).
- **Regression:** Predicting a "Quantity" (How much does the dog weigh?).

So, **Linear Regression** literally means: *"Using a straight line to predict a quantity."*

---

## Ordinary Least Squares (OLS)

Linear Regression is also called **Ordinary Least Squares (OLS)**. It is the formal, statistical name for the exact Linear Regression method.

- **"Ordinary":** We are using the standard, unweighted approach. Every data point is equally important and the variance of the errors is constant.
- **"Least":** We are minimizing our Cost Function (finding the bottom of the bowl).
- **"Squares":** We are minimizing the sum of the **squared** errors (the residuals).

OLS is the mathematical framework that dictates that squaring errors and minimizing them is the optimal way to fit a straight line to data.

### OLS Assumptions

OLS guarantees the "best" fit, but only if these assumptions hold:

1. **Linearity:** The true relationship between x and y is actually a straight line (not a curve).
2. **Independence:** The data points don't influence each other (your 3rd ride doesn't depend on your 2nd ride).
3. **Homoscedasticity:** The variance of the errors is constant across all values of x. (The "spread" of the data around the line is roughly the same whether you're at 100W or 300W.)
4. **Normality of Residuals:** The errors are roughly normally distributed (most errors are small, a few are large).

When these assumptions are violated, OLS still gives you *a* line — it just might not be the *best* line. This is when you reach for more advanced techniques.

### The Closed-Form Solution (Normal Equation)

Because OLS produces a perfect convex parabola, we don't actually *have* to step down the hill. We can use matrix calculus to set the derivative to zero and solve for the bottom in one calculation — the **Normal Equation**:

> **w** = (**X**ᵀ**X**)⁻¹ **X**ᵀ**y**

But computing the inverse of (**X**ᵀ**X**) is incredibly computationally expensive. If you have 100,000 features, calculating that inverse would crash most standard computers. **Gradient Descent** bypasses this bottleneck.

---

## The Model

Imagine you have a spreadsheet of your rides.

- **Feature (x):** Average Power (Watts)
- **Label (y):** Average Speed (km/h)

We assume the relationship is a straight line:

> ŷ = wx + b

- **w (Weight):** The "slope" — how much speed you gain per watt.
- **b (Bias):** The "intercept" — your speed if power was zero (maybe you're coasting down a hill!).

---

## The Cost Function: The "North Star" of Machine Learning

### What Is It?

The **Cost Function** (also called the Loss Function or Objective Function) is the "GPS" for your model. Without it, your algorithm has no way of knowing if it's getting closer to the truth or driving off a cliff.

- It takes the parameters of your model (w and b) as inputs.
- It outputs a **single scalar number** representing the total error.
- **High Cost** = Poor performance (the model is guessing wildly).
- **Low Cost** = High accuracy (the model has "learned" the pattern).

### Why Do We Use It?

We don't use a Cost Function just to see how "bad" we are — we use it to **optimize**. In calculus terms, we are looking for the **Global Minimum**. Because the Cost Function is a surface, we can calculate the "slope" (gradient) at any point. This slope tells us exactly how to nudge w and b to make the error smaller.

### Residuals: The Building Blocks of Error

A **residual** is the vertical distance between a data point and the line — it's what MSE squares and averages. If the point is above the line, the residual is negative; if below, it's positive.

![](linear_regression_speed_vs_power.png)

The goal of Linear Regression is to position the line so that these vertical bars are as small as possible, collectively.

### The Mean Squared Error (MSE)

A real model has to satisfy hundreds of points at once. We use the **Mean Squared Error (MSE)** to find the "Average Wrongness" across the whole dataset:

```
          1    ᵐ
J(w,b) = --- * Σ  (ŷᵢ - yᵢ)²
          2m  ⁱ⁼¹
```

Where:

| Symbol | Meaning |
|---|---|
| **J** | Standard symbol for a Cost Function (Objective Function) |
| **m** | Total number of data points (rows in your spreadsheet) |
| **ŷᵢ** | The model's prediction for data point i |
| **yᵢ** | The actual value recorded for data point i |

The 1/2m denominator (instead of 1/m) is a convenience — when we take the derivative of the squared term, the 2 cancels out, making the math cleaner.

### Why the Shape Is a 3D "Bowl"

Because we are squaring the difference, the math behaves like a parabola (y = x²).

- If you have **one parameter** (w), the cost is a 2D U-shape.
- If you have **two parameters** (w and b), the U-shape rotates into a 3D **"Bowl"** or **Paraboloid**.

### Why "Convex" Is the Gold Standard

A 3D Paraboloid is a **convex function**, which means:

1. It has no "fake" bottoms (no local minima).
2. Any "downhill" path will eventually lead to the same absolute bottom.
3. This **guarantees** that Gradient Descent will always find the best possible version of your model, provided your Learning Rate (α) isn't too high.

---

## The Training Loop (The Big Picture)

The "Life of a Model":

1. **Initialize:** The computer guesses random values for w and b.
2. **Predict:** It calculates ŷ for every row of power data.
3. **Calculate Cost:** It sees how far off the "Average Wrongness" is.
4. **Gradient Descent:** It takes the Partial Derivative of the Cost (J) with respect to w and b.
5. **Update:** It nudges w and b in the direction that makes the cost smaller.
6. **Repeat:** It does this until the line fits the data.

![The Learning Curve](images/rapid_improvement.png)

---

## Gradient Descent

**The core idea:** Gradient Descent is an optimization algorithm that iteratively adjusts your model's parameters to minimize a cost function. Think of it as standing on a hilly landscape in dense fog — you can't see the bottom, but you can feel the slope under your feet, so you take small steps downhill until the ground flattens out.

### Conceptual Exercise: The "Tuning" Phase

Let's say your current model is ŷ = 0.1x + 5. You look at three data points:

| Power (x) | Actual Speed (y) | Prediction (ŷ = 0.1x + 5) | Error (ŷ - y) |
|---|---|---|---|
| 200 | 25 | 25 | 0 |
| 250 | 30 | 30 | 0 |
| 300 | 30 | 35 | +5 |

**Q1:** Does the Weight (w) need to be larger or smaller than 0.1 to fit this data better?

Since the model predicted 35 km/h but you were only doing 30 km/h, the model is "over-shooting." To bring those predictions down, the Weight w needs to be **smaller** than 0.1.

**Q2:** Why does squaring the error (turning -5 into 25) help the computer more than using the raw error?

- **The "Sign" Problem:** If one error is +5 and another is -5, the average error is 0. The computer would think it's a perfect model! Squaring makes them both 25.
- **The "Penalty" Problem:** Squaring "punishes" the model for being way off. Being 1 km/h off is okay (1² = 1), but being 5 km/h off is terrible (5² = 25). This forces the model to prioritize fixing the biggest mistakes first.

---

## The Calculus of Gradient Descent

To train the model, we calculate the partial derivative for both w and b at the same time. This allows the line to shift up/down and tilt left/right **simultaneously**.

### The Setup: The Cost Function (J)

Recall our MSE for m data points:

```
          1    ᵐ
J(w,b) = --- * Σ  (ŷᵢ - yᵢ)²
          2m  ⁱ⁼¹
```

Where ŷᵢ = wxᵢ + b.

### Partial Derivative for the Weight (w)

We want to know how the total error changes if we **tilt** the line. Using the Chain Rule:

- **Outside:** The 2 comes down and cancels the 2 in the denominator.
- **Inside:** The derivative of (wx + b - y) with respect to w is just x.

```
∂J     1   ᵐ
--- = --- * Σ  (ŷᵢ - yᵢ) * xᵢ
∂w     m  ⁱ⁼¹
```

**The Intuition:** The update for w is scaled by the input x. If you have a high-power ride (x is large) and the model is wrong, that specific ride will pull the "slope" of the line much harder than a low-power ride.

### Partial Derivative for the Bias (b)

This tells us how the error changes if we **shift** the whole line up or down.

- **Outside:** Same as above (the 2 cancels).
- **Inside:** The derivative of (wx + b - y) with respect to b is just 1.

```
∂J     1   ᵐ
--- = --- * Σ  (ŷᵢ - yᵢ)
∂b     m  ⁱ⁼¹
```

### The "Aha!" Moment

Look closely at the formulas. They are just the **Average Error** multiplied by the input.

- If predictions are mostly **too high** (error is positive), the gradient is positive.
- The formula **subtracts** that positive number from w and b, effectively **lowering** them.

For a **single data point** (m = 1), the summation disappears and the formulas simplify to:

> ∂J/∂w = (ŷ - y) * x
>
> ∂J/∂b = (ŷ - y)

This is exactly what we compute in the practice problems below.

### The Simultaneous Update (The Gradient Descent Step)

The computer calculates both gradients and then updates the weights:

```
            ∂J
w = w - α * ---
            ∂w

            ∂J
b = b - α * ---
            ∂b
```

---

## The Learning Rate (α)

The **Learning Rate** determines the size of the step you take in the direction the gradient points. Think of it as your "cadence" for learning.

### α Is Too Small ("Tiny Steps")

You are like a hiker taking 1-inch steps down a massive mountain.

- **The Good:** You are almost guaranteed to reach the bottom eventually.
- **The Bad:** It might take 10 million iterations (and a lot of compute) to get there.

### α Is Too Large ("Giant Leaps")

You are like a hiker trying to jump 5 miles at a time.

- **The Result:** You might be at the top of the left side of the bowl, calculate the "downhill" direction, and jump so far that you land even **higher** on the right side.
- **The Technical Term:** This is called **Divergence**. The model's error gets worse over time until the numbers "explode."

![High Learning Rate Causing Oscillation in the 3D Bowl](images/learning_bowl.png)

### The "Goldilocks" Approach: Learning Rate Decay

The "perfect" learning rate actually changes:

- Start with **large steps** to cover ground quickly when you're far from the goal.
- As the gradient gets smaller (the ground gets flatter), **shrink the steps** so you don't overshoot the bottom.

### Quick Conceptual Check

Imagine you are training your cycling model:

- In Iteration 1, your Gradient is **+120**.
- You use a Learning Rate of **0.1**.
- Your "Step" is: 0.1 * 120 = 12. So you subtract 12 from your weight.

**Scenario:** If you check the Gradient in Iteration 2 and it is now **+150** (the error got bigger!), what does that tell you about your Learning Rate of 0.1?

The learning rate is **too high**. We overshot the bottom. If your error (gradient) jumped from 120 to 150, you didn't just walk down the hill — you sprinted so fast you flew across the valley and landed higher up on the opposite side. This is the "Divergence" trap.

---

## Initialization: Where Do We Start?

Before the model can "wander" down the landscape, it needs a starting coordinate (w, b). In Machine Learning, we call this **Initialization**.

### A. The "Safe" Start (Zero Initialization)

For simple Linear Regression, you can start w = 0 and b = 0.

- Prediction: 0(200) + 0 = 0.
- Error: 0 - 18 = -18.
- The model sees it's too low and starts "climbing" out.

### B. The "Standard" Start (Small Random Numbers)

In professional ML, we use **Xavier** or **He Initialization** — pick a random number between -0.01 and 0.01. This keeps initial predictions small so the gradients don't explode on the very first step. For complex Neural Networks, if every weight starts at 0, every neuron learns the exact same thing (**Symmetry**). Small random numbers break that symmetry.

### C. The "Heuristic" Start (Domain Knowledge)

Because you are a cyclist, you might "cheat": "I know 200W usually results in about 30 km/h." So 30 = w(200), giving w = 0.15. You could initialize w = 0.15 to give the model a head start.

### Contour Plots

You'll often see the 3D bowl flattened into a **Contour Plot** (like a topographical map).

- **Circles:** Each circle represents a specific "altitude" of Cost (J).
- **The Center:** The smallest circle in the middle is the Global Minimum.
- **The Path:** Gradient Descent looks like a series of connected dots zig-zagging toward the center.

---

## Feature Scaling

If your features have different "units" or "magnitudes," you **must** scale them.

**Scenario A (No Scaling):** You are predicting house prices using only "Square Footage." Since you only have one feature, Gradient Descent will eventually find the bottom, even if the numbers are huge.

**Scenario B (Must Scale):** You are predicting speed using Power (0–1000W) and Gradient (0–0.15). Because 1000 is so much bigger than 0.15, a tiny nudge in the "Power Weight" (w₁) changes the prediction by 1000, while a nudge in the "Gradient Weight" (w₂) changes it by almost nothing. This makes your 3D cost bowl look like a long, skinny canyon. Your "hiker" (Gradient Descent) will bounce off the walls of the canyon for eternity instead of walking down the center.

### Scaling Methods

#### 1. Min-Max Scaling (Normalization)

Squishes everything between 0 and 1.

```
           x - min(x)
x_scaled = -----------
           max(x) - min(x)
```

#### 2. Standardization (Z-Score)

Centers the data around 0 with a standard deviation of 1.

```
           x - mean(x)
x_scaled = -----------
             std(x)
```

**Intuition:** A value of 1.0 means "This ride was 1 standard deviation stronger than my average."

#### 3. Mean Normalization

Similar to Min-Max, but centers the data so the average is 0.

```
           x - mean(x)
x_scaled = ---------------
           max(x) - min(x)
```

### Which Scaling Method to Use

| Method | Output Range | Handles Outliers? | Best For |
|---|---|---|---|
| **Min-Max** | [0, 1] | No — one outlier squishes everything | Data with known boundaries (e.g., percentages) |
| **Z-Score** | Centered at 0, no fixed range | Yes — outliers stay outliers | Most general ML tasks |
| **Mean Normalization** | Centered at 0, bounded | Partially | When you want both centering and bounding |

---

## Practice Problems

### Problem 1: The Single-Step Update

You are training a model to predict your speed (ŷ) based on your Power (x).

- Current Weight (w): 0.10
- Current Bias (b): 0
- Learning Rate (α): 0.01

**The Data Point:** Power (x) = 200W, Actual Speed (y) = 18 km/h.

Since we have a single data point (m = 1), the gradient formulas simplify to: ∂J/∂w = (ŷ - y) * x.

**Solution:**

**1. Calculate the Prediction (ŷ):**

> ŷ = wx + b = 0.10 * 200 + 0 = **20 km/h**

**2. Calculate the Error:**

> ŷ - y = 20 - 18 = **+2 km/h**

**3. Calculate the Gradient for w:**

> ∂J/∂w = error * x = 2 * 200 = **400**

**4. Update the Weight:**

> w_new = w_old - α * gradient = 0.10 - (0.01 * 400) = 0.10 - 4.0 = **-3.9**

The weight exploded to -3.9! This is what happens when features aren't scaled — the gradient (400) is enormous because x = 200 amplifies the error.

---

### Problem 2: Scaling the Features

In real ML, we would divide that 200W by 1000 so it becomes 0.2. Try the "Mini-Update" instead:

- w: 0.10
- x: 0.2 (Scaled Power)
- y: 0.18 (Scaled Speed)
- α: 0.1

**Solution:**

**1.** ŷ = 0.10 * 0.2 = **0.02**

**2.** Error = 0.02 - 0.18 = **-0.16**

**3.** Gradient = -0.16 * 0.2 = **-0.032**

**4.** w_new = 0.10 - 0.1 * (-0.032) = 0.10 + 0.0032 = **0.1032**

A smooth, sensible update — no explosion.

---

### Problem 3: Multi-Step Gradient Descent

Let's watch the model learn over 5 iterations using the scaled data from Problem 2. We use a single data point (x = 0.2, y = 0.18) with α = 0.1 and b = 0.

| Iteration | w | ŷ = w * 0.2 | Error (ŷ - y) | Gradient (err * x) | w_new = w - α * grad |
|---|---|---|---|---|---|
| 1 | 0.1000 | 0.0200 | -0.1600 | -0.0320 | 0.1032 |
| 2 | 0.1032 | 0.0206 | -0.1594 | -0.0319 | 0.1064 |
| 3 | 0.1064 | 0.0213 | -0.1587 | -0.0317 | 0.1096 |
| 4 | 0.1096 | 0.0219 | -0.1581 | -0.0316 | 0.1127 |
| 5 | 0.1127 | 0.0225 | -0.1575 | -0.0315 | 0.1159 |

**What to notice:**

- The **error shrinks** slightly each step (-0.1600 → -0.1575). The model is learning.
- The **gradient shrinks** too — the slope is getting flatter as we approach the bottom.
- The weight is **slowly increasing** toward the true value (y/x = 0.18/0.2 = 0.9). With one data point it would eventually converge there.
- After just 5 steps, w moved from 0.10 to 0.1159. With hundreds more iterations, it would reach 0.9.

---

### Problem 4: Final Prediction

You've trained your model on your Trek Emonda data and reached the bottom of the bowl. Final parameters:

- w = 0.08
- b = 2.0

**Question:** If you hold a steady 250 Watts, what is your model's predicted speed?

**Solution:**

> ŷ = wx + b = 0.08 * 250 + 2.0 = 20.0 + 2.0 = **22 km/h**

---

### Thought Problem: Scaling vs. Shrinking α

If we have unscaled power (x = 200) and want to prevent the "-3.9" explosion, we have two choices:

1. **Scale x:** Turn 200 into 0.2.
2. **Scale α:** Use a tiny learning rate like 0.00001.

Which is easier to manage with 50 different features?

**Answer:** Scaling the features. If one feature (Power) needs α = 0.00001 to stay stable, and another feature (Gradient %) needs α = 0.1 to actually move, you're stuck — you can only pick **one** α for the whole model. The small α starves one weight; the large α explodes another. By scaling all features to a range like [0, 1] or [-1, 1], you "round out" the cost bowl, and a single sensible α (like 0.01) works for everything.

---

## The Big Picture Checklist

You now understand the entire **Supervised Learning** cycle for Linear Regression:

| Step | What Happens |
|---|---|
| **1. The Model** | ƒ(x) = wx + b |
| **2. The Goal (Cost)** | Minimize J(w, b) using Mean Squared Error |
| **3. The Engine (Calculus)** | Calculate ∂J/∂w and ∂J/∂b to find the slope |
| **4. The Step (Optimization)** | Update w and b using the Learning Rate (α) |
| **5. The Finish** | Repeat until the gradient is nearly zero (the ground is flat) |

## Summary

| Concept | Perspective |
|---|---|
| **The Input** | The model's weights and biases (w, b) |
| **The Output** | A "Penalty Score" (J) |
| **The Shape** | A 3D Bowl (Paraboloid) created by squaring the errors |
| **The Goal** | Find the single coordinate (w, b) at the very bottom of the bowl |
| **The Tool** | Gradient Descent — feeling the slope of the bowl to find the floor |

---

## Further Reading

- [Stanford CS229: Linear Regression Lecture Notes](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf)
- [MIT OpenCourseWare: Lecture 5 — Linear Regression](https://ocw.mit.edu/)
- [Georgia Tech CS 7641: Machine Learning (OMSCS)](https://omscs.gatech.edu/cs-7641-machine-learning)
