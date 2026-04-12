# Course 2 - Lesson 2 - Lab 1: Logistic Regression (Binary Classification) Lab

For this lab, we are going to write the core components of Logistic Regression entirely from scratch using NumPy. We will not use pre-built libraries like scikit-learn for the training loop, because you need to prove you can write the vectorized calculus yourself.

For the derivation of Binary Cross-Entropy and the gradient formulas, see [Lesson 2: Logistic Regression](lesson_2_logistic_regression.md).

---

## The Scenario: The "Bonk" Predictor

We use a realistic synthetic dataset simulating 100 rides on your Trek Emonda. We are looking at two input features:

- **x₁:** Normalized Power (Watts)
- **x₂:** Ride Duration (Hours)

The target variable y is whether you experienced sudden glycogen depletion (bonked) on that ride (y=1) or finished strong (y=0).

---

## Lab Instructions

Below is the Python starter code. The data generation, feature scaling, and plotting logic are provided. Your task is to fill in the mathematical engine in the three **TODO** sections:

1. The **Sigmoid Function**
2. The **Binary Cross-Entropy Cost Function**
3. The **Gradient Descent Update Step**

Copy this into your IDE, complete the TODO blocks, and run the script.

---

## Starter Code

```python
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA PREPARATION (Provided)
# ==========================================
np.random.seed(42)
m = 100  # Number of rides

# Generate realistic cycling data
# Feature 1: Power (150W to 300W)
# Feature 2: Duration (1 to 5 hours)
X_power = np.random.uniform(150, 300, m)
X_duration = np.random.uniform(1, 5, m)
X = np.column_stack((X_power, X_duration))

# Define a hidden "true" boundary:
# Bonk is likely if (Power * 0.05) + (Duration * 1.5) > 16
true_z = (X[:, 0] * 0.05) + (X[:, 1] * 1.5) - 16
probabilities = 1 / (1 + np.exp(-true_z))
y = (np.random.rand(m) < probabilities).astype(int)

# Feature Scaling (Z-score normalization)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# ==========================================
# 2. THE MATH ENGINE (Your Turn)
# ==========================================

def sigmoid(z):
    """
    TODO 1: Implement the Sigmoid function.
    z can be a scalar or a numpy array.
    """
    # YOUR CODE HERE
    pass

def compute_cost(X, y, w, b):
    """
    TODO 2: Implement Binary Cross-Entropy (Log Loss).
    Use a tiny epsilon to prevent np.log(0) errors.
    """
    m = X.shape[0]
    epsilon = 1e-10

    # YOUR CODE HERE
    # 1. Calculate z
    # 2. Calculate predictions (y_hat) using sigmoid
    # 3. Calculate the cost

    return cost

def compute_gradient(X, y, w, b):
    """
    TODO 3: Implement the gradient calculations.
    Returns dj_dw (vector) and dj_db (scalar).
    """
    m = X.shape[0]

    # YOUR CODE HERE
    # 1. Calculate predictions (y_hat)
    # 2. Calculate the error (y_hat - y)
    # 3. Calculate dj_dw (using np.dot for vectorization)
    # 4. Calculate dj_db

    return dj_dw, dj_db

# ==========================================
# 3. TRAINING LOOP (Provided)
# ==========================================

w = np.zeros(X_scaled.shape[1])
b = 0.0
alpha = 0.1
iterations = 1000
cost_history = []

print("Starting Training...")
for i in range(iterations):
    dj_dw, dj_db = compute_gradient(X_scaled, y, w, b)

    w = w - alpha * dj_dw
    b = b - alpha * dj_db

    if i % 100 == 0:
        cost = compute_cost(X_scaled, y, w, b)
        cost_history.append(cost)
        print(f"Iteration {i:4d}: Cost = {cost:.4f}")

print("\nFinal Parameters:")
print(f"w = {w}")
print(f"b = {b:.4f}")

# ==========================================
# 4. ACCURACY & PREDICTION (Provided)
# ==========================================

# Classify all rides using 0.5 threshold
y_hat_final = sigmoid(np.dot(X_scaled, w) + b)
predictions = (y_hat_final >= 0.5).astype(int)
accuracy = np.mean(predictions == y) * 100
print(f"\nAccuracy: {int(accuracy)}% ({np.sum(predictions == y)}/{m} rides classified correctly)")

# Predict on a new ride: 260W for 3.5 hours
new_ride = np.array([260, 3.5])
new_ride_scaled = (new_ride - X_mean) / X_std
bonk_probability = sigmoid(np.dot(new_ride_scaled, w) + b)
print(f"\nNew Ride: 260W for 3.5 hours")
print(f"Bonk Probability: {bonk_probability:.1%}")
print(f"Prediction: {'BONK' if bonk_probability >= 0.5 else 'Finish Strong'}")

# ==========================================
# 5. VISUALIZATION (Provided)
# ==========================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Learning Curve
ax1.plot(range(0, iterations, 100), cost_history, marker='o', color='blue')
ax1.set_title("Learning Curve (Cost vs Iterations)")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Binary Cross-Entropy Cost")
ax1.grid(True)

# Plot 2: Decision Boundary
ax2.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1],
            color='green', label='Finished Strong (y=0)')
ax2.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1],
            color='red', marker='x', label='Bonked (y=1)')

# Decision boundary line: w0*x0 + w1*x1 + b = 0  =>  x1 = -(w0*x0 + b) / w1
x0_vals = np.array([np.min(X_scaled[:, 0]), np.max(X_scaled[:, 0])])
x1_vals = -(w[0] * x0_vals + b) / w[1]

ax2.plot(x0_vals, x1_vals, color='black', linestyle='--',
         linewidth=2, label='Decision Boundary')
ax2.set_title("Logistic Regression Decision Boundary")
ax2.set_xlabel("Normalized Power (Z-score)")
ax2.set_ylabel("Normalized Duration (Z-score)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_lab_results.png')
print("\nGenerated 'logistic_lab_results.png'. Open to view your model's performance!")
```

---

## Solution

<details>
<summary>Click to reveal the completed code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA PREPARATION
# ==========================================
np.random.seed(42)
m = 100

X_power = np.random.uniform(150, 300, m)
X_duration = np.random.uniform(1, 5, m)
X = np.column_stack((X_power, X_duration))

true_z = (X[:, 0] * 0.05) + (X[:, 1] * 1.5) - 16
probabilities = 1 / (1 + np.exp(-true_z))
y = (np.random.rand(m) < probabilities).astype(int)

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# ==========================================
# 2. THE MATH ENGINE
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    m = X.shape[0]
    epsilon = 1e-10

    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    cost = -1/m * np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
    return cost

def compute_gradient(X, y, w, b):
    m = X.shape[0]

    y_hat = sigmoid(np.dot(X, w) + b)
    error = y_hat - y
    dj_dw = (1/m) * np.dot(X.T, error)
    dj_db = (1/m) * np.sum(error)
    return dj_dw, dj_db

# ==========================================
# 3. TRAINING LOOP
# ==========================================

w = np.zeros(X_scaled.shape[1])
b = 0.0
alpha = 0.1
iterations = 1000
cost_history = []

print("Starting Training...")
for i in range(iterations):
    dj_dw, dj_db = compute_gradient(X_scaled, y, w, b)

    w = w - alpha * dj_dw
    b = b - alpha * dj_db

    if i % 100 == 0:
        cost = compute_cost(X_scaled, y, w, b)
        cost_history.append(cost)
        print(f"Iteration {i:4d}: Cost = {cost:.4f}")

print("\nFinal Parameters:")
print(f"w = {w}")
print(f"b = {b:.4f}")

# ==========================================
# 4. ACCURACY & PREDICTION
# ==========================================

y_hat_final = sigmoid(np.dot(X_scaled, w) + b)
predictions = (y_hat_final >= 0.5).astype(int)
accuracy = np.mean(predictions == y) * 100
print(f"\nAccuracy: {int(accuracy)}% ({np.sum(predictions == y)}/{m} rides classified correctly)")

new_ride = np.array([260, 3.5])
new_ride_scaled = (new_ride - X_mean) / X_std
bonk_probability = sigmoid(np.dot(new_ride_scaled, w) + b)
print(f"\nNew Ride: 260W for 3.5 hours")
print(f"Bonk Probability: {bonk_probability:.1%}")
print(f"Prediction: {'BONK' if bonk_probability >= 0.5 else 'Finish Strong'}")

# ==========================================
# 5. VISUALIZATION
# ==========================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(range(0, iterations, 100), cost_history, marker='o', color='blue')
ax1.set_title("Learning Curve (Cost vs Iterations)")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Binary Cross-Entropy Cost")
ax1.grid(True)

ax2.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1],
            color='green', label='Finished Strong (y=0)')
ax2.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1],
            color='red', marker='x', label='Bonked (y=1)')

x0_vals = np.array([np.min(X_scaled[:, 0]), np.max(X_scaled[:, 0])])
x1_vals = -(w[0] * x0_vals + b) / w[1]

ax2.plot(x0_vals, x1_vals, color='black', linestyle='--',
         linewidth=2, label='Decision Boundary')
ax2.set_title("Logistic Regression Decision Boundary")
ax2.set_xlabel("Normalized Power (Z-score)")
ax2.set_ylabel("Normalized Duration (Z-score)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_lab_results.png')
print("\nGenerated 'logistic_lab_results.png'. Open to view your model's performance!")
```

</details>

---

## Math-to-Code Mapping

| Math Formula | Python Code | What It Does |
|---|---|---|
| g(z) = 1 / (1 + e⁻ᶻ) | `1 / (1 + np.exp(-z))` | Squash any number into (0, 1) |
| z = **Xw** + b | `np.dot(X, w) + b` | Linear combination for all m rides at once |
| J = -(1/m) Σ[y ln(ŷ) + (1-y) ln(1-ŷ)] | `-1/m * np.sum(y * np.log(y_hat) + ...)` | Binary Cross-Entropy cost |
| ∂J/∂w = (1/m) **X**ᵀ(ŷ - y) | `(1/m) * np.dot(X.T, error)` | Gradient for all weights simultaneously |
| ∂J/∂b = (1/m) Σ(ŷ - y) | `(1/m) * np.sum(error)` | Gradient for the bias |

The key vectorization insight: `np.dot(X.T, error)` computes **all** weight gradients in a single matrix operation — no Python loop over features needed.

---

## What to Experiment With

The real learning comes from breaking things. Try these modifications:

1. **Explode the learning rate:** Change `alpha = 0.1` to `alpha = 10.0`. Watch the cost oscillate wildly or become `NaN`. This is the divergence you learned about in Lesson 1 — it applies to classification too.

2. **Remove the Z-score scaling:** Replace `X_scaled` with raw `X` in the training loop. Even with `alpha = 0.0001`, the gradients will be unstable because Power (150-300) dwarfs Duration (1-5). The cost bowl becomes a skinny canyon.

3. **Make bonking easier:** Change the true boundary from `- 16` to `- 12` in the data generation. More rides become bonks, and the decision boundary shifts. Does accuracy change?

4. **Starve the data:** Change `m = 100` to `m = 20`. With fewer rides, the decision boundary becomes less stable. Run it a few times (change the seed) and watch how the line wobbles.

5. **Try a different new ride:** Predict on a 180W, 4.5-hour ride (long but easy) vs. a 290W, 1.5-hour ride (short but hard). Which has a higher bonk probability? Does that match your cycling intuition?

---

## Key Implementation Details

| Detail | Why It Matters |
|---|---|
| **epsilon = 1e-10** | Prevents `np.log(0)` which returns `-inf` and crashes the cost calculation. We add a tiny number so ln(0) becomes ln(0.0000000001) ≈ -23 instead of -∞. |
| **Z-score scaling** | Without it, Power (150-300) dominates Duration (1-5), creating the "skinny canyon" problem from Lesson 1. |
| **np.dot(X, w) + b** | Vectorized: computes z for all 100 rides in one operation. A Python `for` loop would do the same thing ~100x slower. |
| **np.dot(X.T, error)** | The transpose trick: X is (100×2), error is (100,). X.T @ error gives a (2,) vector — one gradient per feature. |
