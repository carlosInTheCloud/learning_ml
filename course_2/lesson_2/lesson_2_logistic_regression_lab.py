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
