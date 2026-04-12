import numpy as np

# ==========================================
# 1. DATA PREPARATION (The "Noisy Sine Wave")
# ==========================================
np.random.seed(42)
m = 30
X = np.sort(np.random.uniform(0, 1, m))
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, m)

# Train/Test split — hold out the last 10 points
X_train, X_test = X[:20], X[20:]
y_train, y_test = y[:20], y[20:]

# To overfit, we expand X into a 10th-degree polynomial
def expand_features(X, degree):
    return np.column_stack([X**i for i in range(1, degree + 1)])

degree = 10
X_train_poly = expand_features(X_train, degree)
X_test_poly = expand_features(X_test, degree)

# Scale features using training stats only
X_mean = np.mean(X_train_poly, axis=0)
X_std = np.std(X_train_poly, axis=0)
X_train_scaled = (X_train_poly - X_mean) / X_std
X_test_scaled = (X_test_poly - X_mean) / X_std

# ==========================================
# 2. THE REGULARIZED ENGINE
# ==========================================

def compute_cost_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    error = np.dot(X, w) + b - y
    cost = (1 / (2 * m)) * np.sum(error**2) + (lambda_ / (2 * m)) * np.sum(w**2)
    return cost


def compute_gradient_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    error = np.dot(X, w) + b - y
    dj_dw = (1 / m) * np.dot(X.T, error) + (lambda_ / m) * w
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db


# ==========================================
# 3. TRAINING
# ==========================================

def train(X, y, lambda_val, alpha=0.1, iterations=10000, verbose=False):
    w = np.zeros(X.shape[1])
    b = 0.0
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient_reg(X, y, w, b, lambda_val)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if verbose and i % 2000 == 0:
            cost = compute_cost_reg(X, y, w, b, lambda_val)
            print(f"  Iteration {i:5d}: Cost = {cost:.6f}")
    return w, b


# ==========================================
# 4. LAMBDA SWEEP — Bias-Variance Tradeoff
# ==========================================

lambdas = [0, 0.01, 0.1, 1.0, 10.0]

print("=" * 72)
print(f"{'Lambda':<10} {'Train Cost':>12} {'Test Cost':>12} {'Max |w|':>10} {'Sum w²':>12}")
print("=" * 72)

best_lambda = None
best_test_cost = float('inf')

for lam in lambdas:
    w, b = train(X_train_scaled, y_train, lam)
    train_cost = compute_cost_reg(X_train_scaled, y_train, w, b, 0)
    test_cost = compute_cost_reg(X_test_scaled, y_test, w, b, 0)

    if test_cost < best_test_cost:
        best_test_cost = test_cost
        best_lambda = lam

    print(f"{lam:<10} {train_cost:>12.4f} {test_cost:>12.4f} {np.max(np.abs(w)):>10.2f} {np.sum(w**2):>12.2f}")

print("=" * 72)
print(f"\nBest lambda (lowest test cost): {best_lambda}")

# ==========================================
# 5. DETAILED COMPARISON: Lambda=0 vs Best
# ==========================================

print(f"\n{'=' * 55}")
print(f"Detailed comparison: Lambda=0 vs Lambda={best_lambda}")
print(f"{'=' * 55}")

print(f"\nTraining Lambda=0 (verbose):")
w_overfit, b_overfit = train(X_train_scaled, y_train, 0, verbose=True)

print(f"\nTraining Lambda={best_lambda} (verbose):")
w_reg, b_reg = train(X_train_scaled, y_train, best_lambda, verbose=True)

print(f"\nWeight magnitudes (all {degree} polynomial terms):")
print(f"  Lambda=0:            {np.array2string(w_overfit, precision=2, suppress_small=True)}")
print(f"  Lambda={best_lambda:<14} {np.array2string(w_reg, precision=2, suppress_small=True)}")

X_line = np.linspace(0, 1, 100)
X_line_poly = expand_features(X_line, degree)
X_line_scaled = (X_line_poly - X_mean) / X_std
y_overfit_line = np.dot(X_line_scaled, w_overfit) + b_overfit
y_reg_line = np.dot(X_line_scaled, w_reg) + b_reg

print(f"\nPrediction range across [0, 1]:")
print(f"  Lambda=0:            ({y_overfit_line.min():>6.2f}, {y_overfit_line.max():>5.2f})")
print(f"  Lambda={best_lambda:<14} ({y_reg_line.min():>6.2f}, {y_reg_line.max():>5.2f})")
print(f"  True sine wave:      ( -1.00,  1.00)")

print("\nKey observations:")
print("  - Lambda=0 has LOWER training cost (it memorized the noise).")
print("  - Lambda=0 has HIGHER test cost (it fails on new data — overfitting!).")
print("  - Lambda=0 has MASSIVE weights — the polynomial swings wildly between points.")
print(f"  - Lambda={best_lambda} has smaller, controlled weights — it learned the smooth trend.")
print(f"  - The prediction range for Lambda=0 likely exceeds [-1.5, 1.5];")
print(f"    Lambda={best_lambda} stays close to the true sine wave range [-1, 1].")
