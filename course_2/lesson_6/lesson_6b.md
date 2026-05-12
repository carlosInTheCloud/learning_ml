# Course 2 — Lesson 6b — SVMs in scikit-learn (three scenarios)

This companion note writes **concrete scikit-learn code** for the three architectural paths from Lesson 6.

Imagine a dataset of **cycling rides** where you predict **Bonk** vs **No Bonk** from two features only: **Power** and **Cadence**. Labels are binary (`1` = Bonk, `0` = No Bonk). Features live in `X` with shape `(n_samples, 2)`; labels in `y`.

---

## 1. Dual path — small, complex data (`SVC` + RBF)

If you have on the order of **~10,000 rides** and the Bonk class forms a **nonlinear** pattern (e.g. a blob surrounded by a ring of non-bonks), you want the **dual + kernel** path: **`SVC`** with an **RBF** kernel so the decision boundary can bend in the original 2D feature plot.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the raw historical data into a Pandas DataFrame
df = pd.read_csv("historical_cycling_rides.csv")

# 2. Separate the Features (X) from the Target (y)
# Drop the "Bonk" column to create the feature matrix
X_raw = df.drop(columns=["Bonk"]) 
# Isolate the "Bonk" column to create the target vector
y_raw = df["Bonk"]                

# 3. Perform the Train/Test Split
# test_size=0.2 means 20% goes to the Test Set, 80% to the Train Set.
# random_state=42 ensures the random shuffle is exactly the same every time you run it.
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# --- Now the data is ready for the SVM ---
from sklearn.svm import SVC

# Instantiate and Train
dual_svm = SVC(kernel='rbf', C=1.0)
dual_svm.fit(X_train, y_train)

# Predict on the hidden 20%
predictions = dual_svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
```

**Architect’s note — `gamma`:** Larger **`gamma`** → **narrower** RBF bumps → boundary can hug individual points (**higher variance / overfitting**). Smaller **`gamma`** → **wider**, smoother influence → **smoother** boundary (**higher bias / underfitting**). Tune **`C`** and **`gamma`** together on validation data.

---

## 2. Primal path — massive, linear data (`LinearSVC`)

If you have **millions** of rides and a **linearly separable** (or nearly linear) pattern in Power–Cadence space, use the **primal** path: optimize **`w`** and **`b`** directly with hinge loss — no dense **`n × n`** Gram matrix.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import time

# ==========================================
# 1. DATA INGESTION (Simulating 5 Million Rows)
# ==========================================
print("1. Generating 5 million rows of historical data...")
# In production, this would be: df = pd.read_csv("snowflake_export.csv")
X_raw, y_raw = make_classification(
    n_samples=5000000, 
    n_features=2,       # E.g., Power and Cadence
    n_informative=2, 
    n_redundant=0, 
    random_state=42
)

# ==========================================
# 2. THE TRAIN / TEST SPLIT
# ==========================================
print("2. Slicing data into Training (80%) and Testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# ==========================================
# 3. ARCHITECTURE SELECTION: THE PRIMAL PATH
# ==========================================
# loss='hinge': Use the strict margin loss (not probability).
# dual=False: Force the C++ liblinear solver to optimize 'w' directly.
# C=1.0: Our Box Constraint / Soft Margin strictness.
primal_svm = LinearSVC(loss='hinge', dual=False, C=1.0)

# ==========================================
# 4. TRAINING (Phase 1)
# ==========================================
print("3. Training the Primal SVM...")
start_time = time.time()

# This is where the mathematical magic of the Primal path happens.
# It ignores the alpha shadows and optimizes the weights via Gradient Descent.
primal_svm.fit(X_train, y_train)

end_time = time.time()
print(f"   Training completed in {end_time - start_time:.2f} seconds!")

# ==========================================
# 5. INFERENCE & VALIDATION (Phase 2)
# ==========================================
print("4. Predicting on the hidden test set...")
predictions = primal_svm.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")
```

**Note:** `LinearSVC` is backed by **liblinear** (coordinate-style optimization), not a naive “one giant gradient step” story — but operationally it is still the **fast linear primal** tool for huge **`n`**.

---

## 3. The hack — massive **and** complex (`RBFSampler` + `LinearSVC` pipeline)

If you have **millions** of rows **and** a **nonlinear** boundary, avoid materializing a dual **Gram** matrix. **Approximate** RBF geometry with **Random Fourier Features** (**Bochner / RFF**), then run **`LinearSVC`** on the expanded features.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import time

# ==========================================
# 1. DATA INGESTION (Simulating 5 Million Rows)
# ==========================================
print("1. Generating 5 million rows of complex historical data...")
X_raw, y_raw = make_classification(
    n_samples=5000000, 
    n_features=2,       # Raw data starts in 2D (Power, Cadence)
    n_informative=2, 
    n_redundant=0, 
    random_state=42
)

# ==========================================
# 2. THE TRAIN / TEST SPLIT
# ==========================================
print("2. Slicing data into Training (80%) and Testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# ==========================================
# 3. ARCHITECTURE SELECTION: THE PIPELINE HACK
# ==========================================
# Step A: The Mapper. Converts 2 features into 5,000 Fourier features.
# Step B: The Solver. Uses the blazing-fast Primal linear optimizer.
advanced_svm = Pipeline([
    ("fourier_mapper", RBFSampler(gamma=1.0, n_components=5000, random_state=42)),
    ("primal_solver", LinearSVC(loss='hinge', dual=False, C=1.0))
])

# ==========================================
# 4. TRAINING (Phase 1)
# ==========================================
print("3. Training the Pipeline (Mapping to 5,000D -> Primal Optimization)...")
start_time = time.time()

# This streams the data through the mapper and directly into the solver.
advanced_svm.fit(X_train, y_train)

end_time = time.time()
print(f"   Training completed in {end_time - start_time:.2f} seconds!")

# ==========================================
# 5. INFERENCE & VALIDATION (Phase 2)
# ==========================================
print("4. Predicting on the hidden test set...")
# The test data is also automatically mapped to 5,000D before prediction!
predictions = advanced_svm.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")
```

**Reminder:** **`n_components`** and **`gamma`** control approximation quality vs. cost; **`C`** still controls margin strictness on the **mapped** hinge problem. Cross-validate.

---

## Quick map

| Situation | API | Why |
|---|---|---|
| Small **`n`**, nonlinear boundary | `SVC(kernel="rbf", …)` | Dual + kernel; OK while Gram-style work fits in memory. |
| Huge **`n`**, linear boundary | `LinearSVC(dual=False, …)` | Primal on few features; scales to large **`n`**. |
| Huge **`n`**, nonlinear boundary | `Pipeline(RBFSampler, LinearSVC)` | Approximate kernel features + fast linear primal. |
