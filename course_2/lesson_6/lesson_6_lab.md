# Course 2 — Lab 6 — SVM primal vs dual (circles dataset)

## Lab assignment

You have **nested** classes (Bonk vs No Bonk in feature space)—the canonical **non-linear** pattern where a **hyperplane in the original space** fails. Beneath the setup block below, write **Python** that completes five **architectural** comparisons.

Your answers should mirror the vocabulary from **Lesson 6** (Primal vs Dual, soft margin `$C$,` support vectors).

---

### Setup block (paste this above your solutions)

Run `lesson_6_lab_training_data.py` first if you want `lesson_6_lab_data.csv` on disk; the following **numpy + scikit-learn** block reproduces the same split without reading the CSV (same RNG seeds and `make_circles` arguments):

```python
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# 1_000 rides, inner vs outer circle (factor & noise match the data generator lab script)
X_raw, y_raw = make_circles(
    n_samples=1000, factor=0.3, noise=0.05, random_state=42
)

# 80 % train / 20 % test → 800 training rows, 200 test rows
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)
```

---

### Task 1: Prove the primal limitation

- Instantiate **`LinearSVC`** (Primal path). Train it on `X_train, y_train`.
- Print the **accuracy** on `X_test, y_test` (e.g. `accuracy_score`).
- **Hypothesis:** Because the model only induces a **linear** boundary in the original 2-D feature space, it should **fail** to separate the circles (accuracy near chance or otherwise poor—not a tight ring).

---

### Task 2: Prove the dual + kernel path

- Instantiate **`SVC`** with **`kernel="rbf"`** (dual + kernel path). Train on the **same** `X_train, y_train`.
- Print **test accuracy**.
- **Hypothesis:** With the RBF kernel, the classifier can exploit the **RKHS** induced by `rbf`; it should **nearly** solve the circles problem (**near-perfect** accuracy on this synthetic setup).

---

### Task 3: Verify sparsity

- Using the trained **RBF `SVC` from Task 2**, print the **exact number of support vectors** (how many training points ended up non-trivially in the dual solution).
- **Hint:** Inspect the estimator attribute that holds **support vectors** or the per-class support counts—e.g. the shape of the support-vector matrix, or **`n_support_`**. Interpret: relative to **800** training rows, how many rows were effectively “inactive” (`800 - n_sv`)?

---

### Task 4: Test the box constraint (`$C$`)

- Train a **third** model: **`SVC(kernel="rbf", C=0.001)`**—a **very relaxed** soft margin (`$C \ll 1$`).
- Report **test accuracy** and **number of support vectors**.
- **Hypothesis:** Because the penalty for margin violations with **`C=0.001`** is so small, the algorithm can behave **much more permissively** toward training errors; **accuracy should drop**, and **the number of support vectors should change sharply** versus Task 2 (report counts and sanity-check).

---

### Task 5: Bounded support vectors (soft-margin “violators”)

- Reuse the **fitted `SVC` from Task 4** (same object you used to report SV count and accuracy).
- For **binary** classification, **`dual_coef_`** stores **signed** dual weights **\(\alpha_i y_i\)** for each support vector; **`np.abs(...)`** recovers **\(\alpha_i\)** (up to floating-point noise).
- Count how many support vectors have **\(\alpha_i \approx C\)** (the **box ceiling** from the soft-margin dual). Those are the **bounded** support vectors in KKT terms—often described as points that **incur slack** / sit **inside or beyond the ideal margin** relative to the separating construction.
- Also report how many support vectors remain **free** (\(0 < \alpha_i < C\)).
- **Hint:** compare to **`model.C`** with **`np.isclose(..., rtol=0, atol=1e-6)`** rather than testing exact equality.

---

When you finish, optionally compare outcomes in one sentence per task versus the hypotheses above.
