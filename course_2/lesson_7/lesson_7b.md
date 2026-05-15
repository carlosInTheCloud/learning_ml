# Course 2 — Lesson 7b — Random Forest & XGBoost in Python

You never deploy a model with default settings. In practice you build a **pipeline** that forces the computer to test many hyperparameter combinations and pick a strong configuration for **your** data.

This note uses a **simulated** dataset of **10,000** historical cycling-style rows (classification) so the code runs anywhere without a CSV. The ideas transfer directly to real ride tables.

---

## Part 1: The parallel factory (Random Forest)

When you tune a Random Forest, you are really tuning **bootstrapping** (how each tree sees the data) and **tree geometry** (depth, splits, how many features each split may consider). Because the trees are **independent**, you can use all CPU cores to train them in parallel. In scikit-learn, `n_jobs=-1` does that.

```python
import time

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# 1. Simulate 10,000 rides with 10 sensors (features)
X_raw, y_raw = make_classification(n_samples=10000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# 2. Base estimator — n_jobs=-1 uses all available CPU cores for tree training
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# 3. Grid search space
rf_param_grid = {
    "n_estimators": [100, 300, 500],  # Board size: how many trees?
    "max_depth": [None, 10, 20],  # Pruning — None means no depth limit here
    "max_features": ["sqrt", "log2"],  # Feature subsampling per split
    "min_samples_split": [2, 5, 10],  # Pruning — minimum rows to split a node
}

# 4. Run the tuner
print("Initiating Random Forest Grid Search...")
start_time = time.time()

# cv=5 → 5-fold cross-validation.
# Cartesian product: 3 × 3 × 2 × 3 = 54 configs × 5 folds = 270 fits.
rf_tuner = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
rf_tuner.fit(X_train, y_train)

print(f"Search completed in {time.time() - start_time:.2f} seconds.")
print(f"Best RF parameters: {rf_tuner.best_params_}")
print(f"Best validation accuracy: {rf_tuner.best_score_ * 100:.2f}%")

best_rf_model = rf_tuner.best_estimator_
```

### Why does `n_estimators` list three values: 100, 300, 500?

Split the idea in two: what the number **means**, and why you pass a **list** into `GridSearchCV`.

**What `n_estimators` means.** Each “estimator” here is one decision tree. So `n_estimators` is how many trees sit on the **committee**. With 100, the final prediction aggregates 100 votes; with 500, 500 votes.

**Why three entries.** You hand the dictionary to **`GridSearchCV`**, which tries **combinations** of hyperparameters. You are telling the computer: “I do not know the best committee size for this dataset — try **100**, **300**, and **500** and report what wins.” The search builds and scores forests at those sizes (along with every combination of the other grid keys), which is why grid search can take noticeable time.

**Diminishing returns.** Extra trees **do not** cause the same kind of overfitting spiral as an unconstrained depth in a single tree, but they **do** cost compute and latency. Often you gain a lot going from tens to hundreds of trees, less going from 100 to 300, and almost nothing going from 500 to 5,000 while bills and prediction time balloon. The grid `[100, 300, 500]` is a way to probe the **elbow**: the point where more trees barely help, so you can deploy something **lean**.

### After the search, should I zoom in around the best tree count (200, 300, 400)?

That “coarse then fine” instinct is central to ML engineering — but for **`n_estimators` in Random Forest**, the usual answer is **no**.

**Plateau, not peak.** For forest size, the validation curve often **flattens**: once you are past the elbow, 287 versus 300 versus 314 trees may differ by **tiny** fractions of accuracy. Spending compute on a tight grid around 300 buys almost nothing except a prettier number.

**Pick a round number and move on.** Find the rough neighborhood from a **coarse** grid, commit to something like **300**, and redirect effort elsewhere.

### When coarse-to-fine search *does* pay off

For hyperparameters that **shape the math** of the fit — where you can genuinely over- or undershoot — the zoom-in strategy is appropriate. Examples:

- **`learning_rate`** in boosting (shown in Part 2)
- **`C`**, **`gamma`** (SVMs)
- **`max_depth`** / min-leaf constraints when they strongly control complexity

Typical pattern:

1. **Coarse:** wide steps, sometimes logarithmic (e.g. `learning_rate`: `[0.001, 0.01, 0.1]`).
2. **Fine:** once you see which region wins, search narrowly (e.g. around `0.1`).

**Rule of thumb:** **volume-style** knobs (mostly “how many trees”) → coarse grid, find the plateau, lock a round value.**Math-style** knobs → coarse then fine.

### Why can `max_features` have two entries while `n_estimators` has three?

`GridSearchCV` does **not** run “column 1 of every list together, then column 2…”. It takes the **Cartesian product**: every choice from one key is paired with every choice from the others.

With the RF grid above:

$$
3 \times 3 \times 2 \times 3 = 54
$$

unique hyperparameter tuples, each evaluated with **`cv=5`**, hence **\(54 \times 5 = 270\)** fits. List lengths **do not** need to match.

**What `'sqrt'` and `'log2'` mean.** These are built-in rules for **how many features** a split may consider when you have \(M\) columns total (`max_features` in the sense of blindfold strength):

- **`sqrt`** — about \(\sqrt{M}\) features at each split (very common default scale).
- **`log2`** — about \(\log_2(M)\) features — a **stronger** blindfold (fewer features per split).

Example: \(M = 100\) → \(\sqrt{100} = 10\) vs \(\log_2(100) \approx 6.64\).

If **`max_features=None`**, trees can consider **all** features at every split. Then dominant signals (say, power on every ride) can make forests of **highly correlated** trees, which defeats the diversification you wanted. The grid compares **`sqrt`** vs **`log2`** across all other combinations to see how aggressive the blindfold should be **for your data**.

---

## Part 2: The sequential chain (XGBoost)

Boosting trains trees **one after another**; each stage depends on residuals or gradients from the previous one. **`GridSearchCV`** is **model-agnostic**: it performs the **same Cartesian product** on whichever parameter dictionary you give it.

```python
import time

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

# Same synthetic setup as Part 1 (swap for real X, y in production)
X_raw, y_raw = make_classification(n_samples=10000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

xgb_base = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

xgb_param_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 6, 10],
    "subsample": [0.8, 1.0],
}

print("Initiating XGBoost Grid Search...")
start_time = time.time()

# 3 × 3 × 3 × 2 = 54 configs × 5 folds = 270 fits — same counting story as RF
xgb_tuner = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
xgb_tuner.fit(X_train, y_train)

print(f"Search completed in {time.time() - start_time:.2f} seconds.")
print(f"Best XGB parameters: {xgb_tuner.best_params_}")
print(f"Best validation accuracy: {xgb_tuner.best_score_ * 100:.2f}%")

best_xgb_model = xgb_tuner.best_estimator_
```

### Does `GridSearchCV` take the Cartesian product for XGBoost too?

Yes. Example with the grid above:

$$
3 \times 3 \times 3 \times 2 = 54
$$

configs, times **`cv=5`** → **270** boosted models trained **at the tuner level**.

### Why RF and XGBoost can feel different on the clock

Fit counts **look** symmetric (both 270 in this toy setup), but **architecture** differs.

- **Random Forest:** trees in one forest fit are **parallel** internally (`n_jobs` on the forest); stages are pleasantly parallel overall.
- **XGBoost:** **within one training run**, boosting is **sequential** — tree 2 waits on tree 1. Cross-validation still parallelizes **folds/config jobs** according to **`GridSearchCV(n_jobs=...)`**, but each inner boost stays ordered.

So RF grid search often **finishes sooner** than XGBoost at the **same** outer fit count — another reason to keep grids **sparse** or move to randomized search when the product explodes.

### Why `[0.01, 0.1, 0.2]` for `learning_rate` instead of `[0.01, 0.1, 1]`?

Pure coarse exploration sometimes uses logarithmic ladders like **0.01 → 0.1 → 1.0**.

For gradient boosting in practice:

1. **\(\nu = 1\) is “full step” boosting** — \(F_m(x) = F_{m-1}(x) + \nu \times \text{(tree)}\). At \(\nu = 1\) you apply the full correction with **no** shrinkage; updates can **oscillate** and overshoot (“ping-pong” around good solutions).

2. **Empirical ceiling.** On many problems, strong settings for XGBoost-style boosters rarely need learning rates **above ~0.2–0.3**.

3. **Compute budget.** Every bad value in the grid multiplies the Cartesian product. If **1.0** is a known dead zone for stable boosting, including it **wastes** a slice of the search on configurations you can predict will behave poorly.

So: **textbook** coarse grids may span orders of magnitude; **production** grids often **trim** known-bad regions for that algorithm and spend degrees of freedom where the response surface is actually informative — then **refine** around the winner (coarse-to-fine for **math** parameters).

---

## Closing

**Random Forest:** parallel tree factory, `n_jobs=-1`, grid over committee size, depth, split rules, and **`max_features`** blindfolds — remember the **Cartesian product** and **plateau** behavior for tree count.

**XGBoost:** same **`GridSearchCV`** machinery, but **sequential** boosting inside each fit; tune **`learning_rate`** with the same respect you’d give other **sharp** knobs, and keep grids small enough that the search finishes on your hardware.
