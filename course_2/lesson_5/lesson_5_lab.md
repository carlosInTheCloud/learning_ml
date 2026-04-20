# Course 2 — Lab 5 — Decision tree induction from scratch

## Objective

You will implement a **greedy, recursive** decision-tree learner using **Python** and **NumPy** only. The code must:

- **Dynamically** evaluate candidate **numeric** splits on the training data.
- Compute **information gain** for each candidate.
- **Recursively** grow branches until nodes are **pure** (entropy **$= 0$**).

## Dataset

The lab uses **$m = 6$** rides so the tree must perform **at least one** meaningful depth beyond the root.

- **Features $X$:** column **0** = power (watts), column **1** = cadence (rpm).
- **Labels $y$:** **$1$** = Bonk, **$0$** = No Bonk.

```python
import numpy as np

# Shape: (6 examples, 2 features)
X = np.array([
    [190, 85],
    [210, 90],
    [250, 75],
    [180, 80],
    [205, 95],
    [230, 85],
])

# Shape: (6 examples,)
y = np.array([0, 0, 1, 0, 0, 1])
```

## Your tasks

Implement **four** functions that form the core of the learner:

1. **`calculate_entropy(y)`** — Shannon entropy (base 2) of a 1-D label array.
2. **`information_gain(y, y_left, y_right)`** — information gain from sending parent labels `y` into left/right child label sets `y_left` and `y_right` (use counts to weight child entropies).
3. **`find_best_split(X, y)`** — for **each** feature and **each** sensible threshold derived from **unique** values in that column, compute information gain; return the split (feature index, threshold, and gain) that **maximizes** gain.
4. **`build_tree(X, y, depth=0)`** — recursive routine: if the node is pure (or you add a base case), return a leaf; otherwise use `find_best_split`, partition `X` and `y` into left/right buckets, and recurse on each child.
