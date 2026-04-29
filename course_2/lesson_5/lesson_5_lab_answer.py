import numpy as np
from dataclasses import dataclass
from typing import Optional

# Expected structure on the lab dataset (X, y below):
#   Depth 0: Split on Feature 0 >= 230.0 (IG: 0.4591)
#     Depth 1 (Left): Leaf Node predicting class 1
#     Depth 1 (Right): Leaf Node predicting class 0


@dataclass
class TreeNode:
    """Leaf: is_leaf True, prediction set. Internal: feature, threshold, left, right."""

    is_leaf: bool
    prediction: Optional[int] = None
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


def predict_one(x: np.ndarray, node: TreeNode) -> int:
    """Classify a single feature vector (1D) of shape (n_features,)."""
    if node.is_leaf:
        return int(node.prediction)
    assert node.feature is not None and node.threshold is not None
    if x[node.feature] >= node.threshold:
        return predict_one(x, node.left)
    return predict_one(x, node.right)


def predict(X: np.ndarray, root: TreeNode) -> np.ndarray:
    """Classify one row (1D) or a batch of rows (2D). Returns integer class labels."""
    if X.ndim == 1:
        return np.array(predict_one(X, root))
    return np.array([predict_one(row, root) for row in X])


def print_tree(node: TreeNode, depth: int = 0) -> None:
    """Print a compact view of the learned tree (no training-data statistics)."""
    pad = "  " * depth
    if node.is_leaf:
        print(f"{pad}predict class {node.prediction}")
        return
    print(f"{pad}if x[{node.feature}] >= {node.threshold}:")
    print(f"{pad}  true branch:")
    print_tree(node.left, depth + 1)
    print(f"{pad}  false branch:")
    print_tree(node.right, depth + 1)


def calculate_entropy(y):
    """Calculates the Shannon Entropy of a label array."""
    # If the bucket is empty, entropy is 0
    if len(y) == 0:
        return 0.0

    # Calculate probabilities of class 1 and class 0
    p1 = np.sum(y == 1) / len(y)
    p0 = 1.0 - p1
    # Prevent log(0) errors
    if p1 == 0 or p0 == 0:
        return 0.0

    entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
    return entropy

def information_gain(y, y_left, y_right):
    """ Calculates IG of a parent splitting into two children."""
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)

    # Weighted average of the children's entropy
    y_left_entropy = calculate_entropy(y_left)
    y_right_entropy = calculate_entropy(y_right)
    weighted_entropy = (p_left * y_left_entropy) + (p_right * y_right_entropy)
    y_entropy = calculate_entropy(y)
    ig = y_entropy - weighted_entropy
    return ig

def find_best_split(X, y):
    """Loops through all features and thresholds to find the highest IG."""
    best_ig = -1
    best_feature = None
    best_threshold = None
    
    _, n_features = X.shape

    for feature_idx in range(n_features):
        # Get all unique values in this column to test as thresholds
        thresholds = np.unique(X[:, feature_idx])

        for threshold in thresholds:
            # Physically split the data
            left_mask = X[:, feature_idx] >= threshold
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            # Skip if a split puts all data in one bucket
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            ig = information_gain(y, y_left, y_right)
            # If this is the best split we've seen, save it
            if ig > best_ig:
                best_ig = ig
                best_feature = feature_idx
                best_threshold = threshold
    return best_feature, best_threshold, best_ig

def build_tree(X, y) -> TreeNode:
    """Recursively builds the tree until a stopping condition is met. Returns the subtree root."""
    # Stopping Condition: The node is perfectly pure (Entropy = 0)
    entropy = calculate_entropy(y)
    if entropy == 0:
        # Predict the class that exists in this pure bucket
        prediction = int(y[0])
        return TreeNode(is_leaf=True, prediction=prediction)

    # Find the best mathematical split
    feature, threshold, ig = find_best_split(X, y)
    # If no split improves the entropy, stop and vote
    if ig == 0 or feature is None:
        prediction = int(np.bincount(y).argmax())  # Majority vote
        return TreeNode(is_leaf=True, prediction=prediction)

    # Execute the split
    left_mask = X[:, feature] >= threshold
    right_mask = ~left_mask
    # Recursively build the Left and Right branches
    left_child = build_tree(X[left_mask], y[left_mask])
    right_child = build_tree(X[right_mask], y[right_mask])

    return TreeNode(
        is_leaf=False,
        feature=feature,
        threshold=float(threshold),
        left=left_child,
        right=right_child,
    )


# Execute the lab
X = np.array([[190, 85], [210, 90], [250, 75], [180, 80], [205, 95], [230, 85]])
y = np.array([0, 0, 1, 0, 0, 1])
tree = build_tree(X, y)

print("\n--- Learned tree (structure) ---")
print_tree(tree)

# Example predictions: training rows, one novel point per branch, and one in-between
examples = np.array(
    [
        [250, 75],  # class 1 in training — expect left branch
        [180, 80],  # class 0 — expect right
        [240, 80],  # just below root threshold on feature 0 -> right / "no"
        [100, 50],  # far from data — still follow thresholds
    ]
)
print("\n--- Predictions (one row at a time) ---")
for x_row in examples:
    pred = predict(x_row, tree)
    print(f"  x = {x_row!r}  ->  class {int(pred)}")

print("\n--- Same rows via batch predict (2D) ---")
print("  classes:", predict(examples, tree))