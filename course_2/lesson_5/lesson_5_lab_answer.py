import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Node :
    is_leaf: bool
    left_node: Optional["Node"] = None
    right_node: Optional["Node"] = None
    prediction: Optional[int] = None
    feature_index: Optional[int] = None
    threshold: Optional[float] = None

def calculate_entropy(y):
    y_length = len(y)
    
    if y_length <= 0:
        return 0

    p1 = np.sum(y == 1)/y_length
    p0 = 1.0 - p1

    if p1 == 0:
        return 0

    if p0 == 0:
        return 0

    entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
    return entropy

def calculated_weighted_entropy(y_left, y_right, y_length):
    if y_length == 0:
        return 0

    pl = len(y_left)/y_length
    pr = len(y_right)/y_length

    left_entropy = calculate_entropy(y_left)
    right_entropy = calculate_entropy(y_right)
    w_entropy = pl * left_entropy + pr * right_entropy 
    return w_entropy

def find_best_split(X,y):
    best_feature_index = None
    best_threshold = None
    best_information_gain = -1

    ye = calculate_entropy(y)
    y_len = len(y)
    _, feature_count = X.shape
 
    for feature_index in range(feature_count):
        unique_thresholds = np.unique(X[:,feature_index])
        
        for threshold in unique_thresholds:
            lm = X[:,feature_index] >= threshold
            rm = ~lm

            yl = y[lm]
            yr = y[rm]

            if len(yl) == 0:
                continue

            if len(yr) == 0:
                continue

            we = calculated_weighted_entropy(yl, yr, y_len)
            ig = ye - we 
            if ig > best_information_gain:
                best_information_gain = ig
                best_feature_index = feature_index
                best_threshold = threshold

    return best_information_gain, best_feature_index, best_threshold

def split_tree(X, feature_index, threshold,y) -> tuple[Node, Node]:
    left_mask = X[:, feature_index] >= threshold
    right_mask = ~left_mask
    left_node = build_tree(X[left_mask], y[left_mask])
    right_node = build_tree(X[right_mask], y[right_mask])

    return left_node, right_node
    
def build_tree(X,y) -> Node:
    y_entropy = calculate_entropy(y)

    if y_entropy == 0:
        prediction = y[0]
        return Node(is_leaf=True, prediction=prediction)

    big, bfi, bt = find_best_split(X,y)

    if big == 0 or bfi is None:
        prediction = int(np.bincount(y).argmax())  # Majority vote
        return Node(is_leaf=True, prediction=prediction)
    
    left_node, right_node = split_tree(X, bfi, bt, y)
    n = Node(
        is_leaf=False,
        left_node=left_node, 
        right_node=right_node, 
        feature_index=bfi,
        threshold=float(bt),
        )
    
    return n

def predict_one(x: np.ndarray, node: Node) -> int:
    """Classify a single feature vector (1D) of shape (n_features,)."""
    if node.is_leaf:
        return int(node.prediction)
    assert node.feature_index is not None and node.threshold is not None
    if x[node.feature_index] >= node.threshold:
        return predict_one(x, node.left_node)
    return predict_one(x, node.right_node)


def predict(X: np.ndarray, root: Node) -> np.ndarray:
    """Classify one row (1D) or a batch of rows (2D). Returns integer class labels."""
    if X.ndim == 1:
        return np.array(predict_one(X, root))
    return np.array([predict_one(row, root) for row in X])


def print_tree(node: Node, depth: int = 0) -> None:
    """Print a compact view of the learned tree (no training-data statistics)."""
    pad = "  " * depth
    if node.is_leaf:
        print(f"{pad}predict class {node.prediction}")
        return
    print(f"{pad}if x[{node.feature_index}] >= {node.threshold}:")
    print(f"{pad}  true branch:")
    print_tree(node.left_node, depth + 1)
    print(f"{pad}  false branch:")
    print_tree(node.right_node, depth + 1)

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