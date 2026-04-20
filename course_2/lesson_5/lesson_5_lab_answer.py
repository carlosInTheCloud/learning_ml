import numpy as np

# Expected structure on the lab dataset (X, y below):
#   Depth 0: Split on Feature 0 >= 230.0 (IG: 0.4591)
#     Depth 1 (Left): Leaf Node predicting class 1
#     Depth 1 (Right): Leaf Node predicting class 0

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
        
    return -p1 * np.log2(p1) - p0 * np.log2(p0)

def information_gain(y, y_left, y_right):
    """Calculates IG of a parent splitting into two children."""
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)
    
    # Weighted average of the children's entropy
    weighted_entropy = (p_left * calculate_entropy(y_left)) + (p_right * calculate_entropy(y_right))
    
    return calculate_entropy(y) - weighted_entropy

def find_best_split(X, y):
    """Loops through all features and thresholds to find the highest IG."""
    best_ig = -1
    best_feature = None
    best_threshold = None
    
    n_examples, n_features = X.shape
    
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

def build_tree(X, y, depth=0):
    """Recursively builds the tree until a stopping condition is met."""
    # Stopping Condition: The node is perfectly pure (Entropy = 0)
    if calculate_entropy(y) == 0:
        # Predict the class that exists in this pure bucket
        prediction = y[0]
        print(f"{'  ' * depth}Depth {depth}: Leaf Node predicting class {prediction}")
        return
    
    # Find the best mathematical split
    feature, threshold, ig = find_best_split(X, y)
    
    # If no split improves the entropy, stop and vote
    if ig == 0 or feature is None:
        prediction = np.bincount(y).argmax() # Majority vote
        print(f"{'  ' * depth}Depth {depth}: Leaf Node predicting class {prediction}")
        return
        
    print(f"{'  ' * depth}Depth {depth}: Split on Feature {feature} >= {threshold} (IG: {ig:.4f})")
    
    # Execute the split
    left_mask = X[:, feature] >= threshold
    right_mask = ~left_mask
    
    # Recursively build the Left and Right branches
    print(f"{'  ' * depth}  --> Building Left Branch:")
    build_tree(X[left_mask], y[left_mask], depth + 1)
    
    print(f"{'  ' * depth}  --> Building Right Branch:")
    build_tree(X[right_mask], y[right_mask], depth + 1)

# Execute the lab
X = np.array([[190, 85], [210, 90], [250, 75], [180, 80], [205, 95], [230, 85]])
y = np.array([0, 0, 1, 0, 0, 1])

build_tree(X, y)