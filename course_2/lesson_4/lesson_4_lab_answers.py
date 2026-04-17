import numpy as np

# Expected results (match your implementation against these):
#   Iteration 0 | Cost: 0.6931
#   Iteration 999 | Cost: 0.2036

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(42)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return W1, b1, W2, b2

def train(X, Y, num_iterations, alpha, lambd):
    n_x = X.shape[0] # 2 features
    n_h = 3          # 3 hidden neurons
    n_y = Y.shape[0] # 1 output neuron
    m = X.shape[1]   # 4 examples
    
    W1, b1, W2, b2 = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(num_iterations):
        
        # --- 1. FORWARD PROPAGATION ---
        Z1 = np.dot(W1, X) + b1
        A1 = relu(Z1)
        
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        
        # --- 2. CALCULATE COST (with L2 Regularization) ---
        cross_entropy = -(1/m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
        L2_penalty = (lambd / (2*m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        cost = cross_entropy + L2_penalty
        
        # --- 3. BACKPROPAGATION ---
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T) + (lambd/m) * W2
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Derivative of ReLU gate
        relu_derivative = (Z1 > 0).astype(float)
        
        dZ1 = np.dot(W2.T, dZ2) * relu_derivative
        dW1 = (1/m) * np.dot(dZ1, X.T) + (lambd/m) * W1
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # --- 4. GRADIENT DESCENT UPDATE ---
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        
        # Print progress
        if i == 0 or i == num_iterations - 1:
            print(f"Iteration {i} | Cost: {cost:.4f}")

# Execute the lab
X = np.array([[0.9, 0.8, 0.2, 0.1],
              [0.9, 0.2, 0.8, 0.1]])
Y = np.array([[1, 0, 0, 0]])

train(X, Y, num_iterations=1000, alpha=0.1, lambd=0.05)