import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Ensure NumPy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D, dtype=float)
    b = 0.0
    
    # Gradient descent loop
    for _ in range(steps):
        # Linear combination
        z = X @ w + b  # shape (N,)
        
        # Predictions
        y_hat = _sigmoid(z)  # shape (N,)
        
        # Gradients
        error = y_hat - y  # shape (N,)
        
        dw = (1 / N) * (X.T @ error)  # shape (D,)
        db = (1 / N) * np.sum(error)  # scalar
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
    
    return w, float(b)# Write code here
    pass