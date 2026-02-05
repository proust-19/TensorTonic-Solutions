import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    w = np.zeros(X.shape[1], dtype=np.float32)
    b = 0.0

    for _ in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)

        er = p - y
        dw = np.dot(X.T, er) / len(y)
        db = np.mean(er)

        w -= lr * dw
        b -= lr * db

    return w, b