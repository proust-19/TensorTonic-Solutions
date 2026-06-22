import numpy as np

def forward(X, W1, b1, W2, b2):
    # First layer
    z1 = np.matmul(W1, X) + b1  # Linear transformation
    a1 = np.maximum(0, z1)  # ReLU activation

    # Second layer
    z2 = np.matmul(W2, a1) + b2  # Linear transformation
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation

    cache = [z1, a1, z2, a2]
    return a2, cache

np.random.seed(42)
X = np.random.randn(3, 5)   # 3 features, 5 samples
W1 = np.random.randn(4, 3)
b1 = np.zeros((4, 1))
W2 = np.random.randn(1, 4)
b2 = np.zeros((1, 1))

A2, cache = forward(X, W1, b1, W2, b2)
print(A2.shape)