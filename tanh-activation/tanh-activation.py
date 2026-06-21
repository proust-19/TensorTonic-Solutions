import numpy as np

def tanh(x):
    """
    Implement the tanh activation function.
    """

    x = np.asarray(x, dtype=float)
    return np.tanh(x)