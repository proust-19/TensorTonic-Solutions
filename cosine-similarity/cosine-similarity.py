import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    mod_a = np.linalg.norm(a)
    mod_b = np.linalg.norm(b)
    if mod_a == 0 or mod_b == 0:
        return 0
    cosine = np.dot(a, b) / (mod_a * mod_b)
    return cosine