import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.array(A, dtype=float)
    
    a, b = A.shape
    det = np.linalg.det(A)

    if a != b or det == 0:
        return None

    A_inv = np.linalg.inv(A)

    return A_inv  
