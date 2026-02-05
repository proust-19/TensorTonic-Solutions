import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    n, m = np.shape(A)
    T = np.zeros((m,n))
    
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T
