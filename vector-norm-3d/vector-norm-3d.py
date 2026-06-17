import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.asarray(v, dtype=float)

    norm = np.linalg.norm(v, axis = -1)
    if norm.ndim == 0:
        return float(norm)
        
    return norm