import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.asarray(p)
    y = np.asarray(y)

    eps = 1e-15
    p = np.clip(p, eps, 1-eps)
    
    fl = (1-p)**gamma * y * np.log(p) + p**gamma * np.log(1-p) * (1-y)
    return np.mean(-fl)