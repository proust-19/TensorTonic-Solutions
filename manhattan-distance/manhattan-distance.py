import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x = np.array(x)
    y = np.array(y)

    dis = np.abs(x - y)
    return float(np.sum(dis))