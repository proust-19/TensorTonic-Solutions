import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    y = x.shape
    h = y[-2]
    w = y[-1]

    if len(y) == 3:
        c = y[0]
        out = np.zeros(c, dtype=x.dtype)
        for i in range(c):
            s = 0.0
            for hi in range(h):
                for wi in range(w):
                    s += x[i, hi, wi]
            out[i] = s / (h * w)
        return out
    elif len(y) == 4:
        n = y[0]
        c = y[1]
        out = np.zeros((n,c), dtype=x.dtype)
        for ni in range(n):
            for i in range(c):
                s = 0.0
                for hi in range(h):
                    for wi in range(w):
                        s += x[ni, i, hi, wi]
                out[ni, i] = s / (h * w)
        return out
    else:
        raise ValueError("Input must have shape (C,H,W) or (N,C,H,W)")