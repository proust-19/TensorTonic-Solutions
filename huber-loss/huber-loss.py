import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    abs_e = np.abs(y_pred-y_true)
    
    Loss = np.where(abs_e <= delta,
            np.square(abs_e)/2,
            delta*(abs_e - 0.5*delta)
            )
    return np.mean(Loss)
    