import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    
    n = y_pred.shape[0]
    tru = y_pred[np.arange(n), y_true]

    loss = - np.mean(np.log(tru))

    return float(loss)