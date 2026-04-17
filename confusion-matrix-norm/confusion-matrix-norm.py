import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
        
    C = np.zeros((num_classes, num_classes), dtype=float)
    
    for t,p in zip(y_true, y_pred):
        C[t][p] += 1

    if normalize == 'true':
        row = C.sum(axis=1, keepdims=True)
        C = np.divide(C, row, where=row != 0)  
    elif normalize == 'pred':
        col = C.sum(axis=0, keepdims=True)
        C = np.divide(C, col, where=col != 0)
    elif normalize == 'all':
        total = C.sum()
        if total != 0:
            C = C / total
    elif normalize == 'none':
        C = C
    return C