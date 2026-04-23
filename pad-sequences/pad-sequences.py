import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    L = max_len if max_len is not None else max((len(seq) for seq in seqs), default=0)
    
    if N == 0:
        return np.empty((0, 0), dtype=int)
    out = np.full((N, L), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        n = min(len(seq), L)
        out[i, :n] = seq[:n]
    return out