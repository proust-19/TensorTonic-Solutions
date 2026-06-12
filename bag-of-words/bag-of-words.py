import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    out = []
    for voc in vocab:
        out.append(tokens.count(voc))
        
    return np.array(out, dtype=int)