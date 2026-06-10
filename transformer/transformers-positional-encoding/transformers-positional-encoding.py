import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos_encod = np.zeros((seq_length, d_model))
    
    for pos in range(seq_length):
        for i in range(d_model):
            if i%2 == 0:
                l = pow(10000, i / d_model)
                pos_encod[pos][i] = np.sin(pos/l)
            else:
                l = pow(10000, (i-1) / d_model)
                pos_encod[pos][i] = np.cos(pos/l)

    return pos_encod