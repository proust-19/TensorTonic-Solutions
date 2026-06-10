import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos_encod = np.zeros((seq_length, d_model))
    
    pos = np.arange(seq_length).reshape(-1, 1)

    div_t = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pos_encod[:, 0::2] = np.sin(pos*div_t)
    pos_encod[:, 1::2] = np.cos(pos*div_t)

    return pos_encod