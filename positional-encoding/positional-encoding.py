import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos_encod = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(d_model):
            if i%2 == 0:
                l = pow(base, i / d_model)
                pos_encod[pos][i] = np.sin(pos/l)
            else:
                l = pow(base, (i-1) / d_model)
                pos_encod[pos][i] = np.cos(pos/l)

    return pos_encod   