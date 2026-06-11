import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    q_w, k_w, v_w = np.dot(Q, W_q), np.dot(K, W_k), np.dot(V, W_v)

    Q_head = q_w.reshape(batch_size, seq_len, num_heads,  d_k).transpose(0, 2, 1, 3)
    K_head = k_w.reshape(batch_size, seq_len, num_heads,  d_k).transpose(0, 2, 1, 3)
    V_head = v_w.reshape(batch_size, seq_len, num_heads,  d_k).transpose(0, 2, 1, 3)

    score = np.matmul(Q_head, K_head.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    cont_head = np.matmul(softmax(score, axis=-1), V_head)

    context = cont_head.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    output = np.dot(context, W_o)
    return output