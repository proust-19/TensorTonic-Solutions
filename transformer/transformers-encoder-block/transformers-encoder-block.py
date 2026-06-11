import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    miu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    layer = gamma * ((x - miu) / np.sqrt(var + eps)) + beta

    return layer

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    q_w, k_w, v_w = np.dot(Q, W_q), np.dot(K, W_k), np.dot(V, W_v)
    
    Q_head = q_w.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_head = k_w.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_head = v_w.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    score = np.matmul(Q_head, K_head.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    con_head = np.matmul(softmax(score, axis=-1), V_head)

    context = con_head.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    return np.dot(context, W_o)
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    ffn = np.dot(np.maximum(0, np.dot(x, W1) + b1), W2) + b2
    return ffn

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    xf = layer_norm(x + attn_out, gamma1, beta1)

    ffn_out = feed_forward(xf, W1, b1, W2, b2)
    output = layer_norm(xf + ffn_out, gamma2, beta2)
    return output