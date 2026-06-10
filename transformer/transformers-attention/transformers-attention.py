import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    mat_mul = torch.matmul(Q, K.transpose(-2, -1))
    d_k = K.shape[-1]
    att = F.softmax(mat_mul / math.sqrt(d_k), dim=-1)

    out = torch.matmul(att, V)
    
    return out