import torch
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Q: [batch, seq_len_q, d_k]
    K: [batch, seq_len_k, d_k]
    V: [batch, seq_len_k, d_v]
    
    Returns:
        [batch, seq_len_q, d_v]
    """
    d_k = Q.size(-1)
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output