import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    
    batch, q_len, d_model = Q.shape
    k_len = K.shape[1]
    head_width = d_model // num_heads

    # 1. Project (combine all heads into one model-width matrix multiply)
    Qp = Q @ W_q   # (batch, q_len, d_model)
    Kp = K @ W_k   # (batch, k_len, d_model)
    Vp = V @ W_v   # (batch, k_len, d_model)

    # 2. Split into heads: (batch, seq, heads, head_width)
    Qh = Qp.reshape(batch, q_len, num_heads, head_width).transpose(0, 2, 1, 3)
    Kh = Kp.reshape(batch, k_len, num_heads, head_width).transpose(0, 2, 1, 3)
    Vh = Vp.reshape(batch, k_len, num_heads, head_width).transpose(0, 2, 1, 3)
    # now: (batch, heads, seq, head_width)

    # 3. Scaled dot-product attention per head
    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(head_width)  # (batch, heads, q_len, k_len)
    scores = scores - scores.max(axis=-1, keepdims=True)          # stability
    exp = np.exp(scores)
    weights = exp / exp.sum(axis=-1, keepdims=True)               # softmax over keys

    heads = weights @ Vh                                          # (batch, heads, q_len, head_width)

    # 4. Concatenate heads back to (batch, q_len, d_model)
    heads = heads.transpose(0, 2, 1, 3).reshape(batch, q_len, d_model)

    # 5. Output projection
    return heads @ W_o