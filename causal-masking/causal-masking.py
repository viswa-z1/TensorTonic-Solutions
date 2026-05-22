import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    Apply causal masking to attention scores.

    Parameters:
        scores     : np.ndarray with shape (..., T, T)
        mask_value : value used for masked positions

    Returns:
        Masked scores with same shape
    """

    # Convert to float array
    scores = np.asarray(scores, dtype=float)

    # Sequence length
    T = scores.shape[-1]

    # Create upper-triangular mask (excluding diagonal)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    # Create masked copy
    masked_scores = scores.copy()

    # Apply mask using broadcasting
    masked_scores[..., mask] = mask_value

    return masked_scores