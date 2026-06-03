import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.

    Parameters:
        Z1 : array-like, shape (N, D)
        Z2 : array-like, shape (N, D)
        temperature : float > 0

    Returns:
        float
    """

    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    if Z1.ndim != 2 or Z2.ndim != 2:
        raise ValueError("Z1 and Z2 must be 2D arrays")

    if Z1.shape != Z2.shape:
        raise ValueError("Z1 and Z2 must have the same shape")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # Similarity matrix
    S = (Z1 @ Z2.T) / temperature

    # Numerically stable softmax
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max

    exp_S = np.exp(S_stable)

    # log(sum(exp(.)))
    log_denom = np.log(np.sum(exp_S, axis=1))

    # Positive pair logits (diagonal)
    pos_logits = np.diag(S_stable)

    # InfoNCE loss per sample
    losses = -(pos_logits - log_denom)

    return float(np.mean(losses))