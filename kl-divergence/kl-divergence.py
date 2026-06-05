import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).

    Parameters:
        p : array-like, probability distribution
        q : array-like, probability distribution
        eps : float, numerical stability constant

    Returns:
        float
    """

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")

    # Stabilize q to avoid division by zero / log(0)
    q_stable = q + eps

    # Only terms with p > 0 contribute
    mask = p > 0

    kl = np.sum(
        p[mask] * np.log(p[mask] / q_stable[mask])
    )

    return float(kl)