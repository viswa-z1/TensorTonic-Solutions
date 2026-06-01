import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.

    Parameters:
        g : array-like
            Gradient tensor of any shape
        max_norm : float
            Maximum allowed L2 norm

    Returns:
        np.ndarray with same shape as g
    """

    g = np.asarray(g, dtype=float)

    # Handle invalid threshold
    if max_norm <= 0:
        return g.copy()

    # Global L2 norm
    norm = np.linalg.norm(g)

    # Zero norm or already within limit
    if norm == 0 or norm <= max_norm:
        return g.copy()

    # Scale factor
    scale = max_norm / norm

    return g * scale