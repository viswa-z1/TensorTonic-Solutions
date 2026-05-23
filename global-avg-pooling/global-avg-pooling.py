import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dimensions.

    Supports:
        (C, H, W)     -> (C,)
        (N, C, H, W)  -> (N, C)

    Returns:
        Float NumPy array
    """

    x = np.asarray(x, dtype=float)

    # (C, H, W)
    if x.ndim == 3:
        return np.mean(x, axis=(1, 2))

    # (N, C, H, W)
    elif x.ndim == 4:
        return np.mean(x, axis=(2, 3))

    else:
        raise ValueError(
            "Input must have shape (C,H,W) or (N,C,H,W)"
        )