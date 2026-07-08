import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.asarray(v, dtype=float)

    if v.ndim == 1:
        norm = np.sqrt(np.sum(v ** 2))
        if norm > 1e-10:
            return v / norm
        return np.zeros_like(v)

    elif v.ndim == 2:
        norms = np.sqrt(np.sum(v ** 2, axis=1, keepdims=True))
        return np.where(norms > 1e-10, v / norms, 0.0)

    else:
        raise ValueError("Input must be a 3D vector or a batch of 3D vectors")