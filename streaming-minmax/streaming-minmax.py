import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    return {
        "min": np.full(D, np.inf, dtype=float),
        "max": np.full(D, -np.inf, dtype=float),
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X_batch = np.asarray(X_batch, dtype=float)

    if X_batch.ndim == 1:
        X_batch = X_batch.reshape(-1, 1)

    # Batch statistics
    batch_min = np.min(X_batch, axis=0)
    batch_max = np.max(X_batch, axis=0)

    # Update running statistics
    state["min"] = np.minimum(state["min"], batch_min)
    state["max"] = np.maximum(state["max"], batch_max)

    # Normalize using UPDATED statistics
    denom = np.maximum(state["max"] - state["min"], eps)
    X_norm = (X_batch - state["min"]) / denom

    return X_norm