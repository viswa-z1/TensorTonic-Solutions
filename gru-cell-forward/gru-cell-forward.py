import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)

    if a.ndim == 1:
        return a.reshape(1, feat), True

    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    GRU forward pass for one time step.

    Supports:
        x: (D,) or (N,D)
        h_prev: (H,) or (N,H)

    Returns:
        h_t with same batch structure as input
    """

    # Infer feature sizes
    D = params["Wz"].shape[0]
    H = params["Wz"].shape[1]

    # Convert to 2D
    x, x_was_1d = _as2d(x, D)
    h_prev, _ = _as2d(h_prev, H)

    # Parameters
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]

    # Update gate
    z_t = _sigmoid(x @ Wz + h_prev @ Uz + bz)

    # Reset gate
    r_t = _sigmoid(x @ Wr + h_prev @ Ur + br)

    # Candidate hidden state
    h_candidate = np.tanh(
        x @ Wh + (r_t * h_prev) @ Uh + bh
    )

    # Final hidden state
    h_t = (1.0 - z_t) * h_prev + z_t * h_candidate

    # Restore original shape if input was 1D
    if x_was_1d:
        return h_t[0]

    return h_t