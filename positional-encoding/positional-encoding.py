import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return positional encoding matrix of shape (seq_len, d_model)
    using sinusoidal formulation.

    Odd d_model -> last column is sin.
    """

    # Position indices: shape (seq_len, 1)
    positions = np.arange(seq_len, dtype=float).reshape(-1, 1)

    # Even dimension indices: 0, 2, 4, ...
    div_terms = np.arange(0, d_model, 2, dtype=float)

    # Compute denominator terms
    div_terms = np.power(base, div_terms / d_model)

    # Compute angles
    angles = positions / div_terms  # Broadcasting

    # Initialize output
    pe = np.zeros((seq_len, d_model), dtype=float)

    # Fill even columns with sin
    pe[:, 0::2] = np.sin(angles)

    # Fill odd columns with cos
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe