import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution forward pass.
    Valid padding, stride=1.

    Parameters:
        x : input tensor of shape (N, C_in, H, W)
        W : filters of shape (C_out, C_in, KH, KW)
        b : bias of shape (C_out,)

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """

    # Convert to float arrays
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape

    # Output spatial dimensions
    H_out = H - KH + 1
    W_out = W_in - KW + 1

    # Initialize output
    out = np.zeros((N, C_out, H_out, W_out), dtype=float)

    # Convolution
    for i in range(H_out):
        for j in range(W_out):

            # Extract patch
            patch = x[:, :, i:i+KH, j:j+KW]
            # shape: (N, C_in, KH, KW)

            # Compute convolution for all N and C_out
            # Broadcasting:
            # patch -> (N, 1, C_in, KH, KW)
            # W     -> (1, C_out, C_in, KH, KW)
            conv = np.sum(
                patch[:, None, :, :, :] * W[None, :, :, :, :],
                axis=(2, 3, 4)
            )

            out[:, :, i, j] = conv + b

    return out