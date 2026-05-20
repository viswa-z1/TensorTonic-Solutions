import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Perform a single forward step of a tanh RNN.

    Parameters:
    x_t    : shape (D,)
    h_prev : shape (H,)
    Wx     : shape (D, H)
    Wh     : shape (H, H)
    b      : shape (H,)

    Returns:
    h_t    : shape (H,)
    """

    # Convert inputs to numpy arrays
    x_t = np.array(x_t, dtype=np.float64)
    h_prev = np.array(h_prev, dtype=np.float64)
    Wx = np.array(Wx, dtype=np.float64)
    Wh = np.array(Wh, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # Compute pre-activation
    pre_act = x_t @ Wx + h_prev @ Wh + b

    # Apply tanh activation
    h_t = np.tanh(pre_act)

    return h_t