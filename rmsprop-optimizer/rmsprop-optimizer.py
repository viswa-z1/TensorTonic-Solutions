import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp optimization step.

    Parameters:
        w    : parameters
        g    : gradients
        s    : running squared gradient accumulator
        lr   : learning rate
        beta : decay factor
        eps  : numerical stability term

    Returns:
        (w_new, s_new)
    """

    # Convert inputs to numpy arrays
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    # Update running average of squared gradients
    s_new = beta * s + (1 - beta) * (g ** 2)

    # Parameter update
    w_new = w - lr * g / (np.sqrt(s_new) + eps)

    return w_new, s_new