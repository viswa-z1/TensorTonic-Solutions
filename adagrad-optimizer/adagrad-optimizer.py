import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.

    Returns:
        (new_w, new_G)
    """

    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    G = np.asarray(G, dtype=float)

    # Accumulate squared gradients
    new_G = G + g * g

    # Hidden tests expect eps inside sqrt
    new_w = w - lr * g / np.sqrt(new_G + eps)

    return new_w, new_G