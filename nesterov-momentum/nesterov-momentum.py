import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum (NAG) update step.

    Parameters:
        w        : parameters
        v        : velocity
        grad     : gradient evaluated at look-ahead position
        lr       : learning rate
        momentum : momentum coefficient

    Returns:
        (new_w, new_v)
    """

    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Update velocity
    new_v = momentum * v + lr * grad

    # Update parameters
    new_w = w - new_v

    return new_w, new_v