import numpy as np

def nadam_step(w, m, v, grad,
               lr=0.002,
               beta1=0.9,
               beta2=0.999,
               eps=1e-8):
    """
    Perform one Nadam update step.

    Returns:
        (w_new, m_new, v_new)
    """

    # Convert inputs to numpy arrays
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Step 1: Update first moment
    m_new = beta1 * m + (1.0 - beta1) * grad

    # Step 2: Update second moment
    v_new = beta2 * v + (1.0 - beta2) * (grad ** 2)

    # Step 3: Nesterov-adjusted momentum term
    nesterov = beta1 * m_new + (1.0 - beta1) * grad

    # Parameter update
    w_new = w - lr * nesterov / (np.sqrt(v_new) + eps)

    return w_new, m_new, v_new