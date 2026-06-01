import numpy as np

def adamw_step(
    w,
    m,
    v,
    grad,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    eps=1e-8
):
    """
    Perform one AdamW update step.

    Returns:
        (new_w, new_m, new_v)
    """

    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # First moment
    new_m = beta1 * m + (1.0 - beta1) * grad

    # Second moment
    new_v = beta2 * v + (1.0 - beta2) * (grad ** 2)

    # Decoupled weight decay + adaptive update
    new_w = (
        w
        - lr * weight_decay * w
        - lr * new_m / (np.sqrt(new_v) + eps)
    )

    return new_w, new_m, new_v