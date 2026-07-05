import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    V = np.asarray(V, dtype=float)
    V_new = V.copy()

    # TD error
    td_error = r + gamma * V[s_next] - V[s]

    # TD(0) update
    V_new[s] += alpha * td_error

    return V_new