import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    q_values = np.asarray(q_values)

    n_actions = q_values.shape[0]

    if rng is not None:
        if rng.random() < epsilon:
            return int(rng.integers(n_actions))
        else:
            return int(np.argmax(q_values))
    else:
        if np.random.random() < epsilon:
            return int(np.random.randint(n_actions))
        else:
            return int(np.argmax(q_values))