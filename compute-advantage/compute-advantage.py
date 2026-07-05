import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    states = np.asarray(states, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    V = np.asarray(V, dtype=float)

    T = len(rewards)

    # Compute discounted returns
    returns = np.zeros(T, dtype=float)
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Advantage = Return - Value(state)
    advantages = returns - V[states]

    return advantages