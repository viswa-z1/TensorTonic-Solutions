def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration.

    Parameters:
        values       : current value estimates
        transitions  : transition probabilities
        rewards      : immediate rewards
        gamma        : discount factor

    Returns:
        Updated value list
    """

    num_states = len(values)
    new_values = []

    # Loop over states
    for s in range(num_states):

        best = float('-inf')

        # Loop over actions
        for a in range(len(transitions[s])):

            # Immediate reward
            q = rewards[s][a]

            # Expected future value
            future = 0.0

            for s_next in range(num_states):
                future += transitions[s][a][s_next] * values[s_next]

            q += gamma * future

            # Take max over actions
            if q > best:
                best = q

        new_values.append(float(best))

    return new_values