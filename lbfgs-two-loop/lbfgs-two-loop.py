def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    
    Args:
        grad (list of float): Current gradient vector.
        s_list (list of list of float): List of past step differences (s_i = x_{i+1} - x_i).
        y_list (list of list of float): List of past gradient differences (y_i = g_{i+1} - g_i).
    
    Returns:
        list of float: The descent direction (negative search direction).
    """
    m = len(s_list)  # Number of history pairs
    q = grad[:]  # Start with a copy of the gradient
    alpha = [0] * m  # To store alpha values
    rho = [0] * m  # To store rho values

    # First loop: Backward pass
    for i in range(m - 1, -1, -1):  # From m-1 to 0
        rho[i] = 1.0 / _dot(y_list[i], s_list[i])  # Compute rho_i
        alpha[i] = rho[i] * _dot(s_list[i], q)  # Compute alpha_i
        q = [q_j - alpha[i] * y_j for q_j, y_j in zip(q, y_list[i])]  # Update q

    # Initial scaling of the Hessian approximation
    y_last = y_list[-1]
    s_last = s_list[-1]
    gamma = _dot(s_last, y_last) / _dot(y_last, y_last)  # Scaling factor
    r = [gamma * q_j for q_j in q]  # Initial r = gamma * q

    # Second loop: Forward pass
    for i in range(m):
        beta = rho[i] * _dot(y_list[i], r)  # Compute beta
        r = [r_j + s_j * (alpha[i] - beta) for r_j, s_j in zip(r, s_list[i])]  # Update r

    # Return the negated result as the descent direction
    return [-r_j for r_j in r]
