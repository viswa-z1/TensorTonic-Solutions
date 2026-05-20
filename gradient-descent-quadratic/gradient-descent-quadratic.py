def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Minimize f(x) = ax^2 + bx + c using gradient descent.

    Returns:
        Final x value after 'steps' iterations.
    """

    x = float(x0)

    for _ in range(steps):
        # Compute gradient
        grad = 2 * a * x + b

        # Gradient descent update
        x = x - lr * grad

    return float(x)