import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).

    Parameters
    ----------
    A : np.ndarray of shape (N, N)
        Square matrix.

    Returns
    -------
    scalar
        Sum of diagonal elements.
    """
    A = np.asarray(A)

    n, m = A.shape
    if n != m:
        raise ValueError("Input matrix must be square")

    trace = 0
    for i in range(n):
        trace += A[i, i]

    return trace