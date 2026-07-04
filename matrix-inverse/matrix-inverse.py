import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I.
    Returns None for non-square or singular matrices.
    """
    A = np.asarray(A, dtype=float)

    # Validate input
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    # Check for singularity
    if abs(np.linalg.det(A)) < 1e-10:
        return None

    return np.linalg.inv(A)