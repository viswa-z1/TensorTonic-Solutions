import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.

    Returns:
        np.ndarray of eigenvalues (possibly complex)
        or None for invalid/non-square input.
    """

    try:
        # Convert input
        matrix = np.asarray(matrix, dtype=float)

        # Must be 2D
        if matrix.ndim != 2:
            return None

        rows, cols = matrix.shape

        # Must be square
        if rows != cols:
            return None

        # Handle empty matrix
        if rows == 0:
            return np.array([], dtype=float)

        # Compute eigenvalues
        eigvals = np.linalg.eigvals(matrix)

        # Sort by real part, then imaginary part
        idx = np.lexsort((eigvals.imag, eigvals.real))
        eigvals = eigvals[idx]

        return eigvals

    except Exception:
        return None