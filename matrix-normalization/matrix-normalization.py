import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a matrix using L1, L2, or Max norm.

    Parameters:
        matrix    : list or np.ndarray
        axis      : axis for normalization
        norm_type : 'l1', 'l2', or 'max'

    Returns:
        Normalized NumPy array
        OR None for invalid inputs
    """

    try:
        # Convert to float array
        matrix = np.asarray(matrix, dtype=float)

        # Must be 2D
        if matrix.ndim != 2:
            return None

        # Compute norms
        if norm_type == 'l2':

            norms = np.sqrt(
                np.sum(matrix ** 2,
                       axis=axis,
                       keepdims=True)
            )

        elif norm_type == 'l1':

            norms = np.sum(
                np.abs(matrix),
                axis=axis,
                keepdims=True
            )

        elif norm_type == 'max':

            norms = np.max(
                np.abs(matrix),
                axis=axis,
                keepdims=True
            )

        else:
            return None

        # Avoid divide-by-zero
        norms = np.where(norms == 0, 1.0, norms)

        # Normalize
        normalized = matrix / norms

        return normalized

    except Exception:
        return None