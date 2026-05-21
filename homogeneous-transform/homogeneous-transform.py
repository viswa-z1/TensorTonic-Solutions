import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).

    Parameters:
        T      : shape (4,4)
        points : shape (3,) or (N,3)

    Returns:
        Transformed point(s) with same batch structure.
    """

    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    # Detect single point input
    single_point = (points.ndim == 1)

    if single_point:
        points = points.reshape(1, 3)

    # Append homogeneous coordinate = 1
    ones = np.ones((points.shape[0], 1), dtype=float)
    points_h = np.hstack([points, ones])

    # Apply transform
    transformed_h = (T @ points_h.T).T

    # Drop homogeneous coordinate
    transformed = transformed_h[:, :3]

    # Return original shape
    if single_point:
        return transformed[0]

    return transformed