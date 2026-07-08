import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    points = np.asarray(points, dtype=float)
    single = (points.ndim == 1)

    if single:
        points = points.reshape(1, 3)

    c = np.cos(theta)
    s = np.sin(theta)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    rotated = np.empty_like(points)
    rotated[:, 0] = x * c - y * s
    rotated[:, 1] = x * s + y * c
    rotated[:, 2] = z

    if single:
        return rotated[0]
    return rotated