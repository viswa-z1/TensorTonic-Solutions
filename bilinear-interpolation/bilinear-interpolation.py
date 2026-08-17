def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.

    Args:
        image (list of list of float): Input 2D grid (image).
        new_h (int): Target height of the resized grid.
        new_w (int): Target width of the resized grid.

    Returns:
        list of list of float: Resized 2D grid.
    """
    # Original dimensions
    H, W = len(image), len(image[0])

    # Handle edge case where new_h or new_w is 1
    if new_h == 1:
        src_y_ratio = 0
    else:
        src_y_ratio = (H - 1) / (new_h - 1)
    
    if new_w == 1:
        src_x_ratio = 0
    else:
        src_x_ratio = (W - 1) / (new_w - 1)

    # Initialize the output grid
    resized = [[0.0 for _ in range(new_w)] for _ in range(new_h)]

    for i in range(new_h):
        for j in range(new_w):
            # Map output pixel (i, j) to source coordinates
            src_y = i * src_y_ratio
            src_x = j * src_x_ratio

            # Integer and fractional parts of the source coordinates
            y0 = int(src_y)
            x0 = int(src_x)
            dy = src_y - y0
            dx = src_x - x0

            # Clamp the neighbor coordinates to stay within bounds
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)

            # Interpolate using the four nearest neighbors
            top_left = image[y0][x0]
            top_right = image[y0][x1]
            bottom_left = image[y1][x0]
            bottom_right = image[y1][x1]

            # Bilinear interpolation formula
            resized[i][j] = (
                top_left * (1 - dy) * (1 - dx) +
                top_right * (1 - dy) * dx +
                bottom_left * dy * (1 - dx) +
                bottom_right * dy * dx
            )

    return resized
