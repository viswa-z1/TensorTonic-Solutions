import math

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    
    Args:
        image (list of list of int): 2D grayscale image.
        angle_degrees (float): Rotation angle in degrees (counterclockwise).
    
    Returns:
        list of list of int: Rotated image with the same dimensions as the input.
    """
    # Dimensions of the input image
    H, W = len(image), len(image[0])
    
    # Compute the center of the image
    cy, cx = (H - 1) / 2, (W - 1) / 2
    
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)
    
    # Initialize the output image with zeros
    rotated_image = [[0 for _ in range(W)] for _ in range(H)]
    
    # Iterate over each pixel in the output image
    for i in range(H):
        for j in range(W):
            # Compute the offset from the center
            dy, dx = i - cy, j - cx
            
            # Apply the inverse rotation to find the source pixel
            src_y = cy + dy * cos_theta + dx * sin_theta
            src_x = cx - dy * sin_theta + dx * cos_theta
            
            # Round to the nearest integer to find the nearest neighbor
            src_y = round(src_y)
            src_x = round(src_x)
            
            # Check if the source pixel is within bounds
            if 0 <= src_y < H and 0 <= src_x < W:
                rotated_image[i][j] = image[src_y][src_x]
            else:
                # Out-of-bounds pixels are filled with 0
                rotated_image[i][j] = 0
    
    return rotated_image
