import math

def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.

    Parameters:
        image: 2D list of numbers

    Returns:
        2D list of floats containing gradient magnitudes
    """

    rows = len(image)
    cols = len(image[0])

    # Sobel kernels
    Kx = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    Ky = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]

    # Zero-padded image
    padded = [[0] * (cols + 2) for _ in range(rows + 2)]

    for i in range(rows):
        for j in range(cols):
            padded[i + 1][j + 1] = image[i][j]

    # Output image
    output = [[0.0 for _ in range(cols)] for _ in range(rows)]

    # Apply Sobel operator
    for i in range(rows):
        for j in range(cols):

            gx = 0
            gy = 0

            # 3x3 convolution
            for ki in range(3):
                for kj in range(3):

                    pixel = padded[i + ki][j + kj]

                    gx += pixel * Kx[ki][kj]
                    gy += pixel * Ky[ki][kj]

            # Gradient magnitude
            output[i][j] = math.sqrt(gx * gx + gy * gy)

    return output