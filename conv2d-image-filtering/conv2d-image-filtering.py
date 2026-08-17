def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.

    Args:
        image (list of list of float): Input 2D image.
        kernel (list of list of float): 2D convolution kernel.
        stride (int): Stride of the convolution.
        padding (int): Amount of zero-padding on all sides.

    Returns:
        list of list of float: The result of the convolution.
    """
    H = len(image)
    W = len(image[0])
    k_h = len(kernel)
    k_w = len(kernel[0])

    # Step 1: Zero-padding
    padded_H = H + 2 * padding
    padded_W = W + 2 * padding
    padded = [[0.0 for _ in range(padded_W)] for _ in range(padded_H)]
    for i in range(H):
        for j in range(W):
            padded[i + padding][j + padding] = float(image[i][j])

    # Step 2: Output dimensions
    out_H = ((padded_H - k_h) // stride) + 1
    out_W = ((padded_W - k_w) // stride) + 1
    output = [[0.0 for _ in range(out_W)] for _ in range(out_H)]

    # Step 3: Convolution
    for i in range(out_H):
        for j in range(out_W):
            acc = 0.0
            for m in range(k_h):
                for n in range(k_w):
                    acc += padded[i * stride + m][j * stride + n] * kernel[m][n]
            output[i][j] = acc

    return output
