def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    
    Args:
        image (list of list of int): Binary image (0s and 1s).
        kernel (list of list of int): Binary structuring element (0s and 1s).
        operation (str): Either "erode" or "dilate".
    
    Returns:
        list of list of int: Processed binary image with the same dimensions as the input.
    """
    # Dimensions of the image and kernel
    image_h, image_w = len(image), len(image[0])
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # Zero-padding the image
    padded_image = [[0] * (image_w + 2 * pad_w) for _ in range(image_h + 2 * pad_h)]
    for i in range(image_h):
        for j in range(image_w):
            padded_image[i + pad_h][j + pad_w] = image[i][j]

    # Output image
    output = [[0] * image_w for _ in range(image_h)]

    # Perform the morphological operation
    for i in range(image_h):
        for j in range(image_w):
            # Extract the region of interest (ROI) from the padded image
            roi = [
                [padded_image[i + di][j + dj] for dj in range(kernel_w)]
                for di in range(kernel_h)
            ]

            if operation == "erode":
                # Erosion: Check if all kernel-1 positions match image-1 positions
                match = all(
                    roi[di][dj] == 1
                    for di in range(kernel_h)
                    for dj in range(kernel_w)
                    if kernel[di][dj] == 1
                )
                output[i][j] = 1 if match else 0

            elif operation == "dilate":
                # Dilation: Check if any kernel-1 position matches an image-1 position
                match = any(
                    roi[di][dj] == 1
                    for di in range(kernel_h)
                    for dj in range(kernel_w)
                    if kernel[di][dj] == 1
                )
                output[i][j] = 1 if match else 0

    return output
