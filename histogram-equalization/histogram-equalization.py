def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    
    Args:
        image (list of list of int): 2D grayscale image with pixel values in [0, 255].
    
    Returns:
        list of list of int: Transformed image after histogram equalization.
    """
    # Step 1: Compute the histogram
    hist = [0] * 256
    for row in image:
        for pixel in row:
            hist[pixel] += 1

    # Step 2: Compute the cumulative distribution function (CDF)
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Step 3: Find cdf_min (smallest non-zero value in the CDF)
    cdf_min = next(c for c in cdf if c > 0)

    # Step 4: Total number of pixels
    total_pixels = len(image) * len(image[0])

    # Step 5: Compute the new pixel values using the equalization formula
    equalized_map = [0] * 256
    for i in range(256):
        equalized_map[i] = round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255) if total_pixels != cdf_min else 0

    # Step 6: Apply the new pixel values to the image
    equalized_image = [[equalized_map[pixel] for pixel in row] for row in image]

    return equalized_image
