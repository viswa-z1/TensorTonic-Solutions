def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    
    Args:
        image (list of list of int): 2D grayscale image with pixel values in [0, 255].
    
    Returns:
        list of int: Histogram with 256 bins, where each bin counts pixels of that intensity.
    """
    histogram = [0] * 256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return histogram
