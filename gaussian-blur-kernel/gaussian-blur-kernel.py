import math

def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian kernel.
    
    Args:
        size (int): Odd positive integer, kernel size.
        sigma (float): Standard deviation of the Gaussian.
    
    Returns:
        list of list of float: 2D Gaussian kernel of dimensions size x size.
    """
    center = size // 2
    kernel = []
    sum_val = 0.0
    
    for i in range(size):
        row = []
        for j in range(size):
            x = j - center
            y = i - center
            # Gaussian function (unnormalized)
            val = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            row.append(val)
            sum_val += val
        kernel.append(row)
    
    # Normalize so that sum of all elements is 1
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= sum_val
    
    return kernel
