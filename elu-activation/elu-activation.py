import math

def elu(x, alpha):
    """
    Apply ELU activation to each element in the list x.
    
    Args:
        x (list of float): Input values.
        alpha (float): ELU alpha parameter (>= 0).
    
    Returns:
        list of float: ELU-activated values.
    """
    result = []
    for v in x:
        if v > 0:
            result.append(v)
        else:
            result.append(alpha * (math.exp(v) - 1))
    return result
