import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D vectors
    without using np.linalg.norm().
    """

    # Convert to NumPy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Compute magnitudes manually
    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))

    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Compute cosine similarity
    return float(np.dot(a, b) / (norm_a * norm_b))