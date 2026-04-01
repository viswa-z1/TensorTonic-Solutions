import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Handle empty input
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=int)
    
    # Determine max length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0
    
    N = len(seqs)
    
    # Initialize output array with pad_value
    out = np.full((N, max_len), pad_value, dtype=int)
    
    # Fill values (truncate if needed)
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        if length > 0:
            out[i, :length] = seq[:length]
    
    return out
    pass