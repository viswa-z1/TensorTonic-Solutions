import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    bow = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        if token in word_to_idx:
            bow[word_to_idx[token]] += 1

    return bow