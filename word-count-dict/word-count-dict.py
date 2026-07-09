def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    counts = {}

    for sentence in sentences:
        for word in sentence:
            counts[word] = counts.get(word, 0) + 1

    return counts