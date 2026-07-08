from collections import Counter

def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Build vocabulary
    vocab = sorted(set(tokens))
    V = len(vocab)

    # Count bigrams
    counts = Counter()
    for i in range(len(tokens) - 1):
        counts[(tokens[i], tokens[i + 1])] += 1

    # Count outgoing bigrams for each context word
    context_counts = Counter()
    for (w1, w2), c in counts.items():
        context_counts[w1] += c

    # Compute add-1 smoothed probabilities
    probs = {}
    for w1 in vocab:
        denom = context_counts.get(w1, 0) + V
        for w2 in vocab:
            c = counts.get((w1, w2), 0)
            probs[(w1, w2)] = (c + 1) / denom

    return dict(counts), probs