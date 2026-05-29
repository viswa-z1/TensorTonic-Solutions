import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.

    Parameters
    ----------
    query_tokens : list[str]
    docs         : list[list[str]]
    k1           : float
    b            : float

    Returns
    -------
    np.ndarray of shape (len(docs),)
    """

    # Empty corpus
    if len(docs) == 0:
        return np.array([], dtype=float)

    N = len(docs)

    # Document lengths
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = np.mean(doc_lengths) if N > 0 else 0.0

    # Document frequency
    df = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    # Deduplicate query terms while preserving order
    query_terms = list(dict.fromkeys(query_tokens))

    # Precompute IDF values
    idf = {}
    for term in query_terms:
        if term in df:
            idf[term] = math.log(
                (N - df[term] + 0.5) /
                (df[term] + 0.5) + 1.0
            )
        else:
            idf[term] = 0.0

    scores = np.zeros(N, dtype=float)

    # BM25 scoring
    for i, doc in enumerate(docs):

        tf = Counter(doc)
        dl = len(doc)

        if avgdl == 0:
            continue

        norm = k1 * (1.0 - b + b * dl / avgdl)

        score = 0.0

        for term in query_terms:

            freq = tf.get(term, 0)

            if freq == 0:
                continue

            score += (
                idf[term]
                * freq * (k1 + 1.0)
                / (freq + norm)
            )

        scores[i] = score

    return scores