import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from text documents.

    Parameters:
        documents : list[str]

    Returns:
        (tfidf_matrix, vocabulary)
    """

    # Handle empty corpus
    if len(documents) == 0:
        return np.zeros((0, 0), dtype=float), []

    # Tokenize documents
    tokenized_docs = [
        doc.lower().split()
        for doc in documents
    ]

    # Build vocabulary
    vocab = sorted(set(
        token
        for doc in tokenized_docs
        for token in doc
    ))

    vocab_size = len(vocab)
    n_docs = len(documents)

    # Handle empty vocabulary
    if vocab_size == 0:
        return np.zeros((n_docs, 0), dtype=float), []

    # Word -> index mapping
    word_to_idx = {
        word: idx
        for idx, word in enumerate(vocab)
    }

    # Document frequency
    df = Counter()

    for doc in tokenized_docs:
        unique_terms = set(doc)

        for term in unique_terms:
            df[term] += 1

    # Compute IDF
    idf = {}

    for term in vocab:
        idf[term] = math.log(n_docs / df[term])

    # Initialize TF-IDF matrix
    tfidf = np.zeros((n_docs, vocab_size), dtype=float)

    # Fill matrix
    for doc_idx, doc in enumerate(tokenized_docs):

        if len(doc) == 0:
            continue

        term_counts = Counter(doc)
        total_terms = len(doc)

        for term, count in term_counts.items():

            tf = count / total_terms

            tfidf[doc_idx, word_to_idx[term]] = (
                tf * idf[term]
            )

    return tfidf, vocab