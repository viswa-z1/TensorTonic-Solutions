def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    seqs = [list(s.encode("utf-8")) for s in corpus]
    token_bytes = {i: bytes([i]) for i in range(256)}

    vocab = []
    merges = []
    next_id = 256

    def count_pairs():
        counts = {}
        for seq in seqs:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    while next_id < vocab_size:
        counts = count_pairs()
        if not counts:
            break
        # Most frequent; ties -> lexicographically greater (left_bytes, right_bytes)
        best = max(
            counts,
            key=lambda p: (
                counts[p],
                (token_bytes[p[0]], token_bytes[p[1]]),
            ),
        )
        left, right = best
        new_bytes = token_bytes[left] + token_bytes[right]

        for idx in range(len(seqs)):
            seq = seqs[idx]
            out = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
                    out.append(next_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seqs[idx] = out

        token_bytes[next_id] = new_bytes
        vocab.append([next_id, list(new_bytes)])
        merges.append([left, right, next_id])
        next_id += 1

    return {"vocab": vocab, "merges": merges}