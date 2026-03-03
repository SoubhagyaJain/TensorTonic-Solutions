import numpy as np
from collections import Counter

def bm25_score(query_tokens, docs, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    BM25 scores for tokenized query vs tokenized documents.

    Accepts:
      query_tokens: array-like of str
      docs: array-like of token lists OR 2D array (n_docs, doc_len)

    Returns:
      np.ndarray of shape (n_docs,), dtype float64
    """
    if docs is None:
        return np.zeros((0,), dtype=np.float64)

    docs_np = np.asarray(docs, dtype=object)

    # Determine how many documents and how to iterate them
    if docs_np.size == 0:
        return np.zeros((0,), dtype=np.float64)

    if docs_np.ndim == 1:
        # Each element is a document (likely a list of tokens)
        n_docs = docs_np.shape[0]
        def get_doc(i):
            d = docs_np[i]
            if d is None:
                return []
            # If it's already a list/tuple/np array of tokens
            return list(d) if not isinstance(d, (str, bytes)) else [d]
    elif docs_np.ndim == 2:
        # Each row is a document
        n_docs = docs_np.shape[0]
        def get_doc(i):
            return list(docs_np[i])
    else:
        raise ValueError("docs must be 1D (list of docs) or 2D (n_docs, doc_len)")

    # Query terms (dedupe, preserve order)
    if query_tokens is None:
        q_list = []
    else:
        q_list = np.asarray(query_tokens, dtype=object).ravel().tolist()

    if len(q_list) == 0:
        return np.zeros((n_docs,), dtype=np.float64)

    q_unique = []
    seen = set()
    for t in q_list:
        ts = str(t)
        if ts not in seen:
            seen.add(ts)
            q_unique.append(ts)

    q_set = set(q_unique)

    # Build TF per doc and DF for query terms
    doc_lens = np.zeros(n_docs, dtype=np.float64)
    tfs = [None] * n_docs
    df = {t: 0 for t in q_set}

    for i in range(n_docs):
        raw_tokens = get_doc(i)
        tokens = [str(x) for x in raw_tokens if x is not None]  # normalize to str
        doc_lens[i] = float(len(tokens))

        c = Counter(tokens)
        tfs[i] = c

        # DF counts (only for query terms)
        for t in q_set:
            if c.get(t, 0) > 0:
                df[t] += 1

    avgdl = float(doc_lens.mean()) if n_docs > 0 else 0.0
    if avgdl == 0.0:
        return np.zeros((n_docs,), dtype=np.float64)

    norm = k1 * (1.0 - b + b * (doc_lens / avgdl))
    scores = np.zeros((n_docs,), dtype=np.float64)

    for t in q_unique:
        dft = df.get(t, 0)
        if dft == 0:
            continue  # idf treated as 0
        idf = np.log((n_docs - dft + 0.5) / (dft + 0.5) + 1.0)

        for i in range(n_docs):
            tf = tfs[i].get(t, 0)
            if tf == 0:
                continue
            tf = float(tf)
            scores[i] += idf * (tf * (k1 + 1.0)) / (tf + norm[i])

    return scores